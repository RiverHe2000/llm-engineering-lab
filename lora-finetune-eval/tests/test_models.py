from __future__ import annotations

import pytest
import torch
from torch import nn

from loraeval.lora import LoRAConfig, count_parameters
from loraeval.models import apply_strategy, load_trainable_state_dict, trainable_state_dict


def test_full_trains_everything(tiny_model: nn.Module) -> None:
    counts = apply_strategy(tiny_model, "full")
    assert counts.trainable == counts.total


def test_head_only(tiny_model: nn.Module) -> None:
    counts = apply_strategy(tiny_model, "head")
    head_params = sum(p.numel() for n, p in tiny_model.named_parameters() if "classifier" in n)
    assert counts.trainable == head_params
    assert all(("classifier" in n) == p.requires_grad for n, p in tiny_model.named_parameters())


def test_lora_budget_is_small(tiny_model: nn.Module) -> None:
    total_before = count_parameters(tiny_model).total
    counts = apply_strategy(tiny_model, "lora", LoRAConfig(r=2))
    # 4 wrapped 32x32 linears, each adds r*(in+out) = 2*64 = 128 params -> 512, plus the head.
    head_params = sum(p.numel() for n, p in tiny_model.named_parameters() if "classifier" in n)
    assert counts.total == total_before + 512
    assert counts.trainable == 512 + head_params
    assert counts.trainable_fraction < 0.2


def test_invalid_strategies(tiny_model: nn.Module) -> None:
    with pytest.raises(ValueError, match="requires a LoRAConfig"):
        apply_strategy(tiny_model, "lora", None)
    with pytest.raises(ValueError, match="unknown strategy"):
        apply_strategy(tiny_model, "adapter")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="matched no parameters"):
        apply_strategy(tiny_model, "head", head_pattern="nothing_here")


def test_trainable_state_dict_round_trip(tiny_model: nn.Module) -> None:
    apply_strategy(tiny_model, "lora", LoRAConfig(r=2))
    state = trainable_state_dict(tiny_model)
    assert set(state) == {n for n, p in tiny_model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for p in tiny_model.parameters():
            if p.requires_grad:
                p.add_(1.0)
    load_trainable_state_dict(tiny_model, state)
    for k, v in trainable_state_dict(tiny_model).items():
        assert torch.equal(v, state[k])
    with pytest.raises(KeyError):
        load_trainable_state_dict(tiny_model, {"bogus": torch.zeros(1)})
