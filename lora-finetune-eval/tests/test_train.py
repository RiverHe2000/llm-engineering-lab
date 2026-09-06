from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from transformers import PreTrainedTokenizerFast

from loraeval.data import LABEL_NAMES, Example, make_loader, stratified_split
from loraeval.lora import LoRAConfig
from loraeval.models import apply_strategy, trainable_state_dict
from loraeval.train import (
    TrainConfig,
    autocast_dtype,
    build_optimizer,
    resolve_device,
    set_seed,
    train_model,
    warmup_linear_decay,
)


class TestSchedule:
    def test_warmup_then_linear_decay(self) -> None:
        assert warmup_linear_decay(0, 100, 10) == pytest.approx(0.1)
        assert warmup_linear_decay(9, 100, 10) == pytest.approx(1.0)
        assert warmup_linear_decay(10, 100, 10) == pytest.approx(1.0)
        assert warmup_linear_decay(55, 100, 10) == pytest.approx(0.5)
        assert warmup_linear_decay(100, 100, 10) == 0.0
        assert warmup_linear_decay(500, 100, 10) == 0.0
        assert warmup_linear_decay(0, 10, 0) == 1.0

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            warmup_linear_decay(-1, 10, 1)
        with pytest.raises(ValueError):
            warmup_linear_decay(0, 0, 0)


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epochs": 0},
            {"lr": 0.0},
            {"weight_decay": -1},
            {"warmup_ratio": 1.0},
            {"precision": "fp16"},
            {"patience": -1},
        ],
    )
    def test_invalid(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)  # type: ignore[arg-type]

    def test_helpers(self) -> None:
        assert resolve_device("cpu").type == "cpu"
        assert autocast_dtype("fp32", torch.device("cpu")) is None
        assert autocast_dtype("bf16", torch.device("cpu")) == torch.bfloat16
        assert autocast_dtype("auto", torch.device("cpu")) is None


def test_optimizer_groups(tiny_model: nn.Module) -> None:
    apply_strategy(tiny_model, "lora", LoRAConfig(r=2))
    opt = build_optimizer(tiny_model, lr=1e-3, weight_decay=0.1)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"]) and all(
        p.dim() < 2 for p in no_decay["params"]
    )
    n = len(decay["params"]) + len(no_decay["params"])
    assert n == sum(1 for p in tiny_model.parameters() if p.requires_grad)
    for p in tiny_model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable"):
        build_optimizer(tiny_model, 1e-3, 0.0)


@pytest.fixture
def loaders(
    fake_tokenizer: PreTrainedTokenizerFast, synthetic_examples: list[Example]
) -> tuple[list[Example], list[Example]]:
    split = stratified_split(synthetic_examples, val_fraction=0.2, test_fraction=0.0, seed=0)
    return split.train, split.val


def _run(
    model: nn.Module,
    data: tuple[list[Example], list[Example]],
    tok: PreTrainedTokenizerFast,
    cfg: TrainConfig,
) -> tuple[nn.Module, float, int]:
    train, val = data
    kw = {"batch_size": cfg.batch_size, "max_length": 16}
    result = train_model(
        model,
        make_loader(train, tok, shuffle=True, seed=cfg.seed, **kw),
        make_loader(val, tok, shuffle=False, **kw),
        cfg,
        n_classes=len(LABEL_NAMES),
    )
    return model, result.best_val_macro_f1, len(result.history)


def test_full_finetune_learns_synthetic_task(
    tiny_model: nn.Module,
    loaders: tuple[list[Example], list[Example]],
    fake_tokenizer: PreTrainedTokenizerFast,
) -> None:
    apply_strategy(tiny_model, "full")
    cfg = TrainConfig(
        epochs=12, lr=2e-3, batch_size=16, device="cpu", precision="fp32", patience=0, seed=0
    )
    _, best_f1, _ = _run(tiny_model, loaders, fake_tokenizer, cfg)
    assert best_f1 > 0.8  # chance level is 0.33


def test_lora_learns_and_only_adapter_changes(
    tiny_model: nn.Module,
    loaders: tuple[list[Example], list[Example]],
    fake_tokenizer: PreTrainedTokenizerFast,
) -> None:
    frozen_before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    apply_strategy(tiny_model, "lora", LoRAConfig(r=8, alpha=16))
    cfg = TrainConfig(
        epochs=12, lr=5e-3, batch_size=16, device="cpu", precision="fp32", patience=0, seed=0
    )
    _, best_f1, _ = _run(tiny_model, loaders, fake_tokenizer, cfg)
    assert best_f1 > 0.5  # a random encoder limits LoRA; observed ~0.62
    for name, p in tiny_model.named_parameters():
        if not p.requires_grad:
            base_name = name.replace(".base.", ".")
            assert torch.equal(p, frozen_before[base_name]), f"frozen weight {name} changed"


def test_early_stopping_and_best_restore(
    tiny_model: nn.Module,
    loaders: tuple[list[Example], list[Example]],
    fake_tokenizer: PreTrainedTokenizerFast,
) -> None:
    apply_strategy(tiny_model, "head")
    train, val = loaders
    kw = {"batch_size": 16, "max_length": 16}
    cfg = TrainConfig(
        epochs=30, lr=1e-2, batch_size=16, device="cpu", precision="fp32", patience=1, seed=0
    )
    result = train_model(
        tiny_model,
        make_loader(train, fake_tokenizer, shuffle=True, seed=0, **kw),
        make_loader(val, fake_tokenizer, shuffle=False, **kw),
        cfg,
        n_classes=3,
    )
    assert result.stopped_early or len(result.history) == 30
    assert result.best_epoch == max(result.history, key=lambda h: h.val_macro_f1).epoch
    assert result.total_seconds > 0
    assert "history" in result.to_dict()


def test_training_is_reproducible(
    tiny_model: nn.Module,
    loaders: tuple[list[Example], list[Example]],
    fake_tokenizer: PreTrainedTokenizerFast,
) -> None:
    cfg = TrainConfig(epochs=2, lr=1e-3, batch_size=16, device="cpu", precision="fp32", seed=7)
    weights = []
    for _ in range(2):
        set_seed(cfg.seed)
        model = copy.deepcopy(tiny_model)
        apply_strategy(model, "lora", LoRAConfig(r=2))
        _run(model, loaders, fake_tokenizer, cfg)
        weights.append(trainable_state_dict(model))
    for k in weights[0]:
        assert torch.equal(weights[0][k], weights[1][k]), k
