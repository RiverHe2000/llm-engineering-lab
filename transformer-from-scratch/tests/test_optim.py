from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from nanoformer.config import ModelConfig
from nanoformer.model import Transformer
from nanoformer.optim import build_param_groups, configure_optimizer, lr_at


def sched(step: int) -> float:
    return lr_at(step, base_lr=1e-3, warmup_steps=10, total_steps=110, min_lr=1e-4)


class TestSchedule:
    def test_warmup_ramps_linearly_to_peak(self) -> None:
        assert sched(0) == pytest.approx(1e-4)  # never exactly zero
        assert sched(4) == pytest.approx(5e-4)
        assert sched(9) == pytest.approx(1e-3)
        assert sched(10) == pytest.approx(1e-3)

    def test_cosine_decay_is_monotone_and_bounded(self) -> None:
        values = [sched(s) for s in range(10, 111)]
        assert all(a >= b for a, b in pairwise(values))
        assert values[-1] == pytest.approx(1e-4)
        assert min(values) >= 1e-4 - 1e-12
        # Half-way through the decay the cosine is at its midpoint.
        assert sched(60) == pytest.approx((1e-3 + 1e-4) / 2)

    def test_holds_at_min_after_total(self) -> None:
        assert sched(1_000) == pytest.approx(1e-4)

    def test_no_warmup(self) -> None:
        assert lr_at(0, base_lr=1.0, warmup_steps=0, total_steps=10, min_lr=0.0) == 1.0

    def test_negative_step_rejected(self) -> None:
        with pytest.raises(ValueError):
            sched(-1)


class TestParamGroups:
    def test_split_by_dimensionality_and_covers_everything(self) -> None:
        model = Transformer(ModelConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2))
        groups = build_param_groups(model, weight_decay=0.1)
        decay, no_decay = groups[0]["params"], groups[1]["params"]
        assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
        assert all(p.dim() >= 2 for p in decay)
        assert all(p.dim() < 2 for p in no_decay)
        all_ids = {id(p) for p in model.parameters()}
        grouped_ids = {id(p) for p in decay} | {id(p) for p in no_decay}
        assert all_ids == grouped_ids
        assert len(decay) + len(no_decay) == len(all_ids)  # tied weight appears once

    def test_frozen_params_are_excluded(self) -> None:
        model = Transformer(ModelConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2))
        model.tok_emb.weight.requires_grad_(False)
        groups = build_param_groups(model, weight_decay=0.1)
        assert all(id(p) != id(model.tok_emb.weight) for p in groups[0]["params"])

    def test_configure_optimizer(self) -> None:
        model = Transformer(ModelConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2))
        opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.99))
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["betas"] == (0.9, 0.99)
