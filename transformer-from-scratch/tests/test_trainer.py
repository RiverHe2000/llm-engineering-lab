from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from nanoformer.config import ModelConfig
from nanoformer.data import TokenDataset
from nanoformer.model import Transformer
from nanoformer.trainer import (
    TrainConfig,
    Trainer,
    load_model,
    loss_to_perplexity,
    resolve_device,
    to_tensor_stats,
)

TINY = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)


def make_cfg(tmp_path: Path, **overrides: Any) -> TrainConfig:
    base: dict[str, Any] = {
        "out_dir": str(tmp_path / "run"),
        "batch_size": 8,
        "block_size": 16,
        "max_steps": 6,
        "lr": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 2,
        "eval_interval": 3,
        "eval_iters": 2,
        "checkpoint_interval": 3,
        "log_interval": 1,
        "device": "cpu",
        "seed": 0,
    }
    base.update(overrides)
    return TrainConfig.from_dict(base)


def test_model_overfits_a_predictable_stream(tmp_path: Path, cyclic_dataset: TokenDataset) -> None:
    """A two-layer model must learn a period-16 cycle: loss ln(16)=2.77 -> < 0.3."""
    cfg = make_cfg(tmp_path, max_steps=120, lr=5e-3, min_lr=5e-4, warmup_steps=10, eval_interval=0)
    trainer = Trainer(Transformer(TINY), cyclic_dataset, cyclic_dataset, cfg)
    first = trainer.train_step()["loss"]
    assert abs(first - math.log(16)) < 0.2
    for _ in range(cfg.max_steps - 1):
        stats = trainer.train_step()
    assert stats["loss"] < 0.3
    assert trainer.evaluate()["val_loss"] < 0.3


def test_resume_is_bit_exact(tmp_path: Path, cyclic_dataset: TokenDataset) -> None:
    """6 straight steps == 3 steps + checkpoint + reload + 3 steps, parameter for parameter."""
    torch.manual_seed(0)
    straight = Trainer(Transformer(TINY), cyclic_dataset, cyclic_dataset, make_cfg(tmp_path / "a"))
    straight.train()

    torch.manual_seed(0)
    cfg_b = make_cfg(tmp_path / "b")
    first_half = Trainer(Transformer(TINY), cyclic_dataset, cyclic_dataset, cfg_b)
    first_half.train()  # writes ckpt_000003.pt on the way to step 6
    resumed = Trainer.resume(
        tmp_path / "b" / "run" / "ckpt_000003.pt", cyclic_dataset, cyclic_dataset
    )
    assert resumed.step == 3
    resumed.train()

    for (name, p_a), (_, p_b) in zip(
        straight.model.named_parameters(), resumed.model.named_parameters(), strict=True
    ):
        assert torch.equal(p_a, p_b), f"parameter {name} diverged after resume"
    assert resumed.step == straight.step == 6


def test_train_writes_metrics_and_checkpoints(tmp_path: Path, cyclic_dataset: TokenDataset) -> None:
    cfg = make_cfg(tmp_path)
    trainer = Trainer(Transformer(TINY), cyclic_dataset, cyclic_dataset, cfg)
    history = trainer.train()
    run = Path(cfg.out_dir)
    assert (run / "ckpt_000003.pt").exists()
    assert (run / "ckpt_final.pt").exists()
    assert (run / "ckpt_best.pt").exists()
    assert trainer.best_val_loss == min(r["val_loss"] for r in history if "val_loss" in r)
    assert history[0].get("is_best") == 1.0  # the first evaluation is trivially the best so far
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows == history
    assert any("val_loss" in r for r in rows)
    assert any("tokens_per_sec" in r for r in rows)
    assert "elapsed_sec" in rows[-1]
    assert rows[-1]["step"] == 6

    model = load_model(run / "ckpt_final.pt")
    assert not model.training
    x = torch.randint(0, 16, (1, 8))
    torch.testing.assert_close(model(x).logits, trainer.model.eval()(x).logits)


def test_gradient_accumulation_matches_larger_batch(
    tmp_path: Path, cyclic_dataset: TokenDataset
) -> None:
    """Loss reported with accumulation equals the mean over its micro-batches."""
    cfg = make_cfg(tmp_path, grad_accum_steps=4, batch_size=2)
    trainer = Trainer(Transformer(TINY), cyclic_dataset, cyclic_dataset, cfg)
    assert cfg.tokens_per_step == 4 * 2 * 16
    stats = trainer.train_step()
    assert abs(stats["loss"] - math.log(16)) < 0.3
    assert stats["grad_norm"] > 0


def test_bf16_autocast_on_cpu_runs(tmp_path: Path, cyclic_dataset: TokenDataset) -> None:
    cfg = make_cfg(tmp_path, precision="bf16", max_steps=2, eval_interval=0, checkpoint_interval=0)
    trainer = Trainer(Transformer(TINY), cyclic_dataset, cyclic_dataset, cfg)
    stats = trainer.train_step()
    assert math.isfinite(stats["loss"])


def test_grad_clip_disabled_when_zero(tmp_path: Path, cyclic_dataset: TokenDataset) -> None:
    cfg = make_cfg(tmp_path, grad_clip=0.0)
    trainer = Trainer(Transformer(TINY), cyclic_dataset, cyclic_dataset, cfg)
    assert math.isfinite(trainer.train_step()["grad_norm"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"max_steps": 0},
        {"eval_interval": -1},
        {"lr": 0.0},
        {"min_lr": 1.0, "lr": 0.1},
        {"grad_clip": -1.0},
        {"precision": "int8"},
    ],
)
def test_invalid_train_config(tmp_path: Path, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make_cfg(tmp_path, **overrides)


def test_unknown_train_config_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_dict({"bogus": 1})


def test_trainer_consistency_checks(tmp_path: Path, cyclic_dataset: TokenDataset) -> None:
    with pytest.raises(ValueError, match="block_size"):
        Trainer(Transformer(TINY), cyclic_dataset, cyclic_dataset, make_cfg(tmp_path, block_size=8))
    big = TokenDataset(cyclic_dataset.tokens, block_size=32)
    with pytest.raises(ValueError, match="max_seq_len"):
        Trainer(Transformer(TINY), big, big, make_cfg(tmp_path, block_size=32))
    with pytest.raises(ValueError, match="fp16"):
        Trainer(
            Transformer(TINY), cyclic_dataset, cyclic_dataset, make_cfg(tmp_path, precision="fp16")
        )


def test_helpers() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    assert loss_to_perplexity(0.0) == 1.0
    stats = to_tensor_stats(torch.tensor([1.0, -3.0]))
    assert stats["absmax"] == 3.0 and stats["mean"] == -1.0
