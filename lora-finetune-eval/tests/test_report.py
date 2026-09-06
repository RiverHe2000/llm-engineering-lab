from __future__ import annotations

import copy
from pathlib import Path

import pytest
from torch import nn
from transformers import PreTrainedTokenizerFast

from loraeval.data import Example
from loraeval.experiment import DataConfig, ExperimentConfig, run_experiment
from loraeval.lora import LoRAConfig
from loraeval.report import RunSummary, load_run, render_comparison
from loraeval.train import TrainConfig


@pytest.fixture
def two_runs(
    tmp_path: Path,
    tiny_model: nn.Module,
    fake_tokenizer: PreTrainedTokenizerFast,
    synthetic_examples: list[Example],
) -> list[RunSummary]:
    common = {
        "data": DataConfig(val_fraction=0.2, test_fraction=0.2, max_length=16, seed=0),
        "train": TrainConfig(
            epochs=2, lr=5e-3, batch_size=16, device="cpu", precision="fp32", seed=0
        ),
        "out_dir": str(tmp_path / "runs"),
        "n_boot": 30,
    }
    head = ExperimentConfig(name="head", strategy="head", lora=None, **common)  # type: ignore[arg-type]
    lora = ExperimentConfig(name="lora", strategy="lora", lora=LoRAConfig(r=2), **common)  # type: ignore[arg-type]
    for cfg in (head, lora):
        run_experiment(
            cfg,
            model=copy.deepcopy(tiny_model),
            tokenizer=fake_tokenizer,
            examples=synthetic_examples,
        )
    return [load_run(tmp_path / "runs" / "head"), load_run(tmp_path / "runs" / "lora")]


def test_load_run_fields(two_runs: list[RunSummary]) -> None:
    head, lora = two_runs
    assert head.strategy == "head" and lora.strategy == "lora"
    assert lora.extra["lora"]["r"] == 2 and head.extra["lora"] is None
    assert head.epochs_run == 2 and head.best_epoch in (1, 2)
    assert head.predictions.labels.shape == lora.predictions.labels.shape


def test_render_comparison_contents(two_runs: list[RunSummary]) -> None:
    md = render_comparison(two_runs)
    assert "| head | head | - |" in md
    assert "| lora | lora | r=2, alpha=16 |" in md
    assert "McNemar" in md and "Per-class F1" in md
    assert "negative | neutral | positive" in md
    assert "p-value" in md and ("significant" in md)
    # Baseline selection by name flips the pairing direction.
    md2 = render_comparison(two_runs, baseline="lora")
    assert "vs. `lora`" in md2 and "| head |" in md2


def test_render_comparison_errors(two_runs: list[RunSummary]) -> None:
    with pytest.raises(ValueError, match="no runs"):
        render_comparison([])
    with pytest.raises(ValueError, match="unique"):
        render_comparison([two_runs[0], two_runs[0]])
    with pytest.raises(ValueError, match="baseline"):
        render_comparison(two_runs, baseline="missing")
