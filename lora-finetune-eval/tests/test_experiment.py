from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from transformers import PreTrainedTokenizerFast

from loraeval.data import Example
from loraeval.experiment import (
    DataConfig,
    ExperimentConfig,
    apply_overrides,
    load_run_model,
    run_experiment,
)
from loraeval.lora import LoRAConfig
from loraeval.train import TrainConfig


class TestConfig:
    def test_round_trip_and_defaults(self) -> None:
        cfg = ExperimentConfig(name="x")
        assert cfg.strategy == "lora" and isinstance(cfg.lora, LoRAConfig)
        again = ExperimentConfig.from_dict(cfg.to_dict())
        assert again == cfg
        assert cfg.run_dir == Path("runs") / "x"

    def test_from_dict_infers_lora_presence(self) -> None:
        head = ExperimentConfig.from_dict({"name": "h", "strategy": "head"})
        assert head.lora is None
        lora = ExperimentConfig.from_dict({"name": "l", "strategy": "lora", "lora": {"r": 2}})
        assert lora.lora is not None and lora.lora.r == 2
        with pytest.raises(ValueError, match="requires a lora"):
            ExperimentConfig.from_dict({"name": "l", "strategy": "lora", "lora": None})

    @pytest.mark.parametrize("name", ["", "a/b", "a\\b"])
    def test_bad_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(name=name, strategy="head", lora=None)

    def test_n_boot_validation(self) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(name="x", n_boot=0)

    def test_yaml_and_overrides(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(
            yaml.safe_dump(
                {"name": "y", "strategy": "lora", "lora": {"r": 4}, "train": {"epochs": 2}}
            )
        )
        cfg = ExperimentConfig.from_yaml(
            path, ["train.epochs=5", "lora.alpha=32", "data.max_length=64"]
        )
        assert cfg.train.epochs == 5 and cfg.lora is not None and cfg.lora.alpha == 32
        assert cfg.data.max_length == 64
        with pytest.raises(ValueError):
            apply_overrides({}, ["novalue"])


def _config(tmp_path: Path, name: str, strategy: str, epochs: int, lr: float) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        strategy=strategy,  # type: ignore[arg-type]
        lora=LoRAConfig(r=4, alpha=8) if strategy == "lora" else None,
        data=DataConfig(val_fraction=0.2, test_fraction=0.2, max_length=16, seed=0),
        train=TrainConfig(
            epochs=epochs, lr=lr, batch_size=16, device="cpu", precision="fp32", seed=0
        ),
        out_dir=str(tmp_path / "runs"),
        n_boot=50,
    )


def test_full_finetune_experiment_learns(
    tmp_path: Path,
    tiny_model: nn.Module,
    fake_tokenizer: PreTrainedTokenizerFast,
    synthetic_examples: list[Example],
) -> None:
    cfg = _config(tmp_path, "full", "full", epochs=5, lr=3e-3)
    result = run_experiment(
        cfg, model=tiny_model, tokenizer=fake_tokenizer, examples=synthetic_examples
    )
    assert result.report.macro_f1.point > 0.8  # chance is 0.33
    assert result.report.accuracy.lower <= result.report.accuracy.point
    assert result.params.trainable == result.params.total


def test_lora_experiment_writes_auditable_artifacts(
    tmp_path: Path,
    tiny_model: nn.Module,
    fake_tokenizer: PreTrainedTokenizerFast,
    synthetic_examples: list[Example],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = copy.deepcopy(tiny_model)
    cfg = _config(tmp_path, "lora_test", "lora", epochs=2, lr=5e-3)
    result = run_experiment(
        cfg, model=tiny_model, tokenizer=fake_tokenizer, examples=synthetic_examples
    )
    run_dir = result.run_dir
    assert {p.name for p in run_dir.iterdir()} >= {
        "config.json",
        "metrics.json",
        "predictions.npz",
        "weights.safetensors",
    }

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["params"]["trainable"] == result.params.trainable
    assert metrics["params"]["trainable"] < 0.3 * metrics["params"]["total"]
    assert metrics["split"]["sizes"] == result.split.sizes()
    assert len(metrics["train"]["history"]) == len(result.train.history) == 2
    assert metrics["test"]["n"] == len(result.split.test)
    assert "torch" in metrics["environment"]
    assert json.loads((run_dir / "config.json").read_text())["lora"]["r"] == 4

    # The saved adapter, applied to a fresh copy of the base model, reproduces the
    # in-memory model's predictions exactly.
    monkeypatch.setattr("loraeval.experiment._load_model", lambda _cfg: copy.deepcopy(template))
    monkeypatch.setattr("loraeval.experiment._load_tokenizer", lambda _cfg: fake_tokenizer)
    model, tok, loaded_cfg = load_run_model(run_dir)
    assert loaded_cfg.name == "lora_test" and not model.training
    enc = tok(
        [e.text for e in result.split.test[:5]],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits.float(), dim=-1).numpy()
    np.testing.assert_allclose(probs, result.predictions.probs[:5], atol=1e-5)
