from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from torch import nn
from transformers import PreTrainedTokenizerFast

from loraeval.cli import main
from loraeval.data import Example


@pytest.fixture
def offline(
    monkeypatch: pytest.MonkeyPatch,
    tiny_model: nn.Module,
    fake_tokenizer: PreTrainedTokenizerFast,
    synthetic_examples: list[Example],
) -> None:
    """Route the CLI's model/tokenizer/data loading to in-memory fixtures."""
    monkeypatch.setattr("loraeval.experiment._load_model", lambda _cfg: copy.deepcopy(tiny_model))
    monkeypatch.setattr("loraeval.experiment._load_tokenizer", lambda _cfg: fake_tokenizer)
    monkeypatch.setattr("loraeval.experiment._load_examples", lambda _cfg: list(synthetic_examples))


def _write_config(tmp_path: Path, name: str, strategy: str) -> Path:
    cfg = {
        "name": name,
        "strategy": strategy,
        "lora": {"r": 2, "alpha": 4} if strategy == "lora" else None,
        "data": {"val_fraction": 0.2, "test_fraction": 0.2, "max_length": 16, "seed": 0},
        "train": {
            "epochs": 2,
            "lr": 5e-3,
            "batch_size": 16,
            "device": "cpu",
            "precision": "fp32",
            "seed": 0,
        },
        "out_dir": str(tmp_path / "runs"),
        "n_boot": 20,
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.usefixtures("offline")
def test_cli_run_compare_predict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    head_cfg = _write_config(tmp_path, "head", "head")
    lora_cfg = _write_config(tmp_path, "lora", "lora")
    assert main(["run", "--config", str(head_cfg)]) == 0
    assert main(["run", "--config", str(lora_cfg), "--override", "train.epochs=1"]) == 0
    runs = tmp_path / "runs"
    assert (runs / "lora" / "weights.safetensors").exists()

    out_md = tmp_path / "RESULTS.md"
    assert (
        main(
            [
                "compare",
                str(runs / "head"),
                str(runs / "lora"),
                "--baseline",
                "head",
                "--out",
                str(out_md),
            ]
        )
        == 0
    )
    text = out_md.read_text(encoding="utf-8")
    assert "| lora |" in text and "McNemar" in text

    assert main(["compare", str(runs / "head"), str(runs / "lora")]) == 0
    assert "Per-class F1" in capsys.readouterr().out

    assert (
        main(
            [
                "predict",
                "--run",
                str(runs / "lora"),
                "--text",
                "profit rose strong",
                "--text",
                "loss fell",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert all("negative=" in ln and "positive=" in ln for ln in lines)
