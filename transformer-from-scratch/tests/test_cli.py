from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nanoformer.cli import apply_overrides, load_yaml_config, main
from nanoformer.tokenizer import BPETokenizer

TEXT = ("to be or not to be, that is the question. " * 60) + (
    "whether tis nobler in the mind. " * 60
)


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"

    assert (
        main(
            [
                "train-tokenizer",
                "--input",
                str(corpus),
                "--vocab-size",
                "300",
                "--out",
                str(tok_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "prepare-data",
                "--input",
                str(corpus),
                "--tokenizer",
                str(tok_path),
                "--out-dir",
                str(data_dir),
            ]
        )
        == 0
    )
    assert (data_dir / "train.bin").exists() and (data_dir / "val.bin").exists()
    vocab = BPETokenizer.load(tok_path).vocab_size  # BPE may stop early on a tiny corpus

    config = {
        "model": {
            "vocab_size": vocab,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "max_seq_len": 16,
        },
        "train": {
            "out_dir": str(run_dir),
            "batch_size": 4,
            "block_size": 16,
            "max_steps": 4,
            "warmup_steps": 1,
            "eval_interval": 2,
            "eval_iters": 1,
            "checkpoint_interval": 2,
            "log_interval": 1,
            "device": "cpu",
        },
        "data": {
            "train_bin": str(data_dir / "train.bin"),
            "val_bin": str(data_dir / "val.bin"),
            "tokenizer": str(tok_path),
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["train", "--config", str(cfg_path)]) == 0
    assert (run_dir / "ckpt_final.pt").exists()
    assert (run_dir / "model_config.json").exists()
    assert json.loads((run_dir / "train_config.json").read_text())["max_steps"] == 4

    # Resume from the intermediate checkpoint with an override that extends training.
    assert (
        main(
            [
                "train",
                "--config",
                str(cfg_path),
                "--resume",
                str(run_dir / "ckpt_000002.pt"),
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "generate",
                "--checkpoint",
                str(run_dir / "ckpt_final.pt"),
                "--tokenizer",
                str(tok_path),
                "--prompt",
                "to be",
                "--max-new-tokens",
                "5",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("to be")

    # Empty prompt falls back to the EOS token as the start symbol.
    assert (
        main(
            [
                "generate",
                "--checkpoint",
                str(run_dir / "ckpt_final.pt"),
                "--tokenizer",
                str(tok_path),
                "--max-new-tokens",
                "2",
                "--device",
                "cpu",
                "--temperature",
                "0",
            ]
        )
        == 0
    )


def test_vocab_mismatch_is_caught(tmp_path: Path) -> None:
    corpus = tmp_path / "c.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    main(["train-tokenizer", "--input", str(corpus), "--vocab-size", "300", "--out", str(tok_path)])
    main(
        [
            "prepare-data",
            "--input",
            str(corpus),
            "--tokenizer",
            str(tok_path),
            "--out-dir",
            str(tmp_path / "d"),
        ]
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocab_size": 999,
                    "d_model": 32,
                    "n_layers": 1,
                    "n_heads": 2,
                    "max_seq_len": 16,
                },
                "train": {
                    "out_dir": str(tmp_path / "r"),
                    "block_size": 16,
                    "max_steps": 1,
                    "device": "cpu",
                },
                "data": {
                    "train_bin": str(tmp_path / "d" / "train.bin"),
                    "val_bin": str(tmp_path / "d" / "val.bin"),
                    "tokenizer": str(tok_path),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="vocab"):
        main(["train", "--config", str(cfg_path)])


def test_overrides() -> None:
    cfg = apply_overrides(
        {"train": {"max_steps": 1}},
        ["train.max_steps=10", "model.dropout=0.1", "train.compile=true"],
    )
    assert cfg["train"]["max_steps"] == 10
    assert cfg["model"]["dropout"] == 0.1
    assert cfg["train"]["compile"] is True
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_load_yaml_config_fills_sections(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    cfg = load_yaml_config(p, ["train.lr=0.01"])
    assert cfg["model"] == {} and cfg["data"] == {}
    assert cfg["train"] == {"lr": 0.01}
