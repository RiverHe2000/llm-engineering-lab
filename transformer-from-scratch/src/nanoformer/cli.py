"""Command-line entry points: ``nanoformer {train-tokenizer,prepare-data,train,generate}``.

The CLI is deliberately thin: it parses arguments and YAML, then delegates to the
library so that everything it does is also reachable (and tested) programmatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from nanoformer.config import ModelConfig
from nanoformer.data import TokenDataset, prepare_corpus
from nanoformer.generate import generate
from nanoformer.model import Transformer
from nanoformer.tokenizer import BPETokenizer
from nanoformer.trainer import TrainConfig, Trainer, load_model, loss_to_perplexity

log = logging.getLogger("nanoformer")


def _coerce(value: str) -> Any:
    """Parse a ``key=value`` override value as YAML so numbers/bools/null work."""
    return yaml.safe_load(value)


def apply_overrides(config: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (e.g. ``train.max_steps=10``)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return config


def load_yaml_config(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("model", "train", "data"):
        raw.setdefault(section, {})
    return apply_overrides(raw, overrides)


# ----- sub-commands ---------------------------------------------------------------------


def cmd_train_tokenizer(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8")
    tok = BPETokenizer.train(text, vocab_size=args.vocab_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    n_tokens = len(tok.encode(text))
    log.info(
        "vocab=%d, corpus=%d chars -> %d tokens (%.2f chars/token), saved to %s",
        tok.vocab_size,
        len(text),
        n_tokens,
        len(text) / max(n_tokens, 1),
        args.out,
    )
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    tok = BPETokenizer.load(args.tokenizer)
    stats = prepare_corpus(tok, args.input, args.out_dir, val_fraction=args.val_fraction)
    log.info("wrote %s", json.dumps(stats))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_yaml_config(args.config, args.override)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    train_cfg = TrainConfig.from_dict(cfg["train"])
    data_cfg = cfg["data"]

    if "tokenizer" in data_cfg:
        tok = BPETokenizer.load(data_cfg["tokenizer"])
        if tok.vocab_size != model_cfg.vocab_size:
            raise ValueError(
                f"model.vocab_size={model_cfg.vocab_size} != tokenizer vocab {tok.vocab_size}"
            )
    train_ds = TokenDataset.from_bin(data_cfg["train_bin"], train_cfg.block_size)
    val_ds = TokenDataset.from_bin(data_cfg["val_bin"], train_cfg.block_size)

    if args.resume:
        trainer = Trainer.resume(args.resume, train_ds, val_ds)
        log.info("resumed from %s at step %d", args.resume, trainer.step)
    else:
        model = Transformer(model_cfg)
        trainer = Trainer(model, train_ds, val_ds, train_cfg)
        Path(train_cfg.out_dir).mkdir(parents=True, exist_ok=True)
        model_cfg.save(Path(train_cfg.out_dir) / "model_config.json")
        (Path(train_cfg.out_dir) / "train_config.json").write_text(
            json.dumps(train_cfg.to_dict(), indent=2), encoding="utf-8"
        )
    history = trainer.train()
    last = history[-1]
    log.info(
        "done: val_loss=%.4f (ppl %.2f)", last["val_loss"], loss_to_perplexity(last["val_loss"])
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    model = load_model(args.checkpoint, device=device)
    tok = BPETokenizer.load(args.tokenizer)
    ids = tok.encode(args.prompt) if args.prompt else [tok.eos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out = generate(
        model,
        idx,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=tok.eos_id,
        generator=gen,
    )
    text = tok.decode(out[0].tolist())
    sys.stdout.write(text + "\n")
    return 0


# ----- parser ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nanoformer", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train-tokenizer", help="learn a byte-level BPE vocabulary")
    t.add_argument("--input", required=True)
    t.add_argument("--vocab-size", type=int, default=1024)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_train_tokenizer)

    d = sub.add_parser("prepare-data", help="encode a text file into train/val .bin files")
    d.add_argument("--input", required=True)
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--val-fraction", type=float, default=0.1)
    d.set_defaults(func=cmd_prepare_data)

    tr = sub.add_parser("train", help="train from a YAML config")
    tr.add_argument("--config", required=True)
    tr.add_argument("--resume", default=None, help="checkpoint to resume from")
    tr.add_argument(
        "--override", action="append", default=[], help="section.key=value (repeatable)"
    )
    tr.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="sample text from a checkpoint")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--prompt", default="")
    g.add_argument("--max-new-tokens", type=int, default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto")
    g.set_defaults(func=cmd_generate)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
