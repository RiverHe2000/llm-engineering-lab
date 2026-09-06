"""``loraeval {run,compare,predict}``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from loraeval.data import LABEL_NAMES
from loraeval.experiment import ExperimentConfig, load_run_model, run_experiment
from loraeval.report import load_run, render_comparison
from loraeval.train import resolve_device

log = logging.getLogger("loraeval")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig.from_yaml(args.config, args.override)
    result = run_experiment(cfg)
    log.info("run written to %s", result.run_dir)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    runs = [load_run(p) for p in args.run_dirs]
    md = render_comparison(runs, baseline=args.baseline)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        log.info("report written to %s", args.out)
    else:
        sys.stdout.write(md)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model, tokenizer, cfg = load_run_model(args.run)
    device = resolve_device(cfg.train.device if args.device == "auto" else args.device)
    model.to(device)
    enc = tokenizer(
        list(args.text),
        padding=True,
        truncation=True,
        max_length=cfg.data.max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)
        ).logits
    probs = torch.softmax(logits.float(), dim=-1).cpu()
    for text, row in zip(args.text, probs, strict=True):
        label = LABEL_NAMES[int(row.argmax())]
        dist = ", ".join(f"{n}={float(p):.3f}" for n, p in zip(LABEL_NAMES, row, strict=True))
        sys.stdout.write(f"{label:9s} | {dist} | {text}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loraeval", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="train + evaluate one configuration")
    r.add_argument("--config", required=True)
    r.add_argument("--override", action="append", default=[], help="key.path=value (repeatable)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="render a Markdown comparison of finished runs")
    c.add_argument("run_dirs", nargs="+")
    c.add_argument("--baseline", default=None, help="run name used for McNemar pairing")
    c.add_argument("--out", default=None, help="write Markdown here instead of stdout")
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("predict", help="classify sentences with a finished run")
    d.add_argument("--run", required=True)
    d.add_argument("--text", action="append", required=True)
    d.add_argument("--device", default="auto")
    d.set_defaults(func=cmd_predict)
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
