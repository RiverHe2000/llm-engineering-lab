"""Plot train/val loss from a run's ``metrics.jsonl``.

Usage: python scripts/plot_loss.py runs/small/metrics.jsonl --out docs/loss_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--out", type=Path, default=Path("loss_curve.png"))
    parser.add_argument("--title", default="nanoformer training")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines()]
    step_rows = [r for r in rows if "loss" in r]
    eval_rows = [r for r in rows if "val_loss" in r]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(
        [r["step"] for r in step_rows],
        [r["loss"] for r in step_rows],
        lw=0.8,
        alpha=0.6,
        label="train (per step)",
    )
    ax.plot(
        [r["step"] for r in eval_rows],
        [r["train_loss"] for r in eval_rows],
        "o-",
        ms=3,
        label="train (eval)",
    )
    ax.plot(
        [r["step"] for r in eval_rows],
        [r["val_loss"] for r in eval_rows],
        "s-",
        ms=3,
        label="val (eval)",
    )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}; final val_loss={eval_rows[-1]['val_loss']:.4f}")


if __name__ == "__main__":
    main()
