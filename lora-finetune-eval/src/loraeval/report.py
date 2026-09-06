"""Load finished runs and render a Markdown comparison with confidence intervals and
paired significance tests. This is the artefact a reviewer reads, so it is generated,
never hand-edited."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loraeval.evaluate import EvalReport, Predictions
from loraeval.lora import ParamCount
from loraeval.metrics import ConfidenceInterval, mcnemar_test


@dataclass
class RunSummary:
    name: str
    strategy: str
    params: ParamCount
    best_epoch: int
    epochs_run: int
    train_seconds: float
    report: EvalReport
    predictions: Predictions
    extra: dict[str, Any]


def load_run(run_dir: str | Path) -> RunSummary:
    run_dir = Path(run_dir)
    metrics: dict[str, Any] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config: dict[str, Any] = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return RunSummary(
        name=str(config["name"]),
        strategy=str(config["strategy"]),
        params=ParamCount(**metrics["params"]),
        best_epoch=int(metrics["train"]["best_epoch"]),
        epochs_run=len(metrics["train"]["history"]),
        train_seconds=float(metrics["train"]["total_seconds"]),
        report=EvalReport.from_dict(metrics["test"]),
        predictions=Predictions.load(run_dir / "predictions.npz"),
        extra={"lora": config.get("lora"), "model_name": config.get("model_name")},
    )


def _pct(ci: ConfidenceInterval) -> str:
    return f"{100 * ci.point:.1f} [{100 * ci.lower:.1f}, {100 * ci.upper:.1f}]"


def _lora_desc(extra: dict[str, Any]) -> str:
    lora = extra.get("lora")
    if not lora:
        return "-"
    return f"r={lora['r']}, alpha={lora['alpha']:g}"


def render_comparison(runs: list[RunSummary], baseline: str | None = None) -> str:
    """Markdown report: headline metrics with 95% CIs, McNemar vs. the baseline, per-class F1."""
    if not runs:
        raise ValueError("no runs to compare")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError("run names must be unique")
    base = runs[0] if baseline is None else next((r for r in runs if r.name == baseline), None)
    if base is None:
        raise ValueError(f"baseline {baseline!r} not among runs {names}")

    lines: list[str] = []
    lines.append(
        f"Test set: n = {base.report.n} sentences. Intervals are 95% percentile bootstrap."
    )
    lines.append("")
    lines.append(
        "| Run | Strategy | LoRA | Trainable params | Accuracy % [95% CI] | Macro-F1 % [95% CI] "
        "| ECE | NLL | Best epoch | Train time |"
    )
    lines.append("|---|---|---|---:|---|---|---:|---:|---:|---:|")
    for r in runs:
        lines.append(
            f"| {r.name} | {r.strategy} | {_lora_desc(r.extra)} | "
            f"{r.params.trainable:,} ({100 * r.params.trainable_fraction:.2f}%) | "
            f"{_pct(r.report.accuracy)} | {_pct(r.report.macro_f1)} | "
            f"{r.report.ece:.3f} | {r.report.nll:.3f} | {r.best_epoch}/{r.epochs_run} | "
            f"{r.train_seconds:.0f}s |"
        )

    lines.append("")
    lines.append(f"### Paired comparison vs. `{base.name}` (McNemar's test)")
    lines.append("")
    lines.append(
        "| Run | Delta accuracy (pts) | baseline right / run wrong "
        "| baseline wrong / run right | p-value | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in runs:
        if r is base:
            continue
        res = mcnemar_test(base.predictions.labels, base.predictions.preds, r.predictions.preds)
        delta = 100 * (r.report.accuracy.point - base.report.accuracy.point)
        verdict = "significant (p < 0.05)" if res.p_value < 0.05 else "not significant"
        lines.append(
            f"| {r.name} | {delta:+.1f} | {res.b} | {res.c} "
            f"| {res.p_value:.4f} ({res.method}) | {verdict} |"
        )

    lines.append("")
    lines.append("### Per-class F1 %")
    lines.append("")
    classes = base.report.class_names
    lines.append("| Run | " + " | ".join(classes) + " |")
    lines.append("|---|" + "---:|" * len(classes))
    for r in runs:
        cells = [f"{100 * r.report.per_class[c].f1:.1f}" for c in classes]
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
