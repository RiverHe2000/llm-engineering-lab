"""Inference over a DataLoader and the structured evaluation report."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from loraeval.data import Example
from loraeval.metrics import (
    ClassMetrics,
    ConfidenceInterval,
    FloatArray,
    IntArray,
    accuracy,
    bootstrap_ci,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
    per_class_metrics,
)


@dataclass
class Predictions:
    probs: FloatArray
    """``[N, C]`` softmax probabilities."""
    labels: IntArray
    """``[N]`` gold labels."""

    def __post_init__(self) -> None:
        self.probs = np.asarray(self.probs, dtype=np.float64)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        if self.probs.ndim != 2 or self.labels.ndim != 1:
            raise ValueError("probs must be [N, C] and labels [N]")
        if self.probs.shape[0] != self.labels.shape[0]:
            raise ValueError("probs and labels disagree on N")

    @property
    def preds(self) -> IntArray:
        return self.probs.argmax(axis=1).astype(np.int64)

    @property
    def n_classes(self) -> int:
        return int(self.probs.shape[1])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, probs=self.probs, labels=self.labels)

    @classmethod
    def load(cls, path: str | Path) -> Predictions:
        with np.load(path) as data:
            return cls(probs=data["probs"], labels=data["labels"])


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader[Example],
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> Predictions:
    """Run the model over ``loader``; softmax is taken in float32 regardless of autocast."""
    was_training = model.training
    model.eval()
    probs: list[FloatArray] = []
    labels: list[IntArray] = []
    try:
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(**inputs).logits
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64))
            labels.append(batch["labels"].numpy().astype(np.int64))
    finally:
        model.train(was_training)
    if not probs:
        raise ValueError("loader yielded no batches")
    return Predictions(probs=np.concatenate(probs), labels=np.concatenate(labels))


@dataclass
class EvalReport:
    n: int
    class_names: list[str]
    accuracy: ConfidenceInterval
    macro_f1: ConfidenceInterval
    per_class: dict[str, ClassMetrics]
    confusion: list[list[int]]
    ece: float
    nll: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "class_names": list(self.class_names),
            "accuracy": vars(self.accuracy),
            "macro_f1": vars(self.macro_f1),
            "per_class": {k: vars(v) for k, v in self.per_class.items()},
            "confusion": self.confusion,
            "ece": self.ece,
            "nll": self.nll,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalReport:
        return cls(
            n=int(d["n"]),
            class_names=list(d["class_names"]),
            accuracy=ConfidenceInterval(**d["accuracy"]),
            macro_f1=ConfidenceInterval(**d["macro_f1"]),
            per_class={k: ClassMetrics(**v) for k, v in d["per_class"].items()},
            confusion=[list(map(int, row)) for row in d["confusion"]],
            ece=float(d["ece"]),
            nll=float(d["nll"]),
        )


def evaluate_predictions(
    pred: Predictions,
    class_names: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> EvalReport:
    n_classes = len(class_names)
    if pred.n_classes != n_classes:
        raise ValueError(f"predictions have {pred.n_classes} classes, expected {n_classes}")
    y_true, y_pred = pred.labels, pred.preds
    cm = confusion_matrix(y_true, y_pred, n_classes)
    per_class = per_class_metrics(cm)
    return EvalReport(
        n=len(y_true),
        class_names=list(class_names),
        accuracy=bootstrap_ci(accuracy, y_true, y_pred, n_boot=n_boot, seed=seed),
        macro_f1=bootstrap_ci(
            lambda t, p: macro_f1(t, p, n_classes), y_true, y_pred, n_boot=n_boot, seed=seed
        ),
        per_class=dict(zip(class_names, per_class, strict=True)),
        confusion=cm.tolist(),
        ece=expected_calibration_error(pred.probs, y_true),
        nll=negative_log_likelihood(pred.probs, y_true),
    )
