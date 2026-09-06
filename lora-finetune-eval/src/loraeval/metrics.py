"""Classification metrics with uncertainty: confusion matrix, per-class P/R/F1,
expected calibration error, percentile-bootstrap confidence intervals and McNemar's
paired significance test.

Everything is plain NumPy so the numbers are auditable line by line, which is exactly
what a model-validation reviewer wants to see.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def _as_int(a: NDArray[np.generic] | list[int]) -> IntArray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError("labels must be 1-D")
    return arr.astype(np.int64)


def confusion_matrix(y_true: IntArray, y_pred: IntArray, n_classes: int) -> IntArray:
    """``cm[i, j]`` = number of examples with true class ``i`` predicted as ``j``."""
    y_true, y_pred = _as_int(y_true), _as_int(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size and (y_true.min() < 0 or y_true.max() >= n_classes):
        raise ValueError("y_true contains labels outside [0, n_classes)")
    if y_pred.size and (y_pred.min() < 0 or y_pred.max() >= n_classes):
        raise ValueError("y_pred contains labels outside [0, n_classes)")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(y_true: IntArray, y_pred: IntArray) -> float:
    y_true, y_pred = _as_int(y_true), _as_int(y_pred)
    if y_true.size == 0:
        raise ValueError("empty input")
    return float((y_true == y_pred).mean())


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


def per_class_metrics(cm: IntArray) -> list[ClassMetrics]:
    """Precision/recall/F1 per class from a confusion matrix; ``0/0`` is defined as 0."""
    out: list[ClassMetrics] = []
    for c in range(cm.shape[0]):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(ClassMetrics(precision, recall, f1, support=tp + fn))
    return out


def macro_f1(y_true: IntArray, y_pred: IntArray, n_classes: int) -> float:
    """Unweighted mean of per-class F1: the right headline metric for imbalanced data
    (Financial PhraseBank is 61% neutral, so accuracy alone rewards a lazy model)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    return float(np.mean([m.f1 for m in per_class_metrics(cm)]))


def negative_log_likelihood(probs: FloatArray, y_true: IntArray, eps: float = 1e-12) -> float:
    y_true = _as_int(y_true)
    p = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.log(p).mean())


def expected_calibration_error(probs: FloatArray, y_true: IntArray, n_bins: int = 15) -> float:
    """ECE (Naeini et al., 2015): ``sum_b |B_b|/N * |acc(B_b) - conf(B_b)|`` over
    equal-width confidence bins of the top-class probability.

    A model that says "90% sure" should be right 90% of the time; ECE measures the
    average gap. It matters wherever a probability feeds a downstream decision
    (thresholding, expected-loss computations, human escalation rules).
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = _as_int(y_true)
    if probs.ndim != 2 or probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs must be [N, C] and match y_true")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in pairwise(edges):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.sum() / n * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    lower: float
    upper: float
    level: float = 0.95

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


def bootstrap_ci(
    metric: Callable[[IntArray, IntArray], float],
    y_true: IntArray,
    y_pred: IntArray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> ConfidenceInterval:
    """Percentile bootstrap over examples (Efron, 1979).

    Resampling the *test set* answers "how much would this number move if I had drawn
    a different test set of the same size?", which is the uncertainty a reader of a
    single accuracy figure actually needs. With N ≈ 340 test sentences the 95% CI on
    accuracy is roughly ±3-4 points, wider than most "improvements" reported in blog posts.
    """
    y_true, y_pred = _as_int(y_true), _as_int(y_pred)
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric(y_true, y_pred)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.array([metric(y_true[i], y_pred[i]) for i in idx])
    alpha = (1 - level) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return ConfidenceInterval(point, float(lower), float(upper), level)


@dataclass(frozen=True)
class McNemarResult:
    b: int
    """Examples model A got right and model B got wrong."""
    c: int
    """Examples model A got wrong and model B got right."""
    statistic: float
    p_value: float
    method: str

    @property
    def n_discordant(self) -> int:
        return self.b + self.c


def mcnemar_test(y_true: IntArray, pred_a: IntArray, pred_b: IntArray) -> McNemarResult:
    """Paired test of "do models A and B have the same error rate?" (McNemar, 1947).

    Only the *discordant* pairs carry information. Uses the exact binomial test when
    ``b + c < 25`` and the continuity-corrected chi-square otherwise (Edwards, 1948).
    Two models evaluated on the same test set are paired, so this is the appropriate
    test; an unpaired comparison of two accuracies throws away the pairing and loses power.
    """
    y_true, pred_a, pred_b = _as_int(y_true), _as_int(pred_a), _as_int(pred_b)
    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("all inputs must have the same length")
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, "exact")
    if n < 25:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5**n
        return McNemarResult(b, c, float(k), float(min(1.0, 2 * tail)), "exact")
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))  # chi-square survival function with 1 dof
    return McNemarResult(b, c, float(stat), float(p), "chi2-corrected")
