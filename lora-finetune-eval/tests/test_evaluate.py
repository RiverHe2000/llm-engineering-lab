from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from transformers import PreTrainedTokenizerFast

from loraeval.data import LABEL_NAMES, Example, make_loader
from loraeval.evaluate import EvalReport, Predictions, evaluate_predictions, predict


def test_predict_shapes_and_probabilities(
    tiny_model: nn.Module,
    fake_tokenizer: PreTrainedTokenizerFast,
    synthetic_examples: list[Example],
) -> None:
    loader = make_loader(
        synthetic_examples[:23], fake_tokenizer, batch_size=8, max_length=16, shuffle=False
    )
    tiny_model.train()
    pred = predict(tiny_model, loader, torch.device("cpu"))
    assert tiny_model.training  # mode restored
    assert pred.probs.shape == (23, 3) and pred.labels.shape == (23,)
    np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-6)
    assert pred.preds.shape == (23,) and pred.n_classes == 3
    assert np.array_equal(pred.labels, np.array([e.label for e in synthetic_examples[:23]]))


def test_predict_empty_loader_raises(
    tiny_model: nn.Module, fake_tokenizer: PreTrainedTokenizerFast
) -> None:
    loader = make_loader([], fake_tokenizer, batch_size=8, max_length=16, shuffle=False)
    with pytest.raises(ValueError, match="no batches"):
        predict(tiny_model, loader, torch.device("cpu"))


def test_predictions_validation_and_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Predictions(probs=np.ones((3, 2)), labels=np.zeros(2, dtype=int))
    with pytest.raises(ValueError):
        Predictions(probs=np.ones(3), labels=np.zeros(3, dtype=int))
    p = Predictions(probs=np.array([[0.2, 0.8], [0.6, 0.4]]), labels=np.array([1, 1]))
    p.save(tmp_path / "p.npz")
    q = Predictions.load(tmp_path / "p.npz")
    assert np.array_equal(p.probs, q.probs) and np.array_equal(p.labels, q.labels)
    assert q.preds.tolist() == [1, 0]


def test_evaluate_predictions_report_round_trip() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 120)
    logits = rng.normal(size=(120, 3))
    logits[np.arange(120), labels] += 2.0
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    report = evaluate_predictions(Predictions(probs, labels), LABEL_NAMES, n_boot=200, seed=0)
    assert report.n == 120
    assert report.accuracy.lower <= report.accuracy.point <= report.accuracy.upper
    assert 0.5 < report.accuracy.point <= 1.0
    assert set(report.per_class) == set(LABEL_NAMES)
    assert sum(sum(row) for row in report.confusion) == 120
    assert 0.0 <= report.ece <= 1.0 and report.nll > 0

    d = report.to_dict()
    again = EvalReport.from_dict(d)
    assert again.to_dict() == d

    with pytest.raises(ValueError, match="classes"):
        evaluate_predictions(
            Predictions(probs[:, :2] / probs[:, :2].sum(1, keepdims=True), labels % 2), LABEL_NAMES
        )
