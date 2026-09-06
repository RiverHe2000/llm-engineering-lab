from __future__ import annotations

import math

import numpy as np
import pytest

from loraeval.metrics import (
    accuracy,
    bootstrap_ci,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    mcnemar_test,
    negative_log_likelihood,
    per_class_metrics,
)

Y_TRUE = np.array([0, 0, 1, 1, 2, 2, 2, 1])
Y_PRED = np.array([0, 1, 1, 1, 2, 0, 2, 1])


class TestBasics:
    def test_confusion_matrix(self) -> None:
        cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
        expected = np.array([[1, 1, 0], [0, 3, 0], [1, 0, 2]])
        assert np.array_equal(cm, expected)
        assert cm.sum() == len(Y_TRUE)

    def test_confusion_matrix_validation(self) -> None:
        with pytest.raises(ValueError):
            confusion_matrix(np.array([0, 1]), np.array([0]), 2)
        with pytest.raises(ValueError):
            confusion_matrix(np.array([0, 5]), np.array([0, 1]), 2)
        with pytest.raises(ValueError):
            confusion_matrix(np.array([[0]]), np.array([[0]]), 2)

    def test_accuracy(self) -> None:
        assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(6 / 8)
        with pytest.raises(ValueError):
            accuracy(np.array([]), np.array([]))

    def test_per_class_and_macro_f1(self) -> None:
        metrics = per_class_metrics(confusion_matrix(Y_TRUE, Y_PRED, 3))
        # class 0: tp=1 fp=1 fn=1 -> P=R=F1=0.5
        assert metrics[0].precision == 0.5 and metrics[0].recall == 0.5 and metrics[0].f1 == 0.5
        assert metrics[0].support == 2
        # class 1: tp=3 fp=1 fn=0 -> P=0.75 R=1 F1=0.857
        assert metrics[1].precision == 0.75 and metrics[1].recall == 1.0
        assert metrics[1].f1 == pytest.approx(2 * 0.75 / 1.75)
        # class 2: tp=2 fp=0 fn=1 -> P=1 R=2/3 F1=0.8
        assert metrics[2].f1 == pytest.approx(0.8)
        assert macro_f1(Y_TRUE, Y_PRED, 3) == pytest.approx((0.5 + 2 * 0.75 / 1.75 + 0.8) / 3)

    def test_absent_class_gets_zero_not_nan(self) -> None:
        metrics = per_class_metrics(confusion_matrix(np.array([0, 0]), np.array([0, 0]), 3))
        assert metrics[1].f1 == 0.0 and metrics[1].support == 0
        assert not math.isnan(macro_f1(np.array([0, 0]), np.array([0, 0]), 3))

    def test_nll(self) -> None:
        probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        assert negative_log_likelihood(probs, np.array([0, 1])) == pytest.approx(
            -(math.log(0.9) + math.log(0.8)) / 2
        )
        assert math.isfinite(negative_log_likelihood(np.array([[1.0, 0.0]]), np.array([1])))


class TestCalibration:
    def test_perfectly_calibrated_is_zero(self) -> None:
        # 10 predictions at 80% confidence, exactly 8 of them right.
        probs = np.tile([0.8, 0.2], (10, 1))
        y = np.array([0] * 8 + [1] * 2)
        assert expected_calibration_error(probs, y, n_bins=10) == pytest.approx(0.0, abs=1e-9)

    def test_overconfident_is_large(self) -> None:
        probs = np.tile([0.99, 0.01], (10, 1))
        y = np.array([0] * 5 + [1] * 5)  # only 50% right at 99% confidence
        assert expected_calibration_error(probs, y, n_bins=10) == pytest.approx(0.49)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error(np.ones((3, 2)), np.zeros(2, dtype=int))
        with pytest.raises(ValueError):
            expected_calibration_error(np.ones((2, 2)), np.zeros(2, dtype=int), n_bins=0)


class TestBootstrap:
    def test_ci_contains_point_and_is_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 200)
        pred = np.where(rng.random(200) < 0.8, y, (y + 1) % 3)
        ci = bootstrap_ci(accuracy, y, pred, n_boot=500, seed=1)
        assert ci.lower <= ci.point <= ci.upper
        assert 0.7 < ci.point < 0.9
        assert 0.02 < ci.upper - ci.lower < 0.15
        again = bootstrap_ci(accuracy, y, pred, n_boot=500, seed=1)
        assert (again.lower, again.upper) == (ci.lower, ci.upper)
        assert "[" in str(ci)

    def test_interval_shrinks_with_more_data(self) -> None:
        rng = np.random.default_rng(0)
        widths = []
        for n in (50, 800):
            y = rng.integers(0, 2, n)
            pred = np.where(rng.random(n) < 0.75, y, 1 - y)
            ci = bootstrap_ci(accuracy, y, pred, n_boot=400, seed=0)
            widths.append(ci.upper - ci.lower)
        assert widths[1] < widths[0] / 2

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci(accuracy, Y_TRUE, Y_PRED, level=1.5)
        with pytest.raises(ValueError):
            bootstrap_ci(accuracy, Y_TRUE, Y_PRED, n_boot=0)


class TestMcNemar:
    def test_identical_models_p_is_one(self) -> None:
        res = mcnemar_test(Y_TRUE, Y_PRED, Y_PRED)
        assert res.b == res.c == 0 and res.p_value == 1.0

    def test_exact_branch_matches_binomial(self) -> None:
        y = np.zeros(30, dtype=int)
        a = np.zeros(30, dtype=int)  # A always right
        b = np.concatenate([np.ones(10, dtype=int), np.zeros(20, dtype=int)])  # B wrong on 10
        res = mcnemar_test(y, a, b)
        assert (res.b, res.c) == (10, 0)
        assert res.method == "exact"
        assert res.p_value == pytest.approx(2 * 0.5**10)
        assert res.n_discordant == 10

    def test_symmetry(self) -> None:
        y = np.zeros(30, dtype=int)
        a = np.concatenate([np.ones(4, dtype=int), np.zeros(26, dtype=int)])
        b = np.concatenate([np.zeros(26, dtype=int), np.ones(4, dtype=int)])
        r1, r2 = mcnemar_test(y, a, b), mcnemar_test(y, b, a)
        assert (r1.b, r1.c) == (r2.c, r2.b)
        assert r1.p_value == r2.p_value == 1.0  # perfectly balanced discordance

    def test_chi2_branch_for_many_discordant_pairs(self) -> None:
        y = np.zeros(100, dtype=int)
        a = np.zeros(100, dtype=int)
        b = np.concatenate([np.ones(40, dtype=int), np.zeros(60, dtype=int)])
        res = mcnemar_test(y, a, b)
        assert res.method == "chi2-corrected"
        assert res.statistic == pytest.approx((40 - 1) ** 2 / 40)
        assert res.p_value < 1e-6

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            mcnemar_test(np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.zeros(2, dtype=int))
