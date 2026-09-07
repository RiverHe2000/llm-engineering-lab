"""Tests for the statistics the promotion gate is allowed to use.

This module is the part of the project a reader is most entitled to distrust, because it is
the code that says "the fine-tune worked". So the tests here are mostly not examples: they
are the defining properties, checked against values worked out by hand where a closed form
exists.

Three of them do most of the work. `normal_quantile` is checked by round-tripping it through
`normal_cdf`, because an inverse that does not invert would widen every confidence interval
by a few per cent and never be noticed. `paired_bootstrap_diff` is checked for antisymmetry
and for independence from dictionary insertion order, which together say that the pairing is
over example ids rather than over whatever order the two reports arrived in. And
`mcnemar_exact` is checked against binomial tail sums computed by hand, including the
degenerate case where neither model won a discordant example and the honest answer is p = 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.eval.stats import (
    Interval,
    bootstrap_ci,
    discordant_counts,
    mcnemar_exact,
    non_inferiority,
    normal_cdf,
    normal_quantile,
    paired_bootstrap_diff,
    wilson_interval,
    z_for_confidence,
)

# Small enough that a property test running a hundred cases stays well under a second, large
# enough that the percentile interval still has distinct ends.
FAST_RESAMPLES = 200


def interval(point: float, low: float, high: float) -> Interval:
    """An interval at the conventional confidence, for the pure-geometry tests."""
    return Interval(point=point, low=low, high=high, confidence=0.95)


# --------------------------------------------------------------------------------------
# The normal distribution helpers
# --------------------------------------------------------------------------------------


def test_normal_cdf_matches_known_points() -> None:
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)
    assert normal_cdf(-1.959963984540054) == pytest.approx(0.025, abs=1e-12)


@settings(max_examples=100, deadline=None)
@given(p=st.floats(min_value=1e-9, max_value=1.0 - 1e-9))
def test_normal_quantile_inverts_the_cdf(p: float) -> None:
    """The Halley refinement is what turns nine correct digits into sixteen."""
    assert normal_cdf(normal_quantile(p)) == pytest.approx(p, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5, 2.0])
def test_normal_quantile_rejects_degenerate_probabilities(p: float) -> None:
    with pytest.raises(ValueError, match="p must be in"):
        normal_quantile(p)


def test_z_for_confidence_matches_the_table() -> None:
    assert z_for_confidence(0.95) == pytest.approx(1.959963984540054, abs=1e-12)
    assert z_for_confidence(0.99) == pytest.approx(2.5758293035489004, abs=1e-12)
    assert z_for_confidence(0.90) == pytest.approx(1.6448536269514722, abs=1e-12)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.2])
def test_z_for_confidence_rejects_impossible_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        z_for_confidence(confidence)


# --------------------------------------------------------------------------------------
# Interval, the shape every result comes back in
# --------------------------------------------------------------------------------------


def test_interval_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="inverted"):
        interval(0.0, 0.4, 0.1)


def test_interval_rejects_confidence_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Interval(point=0.0, low=-0.1, high=0.1, confidence=1.0)


def test_interval_width_and_containment() -> None:
    band = interval(0.20, 0.05, 0.35)
    assert band.width == pytest.approx(0.30)
    assert band.contains(0.05)
    assert band.contains(0.35)
    assert not band.contains(0.36)


@pytest.mark.parametrize(
    ("low", "high", "excludes"),
    [(0.01, 0.30, True), (-0.30, -0.01, True), (-0.05, 0.05, False), (0.0, 0.2, False)],
)
def test_excludes_zero_is_the_boundary_the_gate_reads(
    low: float, high: float, excludes: bool
) -> None:
    """An interval whose end sits exactly on zero has not ruled zero out."""
    assert interval((low + high) / 2, low, high).excludes_zero is excludes


# --------------------------------------------------------------------------------------
# The one-sample bootstrap
# --------------------------------------------------------------------------------------


def test_bootstrap_point_is_the_sample_statistic_not_the_resample_mean() -> None:
    """The point estimate must not move when the seed does."""
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    first = bootstrap_ci(values, n_resamples=FAST_RESAMPLES, seed=0)
    second = bootstrap_ci(values, n_resamples=FAST_RESAMPLES, seed=99)
    assert first.point == pytest.approx(0.625)
    assert second.point == first.point


def test_bootstrap_is_deterministic_under_a_seed() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    first = bootstrap_ci(values, n_resamples=FAST_RESAMPLES, seed=7)
    again = bootstrap_ci(values, n_resamples=FAST_RESAMPLES, seed=7)
    assert (first.low, first.high) == (again.low, again.high)


def test_bootstrap_of_a_constant_sample_has_zero_width() -> None:
    """Every resample of a constant sample is that constant, so there is nothing to resolve."""
    band = bootstrap_ci([0.4] * 12, n_resamples=FAST_RESAMPLES, seed=3)
    assert band.low == pytest.approx(0.4)
    assert band.high == pytest.approx(0.4)
    assert band.width == pytest.approx(0.0)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    values = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    band = bootstrap_ci(values, n_resamples=1000, seed=11)
    assert band.contains(band.point)
    assert band.confidence == 0.95


def test_a_quantile_landing_on_a_replicate_is_not_interpolated() -> None:
    """With 201 replicates the 2.5th and 97.5th percentiles fall exactly on one.

    Every resample of an eight-element 0/1 sample has a mean that is a multiple of an eighth,
    so an end that is not a multiple of an eighth would prove the interpolation ran when it
    should not have.
    """
    band = bootstrap_ci([1.0] * 4 + [0.0] * 4, n_resamples=201, seed=8)
    assert band.low * 8 == round(band.low * 8)
    assert band.high * 8 == round(band.high * 8)


def test_bootstrap_honours_a_custom_statistic() -> None:
    def largest(values: Sequence[float]) -> float:
        return max(values)

    band = bootstrap_ci([1.0, 2.0, 9.0], statistic=largest, n_resamples=FAST_RESAMPLES, seed=1)
    assert band.point == pytest.approx(9.0)
    assert band.high == pytest.approx(9.0)
    assert band.low <= 9.0


def test_bootstrap_narrows_as_the_sample_grows() -> None:
    """More data must buy a narrower interval; that is the only reason to collect it."""
    small = bootstrap_ci([1.0, 0.0] * 5, n_resamples=1000, seed=5)
    large = bootstrap_ci([1.0, 0.0] * 50, n_resamples=1000, seed=5)
    assert large.width < small.width


def test_bootstrap_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


@pytest.mark.parametrize(
    ("n_resamples", "confidence", "message"),
    [(0, 0.95, "n_resamples"), (-1, 0.95, "n_resamples"), (10, 0.0, "confidence")],
)
def test_bootstrap_rejects_impossible_settings(
    n_resamples: int, confidence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_ci([1.0, 0.0], n_resamples=n_resamples, confidence=confidence)


# --------------------------------------------------------------------------------------
# The paired bootstrap, which is what makes the comparison powerful
# --------------------------------------------------------------------------------------


def test_paired_difference_point_is_the_mean_difference() -> None:
    baseline = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0}
    candidate = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    band = paired_bootstrap_diff(baseline, candidate, n_resamples=FAST_RESAMPLES, seed=0)
    assert band.point == pytest.approx(0.5)


def test_paired_difference_of_a_run_against_itself_is_exactly_zero() -> None:
    run = {"a": 1.0, "b": 0.0, "c": 0.6}
    band = paired_bootstrap_diff(run, dict(run), n_resamples=FAST_RESAMPLES, seed=4)
    assert (band.point, band.low, band.high) == (0.0, 0.0, 0.0)


@settings(max_examples=40, deadline=None)
@given(
    scores=st.lists(st.tuples(st.floats(0.0, 1.0), st.floats(0.0, 1.0)), min_size=2, max_size=12)
)
def test_paired_difference_ignores_insertion_order(scores: list[tuple[float, float]]) -> None:
    """Pairing is over example ids, so the order the two reports were written in cannot matter."""
    ids = [f"ex-{index:02d}" for index in range(len(scores))]
    baseline = {key: before for key, (before, _) in zip(ids, scores, strict=True)}
    candidate = {key: after for key, (_, after) in zip(ids, scores, strict=True)}
    reversed_baseline = dict(reversed(list(baseline.items())))
    reversed_candidate = dict(reversed(list(candidate.items())))

    straight = paired_bootstrap_diff(baseline, candidate, n_resamples=FAST_RESAMPLES, seed=2)
    shuffled = paired_bootstrap_diff(
        reversed_baseline, reversed_candidate, n_resamples=FAST_RESAMPLES, seed=2
    )
    assert (straight.point, straight.low, straight.high) == (
        shuffled.point,
        shuffled.low,
        shuffled.high,
    )


def test_paired_difference_is_antisymmetric() -> None:
    """Swapping the two models negates the difference; the pairing has no preferred side."""
    baseline = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": 1.0}
    candidate = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0, "e": 0.0}
    forward = paired_bootstrap_diff(baseline, candidate, n_resamples=FAST_RESAMPLES, seed=13)
    backward = paired_bootstrap_diff(candidate, baseline, n_resamples=FAST_RESAMPLES, seed=13)
    assert backward.point == -forward.point
    assert backward.low == pytest.approx(-forward.high, abs=1e-12)
    assert backward.high == pytest.approx(-forward.low, abs=1e-12)


def test_paired_difference_refuses_runs_over_different_examples() -> None:
    with pytest.raises(ValueError, match="cannot be paired"):
        paired_bootstrap_diff({"a": 1.0}, {"b": 1.0})


def test_paired_difference_refuses_two_empty_runs() -> None:
    with pytest.raises(ValueError, match="share no examples"):
        paired_bootstrap_diff({}, {})


def test_paired_difference_error_summarises_a_long_mismatch() -> None:
    """A hundred mismatched ids must not produce a hundred-line error message."""
    baseline = {f"ex-{index:03d}": 1.0 for index in range(20)}
    with pytest.raises(ValueError, match="and 15 more"):
        paired_bootstrap_diff(baseline, {})


# --------------------------------------------------------------------------------------
# McNemar, exact
# --------------------------------------------------------------------------------------


def test_discordant_counts_reads_the_two_directions() -> None:
    baseline = {"a": True, "b": True, "c": False, "d": False, "e": True}
    candidate = {"a": True, "b": False, "c": True, "d": False, "e": False}
    # b lost by the candidate: "b" and "e". Gained: "c". "a" and "d" agree and are ignored.
    assert discordant_counts(baseline, candidate) == (2, 1)


def test_discordant_counts_of_identical_runs_is_empty() -> None:
    run = {"a": True, "b": False}
    assert discordant_counts(run, dict(run)) == (0, 0)


def test_discordant_counts_refuses_unmatched_runs() -> None:
    with pytest.raises(ValueError, match="cannot be paired"):
        discordant_counts({"a": True}, {"a": True, "b": False})


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        # No discordant pairs: the test has nothing to go on and must say so.
        (0, 0, 1.0),
        # n = 1, one tail is C(1,0) = 1 of 2 outcomes.
        (0, 1, 1.0),
        (1, 0, 1.0),
        # n = 5, tail = C(5,0) = 1 of 32.
        (0, 5, 2 * 1 / 32),
        # n = 6, tail = 1 of 64.
        (0, 6, 2 * 1 / 64),
        # n = 10, tail = C(10,0) + C(10,1) = 11 of 1024.
        (1, 9, 2 * 11 / 1024),
        # n = 10, tail = 1 + 10 + 45 = 56 of 1024.
        (2, 8, 2 * 56 / 1024),
        # n = 10, tail = 638 of 1024, so twice the tail exceeds one and is capped.
        (5, 5, 1.0),
        # n = 3, tail = 1 of 8.
        (3, 0, 2 * 1 / 8),
    ],
)
def test_mcnemar_matches_hand_computed_binomial_tails(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(expected, abs=1e-15)


@settings(max_examples=100, deadline=None)
@given(b=st.integers(0, 40), c=st.integers(0, 40))
def test_mcnemar_is_symmetric_and_a_probability(b: int, c: int) -> None:
    """The null hypothesis names no favourite, so the p-value cannot either."""
    p = mcnemar_exact(b, c)
    assert p == mcnemar_exact(c, b)
    assert 0.0 < p <= 1.0


def test_mcnemar_falls_as_the_split_gets_more_extreme() -> None:
    """With the number of discordant pairs fixed, a more lopsided split is more evidence."""
    n = 12
    values = [mcnemar_exact(k, n - k) for k in range(n // 2 + 1)]
    assert values == sorted(values)


def test_mcnemar_handles_counts_far_beyond_a_float_factorial() -> None:
    """Integer binomials and a single division: a large split must not overflow."""
    assert mcnemar_exact(0, 2000) == pytest.approx(0.0, abs=1e-300)
    assert math.isfinite(mcnemar_exact(900, 1100))


@pytest.mark.parametrize(("b", "c"), [(-1, 0), (0, -1), (-2, -3)])
def test_mcnemar_rejects_negative_counts(b: int, c: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_exact(b, c)


# --------------------------------------------------------------------------------------
# Wilson
# --------------------------------------------------------------------------------------


def test_wilson_matches_published_intervals() -> None:
    zero = wilson_interval(0, 10)
    assert zero.point == 0.0
    assert zero.low == 0.0
    assert zero.high == pytest.approx(0.2775, abs=5e-5)

    half = wilson_interval(5, 10)
    assert half.point == 0.5
    assert half.low == pytest.approx(0.2366, abs=5e-5)
    assert half.high == pytest.approx(0.7634, abs=5e-5)


def test_wilson_saturates_at_both_ends() -> None:
    """The interval must stay inside [0, 1] where a normal interval would not."""
    assert wilson_interval(0, 8).low == 0.0
    assert wilson_interval(8, 8).high == 1.0


@settings(max_examples=100, deadline=None)
@given(data=st.integers(1, 200).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
def test_wilson_contains_the_observation_and_stays_in_range(data: tuple[int, int]) -> None:
    successes, n = data
    band = wilson_interval(successes, n)
    assert 0.0 <= band.low <= band.high <= 1.0
    assert band.contains(band.point)


def test_wilson_narrows_with_more_trials() -> None:
    assert wilson_interval(50, 100).width < wilson_interval(5, 10).width


@pytest.mark.parametrize(
    ("successes", "n", "confidence", "message"),
    [
        (0, 0, 0.95, "n must be positive"),
        (0, -3, 0.95, "n must be positive"),
        (11, 10, 0.95, "successes must be in"),
        (-1, 10, 0.95, "successes must be in"),
        (5, 10, 1.0, "confidence must be in"),
    ],
)
def test_wilson_rejects_impossible_arguments(
    successes: int, n: int, confidence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        wilson_interval(successes, n, confidence=confidence)


# --------------------------------------------------------------------------------------
# Non-inferiority
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "margin", "passed", "superior", "inferior"),
    [
        # Clearly better: the whole interval is above zero.
        (0.05, 0.20, 0.05, True, True, False),
        # Straddles zero but never falls below the margin: not worse, not shown better.
        (-0.03, 0.10, 0.05, True, False, False),
        # Falls below the margin at the low end: the loss is not bounded.
        (-0.12, 0.02, 0.05, False, False, False),
        # Entirely below the margin: a demonstrated loss.
        (-0.30, -0.10, 0.05, False, False, True),
        # A zero margin demands strict non-inferiority, and a lower bound of exactly zero
        # does not clear it.
        (0.0, 0.10, 0.0, False, False, False),
    ],
)
def test_non_inferiority_verdicts(
    low: float, high: float, margin: float, passed: bool, superior: bool, inferior: bool
) -> None:
    verdict = non_inferiority(interval((low + high) / 2, low, high), margin)
    assert verdict.passed is passed
    assert verdict.superior is superior
    assert verdict.inferior is inferior
    assert verdict.margin == margin


def test_non_inferiority_keeps_the_interval_it_judged() -> None:
    band = interval(0.1, 0.02, 0.18)
    assert non_inferiority(band, 0.05).interval == band


@pytest.mark.parametrize("margin", [-0.01, -1.0])
def test_non_inferiority_rejects_a_negative_margin(margin: float) -> None:
    """A negative margin would be a superiority test wearing the wrong name."""
    with pytest.raises(ValueError, match="non-negative"):
        non_inferiority(interval(0.0, -0.1, 0.1), margin)


@pytest.mark.parametrize("margin", [float("nan"), float("inf")])
def test_non_inferiority_rejects_a_nonfinite_margin(margin: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        non_inferiority(interval(0.0, -0.1, 0.1), margin)


def test_a_degenerate_sample_gives_a_zero_width_interval_not_an_error() -> None:
    """Every observation identical is not a pathological input; it is a paired tie.

    Two arms that differ by the same amount on every example make every bootstrap replicate
    equal, so the two quantiles are the same number computed two different ways. Linear
    interpolation can put them an ulp apart in the wrong order, and `Interval` refuses an
    inverted interval, so the comparison raised instead of reporting a zero-width interval.
    Found by hypothesis, on the exact pair of values below.
    """
    identical = [0.11205969356422155 - 0.3595646179376875] * 8
    interval = bootstrap_ci(identical, n_resamples=64, seed=0)

    assert interval.high - interval.low == pytest.approx(0.0, abs=1e-12)
    # The point is *not* clamped into the interval. A percentile bootstrap interval can
    # legitimately exclude the observed statistic when the replicate distribution is biased,
    # and forcing containment would hide that. Here the two differ by one ulp of summation
    # order, which is what the tolerance says and all it says.
    assert interval.point == pytest.approx(interval.low, abs=1e-12)
