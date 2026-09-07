"""The statistics the promotion gate is allowed to use, implemented from their definitions.

There is no SciPy here, and that is a decision rather than an omission. The gate is the part
of the project a reader is most entitled to distrust -- it is the code that says "this
fine-tune worked" -- so every number it produces is computed from a stated definition in a
few lines that can be read end to end, and each one is tested against a value worked out by
hand. A dependency would move the interesting part out of the repository.

Four ideas do the work.

*Percentile bootstrap.* The sampling distribution of a rate over 48 test examples is not
normal, and pretending otherwise is how a difference of two points acquires a confident
interval. Resampling makes no distributional assumption.

*Pairing.* Two models are run over the same examples, so the difference is computed example
by example and the bootstrap resamples the differences. Example difficulty then cancels
before any variance is estimated: the long-context notes that are hard for the baseline are
the same notes that are hard for the candidate, and an unpaired test would have to pay for
that shared difficulty out of the effect it is trying to detect.

*Exact McNemar.* The discordant counts are small -- often fewer than ten examples change
answer -- and the chi-square approximation is unreliable there. The exact binomial test is a
sum of binomial coefficients and needs no approximation at all.

*Non-inferiority.* "Not significantly worse" is not a result. A margin has to be named in
advance, and the claim is then that the lower end of the interval sits above it.

Every function that resamples takes a seed and is a pure function of its inputs and that
seed, so a promotion decision can be reproduced exactly from the two reports that produced
it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from random import Random

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "DEFAULT_RESAMPLES",
    "Interval",
    "NonInferiority",
    "bootstrap_ci",
    "discordant_counts",
    "mcnemar_exact",
    "non_inferiority",
    "normal_cdf",
    "normal_quantile",
    "paired_bootstrap_diff",
    "wilson_interval",
    "z_for_confidence",
]

DEFAULT_RESAMPLES = 2000
"""Enough for a percentile interval whose ends are stable to about three decimal places,
and cheap enough that the gate runs in well under a second on a test split of this size."""


class Interval(BaseModel):
    """A point estimate and the interval around it, with the confidence it was built at.

    The confidence travels with the numbers because an interval quoted without it is not a
    result, and because the gate compares intervals from different calls.
    """

    model_config = ConfigDict(frozen=True)

    point: float
    low: float
    high: float
    confidence: float

    @model_validator(mode="after")
    def _check_order(self) -> Interval:
        if self.low > self.high:
            raise ValueError(f"interval is inverted: [{self.low}, {self.high}]")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        return self

    @property
    def width(self) -> float:
        """How wide the interval is; the honest measure of how much data there was."""
        return self.high - self.low

    def contains(self, value: float) -> bool:
        """Whether `value` lies inside the interval, ends included."""
        return self.low <= value <= self.high

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval rules out no difference at all."""
        return not self.contains(0.0)


class NonInferiority(BaseModel):
    """The verdict of a one-sided comparison against a pre-declared margin.

    Sign convention: the interval is over `candidate - baseline`, and `margin` is a positive
    tolerance. The candidate is non-inferior when the lower end of the interval sits above
    `-margin`, which is the claim "whatever it lost, it lost less than the margin".
    """

    model_config = ConfigDict(frozen=True)

    margin: float
    interval: Interval

    @property
    def passed(self) -> bool:
        """Whether the candidate cleared the margin."""
        return self.interval.low > -self.margin

    @property
    def superior(self) -> bool:
        """Whether the candidate is better, not merely not worse."""
        return self.interval.low > 0.0

    @property
    def inferior(self) -> bool:
        """Whether the whole interval sits at or below the margin: a demonstrated loss."""
        return self.interval.high <= -self.margin


def normal_cdf(x: float) -> float:
    """The standard normal distribution function, via `math.erfc`.

    Used to refine `normal_quantile` and to check it in the tests: an inverse that does not
    invert is the kind of error that silently widens every confidence interval by a few per
    cent and is never noticed.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# Acklam's rational approximation to the inverse normal CDF. Coefficients are reproduced
# verbatim; the approximation alone is good to about 1.15e-9 relative, and the Halley step
# below takes it to full double precision.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def _acklam(p: float) -> float:
    """The rational approximation, before refinement."""
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p > _P_HIGH:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
        * q
        / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    )


def normal_quantile(p: float) -> float:
    """The inverse of `normal_cdf`.

    Args:
        p: A probability, strictly inside (0, 1).

    Returns:
        The value `x` with `normal_cdf(x) == p`, to double precision. The rational
        approximation is followed by one Halley step against the exact CDF, which is what
        turns nine correct digits into sixteen and makes `z_for_confidence(0.95)` agree with
        a table to every digit anyone would quote.

    Raises:
        ValueError: If `p` is not strictly between 0 and 1, where the quantile is infinite.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    x = _acklam(p)
    error = normal_cdf(x) - p
    density_term = error * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - density_term / (1.0 + x * density_term / 2.0)


def z_for_confidence(confidence: float) -> float:
    """The two-sided critical value for a confidence level, e.g. 1.96 at 95 %.

    Raises:
        ValueError: If `confidence` is not strictly inside (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return normal_quantile(0.5 + confidence / 2.0)


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted sequence."""
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _check_resampling(n_resamples: int, confidence: float) -> None:
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for a statistic of one sample.

    Args:
        values: The observations. For a rate, pass one 0.0/1.0 per example.
        statistic: What to compute on each resample; the mean by default.
        n_resamples: How many resamples to draw.
        confidence: Coverage of the returned interval.
        seed: Seed for the resampling. The whole interval is a pure function of the inputs
            and this seed, which is what lets a promotion decision be re-derived later.

    Returns:
        The interval, whose `point` is the statistic of the original sample rather than the
        mean of the resamples: the point estimate should not move when the seed does.

    Raises:
        ValueError: If `values` is empty, `n_resamples` is not positive, or `confidence` is
            not strictly inside (0, 1).
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    _check_resampling(n_resamples, confidence)

    estimate = _mean if statistic is None else statistic
    rng = Random(seed)
    size = len(values)
    # `randrange` per draw rather than `random.choices`: the index sequence is then an
    # explicit consequence of the seed and does not depend on how the standard library
    # happens to implement weighted selection today.
    replicates = sorted(
        estimate([values[rng.randrange(size)] for _ in range(size)]) for _ in range(n_resamples)
    )
    alpha = 1.0 - confidence
    low = _quantile(replicates, alpha / 2.0)
    high = _quantile(replicates, 1.0 - alpha / 2.0)
    # A degenerate sample -- every observation identical, which a paired comparison reaches
    # whenever the two arms differ by the same amount everywhere -- makes every replicate
    # equal, and the two quantiles are then the same number computed two different ways.
    # Linear interpolation can put them an ulp apart in the wrong order, and `Interval`
    # refuses an inverted interval. Ordering them is exact here rather than a fudge: when the
    # replicates are all equal the true interval has zero width. Found by hypothesis.
    if high < low:
        low, high = high, low
    return Interval(
        point=estimate(values),
        low=low,
        high=high,
        confidence=confidence,
    )


def _check_paired_keys(baseline: Mapping[str, object], candidate: Mapping[str, object]) -> None:
    """Refuse two runs that do not cover exactly the same examples.

    Silently intersecting them would be worse than failing: the comparison would then be
    computed over a different population from the one the reports describe, and nothing in
    the output would say so.
    """
    missing = sorted(set(baseline) ^ set(candidate))
    if missing:
        shown = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise ValueError(
            f"the two runs do not cover the same examples: {shown}{suffix} appears in one "
            "but not the other, so the comparison cannot be paired"
        )


def _paired_differences(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
) -> list[float]:
    """Candidate minus baseline, example by example, in sorted id order."""
    _check_paired_keys(baseline, candidate)
    if not baseline:
        raise ValueError("cannot compare two runs that share no examples")
    # Sorted, not insertion-ordered: the difference must not depend on which order the two
    # reports happened to be written in.
    return [candidate[key] - baseline[key] for key in sorted(baseline)]


def paired_bootstrap_diff(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval for the mean paired difference, `candidate - baseline`.

    The resampling is over example ids, not over the two sets of scores independently. That
    is the whole point: each resample keeps the two models' scores for an example together,
    so the variance being estimated is the variance of the difference rather than the sum of
    two much larger variances.

    Args:
        baseline: Per-example score for the incumbent, keyed by example id.
        candidate: Per-example score for the challenger, keyed the same way.
        n_resamples: How many resamples to draw.
        confidence: Coverage of the returned interval.
        seed: Seed for the resampling.

    Returns:
        The interval for the mean difference. A positive `point` means the candidate scored
        higher on average.

    Raises:
        ValueError: If the two mappings do not cover exactly the same example ids, or if
            they are empty.
    """
    differences = _paired_differences(baseline, candidate)
    return bootstrap_ci(
        differences,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def discordant_counts(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
) -> tuple[int, int]:
    """Count the examples whose outcome changed, in each direction.

    Args:
        baseline: Per-example success for the incumbent.
        candidate: Per-example success for the challenger.

    Returns:
        `(b, c)`, where `b` is the number of examples the baseline got right and the
        candidate got wrong, and `c` the reverse. The concordant examples are deliberately
        not returned: McNemar's test does not use them, because an example both models got
        right carries no information about which is better. Two empty runs give `(0, 0)`,
        which `mcnemar_exact` reads as no evidence.

    Raises:
        ValueError: If the two mappings do not cover the same example ids.
    """
    _check_paired_keys(baseline, candidate)
    lost = sum(1 for key in baseline if baseline[key] and not candidate[key])
    gained = sum(1 for key in baseline if not baseline[key] and candidate[key])
    return lost, gained


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant counts.

    Under the null hypothesis that the two models are equally likely to win a discordant
    example, the number of wins is Binomial(b + c, 1/2). The two-sided p-value is twice the
    smaller tail, capped at one. Exact rather than chi-square because b + c is typically
    under twenty on a test split of this size, where the continuity-corrected approximation
    is visibly wrong.

    Args:
        b: Examples the baseline won.
        c: Examples the candidate won.

    Returns:
        The p-value. With no discordant examples the test has nothing to go on and returns
        exactly 1.0, which is the right answer rather than a special case: two models that
        never disagree provide no evidence that they differ.

    Raises:
        ValueError: If either count is negative.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    # Both sides are exact integers and the quotient is at most one, so the single division
    # is correctly rounded and cannot overflow however many discordant pairs there are.
    outcomes = 1 << n
    return min(1.0, 2 * tail / outcomes)


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook normal interval because a small model's JSON validity rate
    lives near the ends of the unit interval, where the normal interval is both too narrow
    and capable of proposing a negative rate. The Wilson interval stays inside [0, 1] and
    behaves at 0 and 1 successes.

    Args:
        successes: Number of successes.
        n: Number of trials.
        confidence: Coverage of the interval.

    Returns:
        The interval, with `point` the observed proportion. Note that the point estimate is
        not the centre of the interval: the Wilson centre is pulled towards one half, which
        is exactly the shrinkage that makes the interval honest at the ends. The interval
        always contains that point, which at zero or n successes is true only in exact
        arithmetic -- there the two terms cancel, and in floating point the cancellation
        leaves a few times 1e-17 of the wrong sign.

    Raises:
        ValueError: If `n` is not positive, if `successes` is outside `[0, n]`, or if the
            confidence is not strictly inside (0, 1).
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    z = z_for_confidence(confidence)
    z2 = z * z
    proportion = successes / n
    denominator = 1.0 + z2 / n
    centre = (proportion + z2 / (2 * n)) / denominator
    half = (z / denominator) * math.sqrt(proportion * (1.0 - proportion) / n + z2 / (4 * n * n))
    # Clamped against the observation as well as against [0, 1]: an interval that excludes
    # its own point estimate is a rendering embarrassment and a trap for any caller that
    # asks `contains`.
    return Interval(
        point=proportion,
        low=min(proportion, max(0.0, centre - half)),
        high=max(proportion, min(1.0, centre + half)),
        confidence=confidence,
    )


def non_inferiority(interval: Interval, margin: float) -> NonInferiority:
    """Judge an interval over `candidate - baseline` against a pre-declared margin.

    Args:
        interval: The interval for the difference.
        margin: How much the candidate is allowed to lose, as a positive number. A margin of
            zero demands strict non-inferiority, which is the right setting when the metric
            is the one the system is for.

    Returns:
        The verdict, which keeps the margin and the interval so a report can show the
        reasoning rather than only the answer.

    Raises:
        ValueError: If the margin is negative or not finite. A negative margin would state
            that a candidate must beat the baseline by a fixed amount merely to be called
            non-inferior, which is a superiority test wearing the wrong name.
    """
    if not math.isfinite(margin):
        raise ValueError(f"margin must be finite, got {margin}")
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    return NonInferiority(margin=margin, interval=interval)
