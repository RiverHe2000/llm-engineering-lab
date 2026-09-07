"""The promotion gate: one function that says whether the fine-tune may ship.

The comparison is paired, because both models are run over the same test split and pairing
removes example difficulty from the difference before any variance is estimated. It is also
deliberately hard to pass. Three things can stop a candidate that looks better on average:

*A floor.* JSON validity has an absolute minimum that a candidate must clear regardless of
how much it improved. A model that went from 0.30 to 0.55 has improved by a lot and is still
useless in production, and an average-based gate would promote it.

*A per-slice non-regression rule.* Slices are reported separately precisely because an
average can hide a loss, so the gate refuses a candidate that lost more than a stated
tolerance on any one slice even when the overall number rose.

*Two significance tests that must agree.* The bootstrap interval for the paired difference
must exclude zero, and the exact McNemar test on the same per-example indicators must clear
alpha. They test the same hypothesis by different routes -- one resamples the examples, the
other conditions on the discordant pairs -- and requiring both is the conservative choice for
a gate whose false positives are shipped models.

The outcome is a `Decision` rather than a boolean: PROMOTE, HOLD when the candidate is not
worse but has not been shown to be better, and REJECT when a floor was breached, a slice
regressed, or the candidate is worse than the non-inferiority margin allows. `exit_code` is
non-zero for anything but PROMOTE, so CI can act on the result without parsing the report.

The Markdown renderer is byte-stable for the same input: every collection is walked in enum
or sorted order, every float goes through one of the formatters below, and negative zero is
normalised, because `-0.0000` and `+0.0000` are different bytes for the same number and a
diff between two runs should show only what changed.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TextIO

from pydantic import BaseModel, ConfigDict, model_validator

from sftdpo.eval.metrics import EvalReport, MetricBlock
from sftdpo.eval.stats import (
    DEFAULT_RESAMPLES,
    Interval,
    NonInferiority,
    discordant_counts,
    mcnemar_exact,
    non_inferiority,
    paired_bootstrap_diff,
    wilson_interval,
)
from sftdpo.schemas import Slice

__all__ = [
    "ComparisonResult",
    "Decision",
    "FieldDelta",
    "Floors",
    "SliceDelta",
    "compare",
    "gate",
]


class Decision(StrEnum):
    """What the gate concluded.

    HOLD exists so that "we could not tell" is a first-class answer. Collapsing it into
    REJECT would make an underpowered run look like a failed one, and collapsing it into
    PROMOTE would ship models on noise.
    """

    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


class Floors(BaseModel):
    """Absolute conditions a candidate must meet whatever the paired statistics say.

    Attributes:
        min_json_valid: Lowest acceptable strict JSON validity rate for the candidate. This
            is the deployability floor: below it, nothing downstream can consume the output
            without a repair step, and a large relative improvement does not change that.
        min_schema_valid: Lowest acceptable schema validity rate. Zero by default, so it is
            opt-in for a project that wants to gate on the stricter number.
        max_slice_regression: How far a single slice may fall before the candidate is
            rejected, in absolute rate. Small by default, because a slice loss is the exact
            failure the per-slice reporting exists to catch.
        max_field_recall_drop: How far one field's recall may fall, in absolute rate. This
            floor exists because a run in this project's own history passed every other rule
            while deleting a field: the preference stage learned to omit the optional `flags`
            array on all 160 test records, taking its recall from 0.61 to 0.00, and because
            the flags it used to emit were wrong more often than the average field, both the
            mean F1 and the mean reward *rose*. No aggregate over records can catch that, and
            neither can a per-slice breakdown, because every slice improved.
        min_field_support: How many gold paths a field must have across the split before its
            recall is allowed to block a promotion. A field gold asks for three times has a
            recall that moves in steps of a third, and rejecting a release on that is noise.
    """

    model_config = ConfigDict(frozen=True)

    min_json_valid: float = 0.0
    min_schema_valid: float = 0.0
    max_slice_regression: float = 0.05
    max_field_recall_drop: float = 0.10
    min_field_support: int = 10

    @model_validator(mode="after")
    def _check_ranges(self) -> Floors:
        for name in (
            "min_json_valid",
            "min_schema_valid",
            "max_slice_regression",
            "max_field_recall_drop",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a rate in [0, 1], got {value}")
        if self.min_field_support < 0:
            raise ValueError(
                f"min_field_support must not be negative, got {self.min_field_support}"
            )
        return self


class SliceDelta(BaseModel):
    """How one difficulty slice moved between the two models."""

    model_config = ConfigDict(frozen=True)

    slice: Slice
    n: int
    baseline: float
    candidate: float
    regressed: bool

    @property
    def delta(self) -> float:
        """Candidate minus baseline success rate on this slice."""
        return self.candidate - self.baseline


class FieldDelta(BaseModel):
    """How the recall of one top-level field moved between the two models.

    Recall rather than F1: the failure this row exists to catch is a field the candidate
    stopped producing, and precision is undefined in the direction that matters -- a model
    that emits nothing has perfect precision on everything it did not emit.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    expected: int
    baseline: float
    candidate: float
    regressed: bool
    below_support: bool = False

    @property
    def delta(self) -> float:
        """Candidate minus baseline recall on this field."""
        return self.candidate - self.baseline


def _support_cell(row: FieldDelta) -> str:
    """The Regressed column, which says why a field was exempt rather than leaving a no."""
    return "below support" if row.below_support else _yes_no(row.regressed)


def _rate(value: float) -> str:
    """Four decimal places, with negative zero normalised so the bytes are stable."""
    cleaned = 0.0 if value == 0 else value
    return f"{cleaned:.4f}"


def _delta(value: float) -> str:
    """Signed four decimal places, with negative zero normalised."""
    cleaned = 0.0 if value == 0 else value
    return f"{cleaned:+.4f}"


def _interval(interval: Interval) -> str:
    return f"[{_delta(interval.low)}, {_delta(interval.high)}]"


def _p_value(value: float) -> str:
    """Fixed notation until it stops being informative, then scientific. Deterministic."""
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.2e}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


class ComparisonResult(BaseModel):
    """Everything the gate looked at, and what it concluded.

    Frozen and self-contained, so the rendered report and the decision cannot drift apart:
    `to_markdown` reads only these fields.
    """

    model_config = ConfigDict(frozen=True)

    baseline_model: str
    candidate_model: str
    n_paired: int
    baseline_metrics: MetricBlock
    candidate_metrics: MetricBlock
    success_diff: Interval
    field_f1_diff: Interval
    baseline_wilson: Interval
    candidate_wilson: Interval
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_p: float
    per_slice: tuple[SliceDelta, ...]
    per_field: tuple[FieldDelta, ...] = ()
    margin: float
    alpha: float
    floors: Floors
    verdict: NonInferiority
    decision: Decision
    reasons: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        """0 for PROMOTE, 1 otherwise, so a CI step can gate on the process status."""
        return 0 if self.decision is Decision.PROMOTE else 1

    @property
    def regressed_slices(self) -> tuple[Slice, ...]:
        """The slices that fell further than the tolerance allows."""
        return tuple(row.slice for row in self.per_slice if row.regressed)

    @property
    def regressed_fields(self) -> tuple[str, ...]:
        """The fields whose recall fell further than the tolerance allows."""
        return tuple(row.field for row in self.per_field if row.regressed)

    def to_markdown(self) -> str:
        """Render the whole comparison as Markdown.

        Returns:
            A report that is byte-identical for byte-identical inputs, so two runs of the
            gate over the same reports produce no diff and a change in the file is always a
            change in the result.
        """
        json_valid = self.candidate_metrics.json_valid_rate
        schema_valid = self.candidate_metrics.schema_valid_rate
        lines: list[str] = [
            f"# Promotion gate: {self.candidate_model} vs {self.baseline_model}",
            "",
            f"**Decision: {self.decision.value.upper()}** (exit code {self.exit_code})",
            "",
        ]
        lines.extend(f"- {reason}" for reason in self.reasons)
        lines.extend(
            [
                "",
                f"## Headline metrics (n = {self.n_paired})",
                "",
                "| Metric | Baseline | Candidate | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        rows: tuple[tuple[str, float, float], ...] = (
            (
                "JSON validity (strict)",
                self.baseline_metrics.json_valid_rate,
                self.candidate_metrics.json_valid_rate,
            ),
            (
                "Schema validity",
                self.baseline_metrics.schema_valid_rate,
                self.candidate_metrics.schema_valid_rate,
            ),
            (
                "Mean field F1",
                self.baseline_metrics.mean_field_f1,
                self.candidate_metrics.mean_field_f1,
            ),
            (
                "Exact match",
                self.baseline_metrics.exact_match_rate,
                self.candidate_metrics.exact_match_rate,
            ),
            (
                "Mean reward",
                self.baseline_metrics.mean_reward,
                self.candidate_metrics.mean_reward,
            ),
        )
        lines.extend(
            f"| {name} | {_rate(before)} | {_rate(after)} | {_delta(after - before)} |"
            for name, before, after in rows
        )
        lines.extend(
            [
                "",
                "## Paired comparison",
                "",
                "| Quantity | Value |",
                "| --- | --- |",
                f"| Success difference | {_delta(self.success_diff.point)} |",
                (
                    f"| Difference CI ({self.success_diff.confidence:.0%}) | "
                    f"{_interval(self.success_diff)} |"
                ),
                f"| Field F1 difference | {_delta(self.field_f1_diff.point)} |",
                f"| Field F1 CI | {_interval(self.field_f1_diff)} |",
                f"| Non-inferiority margin | {_rate(self.margin)} |",
                f"| Non-inferior | {_yes_no(self.verdict.passed)} |",
                f"| McNemar b / c | {self.mcnemar_b} / {self.mcnemar_c} |",
                f"| McNemar p (exact) | {_p_value(self.mcnemar_p)} |",
                f"| Alpha | {_rate(self.alpha)} |",
                (
                    f"| Baseline success (Wilson) | {_rate(self.baseline_wilson.point)} "
                    f"{_interval(self.baseline_wilson)} |"
                ),
                (
                    f"| Candidate success (Wilson) | {_rate(self.candidate_wilson.point)} "
                    f"{_interval(self.candidate_wilson)} |"
                ),
                "",
                "## Floors",
                "",
                "| Floor | Required | Candidate | Cleared |",
                "| --- | ---: | ---: | --- |",
                (
                    f"| JSON validity | {_rate(self.floors.min_json_valid)} | "
                    f"{_rate(json_valid)} | "
                    f"{_yes_no(json_valid >= self.floors.min_json_valid)} |"
                ),
                (
                    f"| Schema validity | {_rate(self.floors.min_schema_valid)} | "
                    f"{_rate(schema_valid)} | "
                    f"{_yes_no(schema_valid >= self.floors.min_schema_valid)} |"
                ),
                "",
                f"## Per slice (tolerance {_rate(self.floors.max_slice_regression)})",
                "",
                "| Slice | n | Baseline | Candidate | Delta | Regressed |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        lines.extend(
            f"| {row.slice.value} | {row.n} | {_rate(row.baseline)} | {_rate(row.candidate)} | "
            f"{_delta(row.delta)} | {_yes_no(row.regressed)} |"
            for row in self.per_slice
        )
        lines.extend(
            [
                "",
                f"## Per field recall (tolerance {_rate(self.floors.max_field_recall_drop)})",
                "",
            ]
        )
        if self.per_field:
            lines.extend(
                [
                    "| Field | Gold paths | Baseline | Candidate | Delta | Regressed |",
                    "| --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            lines.extend(
                f"| {row.field} | {row.expected} | {_rate(row.baseline)} | "
                f"{_rate(row.candidate)} | {_delta(row.delta)} | "
                f"{_support_cell(row)} |"
                for row in self.per_field
            )
        else:
            lines.append(
                "Not reported: one of the two runs predates per-field coverage, so "
                "this check did not run."
            )
        lines.append("")
        return "\n".join(lines)


def _empty_interval(confidence: float) -> Interval:
    """The interval for a comparison with nothing in it."""
    return Interval(point=0.0, low=0.0, high=0.0, confidence=confidence)


def _slice_rows(
    ids: Sequence[str],
    slices: Mapping[str, Slice],
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    tolerance: float,
) -> tuple[SliceDelta, ...]:
    """Per-slice success rates on both sides, in the enum's order."""
    grouped: dict[Slice, list[str]] = {}
    for example_id in ids:
        grouped.setdefault(slices[example_id], []).append(example_id)
    rows: list[SliceDelta] = []
    for name in Slice:
        members = grouped.get(name)
        if not members:
            continue
        before = sum(baseline[key] for key in members) / len(members)
        after = sum(candidate[key] for key in members) / len(members)
        rows.append(
            SliceDelta(
                slice=name,
                n=len(members),
                baseline=before,
                candidate=after,
                regressed=after - before < -tolerance,
            )
        )
    return tuple(rows)


def _field_rows(
    baseline: EvalReport, candidate: EvalReport, floors: Floors
) -> tuple[FieldDelta, ...]:
    """Per-field recall on both sides, for the fields both reports scored.

    Reports written before `per_field` existed carry none, and then this returns nothing and
    the floor cannot fire. That is deliberate: the alternative is inventing a recall of zero
    for an older report and rejecting every candidate compared against one. The rendered
    report says the check did not run rather than implying it passed.
    """
    rows: list[FieldDelta] = []
    for name, before in baseline.per_field.items():
        after = candidate.per_field.get(name)
        if after is None:
            continue
        support = min(before.expected, after.expected)
        below_support = support < floors.min_field_support
        rows.append(
            FieldDelta(
                field=name,
                expected=support,
                baseline=before.recall,
                candidate=after.recall,
                regressed=(
                    not below_support
                    and after.recall - before.recall < -floors.max_field_recall_drop
                ),
                below_support=below_support,
            )
        )
    return tuple(rows)


def _check_same_corpus(baseline: EvalReport, candidate: EvalReport) -> dict[str, Slice]:
    """Confirm the two reports describe the same examples, and return the slice of each.

    Raises:
        ValueError: If an example is filed under a different slice on each side, which means
            the two reports were produced from different corpora and every per-slice number
            below would be comparing unlike with unlike.
    """
    baseline_slices = baseline.slice_by_example()
    candidate_slices = candidate.slice_by_example()
    for example_id, name in sorted(baseline_slices.items()):
        other = candidate_slices.get(example_id)
        if other is not None and other is not name:
            raise ValueError(
                f"example {example_id!r} is {name.value!r} in the baseline report and "
                f"{other.value!r} in the candidate report; the reports come from different "
                "corpora"
            )
    return baseline_slices


def _decide(
    *,
    n_paired: int,
    candidate_metrics: MetricBlock,
    floors: Floors,
    slice_rows: Sequence[SliceDelta],
    field_rows: Sequence[FieldDelta],
    verdict: NonInferiority,
    mcnemar_p: float,
    alpha: float,
) -> tuple[Decision, tuple[str, ...]]:
    """Apply the gate's rules in a fixed order and record why.

    The reasons are built in one order regardless of the data, so the rendered report of two
    similar runs differs only where the runs differ.
    """
    if n_paired == 0:
        return Decision.REJECT, ("the two reports share no examples, so nothing was compared",)

    blocking: list[str] = []
    if candidate_metrics.json_valid_rate < floors.min_json_valid:
        blocking.append(
            f"JSON validity {_rate(candidate_metrics.json_valid_rate)} is below the floor of "
            f"{_rate(floors.min_json_valid)}"
        )
    if candidate_metrics.schema_valid_rate < floors.min_schema_valid:
        blocking.append(
            f"schema validity {_rate(candidate_metrics.schema_valid_rate)} is below the floor "
            f"of {_rate(floors.min_schema_valid)}"
        )
    blocking.extend(
        f"slice {row.slice.value} regressed by {_rate(-row.delta)}, tolerance "
        f"{_rate(floors.max_slice_regression)}"
        for row in slice_rows
        if row.regressed
    )
    blocking.extend(
        f"field {row.field!r} lost {_rate(-row.delta)} of its recall "
        f"({_rate(row.baseline)} to {_rate(row.candidate)} over {row.expected} gold paths), "
        f"tolerance {_rate(floors.max_field_recall_drop)}"
        for row in field_rows
        if row.regressed
    )
    if not verdict.passed:
        blocking.append(
            f"the difference CI lower bound {_delta(verdict.interval.low)} does not clear the "
            f"non-inferiority margin {_delta(-verdict.margin)}"
        )
    if blocking:
        return Decision.REJECT, tuple(blocking)

    if verdict.superior and mcnemar_p <= alpha:
        return Decision.PROMOTE, (
            f"the paired difference is {_delta(verdict.interval.point)} with a "
            f"{verdict.interval.confidence:.0%} CI of {_interval(verdict.interval)}, which "
            "excludes zero",
            f"McNemar p = {_p_value(mcnemar_p)} clears alpha = {_rate(alpha)}",
            (
                "no floor was breached, no slice regressed and no field lost recall"
                if field_rows
                else "no floor was breached and no slice regressed; per-field recall was "
                "not reported by both runs, so that check did not run"
            ),
        )

    holding: list[str] = []
    if not verdict.superior:
        holding.append(
            f"the difference CI {_interval(verdict.interval)} includes zero, so the candidate "
            "is not demonstrably better"
        )
    if mcnemar_p > alpha:
        holding.append(f"McNemar p = {_p_value(mcnemar_p)} does not clear alpha = {_rate(alpha)}")
    holding.append(
        f"the candidate is non-inferior at margin {_rate(verdict.margin)} but has not earned "
        "a promotion"
    )
    return Decision.HOLD, tuple(holding)


def compare(
    baseline: EvalReport,
    candidate: EvalReport,
    *,
    margin: float = 0.0,
    floors: Floors | None = None,
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> ComparisonResult:
    """Compare two evaluation reports and decide whether the candidate may be promoted.

    Args:
        baseline: The incumbent's report.
        candidate: The challenger's report. It must cover exactly the same example ids.
        margin: How much success rate the candidate may lose and still be called
            non-inferior. Zero by default: on the metric the system exists for, any loss is
            a loss.
        floors: Absolute conditions, independent of the comparison. Defaults apply only the
            per-slice non-regression rule.
        alpha: Significance level for the McNemar test.
        confidence: Coverage of the bootstrap and Wilson intervals.
        n_resamples: Bootstrap resamples.
        seed: Seed for the bootstrap, so the decision is reproducible.

    Returns:
        The full comparison, including the decision and the reasons for it.

    Raises:
        ValueError: If the two reports do not cover the same examples, or file an example
            under different slices.
    """
    limits = Floors() if floors is None else floors
    slices = _check_same_corpus(baseline, candidate)
    baseline_success = baseline.success_by_example()
    candidate_success = candidate.success_by_example()

    if not baseline_success and not candidate_success:
        empty = _empty_interval(confidence)
        decision, reasons = _decide(
            n_paired=0,
            candidate_metrics=candidate.overall,
            floors=limits,
            slice_rows=(),
            field_rows=(),
            verdict=non_inferiority(empty, margin),
            mcnemar_p=1.0,
            alpha=alpha,
        )
        return ComparisonResult(
            baseline_model=baseline.model,
            candidate_model=candidate.model,
            n_paired=0,
            baseline_metrics=baseline.overall,
            candidate_metrics=candidate.overall,
            success_diff=empty,
            field_f1_diff=empty,
            baseline_wilson=empty,
            candidate_wilson=empty,
            mcnemar_b=0,
            mcnemar_c=0,
            mcnemar_p=1.0,
            per_slice=(),
            margin=margin,
            alpha=alpha,
            floors=limits,
            verdict=non_inferiority(empty, margin),
            decision=decision,
            reasons=reasons,
        )

    success_diff = paired_bootstrap_diff(
        baseline_success,
        candidate_success,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    field_f1_diff = paired_bootstrap_diff(
        baseline.field_f1_by_example(),
        candidate.field_f1_by_example(),
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    lost, gained = discordant_counts(
        {key: value > 0.0 for key, value in baseline_success.items()},
        {key: value > 0.0 for key, value in candidate_success.items()},
    )
    mcnemar_p = mcnemar_exact(lost, gained)

    ids = sorted(baseline_success)
    rows = _slice_rows(
        ids, slices, baseline_success, candidate_success, limits.max_slice_regression
    )
    field_rows = _field_rows(baseline, candidate, limits)
    verdict = non_inferiority(success_diff, margin)
    decision, reasons = _decide(
        n_paired=len(ids),
        candidate_metrics=candidate.overall,
        floors=limits,
        slice_rows=rows,
        field_rows=field_rows,
        verdict=verdict,
        mcnemar_p=mcnemar_p,
        alpha=alpha,
    )
    return ComparisonResult(
        baseline_model=baseline.model,
        candidate_model=candidate.model,
        n_paired=len(ids),
        baseline_metrics=baseline.overall,
        candidate_metrics=candidate.overall,
        success_diff=success_diff,
        field_f1_diff=field_f1_diff,
        baseline_wilson=wilson_interval(
            round(sum(baseline_success.values())), len(ids), confidence=confidence
        ),
        candidate_wilson=wilson_interval(
            round(sum(candidate_success.values())), len(ids), confidence=confidence
        ),
        mcnemar_b=lost,
        mcnemar_c=gained,
        mcnemar_p=mcnemar_p,
        per_slice=rows,
        per_field=field_rows,
        margin=margin,
        alpha=alpha,
        floors=limits,
        verdict=verdict,
        decision=decision,
        reasons=reasons,
    )


def gate(result: ComparisonResult, stream: TextIO | None = None) -> int:
    """Write the comparison report and return the process exit code.

    Returning the code rather than calling `sys.exit` keeps the gate testable and leaves the
    decision to raise `SystemExit` with the caller, which is where a command-line entry point
    belongs. A CI step runs `raise SystemExit(gate(compare(...)))`.

    Args:
        result: The comparison to report.
        stream: Where to write; standard output by default.

    Returns:
        0 when the decision is PROMOTE, 1 otherwise.
    """
    target = sys.stdout if stream is None else stream
    target.write(result.to_markdown())
    return result.exit_code
