"""Tests for the promotion gate.

This is the code that decides whether a fine-tune ships, so the tests are written from the
adversary's side: each one asks what a candidate would have to do to slip past a rule that was
put there to stop it. A candidate that improved a lot and is still unusable must be refused by
the floor. A candidate that lifted the average by losing a slice must be refused by the
per-slice rule. A candidate that is better only according to one of the two significance tests
must not be promoted on that alone.

The reports are built directly from `ExampleOutcome` rows rather than by running a model,
because the gate's job is arithmetic and rule application over those rows and nothing else.
Building them by hand is what makes it possible to state the expected decision in the test
name.

One consequence of the default settings is worth naming, since two tests turn on it. With
`margin=0.0` the non-inferiority claim degenerates into a superiority claim -- the lower end
of the interval has to sit above zero -- so a candidate that merely failed to get worse is
refused rather than held. HOLD becomes the answer only once a positive margin says how much
loss the project is willing to tolerate.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any, cast

import pytest

from sftdpo.eval.compare import ComparisonResult, Decision, Floors, compare, gate
from sftdpo.eval.metrics import (
    EvalReport,
    ExampleOutcome,
    FieldCoverage,
    MetricBlock,
    ParseGap,
)
from sftdpo.schemas import Slice

# Small enough to keep the whole file well under a second, large enough for a percentile
# interval with distinct ends.
FAST_RESAMPLES = 300


def flags(pattern: str) -> list[bool]:
    """Read a run as a string of ones and zeroes, so a test can state it on one line."""
    return [character == "1" for character in pattern]


def run(
    model: str,
    successes: Sequence[bool],
    *,
    slices: Sequence[Slice] | None = None,
    json_valid: Sequence[bool] | None = None,
) -> EvalReport:
    """A report over `ex-00`, `ex-01`, ... with the outcomes spelled out.

    JSON validity follows schema validity unless the caller separates them, which is what the
    floor tests need: a model can emit an object that parses and still fails the schema.
    """
    where = [Slice.CLEAN] * len(successes) if slices is None else list(slices)
    parsed = list(successes) if json_valid is None else list(json_valid)
    outcomes = tuple(
        ExampleOutcome(
            example_id=f"ex-{index:02d}",
            slice=where[index],
            json_valid=parsed[index],
            schema_valid=ok,
            field_f1=1.0 if ok else 0.0,
            exact_match=ok,
            reward=1.0 if ok else (0.2 if parsed[index] else 0.0),
            lenient_json_valid=parsed[index],
        )
        for index, ok in enumerate(successes)
    )
    return EvalReport(
        model=model,
        overall=MetricBlock.over(outcomes),
        parse_gap=ParseGap.over(outcomes),
        outcomes=outcomes,
    )


BASELINE_PATTERN = "1" * 8 + "0" * 32
CANDIDATE_PATTERN = "1" * 32 + "0" * 8


def promoted() -> ComparisonResult:
    """A candidate that clears every rule: it keeps all eight wins and adds twenty-four."""
    return compare(
        run("base", flags(BASELINE_PATTERN)),
        run("sft", flags(CANDIDATE_PATTERN)),
        floors=Floors(min_json_valid=0.5),
        n_resamples=FAST_RESAMPLES,
        seed=0,
    )


# --------------------------------------------------------------------------------------
# The three decisions
# --------------------------------------------------------------------------------------


def test_a_clear_improvement_is_promoted() -> None:
    result = promoted()
    assert result.decision is Decision.PROMOTE
    assert result.exit_code == 0
    assert result.n_paired == 40
    assert result.success_diff.point == pytest.approx(0.6)
    assert result.success_diff.excludes_zero
    assert result.regressed_slices == ()


def test_the_mcnemar_counts_are_the_examples_that_changed_answer() -> None:
    result = promoted()
    assert (result.mcnemar_b, result.mcnemar_c) == (0, 24)
    assert result.mcnemar_p == pytest.approx(2 / 2**24)


def test_a_run_against_itself_is_held_once_a_margin_says_how_much_loss_is_tolerable() -> None:
    """ "We could not tell" has to be a first-class answer, not a failure."""
    same = run("base", flags("10110100"))
    result = compare(same, same, margin=0.05, n_resamples=FAST_RESAMPLES)
    assert result.decision is Decision.HOLD
    assert result.exit_code == 1
    assert result.success_diff.point == 0.0
    assert result.mcnemar_p == 1.0
    assert any("includes zero" in reason for reason in result.reasons)


def test_a_zero_margin_turns_the_gate_into_a_superiority_test() -> None:
    """With no tolerance declared, a candidate that merely held level is not promoted."""
    same = run("base", flags("10110100"))
    result = compare(same, same, margin=0.0, n_resamples=FAST_RESAMPLES)
    assert result.decision is Decision.REJECT
    assert any("non-inferiority margin" in reason for reason in result.reasons)


def test_a_candidate_that_lost_ground_is_rejected() -> None:
    result = compare(
        run("base", flags(CANDIDATE_PATTERN)),
        run("dpo", flags(BASELINE_PATTERN)),
        margin=0.05,
        n_resamples=FAST_RESAMPLES,
    )
    assert result.decision is Decision.REJECT
    assert result.exit_code == 1
    assert result.success_diff.point == pytest.approx(-0.6)
    assert result.verdict.inferior


def test_one_significance_test_alone_does_not_promote() -> None:
    """The bootstrap and McNemar test the same hypothesis by different routes; both must agree.

    Five examples change hands, all in the candidate's favour. That is enough for the
    percentile interval to clear zero and not enough for the exact test: McNemar's p is
    2 * (1/32) = 0.0625, just above alpha. Promoting on the interval alone would be shipping a
    model on the more optimistic of two tests.
    """
    baseline = run("base", flags("1" * 20 + "0" * 20))
    candidate = run("sft", flags("1" * 25 + "0" * 15))
    result = compare(baseline, candidate, margin=0.10, n_resamples=FAST_RESAMPLES)
    assert result.mcnemar_p == pytest.approx(0.0625)
    assert result.success_diff.excludes_zero
    assert result.decision is Decision.HOLD
    assert result.reasons == (
        "McNemar p = 0.0625 does not clear alpha = 0.0500",
        "the candidate is non-inferior at margin 0.1000 but has not earned a promotion",
    )


def test_the_interval_alone_still_refuses_to_promote_when_mcnemar_is_disabled() -> None:
    """The two conditions are independent, so switching one off does not promote by default.

    An alpha of one accepts any p-value, which is the only way to isolate the interval's half
    of the rule: with binary indicators the exact test is the more powerful of the two, so
    every realistic run where the interval cannot exclude zero also fails on alpha.
    """
    same = run("base", flags("10110100"))
    result = compare(same, same, margin=0.05, alpha=1.0, n_resamples=FAST_RESAMPLES)
    assert result.decision is Decision.HOLD
    assert any("includes zero" in reason for reason in result.reasons)
    assert not any("alpha" in reason for reason in result.reasons)


# --------------------------------------------------------------------------------------
# Floors
# --------------------------------------------------------------------------------------


def test_the_json_floor_refuses_a_large_improvement_that_is_still_unusable() -> None:
    """0.30 to 0.55 is a big lift and still nothing downstream can consume."""
    baseline = run("base", flags("1" * 12 + "0" * 28))
    candidate = run("sft", flags("1" * 22 + "0" * 18))
    result = compare(
        baseline,
        candidate,
        floors=Floors(min_json_valid=0.80),
        n_resamples=FAST_RESAMPLES,
    )
    assert result.decision is Decision.REJECT
    assert result.candidate_metrics.json_valid_rate == pytest.approx(0.55)
    assert any("below the floor" in reason for reason in result.reasons)


def test_the_schema_floor_is_separate_from_the_json_floor() -> None:
    """A candidate can emit objects that parse and still fail the schema on most of them."""
    parses = flags("1" * 40)
    baseline = run("base", flags("1" * 4 + "0" * 36), json_valid=parses)
    candidate = run("sft", flags("1" * 20 + "0" * 20), json_valid=parses)
    result = compare(
        baseline,
        candidate,
        floors=Floors(min_json_valid=0.9, min_schema_valid=0.9),
        n_resamples=FAST_RESAMPLES,
    )
    assert result.decision is Decision.REJECT
    assert result.candidate_metrics.json_valid_rate == 1.0
    assert [reason for reason in result.reasons if "schema validity" in reason]


def test_a_floor_that_is_met_does_not_block() -> None:
    assert promoted().decision is Decision.PROMOTE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_json_valid", 1.5),
        ("min_json_valid", -0.1),
        ("min_schema_valid", 2.0),
        ("max_slice_regression", -0.01),
    ],
)
def test_floors_must_be_rates(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="must be a rate"):
        Floors(**{field: value})


# --------------------------------------------------------------------------------------
# The per-slice non-regression rule
# --------------------------------------------------------------------------------------


def test_a_slice_regression_is_refused_even_when_the_average_rose() -> None:
    """The exact failure the per-slice reporting exists to catch."""
    slices = [Slice.CLEAN] * 20 + [Slice.LONG_CONTEXT] * 20
    baseline = run("base", flags("0" * 20 + "1" * 16 + "0" * 4), slices=slices)
    candidate = run("sft", flags("1" * 20 + "0" * 20), slices=slices)
    result = compare(baseline, candidate, n_resamples=FAST_RESAMPLES)
    assert result.success_diff.point > 0
    assert result.decision is Decision.REJECT
    assert result.regressed_slices == (Slice.LONG_CONTEXT,)
    assert any("slice long_context regressed" in reason for reason in result.reasons)


def test_a_small_slice_dip_inside_the_tolerance_is_allowed() -> None:
    slices = [Slice.CLEAN] * 20 + [Slice.MANY_ITEMS] * 20
    baseline = run("base", flags("0" * 20 + "1" * 10 + "0" * 10), slices=slices)
    candidate = run("sft", flags("1" * 20 + "1" * 9 + "0" * 11), slices=slices)
    result = compare(
        baseline,
        candidate,
        floors=Floors(max_slice_regression=0.10),
        n_resamples=FAST_RESAMPLES,
    )
    assert result.per_slice[1].delta == pytest.approx(-0.05)
    assert not result.per_slice[1].regressed
    assert result.decision is Decision.PROMOTE


def test_slice_rows_follow_the_enum_order_and_carry_their_own_counts() -> None:
    slices = [Slice.MANY_ITEMS] * 10 + [Slice.CLEAN] * 30
    result = compare(
        run("base", flags("0" * 40), slices=slices),
        run("sft", flags("1" * 40), slices=slices),
        n_resamples=FAST_RESAMPLES,
    )
    assert [row.slice for row in result.per_slice] == [Slice.CLEAN, Slice.MANY_ITEMS]
    assert [row.n for row in result.per_slice] == [30, 10]
    assert all(row.delta == pytest.approx(1.0) for row in result.per_slice)


# --------------------------------------------------------------------------------------
# What the gate refuses to compare at all
# --------------------------------------------------------------------------------------


def test_two_runs_over_different_examples_cannot_be_compared() -> None:
    baseline = run("base", flags("111"))
    candidate = run("sft", flags("1111"))
    with pytest.raises(ValueError, match="cannot be paired"):
        compare(baseline, candidate, n_resamples=FAST_RESAMPLES)


def test_an_example_filed_under_two_different_slices_is_a_different_corpus() -> None:
    baseline = run("base", flags("11"), slices=[Slice.CLEAN, Slice.CLEAN])
    candidate = run("sft", flags("11"), slices=[Slice.CLEAN, Slice.DISTRACTOR])
    with pytest.raises(ValueError, match="different corpora"):
        compare(baseline, candidate, n_resamples=FAST_RESAMPLES)


def test_two_empty_reports_are_rejected_rather_than_promoted_by_default() -> None:
    result = compare(run("base", []), run("sft", []), n_resamples=FAST_RESAMPLES)
    assert result.n_paired == 0
    assert result.decision is Decision.REJECT
    assert result.reasons == ("the two reports share no examples, so nothing was compared",)
    assert result.per_slice == ()
    assert "nothing was compared" in result.to_markdown()


# --------------------------------------------------------------------------------------
# Reproducibility and rendering
# --------------------------------------------------------------------------------------


def test_the_same_seed_reproduces_the_decision_byte_for_byte() -> None:
    """A promotion decision has to be re-derivable from the two reports that produced it."""
    assert promoted().to_markdown() == promoted().to_markdown()


def test_the_seed_moves_the_interval_but_never_the_point_estimate() -> None:
    baseline = run("base", flags(BASELINE_PATTERN))
    candidate = run("sft", flags(CANDIDATE_PATTERN))
    first = compare(baseline, candidate, n_resamples=FAST_RESAMPLES, seed=1)
    second = compare(baseline, candidate, n_resamples=FAST_RESAMPLES, seed=2)
    assert first.success_diff.point == second.success_diff.point
    assert first.mcnemar_p == second.mcnemar_p


def test_the_comparison_is_antisymmetric() -> None:
    baseline = run("base", flags(BASELINE_PATTERN))
    candidate = run("sft", flags(CANDIDATE_PATTERN))
    forward = compare(baseline, candidate, n_resamples=FAST_RESAMPLES, seed=3)
    backward = compare(candidate, baseline, n_resamples=FAST_RESAMPLES, seed=3)
    assert backward.success_diff.point == -forward.success_diff.point
    assert (backward.mcnemar_b, backward.mcnemar_c) == (forward.mcnemar_c, forward.mcnemar_b)


def test_the_report_shows_the_decision_the_metrics_and_every_slice() -> None:
    text = promoted().to_markdown()
    assert text.startswith("# Promotion gate: sft vs base")
    assert "**Decision: PROMOTE** (exit code 0)" in text
    assert "| JSON validity (strict) | 0.2000 | 0.8000 | +0.6000 |" in text
    assert "| McNemar b / c | 0 / 24 |" in text
    assert "| clean | 40 | 0.2000 | 0.8000 | +0.6000 | no |" in text


def test_the_report_never_renders_a_negative_zero() -> None:
    """`-0.0000` and `+0.0000` are different bytes for the same number."""
    same = run("base", flags("1010"))
    text = compare(same, same, margin=0.05, n_resamples=FAST_RESAMPLES).to_markdown()
    assert "-0.0000" not in text
    assert "+0.0000" in text


def test_a_tiny_p_value_is_rendered_in_scientific_notation() -> None:
    """Four decimal places stop being informative long before the exact test does."""
    assert "| McNemar p (exact) | 1.19e-07 |" in promoted().to_markdown()


def test_a_p_value_of_one_is_rendered_in_full() -> None:
    same = run("base", flags("1100"))
    text = compare(same, same, margin=0.05, n_resamples=FAST_RESAMPLES).to_markdown()
    assert "| McNemar p (exact) | 1.0000 |" in text


def test_the_wilson_intervals_describe_each_side_on_its_own() -> None:
    result = promoted()
    assert result.baseline_wilson.point == pytest.approx(0.2)
    assert result.candidate_wilson.point == pytest.approx(0.8)
    assert result.baseline_wilson.contains(0.2)


def test_the_field_f1_difference_is_reported_beside_the_success_difference() -> None:
    """The continuous companion to the binary indicator, so a partial gain is still visible."""
    result = promoted()
    assert result.field_f1_diff.point == pytest.approx(0.6)


def test_the_result_is_frozen() -> None:
    """The rendered report reads only these fields, so they must not move after the decision."""
    # Through `Any`, because the type checker already refuses the assignment statically; the
    # test is that the model refuses it at runtime too.
    mutable = cast("Any", promoted())
    with pytest.raises(ValueError, match="frozen"):
        mutable.decision = Decision.HOLD


# --------------------------------------------------------------------------------------
# The CI entry point
# --------------------------------------------------------------------------------------


def test_gate_writes_the_report_and_returns_zero_for_a_promotion() -> None:
    result = promoted()
    stream = io.StringIO()
    assert gate(result, stream) == 0
    assert stream.getvalue() == result.to_markdown()


def test_gate_returns_non_zero_for_anything_but_a_promotion() -> None:
    same = run("base", flags("1100"))
    held = compare(same, same, margin=0.05, n_resamples=FAST_RESAMPLES)
    assert gate(held, io.StringIO()) == 1


def test_gate_writes_to_standard_output_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = promoted()
    assert gate(result) == 0
    assert capsys.readouterr().out == result.to_markdown()


# --------------------------------------------------------------------------------------
# The per-field recall floor
#
# This section exists because of a run that happened. The preference stage produced a model
# that beat its own baseline on every headline number and on all six slices, and it did it
# partly by never emitting the optional `flags` array again -- recall 0.61 to 0.00 across the
# split. Every rule above passed it. These tests are the rule that does not.
# --------------------------------------------------------------------------------------


def with_fields(report: EvalReport, **fields: tuple[int, int, int]) -> EvalReport:
    """The same report with per-field coverage attached, as `(expected, emitted, correct)`."""
    return report.model_copy(
        update={
            "per_field": {
                name: FieldCoverage(field=name, expected=expected, emitted=emitted, correct=correct)
                for name, (expected, emitted, correct) in fields.items()
            }
        }
    )


def compare_with_fields(
    baseline_fields: dict[str, tuple[int, int, int]],
    candidate_fields: dict[str, tuple[int, int, int]],
    *,
    floors: Floors | None = None,
) -> ComparisonResult:
    """The promoting comparison from `promoted()`, with field coverage on both sides."""
    return compare(
        with_fields(run("base", flags(BASELINE_PATTERN)), **baseline_fields),
        with_fields(run("sft", flags(CANDIDATE_PATTERN)), **candidate_fields),
        floors=Floors(min_json_valid=0.5) if floors is None else floors,
        n_resamples=FAST_RESAMPLES,
        seed=0,
    )


def test_a_candidate_that_stopped_emitting_a_field_is_rejected_however_good_the_average() -> None:
    """The regression test for the run described above, with its real numbers."""
    result = compare_with_fields(
        {"client_name": (40, 40, 38), "flags": (200, 147, 122)},
        {"client_name": (40, 40, 40), "flags": (200, 0, 0)},
    )
    assert result.decision is Decision.REJECT
    assert result.regressed_fields == ("flags",)
    assert any("'flags'" in reason and "0.6100" in reason for reason in result.reasons)
    # And the candidate really was better everywhere the other rules look.
    assert result.success_diff.point > 0
    assert result.regressed_slices == ()


def test_a_field_the_candidate_improved_does_not_block() -> None:
    result = compare_with_fields(
        {"flags": (200, 147, 122)},
        {"flags": (200, 190, 180)},
    )
    assert result.decision is Decision.PROMOTE
    assert result.regressed_fields == ()


def test_a_drop_exactly_at_the_tolerance_is_not_a_regression() -> None:
    """The boundary is inclusive, so a field that lost exactly the allowance still passes."""
    result = compare_with_fields(
        {"flags": (100, 100, 100)},
        {"flags": (100, 90, 90)},
        floors=Floors(min_json_valid=0.5, max_field_recall_drop=0.10),
    )
    assert result.decision is Decision.PROMOTE
    row = next(row for row in result.per_field if row.field == "flags")
    assert row.delta == pytest.approx(-0.10)
    assert not row.regressed


def test_a_field_with_too_few_gold_paths_cannot_block_a_promotion() -> None:
    """A field gold asks for three times has a recall that moves in thirds; that is noise."""
    result = compare_with_fields(
        {"rare": (3, 3, 3)},
        {"rare": (3, 0, 0)},
    )
    assert result.decision is Decision.PROMOTE
    row = next(row for row in result.per_field if row.field == "rare")
    assert row.below_support
    assert not row.regressed
    assert "below support" in result.to_markdown()


def test_support_is_the_smaller_of_the_two_sides() -> None:
    """Two reports over one corpus agree on the gold counts; this is what happens if they
    ever do not, and the conservative reading is the smaller sample."""
    result = compare_with_fields(
        {"flags": (200, 200, 200)},
        {"flags": (4, 0, 0)},
    )
    row = next(row for row in result.per_field if row.field == "flags")
    assert row.expected == 4
    assert row.below_support


def test_a_field_only_the_baseline_reported_is_skipped_rather_than_scored_zero() -> None:
    result = compare_with_fields(
        {"flags": (200, 147, 122), "gone": (50, 50, 50)},
        {"flags": (200, 147, 122)},
    )
    assert [row.field for row in result.per_field] == ["flags"]
    assert result.decision is Decision.PROMOTE


def test_reports_without_per_field_coverage_say_the_check_did_not_run() -> None:
    """An older report must not be read as a field recall of zero, nor as a silent pass."""
    result = promoted()
    assert result.per_field == ()
    assert any("did not run" in reason for reason in result.reasons)
    assert "Not reported" in result.to_markdown()


def test_the_report_renders_every_field_row() -> None:
    result = compare_with_fields(
        {"client_name": (40, 40, 38), "flags": (200, 147, 122)},
        {"client_name": (40, 40, 40), "flags": (200, 0, 0)},
    )
    markdown = result.to_markdown()
    assert "## Per field recall (tolerance 0.1000)" in markdown
    assert "| flags | 200 | 0.6100 | 0.0000 | -0.6100 | yes |" in markdown
    assert "| client_name | 40 | 0.9500 | 1.0000 | +0.0500 | no |" in markdown


def test_field_rows_keep_the_baseline_report_order() -> None:
    """Schema declaration order, so two runs of the gate produce no diff."""
    result = compare_with_fields(
        {"a": (20, 20, 20), "b": (20, 20, 20), "c": (20, 20, 20)},
        {"c": (20, 20, 20), "b": (20, 20, 20), "a": (20, 20, 20)},
    )
    assert [row.field for row in result.per_field] == ["a", "b", "c"]


def test_the_field_floor_must_be_a_rate() -> None:
    with pytest.raises(ValueError, match="max_field_recall_drop must be a rate"):
        Floors(max_field_recall_drop=1.5)


def test_the_support_threshold_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="min_field_support must not be negative"):
        Floors(min_field_support=-1)
