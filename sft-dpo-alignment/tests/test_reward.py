"""Tests for the reward.

This number replaces a human preference label, so the tests are mostly about the properties a
label is assumed to have rather than about particular completions: that it is bounded, that it
is monotone in correctness, that it is graded rather than collapsing to a constant across the
failures, and that lenient scoring is a superset of strict scoring so the gap between the two
can be reported as a measurement.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Final

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sftdpo.schemas import AdviceRecord, Example, Reward, Sample, Slice, ViolationKind
from sftdpo.verify.fields import f1_from_scores, field_f1
from sftdpo.verify.reward import (
    DEFAULT_FIELD_WEIGHT,
    DEFAULT_PARSE_WEIGHT,
    DEFAULT_SCHEMA_WEIGHT,
    RewardBreakdown,
    Verifier,
    lenient_verifier,
    strict_verifier,
)

GOLD_JSON: Final[dict[str, Any]] = {
    "client_name": "Ada Lovelace",
    "record_date": "2026-03-31",
    "risk_profile": "balanced",
    "objectives": ["retire at 60", "fund school fees"],
    "recommendations": [
        {"product": "Growth Fund", "action": "buy", "amount": 25000.0},
        {"product": "Cash Option", "action": "sell", "amount": 5000.0},
    ],
    "fees": {"advice_fee": 3300.0, "ongoing_fee_pct": 0.88},
    "review_months": 12,
    "flags": ["tfn missing"],
}

GOLD: Final = AdviceRecord.model_validate(GOLD_JSON)

# Three scalars, two objectives, two recommendations of three fields, two fees, the review
# period and one flag. Pinned by `test_a_perfect_completion_scores_exactly_one`.
N_GOLD: Final = 15

CLEAN: Final = GOLD.model_dump_json()
PROSE: Final = f"Sure, here it is:\n```json\n{CLEAN}\n```\nLet me know if you need changes."
JUNK: Final = "I am not able to produce that record."
TRUNCATED: Final = CLEAN[: CLEAN.index('"fees"')]


def mutate(**patch: Any) -> str:
    """The gold object as a completion, with top-level fields replaced."""
    value = deepcopy(GOLD_JSON)
    value.update(patch)
    return json.dumps(value)


EXTRA_KEY: Final = mutate(adviser="Bo")
WRONG_NAME: Final = mutate(client_name="Someone Else")
FEW_FIELDS: Final = json.dumps({"client_name": "Ada Lovelace"})


def example(example_id: str) -> Example:
    return Example(
        example_id=example_id, split="train", slice=Slice.CLEAN, note="a note", gold=GOLD
    )


def sample(example_id: str, text: str) -> Sample:
    return Sample(example_id=example_id, model="tiny", text=text)


# --------------------------------------------------------------------------------------
# The three stages
# --------------------------------------------------------------------------------------


def test_a_perfect_completion_scores_exactly_one() -> None:
    reward = strict_verifier().score(CLEAN, GOLD)
    assert reward.parsed
    assert reward.schema_valid
    assert reward.field_f1 == 1.0
    assert reward.exact_match
    assert reward.value == 1.0
    assert reward.violations == ()
    assert len(reward.fields) == N_GOLD


def test_the_default_weights_are_a_partition_of_the_reward() -> None:
    total = DEFAULT_PARSE_WEIGHT + DEFAULT_SCHEMA_WEIGHT + DEFAULT_FIELD_WEIGHT
    assert total == pytest.approx(1.0)
    assert strict_verifier().weights == pytest.approx((0.2, 0.2, 0.6))


def test_a_completion_with_no_json_in_it_scores_zero() -> None:
    reward = strict_verifier().score(JUNK, GOLD)
    assert not reward.parsed
    assert not reward.schema_valid
    assert reward.field_f1 == 0.0
    assert not reward.exact_match
    assert reward.value == 0.0
    assert reward.headline_violation is ViolationKind.NO_JSON


def test_a_failed_parse_still_reports_every_field() -> None:
    """Downstream reporting should not have to special-case the completions that failed."""
    reward = strict_verifier().score(JUNK, GOLD)
    assert len(reward.fields) == N_GOLD
    assert not any(score.correct for score in reward.fields)


def test_a_completion_cut_off_by_the_token_budget() -> None:
    """The lenient parser closes it, and the schema then charges for what never arrived."""
    assert strict_verifier().score(TRUNCATED, GOLD).value == 0.0
    lenient = lenient_verifier().score(TRUNCATED, GOLD)
    assert lenient.parsed
    assert not lenient.schema_valid
    assert lenient.headline_violation is ViolationKind.MISSING_FIELD
    assert 0.0 < lenient.field_f1 < 1.0


def test_an_invented_key_costs_the_schema_and_precision() -> None:
    reward = strict_verifier().score(EXTRA_KEY, GOLD)
    assert reward.parsed
    assert not reward.schema_valid
    assert reward.headline_violation is ViolationKind.EXTRA_FIELD
    assert reward.field_f1 == pytest.approx(2 * N_GOLD / ((N_GOLD + 1) + N_GOLD))
    assert reward.value == pytest.approx(0.2 + 0.6 * reward.field_f1)


def test_the_field_score_survives_a_schema_failure() -> None:
    """A completion that got most of it right must not score the same as one that got nothing.

    Early in training the schema check fails on nearly everything. A reward that collapsed to
    a constant there would leave the preference miner with pairs it cannot rank.
    """
    good = strict_verifier().score(EXTRA_KEY, GOLD)
    bad = strict_verifier().score(FEW_FIELDS, GOLD)
    assert not good.schema_valid
    assert not bad.schema_valid
    assert good.value > bad.value > 0.0


def test_a_wrong_value_costs_only_its_own_field() -> None:
    reward = strict_verifier().score(WRONG_NAME, GOLD)
    assert reward.schema_valid
    assert reward.field_f1 == pytest.approx(2 * (N_GOLD - 1) / (2 * N_GOLD))
    assert [score.path for score in reward.fields if not score.correct] == ["client_name"]


def test_a_quoted_number_is_charged_once_by_each_layer() -> None:
    """One mistake, two layers: it is not the schema, and it is not the value gold holds."""
    reward = strict_verifier().score(mutate(review_months="12"), GOLD)
    assert not reward.schema_valid
    assert reward.headline_violation is ViolationKind.WRONG_TYPE
    assert [score.path for score in reward.fields if not score.correct] == ["review_months"]


def test_every_schema_violation_is_carried_onto_the_reward() -> None:
    reward = strict_verifier().score(FEW_FIELDS, GOLD)
    assert len(reward.violations) == 4
    assert {violation.kind for violation in reward.violations} == {ViolationKind.MISSING_FIELD}


# --------------------------------------------------------------------------------------
# Exact match
# --------------------------------------------------------------------------------------


def test_an_exact_match_is_judged_after_normalisation() -> None:
    assert strict_verifier().score(mutate(client_name="ADA  LOVELACE"), GOLD).exact_match


def test_getting_every_field_right_and_inventing_a_ninth_is_not_an_exact_match() -> None:
    reward = strict_verifier().score(EXTRA_KEY, GOLD)
    assert not reward.exact_match
    assert reward.field_f1 < 1.0


def test_a_repaired_completion_can_still_be_an_exact_match() -> None:
    assert lenient_verifier().score(PROSE, GOLD).exact_match


# --------------------------------------------------------------------------------------
# Strict against lenient: the headline gap
# --------------------------------------------------------------------------------------


def test_prose_wrapping_is_the_whole_difference_between_the_verifiers() -> None:
    """The object was right; only the packaging was not. That is what the gap measures."""
    assert strict_verifier().score(PROSE, GOLD).value == 0.0
    assert lenient_verifier().score(PROSE, GOLD).value == 1.0


def test_the_verifiers_agree_on_a_clean_completion() -> None:
    assert strict_verifier().score(CLEAN, GOLD) == lenient_verifier().score(CLEAN, GOLD)


@pytest.mark.parametrize("text", [CLEAN, PROSE, JUNK, EXTRA_KEY, WRONG_NAME, TRUNCATED])
def test_lenient_never_scores_below_strict(text: str) -> None:
    assert lenient_verifier().score(text, GOLD).value >= strict_verifier().score(text, GOLD).value


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weights",
    [
        {"parse_weight": -0.1},
        {"schema_weight": -1.0},
        {"field_weight": float("nan")},
        {"parse_weight": float("inf")},
        {"parse_weight": 0.0, "schema_weight": 0.0, "field_weight": 0.0},
    ],
)
def test_a_reward_that_could_not_be_bounded_is_refused(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="weight"):
        Verifier(strict=True, **weights)


def test_only_the_ratio_between_weights_matters() -> None:
    scaled = Verifier(strict=True, parse_weight=1, schema_weight=1, field_weight=3)
    for text in (CLEAN, PROSE, JUNK, EXTRA_KEY, WRONG_NAME):
        assert scaled.score(text, GOLD).value == pytest.approx(
            strict_verifier().score(text, GOLD).value
        )


def test_weighting_only_the_fields_reproduces_the_field_score() -> None:
    verifier = Verifier(strict=True, parse_weight=0.0, schema_weight=0.0, field_weight=1.0)
    reward = verifier.score(EXTRA_KEY, GOLD)
    assert reward.value == pytest.approx(reward.field_f1)
    assert verifier.weights == pytest.approx((0.0, 0.0, 1.0))


def test_weighting_only_the_parse_reproduces_the_parse_flag() -> None:
    verifier = Verifier(strict=True, parse_weight=1.0, schema_weight=0.0, field_weight=0.0)
    assert verifier.score(EXTRA_KEY, GOLD).value == 1.0
    assert verifier.score(JUNK, GOLD).value == 0.0


# --------------------------------------------------------------------------------------
# Batches
# --------------------------------------------------------------------------------------


def test_score_many_is_positionally_aligned_with_its_samples() -> None:
    samples = [sample("b", JUNK), sample("a", CLEAN), sample("b", EXTRA_KEY)]
    rewards = strict_verifier().score_many(samples, [example("a"), example("b")])
    assert [reward.exact_match for reward in rewards] == [False, True, False]
    assert [reward.parsed for reward in rewards] == [False, True, True]


def test_score_many_does_not_reorder_the_callers_samples() -> None:
    samples = [sample("a", CLEAN), sample("a", JUNK)]
    forward = strict_verifier().score_many(samples, [example("a")])
    backward = strict_verifier().score_many(list(reversed(samples)), [example("a")])
    assert forward == list(reversed(backward))


def test_score_many_of_nothing_is_nothing() -> None:
    assert strict_verifier().score_many([], [example("a")]) == []


def test_a_sample_with_no_example_is_a_dataset_fault() -> None:
    """Dropping it would misalign every reward after it, so it stops the batch instead."""
    with pytest.raises(ValueError, match="no example"):
        strict_verifier().score_many([sample("missing", CLEAN)], [example("a")])


def test_two_examples_with_one_id_is_a_dataset_fault() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        strict_verifier().score_many([], [example("a"), example("a")])


def test_scoring_is_deterministic() -> None:
    verifier = strict_verifier()
    assert verifier.score(PROSE, GOLD) == verifier.score(PROSE, GOLD)


# --------------------------------------------------------------------------------------
# The breakdown
# --------------------------------------------------------------------------------------


def rewards_for(texts: list[str]) -> list[Reward]:
    verifier = strict_verifier()
    return [verifier.score(text, GOLD) for text in texts]


def test_the_breakdown_reports_each_stage_separately() -> None:
    """The mean alone cannot be acted on: a parse problem and a field problem need opposite work."""
    breakdown = RewardBreakdown.over(rewards_for([CLEAN, JUNK, EXTRA_KEY, WRONG_NAME]))
    invented = 2 * N_GOLD / ((N_GOLD + 1) + N_GOLD)
    wrong = 2 * (N_GOLD - 1) / (2 * N_GOLD)
    assert breakdown.count == 4
    assert breakdown.parse_rate == 0.75
    assert breakdown.schema_valid_rate == 0.5
    assert breakdown.exact_match_rate == 0.25
    assert breakdown.mean_field_f1 == pytest.approx((1.0 + 0.0 + invented + wrong) / 4)
    assert breakdown.mean_value == pytest.approx(
        (1.0 + 0.0 + (0.2 + 0.6 * invented) + (0.4 + 0.6 * wrong)) / 4
    )


def test_an_empty_run_reports_zero_and_says_so() -> None:
    breakdown = RewardBreakdown.over([])
    assert breakdown.count == 0
    assert breakdown.mean_value == 0.0
    assert breakdown.violations == {}
    assert breakdown.failures == 0


def test_each_failing_completion_gets_exactly_one_vote() -> None:
    """Four missing fields in one completion must not outvote four completions."""
    breakdown = RewardBreakdown.over(rewards_for([FEW_FIELDS, CLEAN]))
    assert breakdown.violations == {ViolationKind.MISSING_FIELD: 1}
    assert breakdown.failures == 1


def test_the_violation_distribution_is_ordered_most_severe_first() -> None:
    breakdown = RewardBreakdown.over(rewards_for([EXTRA_KEY, JUNK, FEW_FIELDS, CLEAN]))
    assert list(breakdown.violations) == [
        ViolationKind.NO_JSON,
        ViolationKind.MISSING_FIELD,
        ViolationKind.EXTRA_FIELD,
    ]
    assert breakdown.failures == 3
    assert breakdown.count == 4


def test_the_gap_between_the_verifiers_is_a_number_you_can_report() -> None:
    texts = [CLEAN, PROSE, PROSE, JUNK]
    strict = RewardBreakdown.over([strict_verifier().score(text, GOLD) for text in texts])
    lenient = RewardBreakdown.over([lenient_verifier().score(text, GOLD) for text in texts])
    assert strict.parse_rate == 0.25
    assert lenient.parse_rate == 0.75
    assert lenient.mean_value - strict.mean_value == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------

_WEIGHT = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
_COMPLETIONS = st.builds(
    lambda prefix, body, suffix: prefix + body + suffix,
    st.sampled_from(["", "Here is the record:\n", "```json\n", "Sure!\n\n", "{"]),
    st.sampled_from([CLEAN, EXTRA_KEY, WRONG_NAME, FEW_FIELDS, JUNK, TRUNCATED, "{}", "[1]"]),
    st.sampled_from(["", "\n```", "\nLet me know.", "}", "}}}"]),
)

# One damage operation per top-level field, so a set of them composes without interfering, and
# every damaged record is still schema-valid — which is what isolates the field component.
DAMAGE: Final[dict[str, Any]] = {
    "client_name": "Someone Else",
    "record_date": "2020-01-01",
    "risk_profile": "growth",
    "objectives": ["retire at 60", "buy a boat"],
    "recommendations": [{"product": "Mystery Trust", "action": "sell", "amount": 1.0}],
    "fees": {"advice_fee": 1.0, "ongoing_fee_pct": 2.0},
    "review_months": 24,
    "flags": [],
}
DAMAGE_NAMES: Final = sorted(DAMAGE)


def damaged(names: frozenset[str]) -> str:
    return mutate(**{name: DAMAGE[name] for name in sorted(names)})


@settings(max_examples=200, deadline=None)
@given(_COMPLETIONS, _WEIGHT, _WEIGHT, _WEIGHT, st.booleans())
def test_the_reward_is_bounded_for_any_completion_and_any_weights(
    text: str, parse: float, schema: float, field: float, strict: bool
) -> None:
    """A convex combination of three components in [0, 1], and IEEE rounding is monotone.

    The numerator and the denominator are summed in the same order, so no rounding step can
    make the ratio exceed one; the assertion is therefore exact rather than approximate.
    """
    assume(parse + schema + field > 0)
    verifier = Verifier(strict=strict, parse_weight=parse, schema_weight=schema, field_weight=field)
    assert 0.0 <= verifier.score(text, GOLD).value <= 1.0


@settings(max_examples=100, deadline=None)
@given(_COMPLETIONS)
def test_the_components_reconstruct_the_value(text: str) -> None:
    """The scalar is exactly its parts, so a change in it can always be attributed."""
    reward = strict_verifier().score(text, GOLD)
    assert reward.value == pytest.approx(
        DEFAULT_PARSE_WEIGHT * reward.parsed
        + DEFAULT_SCHEMA_WEIGHT * reward.schema_valid
        + DEFAULT_FIELD_WEIGHT * reward.field_f1
    )
    assert reward.field_f1 == pytest.approx(f1_from_scores(reward.fields))


@settings(max_examples=100, deadline=None)
@given(_COMPLETIONS)
def test_a_completion_that_did_not_parse_is_graded_as_every_field_missing(text: str) -> None:
    reward = strict_verifier().score(text, GOLD)
    if not reward.parsed:
        assert reward.value == 0.0
        assert reward.field_f1 == field_f1(None, GOLD)


@settings(max_examples=100, deadline=None)
@given(_COMPLETIONS)
def test_lenient_scoring_is_a_superset_of_strict_scoring(text: str) -> None:
    """Lenient parsing succeeds wherever strict does, and returns the same object when it does."""
    assert lenient_verifier().score(text, GOLD).value >= strict_verifier().score(text, GOLD).value


@settings(max_examples=200, deadline=None)
@given(st.sets(st.sampled_from(DAMAGE_NAMES)), st.sets(st.sampled_from(DAMAGE_NAMES)))
def test_the_reward_is_monotone_in_correctness(kept: set[str], extra: set[str]) -> None:
    """A completion correct wherever another is, and wrong nowhere it is right, never scores lower.

    Stated under that precondition rather than for any superset of fields, because adding a
    *wrong* field does lower the reward and has to: precision is part of the signal.
    """
    verifier = strict_verifier()
    repaired = verifier.score(damaged(frozenset(kept)), GOLD)
    worse = verifier.score(damaged(frozenset(kept | extra)), GOLD)
    assert repaired.schema_valid and worse.schema_valid
    assert repaired.value >= worse.value


@settings(max_examples=50, deadline=None)
@given(st.sets(st.sampled_from(DAMAGE_NAMES), min_size=1))
def test_only_the_undamaged_completion_is_an_exact_match(names: set[str]) -> None:
    verifier = strict_verifier()
    assert not verifier.score(damaged(frozenset(names)), GOLD).exact_match
    assert verifier.score(damaged(frozenset()), GOLD).exact_match
