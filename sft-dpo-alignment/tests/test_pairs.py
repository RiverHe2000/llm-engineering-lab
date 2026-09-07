"""Tests for mining (chosen, rejected) pairs out of sampled completions.

Most of these tests state a rule from the module docstring and then break it on purpose: a
pair whose two sides scored the same, a pair whose two sides are the same string, one easy
prompt allowed to contribute every pair in the dataset. Each rule exists because the
resulting dataset would be well formed and would teach the model the wrong thing, which is
the sort of bug that shows up as a disappointing evaluation weeks later rather than as a
crash, so each is pinned here.

The reward landscapes are supplied by a scripted scorer rather than by the real verifier, so
a test that wants three completions at 0.9, 0.6 and 0.1 can say so instead of hunting for
JSON strings that happen to score that way. The real `Verifier` is used at the end, where the
point is precisely that the strictness of the verifier decides what the dataset teaches.

One scorer here is deliberately not a pure function of the completion text. That is the only
way to hand mining two identical strings with different rewards, and without it the
duplicate-text guard would be untestable -- passing only because the real verifier is
deterministic, not because the guard works.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.prefs.pairs import (
    DEFAULT_MAX_PAIRS_PER_PROMPT,
    DEFAULT_MIN_MARGIN,
    SATURATION_TOLERANCE,
    MiningResult,
    MiningStats,
    PromptOutcome,
    SliceYield,
    mine_pairs,
)
from sftdpo.schemas import (
    AdviceRecord,
    Example,
    PreferencePair,
    Reward,
    Sample,
    SchemaViolation,
    Slice,
    ViolationKind,
)
from sftdpo.task.generate import render_prompt
from sftdpo.verify.reward import lenient_verifier, strict_verifier

MODEL: Final = "tiny"
NOTE: Final = "Ada wants a balanced portfolio and a review in a year."

# Ten compared paths: three scalars, one objective, one recommendation of three fields, two
# fees and the review period. The count matters, because one wrong field costs 0.6/paths and
# that is the number `DEFAULT_MIN_MARGIN` is calibrated against.
GOLD_JSON: Final[dict[str, Any]] = {
    "client_name": "Ada Lovelace",
    "record_date": "2026-03-31",
    "risk_profile": "balanced",
    "objectives": ["retire at 60"],
    "recommendations": [{"product": "Growth Fund", "action": "buy", "amount": 25000.0}],
    "fees": {"advice_fee": 3300.0, "ongoing_fee_pct": 0.88},
    "review_months": 12,
    "flags": [],
}
GOLD: Final = AdviceRecord.model_validate(GOLD_JSON)
CLEAN: Final = GOLD.model_dump_json()
PROSE: Final = f"Certainly! Here is the record:\n```json\n{CLEAN}\n```\nHope that helps."
JUNK: Final = "I am not able to produce that record."


def mutate(source: Mapping[str, Any], **patch: Any) -> str:
    """`source` as a completion string, with top-level fields replaced."""
    value = deepcopy(dict(source))
    value.update(patch)
    return json.dumps(value)


WRONG_NAME: Final = mutate(GOLD_JSON, client_name="Someone Else")
EXTRA_KEY: Final = mutate(GOLD_JSON, adviser="Bo")

# The same record with two objectives, two recommendations and a flag: fifteen paths instead
# of ten, so one wrong field is worth 0.04 rather than 0.06.
WIDE_JSON: Final[dict[str, Any]] = {
    **GOLD_JSON,
    "objectives": ["retire at 60", "fund school fees"],
    "recommendations": [
        {"product": "Growth Fund", "action": "buy", "amount": 25000.0},
        {"product": "Cash Option", "action": "sell", "amount": 5000.0},
    ],
    "flags": ["tfn missing"],
}
WIDE_GOLD: Final = AdviceRecord.model_validate(WIDE_JSON)
WIDE_CLEAN: Final = WIDE_GOLD.model_dump_json()
WIDE_WRONG_NAME: Final = mutate(WIDE_JSON, client_name="Someone Else")


def reward(value: float, *, kind: ViolationKind | None = None) -> Reward:
    """A reward with a chosen scalar and, optionally, a headline violation.

    Only `value` and `headline_violation` are read by the miner, but the other fields are
    filled consistently so that a reader of a failing test is not misled by a reward that
    claims to have parsed while carrying a `NO_JSON` violation.
    """
    return Reward(
        parsed=kind is None,
        schema_valid=kind is None,
        field_f1=value,
        exact_match=kind is None and value >= 1.0,
        value=value,
        violations=() if kind is None else (SchemaViolation(kind=kind, path="$"),),
    )


def make_example(
    example_id: str, *, slice_: Slice = Slice.CLEAN, gold: AdviceRecord = GOLD
) -> Example:
    return Example(example_id=example_id, split="train", slice=slice_, note=NOTE, gold=gold)


def make_sample(example_id: str, text: str, index: int = 0) -> Sample:
    return Sample(example_id=example_id, model=MODEL, text=text, sample_index=index)


class TableScorer:
    """A scorer that reads each completion's reward out of a table keyed by its text.

    Stands in for `Verifier` so that a test can state the reward landscape it wants to mine.
    It also records what it was handed, which is how the tests check that scoring happens
    once for the whole run rather than per prompt.
    """

    def __init__(self, table: Mapping[str, Reward]) -> None:
        self.table = dict(table)
        self.calls = 0
        self.examples_seen: list[str] = []
        self.samples_seen: list[str] = []

    def score_many(self, samples: Sequence[Sample], examples: Iterable[Example]) -> list[Reward]:
        self.calls += 1
        self.examples_seen = [example.example_id for example in examples]
        self.samples_seen = [sample.text for sample in samples]
        return [self.table[sample.text] for sample in samples]


class PositionalScorer:
    """A scorer that answers from a fixed list, ignoring the completion entirely.

    Deliberately inconsistent: it will give two identical strings different rewards, which no
    real verifier would do. That is the point. The duplicate-text guard is a check on an
    assumption about another module, so it has to be tested against a scorer that breaks it.
    """

    def __init__(self, rewards: Sequence[Reward]) -> None:
        self.rewards = list(rewards)
        self.examples_seen: list[str] = []

    def score_many(self, samples: Sequence[Sample], examples: Iterable[Example]) -> list[Reward]:
        self.examples_seen = [example.example_id for example in examples]
        if len(samples) != len(self.rewards):
            raise AssertionError(f"scripted {len(self.rewards)} rewards for {len(samples)} samples")
        return list(self.rewards)


@dataclass(frozen=True, slots=True)
class Group:
    """One prompt, its completions, and the rewards a `TableScorer` should return for them."""

    example: Example
    samples: tuple[Sample, ...]
    table: dict[str, Reward]


def group(
    example_id: str,
    values: Sequence[float],
    *,
    slice_: Slice = Slice.CLEAN,
    kinds: Sequence[ViolationKind | None] | None = None,
) -> Group:
    """One prompt's completions, scoring exactly `values`.

    Texts carry the prompt id, so several prompts can be mined together without colliding in
    the reward table, and are distinct within a prompt, so the duplicate-text guard never
    fires by accident in a test that is about something else.
    """
    kind_list = list(kinds) if kinds is not None else [None] * len(values)
    samples = tuple(
        make_sample(example_id, f"{example_id}:c{index}", index) for index in range(len(values))
    )
    table = {
        sample.text: reward(value, kind=kind)
        for sample, value, kind in zip(samples, values, kind_list, strict=True)
    }
    return Group(example=make_example(example_id, slice_=slice_), samples=samples, table=table)


def mine(
    *groups: Group,
    min_margin: float = DEFAULT_MIN_MARGIN,
    max_pairs_per_prompt: int = DEFAULT_MAX_PAIRS_PER_PROMPT,
    seed: int = 0,
) -> MiningResult:
    """Mine the given groups with a `TableScorer` built from their tables."""
    table: dict[str, Reward] = {}
    for item in groups:
        table.update(item.table)
    return mine_pairs(
        [sample for item in groups for sample in item.samples],
        [item.example for item in groups],
        TableScorer(table),
        min_margin=min_margin,
        max_pairs_per_prompt=max_pairs_per_prompt,
        seed=seed,
    )


VALUES = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=8)


# --------------------------------------------------------------------------------------
# Best against worst, then inwards
# --------------------------------------------------------------------------------------


def test_the_best_completion_is_paired_with_the_worst() -> None:
    result = mine(group("ex", [0.9, 0.5, 0.1]), max_pairs_per_prompt=4)

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.chosen == "ex:c0"
    assert pair.rejected == "ex:c2"
    assert pair.chosen_reward == pytest.approx(0.9)
    assert pair.rejected_reward == pytest.approx(0.1)


def test_pairing_works_inwards_from_both_ends() -> None:
    result = mine(group("ex", [1.0, 0.8, 0.2, 0.0]), max_pairs_per_prompt=4)

    assert [(pair.chosen, pair.rejected) for pair in result.pairs] == [
        ("ex:c0", "ex:c3"),
        ("ex:c1", "ex:c2"),
    ]


def test_pairs_come_out_in_descending_margin() -> None:
    result = mine(group("ex", [1.0, 0.8, 0.2, 0.0]), max_pairs_per_prompt=4)

    margins = [pair.margin for pair in result.pairs]
    assert margins == sorted(margins, reverse=True)
    assert margins == pytest.approx([1.0, 0.6])


def test_the_middle_completion_of_an_odd_group_is_left_unpaired() -> None:
    result = mine(group("ex", [1.0, 0.5, 0.0]), max_pairs_per_prompt=4)

    used = {pair.chosen for pair in result.pairs} | {pair.rejected for pair in result.pairs}
    assert used == {"ex:c0", "ex:c2"}


@settings(max_examples=50, deadline=None)
@given(values=VALUES)
def test_no_completion_is_both_chosen_and_rejected(values: list[float]) -> None:
    """The invariant that stops the optimiser being asked to push one text both ways."""
    result = mine(group("ex", values), min_margin=1e-9, max_pairs_per_prompt=8)

    chosen = [pair.chosen for pair in result.pairs]
    rejected = [pair.rejected for pair in result.pairs]
    assert len(set(chosen)) == len(chosen)
    assert len(set(rejected)) == len(rejected)
    assert not set(chosen) & set(rejected)


@settings(max_examples=50, deadline=None)
@given(values=VALUES)
def test_the_chosen_side_always_outscores_the_rejected_side(values: list[float]) -> None:
    result = mine(group("ex", values), min_margin=0.01, max_pairs_per_prompt=8)

    assert all(pair.margin >= 0.01 for pair in result.pairs)
    assert all(pair.chosen_reward > pair.rejected_reward for pair in result.pairs)


def test_a_pair_carries_the_slice_of_its_prompt() -> None:
    result = mine(group("ex", [1.0, 0.0], slice_=Slice.MANY_ITEMS))

    assert result.pairs[0].slice is Slice.MANY_ITEMS


# --------------------------------------------------------------------------------------
# The minimum margin
# --------------------------------------------------------------------------------------


def test_a_gap_below_the_minimum_yields_no_pair() -> None:
    result = mine(group("ex", [0.54, 0.50]), min_margin=0.05)

    assert result.pairs == []
    assert result.outcomes == {PromptOutcome.NO_PAIR: 1}


def test_a_gap_exactly_at_the_minimum_is_kept() -> None:
    """The threshold is `>=`, and the two values here differ by exactly it in binary."""
    result = mine(group("ex", [0.5, 0.0]), min_margin=0.5)

    assert len(result.pairs) == 1
    assert result.pairs[0].margin == 0.5


def test_an_inner_gap_below_the_minimum_stops_the_pairing() -> None:
    result = mine(group("ex", [1.0, 0.55, 0.45, 0.0]), min_margin=0.5, max_pairs_per_prompt=4)

    assert [(pair.chosen, pair.rejected) for pair in result.pairs] == [("ex:c0", "ex:c3")]
    assert result.pairs_dropped_by_cap == 0


@settings(max_examples=50, deadline=None)
@given(
    values=VALUES,
    lower=st.floats(min_value=0.01, max_value=0.5),
    delta=st.floats(min_value=0.0, max_value=0.5),
)
def test_raising_the_margin_never_increases_the_yield(
    values: list[float], lower: float, delta: float
) -> None:
    at_lower = mine(group("ex", values), min_margin=lower, max_pairs_per_prompt=8)
    at_higher = mine(group("ex", values), min_margin=lower + delta, max_pairs_per_prompt=8)

    assert len(at_higher.pairs) <= len(at_lower.pairs)


def test_the_default_margin_admits_one_wrong_field() -> None:
    """The calibration behind `DEFAULT_MIN_MARGIN`, measured rather than restated.

    On this ten-path gold record a single wrong field costs `0.6 / 10`. The fixture is
    deliberately narrower than anything the generator produces; the corpus-side calibration
    is `test_the_default_margin_clears_the_tightest_gap_this_corpus_produces`.
    """
    example = make_example("ex")
    samples = [make_sample("ex", CLEAN, 0), make_sample("ex", WRONG_NAME, 1)]

    result = mine_pairs(samples, [example], strict_verifier())

    assert len(result.pairs) == 1
    assert result.pairs[0].margin == pytest.approx(0.06)
    assert result.pairs[0].margin > DEFAULT_MIN_MARGIN


def test_the_default_margin_admits_one_wrong_field_on_a_wider_record() -> None:
    """The same single error on a fifteen-path record is worth 0.04 and must still pair.

    This test used to assert the opposite. The default was 0.05, calibrated against a
    ten-path fixture, and this test recorded as intended behaviour that a wider record's
    single-field difference was filtered out. It is not intended behaviour on this corpus:
    the generator never produces a ten-path record at all -- the narrowest is thirteen paths
    and the widest thirty-five -- so the old default sat above every real one-field gap and
    the miner could not produce a single-field pair from any of the 640 examples.
    """
    example = make_example("wide", gold=WIDE_GOLD)
    samples = [make_sample("wide", WIDE_CLEAN, 0), make_sample("wide", WIDE_WRONG_NAME, 1)]

    result = mine_pairs(samples, [example], strict_verifier())

    assert len(result.pairs) == 1
    assert result.pairs[0].margin == pytest.approx(0.04)
    assert result.pairs[0].margin > DEFAULT_MIN_MARGIN


def test_the_default_margin_clears_the_tightest_gap_this_corpus_produces() -> None:
    """Calibration against the data rather than against a fixture.

    A wrong field costs `0.6 / n` where `n` is the gold record's leaf-path count. Measured
    over the shipped corpus that is 13 to 35 paths, so the tightest real one-field gap is
    `0.6 / 35`. The default has to sit under it or single-field pairs are unminable, which is
    exactly what happened when the default was 0.05.
    """
    widest_record_paths = 35
    tightest_one_field_gap = 0.6 / widest_record_paths
    assert tightest_one_field_gap > DEFAULT_MIN_MARGIN
    # ...and comfortably above the float noise a rounding difference would produce.
    assert DEFAULT_MIN_MARGIN > 1e-6


@pytest.mark.parametrize("min_margin", [0.0, -0.1, math.nan, math.inf])
def test_a_margin_that_is_not_a_positive_number_is_refused(min_margin: float) -> None:
    """Zero would permit pairing two equally scored completions, the one forbidden thing."""
    with pytest.raises(ValueError, match="min_margin"):
        mine(group("ex", [1.0, 0.0]), min_margin=min_margin)


# --------------------------------------------------------------------------------------
# Never pair two identical strings
# --------------------------------------------------------------------------------------


def test_two_identical_completions_are_never_paired() -> None:
    """Proved against a scorer that scores the same string differently, since a consistent
    one would make the margin rule do this work and hide a broken guard."""
    samples = [make_sample("ex", "identical", 0), make_sample("ex", "identical", 1)]
    scorer = PositionalScorer([reward(1.0), reward(0.0)])

    result = mine_pairs(samples, [make_example("ex")], scorer)

    assert result.pairs == []
    assert result.identical_text_rejections == 1
    assert result.outcomes == {PromptOutcome.NO_PAIR: 1}


def test_an_identical_candidate_does_not_consume_the_cap() -> None:
    """A discarded candidate is not a pair, so it must not spend one of the prompt's slots."""
    texts = ["same", "b", "c", "same"]
    samples = [make_sample("ex", text, index) for index, text in enumerate(texts)]
    scorer = PositionalScorer([reward(1.0), reward(0.7), reward(0.3), reward(0.0)])

    result = mine_pairs(samples, [make_example("ex")], scorer, max_pairs_per_prompt=1)

    assert [(pair.chosen, pair.rejected) for pair in result.pairs] == [("b", "c")]
    assert result.identical_text_rejections == 1
    assert result.pairs_dropped_by_cap == 0


def test_identical_text_rejections_reach_the_stats() -> None:
    samples = [make_sample("ex", "identical", index) for index in range(2)]
    scorer = PositionalScorer([reward(0.9), reward(0.1)])

    stats = mine_pairs(samples, [make_example("ex")], scorer).stats()

    assert stats.identical_text_rejections == 1
    assert stats.pairs == 0


# --------------------------------------------------------------------------------------
# The cap per prompt
# --------------------------------------------------------------------------------------


def test_pairs_from_one_prompt_are_capped() -> None:
    result = mine(group("ex", [1.0, 0.9, 0.8, 0.2, 0.1, 0.0]), max_pairs_per_prompt=2)

    assert len(result.pairs) == 2
    assert result.pairs_dropped_by_cap == 1


def test_a_cap_of_one_keeps_the_widest_margin() -> None:
    result = mine(group("ex", [1.0, 0.6, 0.4, 0.0]), max_pairs_per_prompt=1)

    assert len(result.pairs) == 1
    assert result.pairs[0].margin == pytest.approx(1.0)


def test_the_cap_applies_per_prompt_and_not_per_run() -> None:
    result = mine(
        group("ex-0", [1.0, 0.7, 0.3, 0.0]),
        group("ex-1", [1.0, 0.7, 0.3, 0.0]),
        max_pairs_per_prompt=1,
    )

    assert len(result.pairs) == 2
    assert {pair.example_id for pair in result.pairs} == {"ex-0", "ex-1"}
    assert result.pairs_dropped_by_cap == 2


@settings(max_examples=50, deadline=None)
@given(values=VALUES, cap=st.integers(min_value=1, max_value=4))
def test_no_prompt_ever_exceeds_the_cap(values: list[float], cap: int) -> None:
    result = mine(group("ex", values), min_margin=1e-9, max_pairs_per_prompt=cap)

    assert len(result.pairs) <= cap


@pytest.mark.parametrize("cap", [0, -1])
def test_a_cap_below_one_is_refused(cap: int) -> None:
    with pytest.raises(ValueError, match="max_pairs_per_prompt"):
        mine(group("ex", [1.0, 0.0]), max_pairs_per_prompt=cap)


# --------------------------------------------------------------------------------------
# Prompts with nothing to learn from
# --------------------------------------------------------------------------------------


def test_a_prompt_whose_completions_all_score_the_same_is_skipped() -> None:
    result = mine(group("ex", [0.4, 0.4, 0.4, 0.4]))

    assert result.pairs == []
    assert result.outcomes == {PromptOutcome.FLAT: 1}


def test_all_equally_bad_is_flat_and_all_perfect_is_saturated() -> None:
    """The two have opposite meanings and a bare pair count cannot tell them apart."""
    flat = mine(group("ex", [0.2, 0.2, 0.2])).stats()
    saturated = mine(group("ex", [1.0, 1.0, 1.0])).stats()

    assert (flat.flat_prompts, flat.saturated_prompts) == (1, 0)
    assert (saturated.flat_prompts, saturated.saturated_prompts) == (0, 1)
    assert flat.pairs == saturated.pairs == 0


def test_a_reward_a_hair_under_one_still_counts_as_saturated() -> None:
    """The reward is a weighted sum normalised by its own weights, so a perfect completion
    can land a rounding step below 1.0 and must not be reported as a failure to find signal.
    """
    almost = 1.0 - SATURATION_TOLERANCE / 2.0
    result = mine(group("ex", [almost, almost, almost]))

    assert result.outcomes == {PromptOutcome.SATURATED: 1}


def test_a_reward_clearly_under_one_counts_as_flat() -> None:
    result = mine(group("ex", [0.999, 0.999]))

    assert result.outcomes == {PromptOutcome.FLAT: 1}


def test_a_prompt_sampled_once_is_counted_apart_from_the_flat_ones() -> None:
    """A k=1 run is all-equal by definition, and reporting it as saturation would claim the
    model had outgrown data it was never given a second look at."""
    result = mine(group("ex", [1.0]))

    assert result.pairs == []
    assert result.outcomes == {PromptOutcome.SINGLE_SAMPLE: 1}
    assert result.stats().saturated_prompts == 0


@settings(max_examples=50, deadline=None)
@given(
    landscapes=st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=5),
        min_size=1,
        max_size=5,
    )
)
def test_every_prompt_lands_in_exactly_one_outcome(landscapes: list[list[float]]) -> None:
    """The counts partition the prompts, so a yield of zero always has a stated cause."""
    groups = [group(f"ex-{index}", values) for index, values in enumerate(landscapes)]

    stats = mine(*groups, min_margin=1e-9, max_pairs_per_prompt=3).stats()

    assert stats.prompts == len(groups)
    assert (
        stats.prompts_with_pairs
        + stats.no_pair_prompts
        + stats.saturated_prompts
        + stats.flat_prompts
        + stats.single_sample_prompts
        == len(groups)
    )


# --------------------------------------------------------------------------------------
# The degenerate case: a policy good enough to mine nothing from
# --------------------------------------------------------------------------------------


def test_an_almost_perfect_policy_yields_almost_no_pairs() -> None:
    """The documented end state of a successful pipeline, not a bug.

    Nineteen prompts on which every sample is already perfect and one on which the model
    still slips. The dataset that comes out has a single pair in it, and the honest reading
    is in `saturation_rate`: 95 per cent of the prompts have nothing left to teach. The
    response is to stop DPO on this data and find harder prompts, not to lower the margin
    until pairs appear.
    """
    groups = [group(f"ex-{index}", [1.0, 1.0, 1.0, 1.0]) for index in range(19)]
    groups.append(group("ex-19", [1.0, 1.0, 1.0, 0.2]))

    stats = mine(*groups).stats()

    assert stats.pairs == 1
    assert stats.prompts == 20
    assert stats.saturated_prompts == 19
    assert stats.saturation_rate == pytest.approx(0.95)
    assert stats.prompt_yield == pytest.approx(0.05)
    assert stats.flat_prompts == 0


def test_a_fully_saturated_run_is_empty_but_explained() -> None:
    groups = [group(f"ex-{index}", [1.0, 1.0, 1.0]) for index in range(8)]

    stats = mine(*groups).stats()

    assert stats.pairs == 0
    assert stats.saturation_rate == 1.0
    assert stats.mean_margin == 0.0
    assert stats.rejected_violations == {}


def test_the_saturation_rate_of_an_empty_run_is_zero() -> None:
    """No prompts means no denominator; a rate of zero beats a division by zero."""
    stats = MiningStats()

    assert stats.saturation_rate == 0.0
    assert stats.prompt_yield == 0.0
    assert stats.pairs_per_prompt == 0.0


def test_lowering_the_margin_cannot_rescue_a_saturated_run() -> None:
    """The tempting wrong fix, shown not to work: identical scores are identical at any
    threshold, because a positive margin can never be met by a gap of zero."""
    groups = [group(f"ex-{index}", [1.0, 1.0, 1.0]) for index in range(4)]

    assert mine(*groups, min_margin=1e-12).pairs == []


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def test_the_stats_count_prompts_with_usable_pairs() -> None:
    result = mine(
        group("ex-0", [1.0, 0.0]),
        group("ex-1", [0.5, 0.5]),
        # A gap under the default margin: two completions that differ, but by less than one
        # wrong field is worth, so there is nothing to prefer.
        group("ex-2", [1.0, 0.995]),
    )

    stats = result.stats()
    assert stats.prompts == 3
    assert stats.prompts_with_pairs == 1
    assert stats.no_pair_prompts == 1
    assert stats.flat_prompts == 1
    assert stats.prompt_yield == pytest.approx(1 / 3)
    assert stats.pairs_per_prompt == pytest.approx(1 / 3)


def test_the_mean_margin_is_the_mean_over_the_pairs() -> None:
    stats = mine(group("ex", [1.0, 0.6, 0.4, 0.0]), max_pairs_per_prompt=2).stats()

    assert stats.pairs == 2
    assert stats.mean_margin == pytest.approx(0.6)


def test_the_mean_margin_without_pairs_is_zero() -> None:
    result = mine(group("ex", [0.5, 0.5]))

    assert result.mean_margin == 0.0
    assert result.stats().mean_margin == 0.0


def test_the_rejected_violations_say_what_the_model_is_taught_to_stop_doing() -> None:
    stats = mine(
        group("ex-0", [1.0, 0.0], kinds=[None, ViolationKind.NO_JSON]),
        group("ex-1", [1.0, 0.1], kinds=[None, ViolationKind.BAD_ENUM]),
        group("ex-2", [1.0, 0.0], kinds=[None, ViolationKind.NO_JSON]),
    ).stats()

    assert stats.rejected_violations == {ViolationKind.NO_JSON: 2, ViolationKind.BAD_ENUM: 1}
    assert stats.rejected_without_violation == 0


def test_the_violation_distribution_is_ordered_most_severe_first() -> None:
    """`ViolationKind` is declared worst first, and the report follows that order however
    the prompts happened to arrive, so the top row is always the most serious failure."""
    stats = mine(
        group("ex-0", [1.0, 0.1], kinds=[None, ViolationKind.BAD_DATE]),
        group("ex-1", [1.0, 0.0], kinds=[None, ViolationKind.NO_JSON]),
    ).stats()

    assert list(stats.rejected_violations) == [ViolationKind.NO_JSON, ViolationKind.BAD_DATE]


def test_a_rejected_completion_with_no_violation_is_counted_separately() -> None:
    """A schema-valid completion can still be the worse of two. Dropping it from the report
    would make every mined dataset look as though it only ever rejected malformed output."""
    stats = mine(group("ex", [1.0, 0.5])).stats()

    assert stats.rejected_violations == {}
    assert stats.rejected_without_violation == 1


@settings(max_examples=50, deadline=None)
@given(
    landscapes=st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=5),
        min_size=1,
        max_size=4,
    )
)
def test_the_rejected_side_votes_add_up_to_the_pairs(landscapes: list[list[float]]) -> None:
    """Every pair contributes exactly one vote, so the distribution's denominator is the one
    its name implies. `MiningResult` enforces this; here it is checked end to end."""
    kinds: list[ViolationKind | None] = [None, ViolationKind.NO_JSON, ViolationKind.WRONG_TYPE]
    groups = [
        group(
            f"ex-{index}",
            values,
            kinds=[kinds[position % len(kinds)] for position in range(len(values))],
        )
        for index, values in enumerate(landscapes)
    ]

    stats = mine(*groups, min_margin=1e-9, max_pairs_per_prompt=3).stats()

    assert sum(stats.rejected_violations.values()) + stats.rejected_without_violation == stats.pairs


def test_the_per_slice_yield_splits_the_prompts() -> None:
    stats = mine(
        group("ex-0", [1.0, 0.0], slice_=Slice.CLEAN),
        group("ex-1", [1.0, 0.5], slice_=Slice.MANY_ITEMS),
        group("ex-2", [1.0, 0.0], slice_=Slice.MANY_ITEMS),
    ).stats()

    assert set(stats.slices) == {Slice.CLEAN, Slice.MANY_ITEMS}
    assert stats.slices[Slice.CLEAN].pairs == 1
    assert stats.slices[Slice.MANY_ITEMS].pairs == 2
    assert stats.slices[Slice.MANY_ITEMS].mean_margin == pytest.approx(0.75)
    assert stats.slices[Slice.MANY_ITEMS].pairs_per_prompt == pytest.approx(1.0)
    assert stats.slices[Slice.MANY_ITEMS].prompt_yield == pytest.approx(1.0)


def test_a_slice_with_prompts_but_no_pairs_is_still_reported() -> None:
    """The most interesting row in the table: a slice the sampler found no signal on."""
    stats = mine(
        group("ex-0", [1.0, 0.0], slice_=Slice.CLEAN),
        group("ex-1", [0.3, 0.3], slice_=Slice.LONG_CONTEXT),
    ).stats()

    assert stats.slices[Slice.LONG_CONTEXT] == SliceYield(prompts=1)
    assert stats.slices[Slice.LONG_CONTEXT].prompt_yield == 0.0
    assert stats.slices[Slice.LONG_CONTEXT].mean_margin == 0.0


def test_slices_with_no_prompts_are_left_out() -> None:
    stats = mine(group("ex", [1.0, 0.0], slice_=Slice.DISTRACTOR)).stats()

    assert list(stats.slices) == [Slice.DISTRACTOR]


def test_an_empty_slice_yield_has_zero_ratios() -> None:
    empty = SliceYield()

    assert empty.pairs_per_prompt == 0.0
    assert empty.prompt_yield == 0.0


@settings(max_examples=50, deadline=None)
@given(
    landscapes=st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=4),
        min_size=1,
        max_size=5,
    )
)
def test_the_slice_rows_conserve_the_prompts_and_the_pairs(landscapes: list[list[float]]) -> None:
    slices = list(Slice)
    groups = [
        group(f"ex-{index}", values, slice_=slices[index % len(slices)])
        for index, values in enumerate(landscapes)
    ]

    stats = mine(*groups, min_margin=1e-9, max_pairs_per_prompt=2).stats()

    assert sum(row.prompts for row in stats.slices.values()) == stats.prompts
    assert sum(row.pairs for row in stats.slices.values()) == stats.pairs
    assert sum(row.prompts_with_pairs for row in stats.slices.values()) == stats.prompts_with_pairs


def test_the_stats_serialise_for_a_run_manifest() -> None:
    stats = mine(
        group("ex-0", [1.0, 0.0], kinds=[None, ViolationKind.NO_JSON]),
        group("ex-1", [1.0, 1.0], slice_=Slice.MANY_ITEMS),
    ).stats()

    payload = json.loads(json.dumps(stats.as_dict()))

    assert payload["pairs"] == 1
    assert payload["saturated_prompts"] == 1
    assert payload["rejected_violations"] == {"no_json": 1}
    assert payload["slices"]["many_items"]["prompts"] == 1
    assert payload["saturation_rate"] == pytest.approx(0.5)
    assert payload["prompt_yield"] == pytest.approx(0.5)
    assert payload["pairs_per_prompt"] == pytest.approx(0.5)


def test_the_stats_of_an_empty_run_are_all_zero() -> None:
    result = mine_pairs([], [make_example("ex")], TableScorer({}))

    assert result.pairs == []
    assert result.prompts_seen == 0
    assert result.stats() == MiningStats()


# --------------------------------------------------------------------------------------
# Accounting invariants on the result object
# --------------------------------------------------------------------------------------


def test_a_result_whose_slice_counts_do_not_add_up_is_refused() -> None:
    """Guarded because a wrong denominator survives review: the percentage still prints."""
    with pytest.raises(ValueError, match="prompts were seen"):
        MiningResult(outcomes={PromptOutcome.PAIRED: 2}, prompts_by_slice={Slice.CLEAN: 1})


def test_a_result_whose_violation_votes_do_not_add_up_is_refused() -> None:
    pair = PreferencePair(
        example_id="ex",
        slice=Slice.CLEAN,
        prompt="prompt",
        chosen="good",
        rejected="bad",
        chosen_reward=1.0,
        rejected_reward=0.0,
    )

    with pytest.raises(ValueError, match="votes"):
        MiningResult(
            pairs=[pair],
            outcomes={PromptOutcome.PAIRED: 1},
            prompts_by_slice={Slice.CLEAN: 1},
        )


def test_an_empty_result_is_consistent() -> None:
    result = MiningResult()

    assert result.prompts_seen == 0
    assert result.prompts_with_pairs == 0
    assert result.mean_margin == 0.0


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_tie_breaking_is_reproducible_under_a_seed() -> None:
    first = mine(group("ex", [0.9, 0.9, 0.0]), max_pairs_per_prompt=1, seed=7)
    second = mine(group("ex", [0.9, 0.9, 0.0]), max_pairs_per_prompt=1, seed=7)

    assert [pair.chosen for pair in first.pairs] == [pair.chosen for pair in second.pairs]


def test_a_different_seed_can_break_a_tie_differently() -> None:
    """Ties are broken at random on purpose: always taking the first sample index would
    over-represent early completions as the chosen side for no reason but arrival order."""
    winners = {
        mine(group("ex", [0.9, 0.9, 0.0]), max_pairs_per_prompt=1, seed=seed).pairs[0].chosen
        for seed in range(20)
    }

    assert winners == {"ex:c0", "ex:c1"}


def test_the_seed_does_not_move_anything_when_no_rewards_tie() -> None:
    first = mine(group("ex", [1.0, 0.6, 0.2, 0.0]), max_pairs_per_prompt=2, seed=0)
    second = mine(group("ex", [1.0, 0.6, 0.2, 0.0]), max_pairs_per_prompt=2, seed=99)

    assert first.pairs == second.pairs


def test_pairs_do_not_depend_on_the_order_the_samples_arrive_in() -> None:
    """Completions are canonicalised before the tie-break shuffle, so a caller can
    concatenate shards or re-sort a file and mine the same dataset."""
    groups = [group(f"ex-{index}", [0.9, 0.9, 0.5, 0.0]) for index in range(3)]
    table: dict[str, Reward] = {}
    for item in groups:
        table.update(item.table)
    samples = [sample for item in groups for sample in item.samples]
    shuffled = list(samples)
    random.Random(0).shuffle(shuffled)
    examples = [item.example for item in groups]

    in_order = mine_pairs(samples, examples, TableScorer(table), max_pairs_per_prompt=2)
    out_of_order = mine_pairs(shuffled, examples, TableScorer(table), max_pairs_per_prompt=2)

    assert sorted(pair.model_dump_json() for pair in in_order.pairs) == sorted(
        pair.model_dump_json() for pair in out_of_order.pairs
    )


def test_mining_a_subset_reproduces_that_subset_of_the_full_run() -> None:
    """The tie-break stream is seeded per example id, not per position, so re-mining one
    prompt out of a run gives back exactly the pairs that prompt contributed to it."""
    groups = [group(f"ex-{index}", [0.8, 0.8, 0.8, 0.1]) for index in range(4)]
    full = mine(*groups, max_pairs_per_prompt=2)
    alone = mine(groups[2], max_pairs_per_prompt=2)

    assert [pair for pair in full.pairs if pair.example_id == "ex-2"] == alone.pairs


# --------------------------------------------------------------------------------------
# Rejected inputs and edges
# --------------------------------------------------------------------------------------


def test_duplicate_example_ids_are_refused() -> None:
    """Two golds under one id would score half the samples against the wrong record."""
    with pytest.raises(ValueError, match="duplicate example_id"):
        mine_pairs(
            [make_sample("ex", "text")],
            [make_example("ex"), make_example("ex", slice_=Slice.MANY_ITEMS)],
            TableScorer({"text": reward(1.0)}),
        )


def test_a_sample_naming_an_unknown_example_is_refused_before_scoring() -> None:
    scorer = TableScorer({"text": reward(1.0)})

    with pytest.raises(ValueError, match="not supplied"):
        mine_pairs([make_sample("ghost", "text")], [make_example("ex")], scorer)

    assert scorer.calls == 0


def test_examples_without_samples_are_not_counted_as_prompts() -> None:
    """A prompt nobody sampled has no yield, and counting it would dilute every rate."""
    result = mine_pairs(
        list(group("ex-0", [1.0, 0.0]).samples),
        [make_example("ex-0"), make_example("ex-1"), make_example("ex-2")],
        TableScorer(group("ex-0", [1.0, 0.0]).table),
    )

    assert result.prompts_seen == 1
    assert result.stats().prompts == 1


def test_the_verifier_is_called_once_for_the_whole_run() -> None:
    """Scoring is batched over the run rather than repeated per prompt, because a real
    verifier parses JSON and a per-prompt call would re-enter it for every group."""
    groups = [group(f"ex-{index}", [1.0, 0.0]) for index in range(3)]
    table: dict[str, Reward] = {}
    for item in groups:
        table.update(item.table)
    scorer = TableScorer(table)

    mine_pairs(
        [sample for item in groups for sample in item.samples],
        [item.example for item in groups],
        scorer,
    )

    assert scorer.calls == 1
    assert scorer.examples_seen == ["ex-0", "ex-1", "ex-2"]
    assert len(scorer.samples_seen) == 6


# --------------------------------------------------------------------------------------
# Against the real verifier
# --------------------------------------------------------------------------------------


def test_the_real_verifier_prefers_clean_json_to_junk() -> None:
    example = make_example("ex")
    samples = [
        make_sample("ex", CLEAN, 0),
        make_sample("ex", EXTRA_KEY, 1),
        make_sample("ex", JUNK, 2),
    ]

    result = mine_pairs(samples, [example], strict_verifier())

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.chosen == CLEAN
    assert pair.rejected == JUNK
    assert pair.chosen_reward == 1.0
    assert pair.rejected_reward == 0.0
    assert result.stats().rejected_violations == {ViolationKind.NO_JSON: 1}


def test_a_mined_pair_carries_the_shared_prompt() -> None:
    """The prompt on the pair is the one `render_prompt` produces, so the DPO batch and the
    supervised batch condition on character-for-character the same string."""
    result = mine_pairs(
        [make_sample("ex", CLEAN, 0), make_sample("ex", JUNK, 1)],
        [make_example("ex")],
        strict_verifier(),
    )

    assert result.pairs[0].prompt == render_prompt(NOTE)


def test_the_choice_of_verifier_decides_what_the_dataset_teaches() -> None:
    """The same two completions are a lesson about prose under the strict verifier and
    nothing at all under the lenient one, which recovers the object from the prose and scores
    both perfect. The verifier is the label, so this difference is the whole design."""
    samples = [make_sample("ex", CLEAN, 0), make_sample("ex", PROSE, 1)]
    examples = [make_example("ex")]

    strict = mine_pairs(samples, examples, strict_verifier())
    lenient = mine_pairs(samples, examples, lenient_verifier())

    assert [pair.rejected for pair in strict.pairs] == [PROSE]
    assert strict.stats().rejected_violations == {ViolationKind.UNPARSEABLE: 1}
    assert lenient.pairs == []
    assert lenient.stats().saturated_prompts == 1


def test_a_policy_that_is_already_right_yields_nothing_under_the_real_verifier() -> None:
    """The end state again, this time with the real reward: three textually different but
    equally correct completions score 1.0 apiece and the prompt is retired as saturated."""
    spaced = json.dumps(json.loads(CLEAN), indent=2)
    reordered = json.dumps(dict(reversed(list(json.loads(CLEAN).items()))))
    samples = [
        make_sample("ex", CLEAN, 0),
        make_sample("ex", spaced, 1),
        make_sample("ex", reordered, 2),
    ]

    stats = mine_pairs(samples, [make_example("ex")], strict_verifier()).stats()

    assert stats.pairs == 0
    assert stats.saturated_prompts == 1
    assert stats.saturation_rate == 1.0
