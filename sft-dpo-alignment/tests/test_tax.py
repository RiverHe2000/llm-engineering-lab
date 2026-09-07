"""Tests for the alignment-tax probe.

The probe exists to measure a cost, so the tests are about whether it can see one. Two
specimen replies stand behind almost every case: a compliant answer that every probe of its
kind must accept, and one JSON advice record, which is what a model trained hard on this task
starts replying to everything with. Every probe must reject that record. If any probe accepted
it, the tax would read as zero on exactly the failure it was built to detect.

The rest is the arithmetic: the comparison is paired over probe ids, a drop is reported with
the same statistics as the promotion gate, and twelve probes are honestly too few to resolve a
small change -- which the interval says out loud rather than the test papering over it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final, cast

import pytest

from sftdpo.eval.tax import (
    PROBES,
    Probe,
    ProbeKind,
    ProbeReport,
    alignment_tax,
    list_items,
    run_probes,
    score_probes,
    word_count,
)

FAST_RESAMPLES = 300

# What a model that still follows instructions replies. One per probe, checked below to be
# exhaustive, so adding a probe without a specimen reply fails rather than going untested.
COMPLIANT: Final[dict[str, str]] = {
    "one_word_capital": "Canberra",
    "one_word_larger": "Twelve.",
    "one_word_colour": "Blue",
    "one_word_season": "Spring",
    "constraint_yes_no": "Yes.",
    "constraint_prefix": "Because rain falls and a dry coat is worth carrying one.",
    "constraint_five_words": "To lend books freely.",
    "constraint_no_letter_e": "A cat naps on a warm mat.",
    "constraint_echo_number": "42",
    "list_three_colours": "red\ngreen\nblue",
    "list_five_cities": "Sydney\nMelbourne\nBrisbane\nPerth\nAdelaide",
    "list_two_habits": "- Track every expense\n- Cancel unused subscriptions",
}

# The failure mode the probe set is for: a model taught that every reply is an advice record.
# Spelled with the separators `json.dumps` uses, because that is what a model emits and because
# a completion with no spaces in it would count as a single word.
ADVICE_RECORD: Final = json.dumps(
    {
        "client_name": "Ada Lovelace",
        "record_date": "2026-03-31",
        "risk_profile": "balanced",
        "objectives": ["retire at 60"],
        "fees": {"advice_fee": 3300.0, "ongoing_fee_pct": 0.88},
        "review_months": 12,
    }
)


def replies(passing: Sequence[str]) -> dict[str, str]:
    """A full set of replies in which exactly `passing` satisfy their probe."""
    return {
        probe.probe_id: COMPLIANT[probe.probe_id] if probe.probe_id in passing else ADVICE_RECORD
        for probe in PROBES
    }


def report(model: str, passing: Sequence[str]) -> ProbeReport:
    return score_probes(replies(passing), model=model)


ALL_IDS: Final = tuple(probe.probe_id for probe in PROBES)


# --------------------------------------------------------------------------------------
# The probe set itself
# --------------------------------------------------------------------------------------


def test_the_probe_ids_are_unique() -> None:
    assert len(set(ALL_IDS)) == len(PROBES)


def test_all_three_kinds_of_instruction_following_are_covered() -> None:
    assert {probe.kind for probe in PROBES} == set(ProbeKind)


def test_the_probes_are_generic_instructions_not_extraction_tasks() -> None:
    """A probe that mentioned the schema would measure the training set, not the tax."""
    for probe in PROBES:
        lowered = probe.instruction.lower()
        assert "json" not in lowered
        assert "schema" not in lowered
        assert "adviser" not in lowered


def test_every_probe_has_a_specimen_compliant_reply() -> None:
    assert set(ALL_IDS) <= set(COMPLIANT)


@pytest.mark.parametrize("probe", PROBES, ids=ALL_IDS)
def test_a_compliant_reply_satisfies_its_probe(probe: Probe) -> None:
    assert probe.passed(COMPLIANT[probe.probe_id])


@pytest.mark.parametrize("probe", PROBES, ids=ALL_IDS)
def test_no_probe_is_fooled_by_an_advice_record(probe: Probe) -> None:
    """The whole point: the aligned model's favourite reply must fail every probe."""
    assert not probe.passed(ADVICE_RECORD)


@pytest.mark.parametrize("probe", PROBES, ids=ALL_IDS)
def test_no_probe_accepts_an_empty_reply(probe: Probe) -> None:
    assert not probe.passed("")


# --------------------------------------------------------------------------------------
# The checkable properties
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("one", 1), ("  two  words ", 2), ("a\nb\tc", 3)],
)
def test_word_count_splits_on_any_whitespace(text: str, expected: int) -> None:
    assert word_count(text) == expected


def test_list_items_strips_every_bullet_style() -> None:
    text = "- red\n* green\n1. blue\n2) black\n\n   \n  white"
    assert list_items(text) == ["red", "green", "blue", "black", "white"]


def test_list_items_of_an_empty_reply_is_empty() -> None:
    assert list_items("\n  \n") == []


def test_a_list_crammed_onto_one_line_has_not_followed_the_instruction() -> None:
    """Every list probe says one per line, so being generous here would measure the parser."""
    assert list_items("red, green, blue") == ["red, green, blue"]


@pytest.mark.parametrize(
    ("reply", "passes"),
    [
        ("Canberra", True),
        ("Canberra.", True),
        ('"Canberra"', True),
        ("well-known", True),
        ("The capital is Canberra", False),
        ("", False),
    ],
)
def test_the_one_word_check_ignores_punctuation_but_not_extra_words(
    reply: str, passes: bool
) -> None:
    probe = next(p for p in PROBES if p.kind is ProbeKind.ONE_WORD)
    assert probe.passed(reply) is passes


@pytest.mark.parametrize(
    ("reply", "passes"),
    [("Yes", True), ("no.", True), ("Yes, it is.", False), ("Salty", False)],
)
def test_the_yes_no_constraint_admits_nothing_else(reply: str, passes: bool) -> None:
    probe = next(p for p in PROBES if p.probe_id == "constraint_yes_no")
    assert probe.passed(reply) is passes


def test_the_prefix_constraint_checks_the_prefix_and_the_budget() -> None:
    probe = next(p for p in PROBES if p.probe_id == "constraint_prefix")
    assert probe.passed("Because it rains.")
    assert not probe.passed("It rains, so people carry one.")
    assert not probe.passed("Because " + "rain " * 25)


def test_the_word_budget_rejects_both_too_many_words_and_none() -> None:
    probe = next(p for p in PROBES if p.probe_id == "constraint_five_words")
    assert probe.passed("Borrowing books and quiet study")
    assert not probe.passed("Borrowing books and having a quiet place to study")
    assert not probe.passed("   ")


def test_the_forbidden_letter_is_checked_in_both_cases() -> None:
    probe = next(p for p in PROBES if p.probe_id == "constraint_no_letter_e")
    assert probe.passed("That cat is soft.")
    assert not probe.passed("The cat is soft.")
    assert not probe.passed("THE CAT IS SOFT.")


def test_the_echo_constraint_admits_only_the_exact_reply() -> None:
    probe = next(p for p in PROBES if p.probe_id == "constraint_echo_number")
    assert probe.passed("  42  ")
    assert not probe.passed("The answer is 42.")


def test_a_list_probe_counts_the_items_it_asked_for() -> None:
    probe = next(p for p in PROBES if p.probe_id == "list_three_colours")
    assert probe.passed("red\ngreen\nblue")
    assert not probe.passed("red\ngreen")
    assert not probe.passed("red\ngreen\nblue\nblack")


# --------------------------------------------------------------------------------------
# Scoring a run
# --------------------------------------------------------------------------------------


def test_scoring_reports_one_outcome_per_probe_in_probe_order() -> None:
    scored = report("base", ALL_IDS)
    assert tuple(outcome.probe_id for outcome in scored.outcomes) == ALL_IDS
    assert scored.pass_rate == 1.0
    assert scored.n == len(PROBES)


def test_scoring_refuses_a_run_with_a_probe_missing() -> None:
    """Skipping it would change the denominator of the pass rate without saying so."""
    partial = replies(ALL_IDS)
    partial.pop(ALL_IDS[0])
    with pytest.raises(ValueError, match="no response for probes"):
        score_probes(partial, model="base")


def test_scoring_refuses_a_reply_to_a_probe_that_does_not_exist() -> None:
    extra = replies(ALL_IDS)
    extra["one_word_invented"] = "Yes"
    with pytest.raises(ValueError, match="unknown probes"):
        score_probes(extra, model="base")


def test_an_empty_report_has_a_pass_rate_of_zero_rather_than_a_division() -> None:
    empty = ProbeReport(model="base")
    assert empty.n == 0
    assert empty.pass_rate == 0.0
    assert empty.per_kind() == {}


def test_the_per_kind_view_follows_the_enum_order() -> None:
    scored = report("base", ["one_word_capital", "list_three_colours"])
    assert list(scored.per_kind()) == [
        ProbeKind.ONE_WORD,
        ProbeKind.CONSTRAINT,
        ProbeKind.LIST_LENGTH,
    ]
    assert scored.per_kind()[ProbeKind.CONSTRAINT] == 0.0
    assert scored.per_kind()[ProbeKind.ONE_WORD] == pytest.approx(0.25)


def test_the_paired_view_is_keyed_by_probe_id() -> None:
    scored = report("base", ["one_word_capital"])
    passed = scored.passed_by_probe()
    assert passed["one_word_capital"] == 1.0
    assert passed["list_two_habits"] == 0.0


def test_a_responder_is_run_over_the_instructions_in_order() -> None:
    seen: list[str] = []

    def respond(instructions: Sequence[str]) -> list[str]:
        seen.extend(instructions)
        return [COMPLIANT[probe.probe_id] for probe in PROBES]

    scored = run_probes(respond, model="base")
    assert seen == [probe.instruction for probe in PROBES]
    assert scored.pass_rate == 1.0


def test_a_responder_that_drops_a_reply_is_refused() -> None:
    """Zipping a short reply list against the probes would misalign every one after it."""

    def respond(instructions: Sequence[str]) -> list[str]:
        return ["Yes"] * (len(instructions) - 1)

    with pytest.raises(ValueError, match="returned 11 replies for 12 probes"):
        run_probes(respond, model="base")


def test_a_custom_probe_set_is_honoured() -> None:
    def is_shouted(text: str) -> bool:
        return text.isupper()

    probes = (
        Probe(
            probe_id="shout",
            kind=ProbeKind.CONSTRAINT,
            instruction="Reply in capitals.",
            check=is_shouted,
        ),
    )
    scored = run_probes(lambda _: ["LOUDLY"], model="base", probes=probes)
    assert scored.pass_rate == 1.0
    assert scored.outcomes[0].probe_id == "shout"


# --------------------------------------------------------------------------------------
# The paired before-and-after comparison
# --------------------------------------------------------------------------------------


def test_a_model_that_did_not_change_pays_no_tax() -> None:
    before = report("base", ALL_IDS)
    result = alignment_tax(before, report("sft", ALL_IDS), n_resamples=FAST_RESAMPLES)
    assert result.n == len(PROBES)
    assert result.delta.point == 0.0
    assert result.tax == 0.0
    assert result.mcnemar_p == 1.0
    assert result.verdict.passed
    assert not result.significant


def test_a_model_that_lost_general_instruction_following_pays_a_measured_tax() -> None:
    """A drop here is the price of the method, and the probe set has to be able to say so."""
    before = report("base", ALL_IDS)
    after = report("sft", [])
    result = alignment_tax(before, after, n_resamples=FAST_RESAMPLES)
    assert result.before_rate == 1.0
    assert result.after_rate == 0.0
    assert result.tax == pytest.approx(1.0)
    assert result.delta.point == pytest.approx(-1.0)
    assert (result.mcnemar_b, result.mcnemar_c) == (12, 0)
    assert result.mcnemar_p == pytest.approx(2 / 2**12)
    assert result.significant
    assert not result.verdict.passed


def test_a_small_drop_over_twelve_probes_is_reported_as_unresolved() -> None:
    """Twelve probes cannot resolve one lost probe, and the interval must not pretend they can.

    The verdict is neither "significant" nor "within the margin": with a single discordant
    probe the interval runs from -0.25 to zero, which is too wide to demonstrate a loss and
    too wide to bound one at twenty points. Reporting both as false is the honest answer, and
    the reason the probe set is described as a cost measurement rather than a benchmark.
    """
    before = report("base", ALL_IDS)
    after = report("sft", ALL_IDS[1:])
    result = alignment_tax(before, after, margin=0.20, n_resamples=FAST_RESAMPLES)
    assert result.tax == pytest.approx(1 / 12)
    assert result.mcnemar_p == 1.0
    assert not result.significant
    assert not result.verdict.passed
    assert result.delta.low < -0.20
    # A margin the twelve probes can actually speak to is cleared by the same data.
    assert alignment_tax(before, after, margin=0.40, n_resamples=FAST_RESAMPLES).verdict.passed


def test_an_improvement_is_not_reported_as_a_negative_tax() -> None:
    before = report("base", [])
    after = report("sft", ALL_IDS)
    result = alignment_tax(before, after, n_resamples=FAST_RESAMPLES)
    assert result.tax == 0.0
    assert result.delta.point == pytest.approx(1.0)


def test_two_runs_over_different_probe_sets_cannot_be_compared() -> None:
    before = report("base", ALL_IDS)
    partial = ProbeReport(model="sft", outcomes=before.outcomes[:5])
    with pytest.raises(ValueError, match="cannot be paired"):
        alignment_tax(before, partial, n_resamples=FAST_RESAMPLES)


def test_the_comparison_is_reproducible_and_frozen() -> None:
    before = report("base", ALL_IDS)
    after = report("sft", ALL_IDS[:6])
    first = alignment_tax(before, after, n_resamples=FAST_RESAMPLES, seed=5)
    again = alignment_tax(before, after, n_resamples=FAST_RESAMPLES, seed=5)
    assert first.to_markdown() == again.to_markdown()
    mutable = cast("Any", first)
    with pytest.raises(ValueError, match="frozen"):
        mutable.after_rate = 0.0


def test_the_report_states_that_a_drop_is_a_cost_not_a_defect() -> None:
    """A table showing a drop invites the wrong conclusion unless the caveat travels with it."""
    result = alignment_tax(
        report("base", ALL_IDS), report("sft", ALL_IDS[:6]), n_resamples=FAST_RESAMPLES
    )
    text = result.to_markdown()
    assert text.startswith("# Alignment tax: sft vs base")
    assert "| Probes | 12 |" in text
    assert "| Pass rate before | 1.0000 |" in text
    assert "| Pass rate after | 0.5000 |" in text
    assert "| Probes lost / gained | 6 / 0 |" in text
    assert "cost of the alignment method, not a fault in the harness" in text


def test_the_report_never_renders_a_negative_zero() -> None:
    before = report("base", ALL_IDS)
    text = alignment_tax(before, report("sft", ALL_IDS), n_resamples=FAST_RESAMPLES).to_markdown()
    assert "-0.0000" not in text
    assert "| Difference | +0.0000 |" in text
