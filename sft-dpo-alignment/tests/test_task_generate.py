"""Tests for the note generator.

The generator is the foundation of every number this project reports, so these tests lean
on invariants rather than golden strings wherever an invariant exists: a golden string
locks in today's wording and tells you nothing when the wording changes on purpose, whereas
"the gold record validates", "the products appear in the order the record lists them" and
"nothing the record omits appears anywhere in the note" stay true across any rewrite of the
prose and fail loudly when the generator actually breaks.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sftdpo.schemas import (
    Action,
    AdviceRecord,
    Recommendation,
    RiskProfile,
    Slice,
    advice_record_json_schema,
)
from sftdpo.task.generate import (
    _ACTIONS_WITH_AMOUNT,
    _BUILDERS,
    _FILLER,
    _FLAG_PROSE,
    _LONG_CONTEXT_TARGET_WORDS,
    _OBJECTIVES,
    GeneratedNote,
    _amount_of,
    _basis_points,
    _date_long,
    _date_relative,
    _date_slash,
    _int_to_words,
    _money_decimal,
    _money_k,
    _money_words,
    _ordinal,
    _sentence,
    _whole_dollars,
    _word_count,
    generate,
    generate_one,
    render_prompt,
)

ALL_SLICES = list(Slice)
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SNAKE_CASE = re.compile(r"^[a-z]+(?:_[a-z0-9]+)*$")


def samples(note_slice: Slice, count: int = 24, seed: int = 11) -> list[GeneratedNote]:
    """A run of notes from one slice, drawn from a single seeded stream."""
    rng = random.Random(seed)
    return [generate_one(note_slice, rng) for _ in range(count)]


def every_slice(count: int = 12, seed: int = 5) -> list[GeneratedNote]:
    """A run covering every slice, for the invariants that hold corpus-wide."""
    return [note for note_slice in Slice for note in samples(note_slice, count, seed)]


def unpriced(note: GeneratedNote) -> list[Recommendation]:
    """Recommendations whose amount is genuinely absent.

    A hold has no amount because holding has no dollar figure, which is an ordinary record
    rather than an omission. Only a buy, sell or switch with no amount is the absent-fields
    slice withholding something the model must not invent.
    """
    return [
        rec
        for rec in note.gold.recommendations
        if rec.amount is None and rec.action is not Action.HOLD
    ]


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_generate_is_reproducible_for_a_seed() -> None:
    assert generate(seed=1, n=10) == generate(seed=1, n=10)


def test_generate_differs_across_seeds() -> None:
    assert generate(seed=1, n=10) != generate(seed=2, n=10)


def test_generate_one_is_a_pure_function_of_its_stream() -> None:
    first = generate_one(Slice.MIXED_FORMATS, random.Random(99))
    second = generate_one(Slice.MIXED_FORMATS, random.Random(99))
    assert first == second


def test_generate_is_a_prefix_of_a_longer_run() -> None:
    """One shared stream means a longer run extends a shorter one rather than replacing it."""
    short = generate(seed=4, n=6)
    assert generate(seed=4, n=12)[:6] == short


def test_generate_cycles_through_every_slice_evenly() -> None:
    produced = [note.slice for note in generate(seed=3, n=len(ALL_SLICES) * 4)]
    counts = {note_slice: produced.count(note_slice) for note_slice in ALL_SLICES}
    assert set(counts.values()) == {4}


def test_generate_honours_an_explicit_slice_subset() -> None:
    chosen = [Slice.CLEAN, Slice.MANY_ITEMS]
    produced = generate(seed=3, n=7, slices=chosen)
    assert [note.slice for note in produced] == [chosen[i % 2] for i in range(7)]


def test_generate_zero_returns_nothing() -> None:
    assert generate(seed=3, n=0) == []


def test_generate_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate(seed=1, n=-1)


def test_generate_rejects_an_empty_slice_sequence() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        generate(seed=1, n=2, slices=[])


def test_every_slice_has_a_builder() -> None:
    assert set(_BUILDERS) == set(Slice)


# --------------------------------------------------------------------------------------
# The gold record
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("note_slice", ALL_SLICES, ids=lambda s: s.value)
def test_gold_validates_against_the_schema(note_slice: Slice) -> None:
    """Every gold record survives a full serialise/validate round trip.

    This is the check the whole package depends on: the verifier scores model output against
    `AdviceRecord`, so a gold record that would not itself validate makes a perfect score
    unreachable.
    """
    for note in samples(note_slice):
        restored = AdviceRecord.model_validate(json.loads(note.gold.model_dump_json()))
        assert restored == note.gold


@pytest.mark.parametrize("note_slice", ALL_SLICES, ids=lambda s: s.value)
def test_gold_record_date_is_an_iso_date(note_slice: Slice) -> None:
    for note in samples(note_slice):
        assert ISO_DATE.match(note.gold.record_date)
        assert dt.date.fromisoformat(note.gold.record_date).year == 2026


def test_gold_flags_are_snake_case_tokens_from_the_catalogue() -> None:
    for note in every_slice():
        for flag in note.gold.flags:
            assert SNAKE_CASE.match(flag)
            assert flag in _FLAG_PROSE


def test_gold_objectives_come_from_the_catalogue() -> None:
    for note in every_slice():
        assert set(note.gold.objectives) <= set(_OBJECTIVES)
        assert len(set(note.gold.objectives)) == len(note.gold.objectives)


def test_gold_fees_and_review_period_are_positive() -> None:
    for note in every_slice():
        assert note.gold.fees.advice_fee > 0
        assert 0 < note.gold.fees.ongoing_fee_pct < 5
        assert note.gold.review_months > 0


def test_gold_risk_profile_is_a_known_enum_member() -> None:
    profiles = {note.gold.risk_profile for note in every_slice()}
    assert profiles <= set(RiskProfile)
    assert len(profiles) > 1


def test_a_hold_never_carries_an_amount() -> None:
    """Holding a position has no dollar figure, so gold must never invent one."""
    for note in every_slice():
        for rec in note.gold.recommendations:
            if rec.action is Action.HOLD:
                assert rec.amount is None


def test_amounts_are_whole_dollars() -> None:
    for note in every_slice():
        for rec in note.gold.recommendations:
            assert rec.amount is None or float(rec.amount).is_integer()


# --------------------------------------------------------------------------------------
# The relationship between note and gold
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("note_slice", ALL_SLICES, ids=lambda s: s.value)
def test_the_note_names_the_client(note_slice: Slice) -> None:
    for note in samples(note_slice):
        assert note.gold.client_name in note.note


@pytest.mark.parametrize("note_slice", ALL_SLICES, ids=lambda s: s.value)
def test_products_appear_once_and_in_the_order_the_record_lists_them(note_slice: Slice) -> None:
    """Ordering is scored by the verifier, so the note has to justify the order in gold."""
    for note in samples(note_slice):
        products = [rec.product for rec in note.gold.recommendations]
        assert len(set(products)) == len(products)
        positions = [note.note.index(product) for product in products]
        assert positions == sorted(positions)
        for product in products:
            assert note.note.count(product) == 1


@pytest.mark.parametrize("note_slice", ALL_SLICES, ids=lambda s: s.value)
def test_notes_are_plain_ascii(note_slice: Slice) -> None:
    """Non-ASCII punctuation would tokenise differently on every model and is never needed."""
    for note in samples(note_slice):
        assert note.note.isascii()


@pytest.mark.parametrize("note_slice", ALL_SLICES, ids=lambda s: s.value)
def test_the_generated_slice_label_matches_what_was_asked_for(note_slice: Slice) -> None:
    assert all(note.slice is note_slice for note in samples(note_slice, count=4))


# --------------------------------------------------------------------------------------
# Slice-specific behaviour
# --------------------------------------------------------------------------------------


def test_clean_notes_state_the_date_already_normalised() -> None:
    for note in samples(Slice.CLEAN):
        assert f"Date of record: {note.gold.record_date}." in note.note


def test_clean_notes_state_every_objective_verbatim() -> None:
    for note in samples(Slice.CLEAN):
        for objective in note.gold.objectives:
            assert objective in note.note


def test_distractor_notes_correct_a_superseded_amount() -> None:
    """The corrected figure is the gold; the superseded one appears earlier and is not."""
    for note in samples(Slice.DISTRACTOR):
        corrected = note.gold.recommendations[0]
        assert corrected.amount is not None
        marker = f"- actually, make that ${corrected.amount:,.0f}."
        assert marker in note.note
        superseded = note.note[: note.note.index(marker)].rsplit("\n", 1)[-1]
        assert f"${corrected.amount:,.0f}" not in superseded


def test_distractor_notes_correct_the_review_period() -> None:
    for note in samples(Slice.DISTRACTOR):
        assert "ignore that, book it for" in note.note
        stale = int(re.findall(r"I said (\d+) months", note.note)[0])
        assert stale != note.gold.review_months
        assert f"book it for {note.gold.review_months} months" in note.note


def test_distractor_notes_carry_chit_chat_that_gold_ignores() -> None:
    """The anecdote must not leak into the record as an objective or a flag."""
    for note in samples(Slice.DISTRACTOR):
        assert len(note.note.splitlines()) > len(note.gold.recommendations) + 5
        assert all(objective in note.note for objective in note.gold.objectives)


def test_mixed_formats_rotates_which_date_style_carries_the_record_date() -> None:
    used: set[str] = set()
    for note in samples(Slice.MIXED_FORMATS, count=30):
        record = dt.date.fromisoformat(note.gold.record_date)
        matched = [
            style.__name__
            for style in (_date_long, _date_slash, _date_relative)
            if style(record) in note.note
        ]
        assert len(matched) == 1
        used.update(matched)
    assert used == {"_date_long", "_date_slash", "_date_relative"}


def test_mixed_formats_uses_all_three_money_styles_in_every_note() -> None:
    for note in samples(Slice.MIXED_FORMATS):
        amounts = [rec.amount for rec in note.gold.recommendations]
        assert all(amount is not None for amount in amounts)
        for style in (_money_words, _money_k, _money_decimal):
            matches = [a for a in amounts if a is not None and style(a) in note.note]
            assert len(matches) == 1


def test_mixed_formats_quotes_the_ongoing_fee_in_basis_points() -> None:
    for note in samples(Slice.MIXED_FORMATS):
        assert _basis_points(note.gold.fees.ongoing_fee_pct) in note.note


def test_mixed_formats_spells_out_the_review_period() -> None:
    for note in samples(Slice.MIXED_FORMATS):
        assert f"Review in {_int_to_words(note.gold.review_months)} months." in note.note


def test_absent_fields_exercises_all_three_kinds_of_absence() -> None:
    drawn = samples(Slice.ABSENT_FIELDS, count=45)
    assert any(not note.gold.objectives for note in drawn)
    assert any(not note.gold.flags for note in drawn)
    assert any(unpriced(note) for note in drawn)


def test_absent_fields_always_omits_exactly_one_thing() -> None:
    for note in samples(Slice.ABSENT_FIELDS, count=45):
        missing = [not note.gold.objectives, not note.gold.flags, bool(unpriced(note))]
        assert sum(missing) == 1


def test_an_empty_objective_list_means_no_objective_survives_in_the_note() -> None:
    """The test that a model inventing an objective would fail: there is nothing to invent
    from, because no catalogue phrase appears anywhere in the text."""
    seen = 0
    for note in samples(Slice.ABSENT_FIELDS, count=45):
        if note.gold.objectives:
            continue
        seen += 1
        for objective in _OBJECTIVES:
            assert objective not in note.note
    assert seen > 0


def test_an_empty_flag_list_means_no_flag_prose_survives_in_the_note() -> None:
    seen = 0
    for note in samples(Slice.ABSENT_FIELDS, count=45):
        if note.gold.flags:
            continue
        seen += 1
        for prose in _FLAG_PROSE.values():
            assert prose not in note.note
    assert seen > 0


def test_a_null_amount_is_stated_as_unconfirmed_rather_than_dropped() -> None:
    seen = 0
    for note in samples(Slice.ABSENT_FIELDS, count=45):
        missing = unpriced(note)
        if not missing:
            continue
        seen += 1
        assert len(missing) == 1
        assert missing[0].action in _ACTIONS_WITH_AMOUNT
        assert "still to be confirmed with the fund" in note.note
    assert seen > 0


def test_long_context_notes_clear_six_hundred_words() -> None:
    for note in samples(Slice.LONG_CONTEXT, count=20):
        assert _word_count(note.note) >= 600


def test_the_filler_catalogue_is_long_enough_to_reach_the_target() -> None:
    """Why the padding loop terminates: the catalogue alone outruns the word target, so it
    can never be exhausted before the target is met and no sentence is ever repeated."""
    assert sum(_word_count(sentence) for sentence in _FILLER) > _LONG_CONTEXT_TARGET_WORDS
    assert len(set(_FILLER)) == len(_FILLER)


def test_long_context_filler_introduces_no_competing_figures() -> None:
    """The padding carries no digits, so length is the only thing this slice varies."""
    for sentence in _FILLER:
        assert not any(character.isdigit() for character in sentence)
        assert "$" not in sentence
        assert "%" not in sentence


def test_many_items_produces_five_to_eight_recommendations() -> None:
    counts = {len(note.gold.recommendations) for note in samples(Slice.MANY_ITEMS, count=40)}
    assert counts <= {5, 6, 7, 8}
    assert len(counts) > 1


def test_many_items_numbers_the_list_in_gold_order() -> None:
    for note in samples(Slice.MANY_ITEMS):
        for position, rec in enumerate(note.gold.recommendations, start=1):
            line = next(line for line in note.note.splitlines() if rec.product in line)
            assert line.startswith(f"{position}. ")


def test_slices_differ_in_length_as_intended() -> None:
    """A one-line ordering of the slices by note length, which is the difficulty knob."""

    def median_words(note_slice: Slice) -> float:
        lengths = sorted(_word_count(note.note) for note in samples(note_slice, count=16))
        return lengths[len(lengths) // 2]

    assert median_words(Slice.CLEAN) < median_words(Slice.MANY_ITEMS)
    assert median_words(Slice.MANY_ITEMS) < median_words(Slice.LONG_CONTEXT)


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (20, "twenty"),
        (24, "twenty-four"),
        (100, "one hundred"),
        (112, "one hundred and twelve"),
        (1000, "one thousand"),
        (12000, "twelve thousand"),
        (2200, "two thousand and two hundred"),
        (150000, "one hundred and fifty thousand"),
        (999999, "nine hundred and ninety-nine thousand and nine hundred and ninety-nine"),
    ],
)
def test_int_to_words_examples(value: int, expected: str) -> None:
    assert _int_to_words(value) == expected


@given(st.integers(min_value=0, max_value=999_999))
def test_int_to_words_is_always_lowercase_words(value: int) -> None:
    words = _int_to_words(value)
    assert words == words.lower()
    assert set(words) <= set("abcdefghijklmnopqrstuvwxyz -")


def test_int_to_words_is_injective_over_the_thousands_used_by_the_generator() -> None:
    """Two different figures must never spell out the same, or the word form would be
    ambiguous and the mixed-formats slice unanswerable."""
    values = list(range(1000)) + [n * 500 for n in range(1, 500)]
    assert len({_int_to_words(value) for value in values}) == len(set(values))


@pytest.mark.parametrize("value", [-1, 1_000_000])
def test_int_to_words_rejects_values_it_cannot_spell(value: int) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1000000\)"):
        _int_to_words(value)


def test_whole_dollars_rejects_cents() -> None:
    with pytest.raises(ValueError, match="whole number of dollars"):
        _whole_dollars(1200.5)


def test_money_words_rejects_cents() -> None:
    with pytest.raises(ValueError, match="whole number of dollars"):
        _money_words(99.99)


def test_money_k_needs_whole_thousands() -> None:
    assert _money_k(12000.0) == "$12k"
    with pytest.raises(ValueError, match="whole number of thousands"):
        _money_k(12500.0)


def test_money_decimal_uses_a_thousands_separator() -> None:
    assert _money_decimal(12000.0) == "12,000.00"


def test_basis_points_rejects_a_fractional_point() -> None:
    assert _basis_points(0.77) == "77 basis points"
    with pytest.raises(ValueError, match="basis points"):
        _basis_points(0.775)


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (28, "28th"),
    ],
)
def test_ordinal_suffixes(day: int, expected: str) -> None:
    assert _ordinal(day) == expected


def test_date_styles_render_the_same_day_three_ways() -> None:
    day = dt.date(2026, 3, 3)
    assert _date_long(day) == "3 March 2026"
    assert _date_slash(day) == "03/03/2026"
    assert _date_relative(day) == "next Tuesday the 3rd of March"


@given(st.dates(min_value=dt.date(2020, 1, 1), max_value=dt.date(2035, 12, 31)))
def test_the_slash_style_is_day_first_and_fixed_width(day: dt.date) -> None:
    rendered = _date_slash(day)
    assert len(rendered) == 10
    assert dt.datetime.strptime(rendered, "%d/%m/%Y").date() == day


def test_sentence_capitalises_without_touching_the_rest() -> None:
    assert _sentence("buy $10 of BHP Group (ASX: BHP)") == "Buy $10 of BHP Group (ASX: BHP)"


def test_amount_of_rejects_a_recommendation_with_no_figure() -> None:
    hold = next(
        rec
        for note in samples(Slice.CLEAN, count=40)
        for rec in note.gold.recommendations
        if rec.action is Action.HOLD
    )
    with pytest.raises(ValueError, match="no amount to supersede"):
        _amount_of(hold)


# --------------------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------------------


def test_render_prompt_embeds_the_note_and_the_schema() -> None:
    prompt = render_prompt("File note for Angus Boland.")
    assert "File note for Angus Boland." in prompt
    assert json.dumps(advice_record_json_schema(), indent=2, sort_keys=True) in prompt


def test_render_prompt_names_every_top_level_field() -> None:
    """A field missing from the schema block is a field the model cannot be expected to emit."""
    prompt = render_prompt("note")
    for field in AdviceRecord.model_fields:
        assert field in prompt


def test_render_prompt_demands_a_single_bare_json_object() -> None:
    prompt = render_prompt("note")
    assert "exactly one JSON object and nothing else" in prompt
    assert "no markdown code fence" in prompt


def test_render_prompt_states_the_rules_the_slices_test() -> None:
    """Each difficulty slice has a matching rule in the instruction, so a failure is a
    failure to follow instructions rather than a failure to guess them."""
    prompt = render_prompt("note")
    assert "YYYY-MM-DD" in prompt
    assert "Never guess a value the note does not contain" in prompt
    assert "record the corrected value" in prompt


@given(st.text(max_size=200))
def test_render_prompt_is_deterministic(note: str) -> None:
    assert render_prompt(note) == render_prompt(note)


@given(st.text(max_size=200))
def test_render_prompt_ends_with_the_note_and_a_cue(note: str) -> None:
    assert render_prompt(note).endswith(f"Adviser note:\n{note}\n\nJSON object:\n")


@given(st.text(max_size=200))
def test_render_prompt_only_varies_by_its_note(note: str) -> None:
    """Training and evaluation share this function, so everything but the note must be fixed."""
    prompt = render_prompt(note)
    head, _, tail = prompt.partition("Adviser note:\n")
    assert head == render_prompt("x").partition("Adviser note:\n")[0]
    assert tail == f"{note}\n\nJSON object:\n"


def test_render_prompt_works_on_a_generated_note() -> None:
    note = generate(seed=2, n=1)[0]
    prompt = render_prompt(note.note)
    assert note.note in prompt
    assert prompt.count("JSON object:") == 1


def test_word_count_ignores_repeated_whitespace() -> None:
    assert _word_count("  two   words \n") == 2


def test_helper_sequences_are_immutable_module_state() -> None:
    """Catalogues are tuples, so a caller cannot mutate the corpus out from under a hash."""
    catalogues: Sequence[Sequence[str]] = (_FILLER, _OBJECTIVES)
    assert all(isinstance(catalogue, tuple) for catalogue in catalogues)
