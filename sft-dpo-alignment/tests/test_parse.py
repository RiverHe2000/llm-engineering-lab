"""Tests for JSON recovery.

The strict/lenient gap is a reported number, so these tests pin down not only whether a
completion is recovered but exactly which repairs it took to recover it. A repair that
silently stopped firing would move the headline without breaking anything else.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from sftdpo.schemas import ViolationKind
from sftdpo.verify.parse import ParseOutcome, Repair, extract_json
from sftdpo.verify.parse import _string_mask as string_mask

OBJECT: dict[str, Any] = {"client_name": "Ada", "review_months": 12}
OBJECT_TEXT = json.dumps(OBJECT)

_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
)
_JSON_OBJECTS = st.dictionaries(st.text(min_size=1, max_size=8), _JSON_SCALARS, max_size=5)
_AWKWARD_TEXT = st.text(alphabet="{}[]\"'\\,: abc\n", max_size=30)


# --------------------------------------------------------------------------------------
# Strict mode
# --------------------------------------------------------------------------------------


def test_strict_accepts_exactly_one_object() -> None:
    outcome = extract_json(OBJECT_TEXT, strict=True)
    assert outcome.ok
    assert outcome.value == OBJECT
    assert outcome.repairs == ()
    assert outcome.violation is None


def test_strict_tolerates_surrounding_whitespace() -> None:
    """Whitespace is not content; a trailing newline is not a formatting failure."""
    outcome = extract_json(f"\n\t {OBJECT_TEXT}  \n", strict=True)
    assert outcome.ok
    assert outcome.value == OBJECT


def test_strict_preserves_nesting_and_types() -> None:
    nested = {"a": [{"b": [1, 2.5, None, True]}], "c": {"d": "e"}}
    outcome = extract_json(json.dumps(nested), strict=True)
    assert outcome.value == nested


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("", ViolationKind.NO_JSON),
        ("   \n\t ", ViolationKind.NO_JSON),
        ("I cannot help with that.", ViolationKind.NO_JSON),
        ("[1, 2]", ViolationKind.NOT_AN_OBJECT),
        ('"just a string"', ViolationKind.NOT_AN_OBJECT),
        ("42", ViolationKind.NOT_AN_OBJECT),
        ("null", ViolationKind.NOT_AN_OBJECT),
        ("true", ViolationKind.NOT_AN_OBJECT),
        (f"```json\n{OBJECT_TEXT}\n```", ViolationKind.UNPARSEABLE),
        (f"Here is the JSON:\n{OBJECT_TEXT}", ViolationKind.UNPARSEABLE),
        (f"{OBJECT_TEXT}\nHope that helps.", ViolationKind.UNPARSEABLE),
        (f"{OBJECT_TEXT} {OBJECT_TEXT}", ViolationKind.UNPARSEABLE),
        (f"{OBJECT_TEXT}}}}}", ViolationKind.UNPARSEABLE),
        ("{'a': 1}", ViolationKind.UNPARSEABLE),
        ('{"a": 1,}', ViolationKind.UNPARSEABLE),
        ('{"a": None}', ViolationKind.UNPARSEABLE),
        ('{"a": 1', ViolationKind.UNPARSEABLE),
    ],
)
def test_strict_rejects_anything_but_one_object(text: str, kind: ViolationKind) -> None:
    outcome = extract_json(text, strict=True)
    assert not outcome.ok
    assert outcome.value is None
    assert outcome.violation is not None
    assert outcome.violation.kind is kind


def test_strict_failure_carries_a_detail() -> None:
    outcome = extract_json("{'a': 1}", strict=True)
    assert outcome.violation is not None
    assert outcome.violation.detail


# --------------------------------------------------------------------------------------
# Lenient mode: one test per named repair
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "repairs"),
    [
        (
            f"```json\n{OBJECT_TEXT}\n```",
            OBJECT,
            (Repair.CODE_FENCE,),
        ),
        (
            f"Here is the JSON:\n{OBJECT_TEXT}",
            OBJECT,
            (Repair.LEADING_PROSE,),
        ),
        (
            f"{OBJECT_TEXT}\nHope that helps.",
            OBJECT,
            (Repair.TRAILING_PROSE,),
        ),
        (
            f"{OBJECT_TEXT}\n{OBJECT_TEXT}",
            OBJECT,
            (Repair.DUPLICATE_OBJECT,),
        ),
        (
            f"{OBJECT_TEXT}}}}}",
            OBJECT,
            (Repair.UNBALANCED_BRACES,),
        ),
        (
            '{"client_name": "Ada", "review_months": 12',
            OBJECT,
            (Repair.CLOSED_UNTERMINATED,),
        ),
        (
            '{"a": None, "b": True, "c": False}',
            {"a": None, "b": True, "c": False},
            (Repair.PYTHON_LITERALS,),
        ),
        (
            '{"a": 1, "b": [1, 2,],}',
            {"a": 1, "b": [1, 2]},
            (Repair.TRAILING_COMMA,),
        ),
        (
            "{'client_name': 'Ada', 'review_months': 12}",
            OBJECT,
            (Repair.SINGLE_QUOTES,),
        ),
    ],
)
def test_each_repair_is_named_individually(
    text: str, expected: dict[str, Any], repairs: tuple[Repair, ...]
) -> None:
    outcome = extract_json(text, strict=False)
    assert outcome.ok
    assert outcome.value == expected
    assert outcome.repairs == repairs


@pytest.mark.parametrize(
    ("text", "expected", "repairs"),
    [
        (
            f"Here you go:\n{OBJECT_TEXT}\nLet me know if you need changes.",
            OBJECT,
            (Repair.LEADING_PROSE, Repair.TRAILING_PROSE),
        ),
        (
            "Sure!\n```json\n{'a': 1,}\n```\nDone.",
            {"a": 1},
            (Repair.CODE_FENCE, Repair.TRAILING_COMMA, Repair.SINGLE_QUOTES),
        ),
        (
            '[{"a": 1}]',
            {"a": 1},
            (Repair.LEADING_PROSE, Repair.UNBALANCED_BRACES),
        ),
        (
            "Here's the JSON: {'a': None,} and that's my answer.",
            {"a": None},
            (
                Repair.LEADING_PROSE,
                Repair.TRAILING_PROSE,
                Repair.PYTHON_LITERALS,
                Repair.TRAILING_COMMA,
                Repair.SINGLE_QUOTES,
            ),
        ),
    ],
)
def test_repairs_compose_and_stay_individually_named(
    text: str, expected: dict[str, Any], repairs: tuple[Repair, ...]
) -> None:
    outcome = extract_json(text, strict=False)
    assert outcome.ok
    assert outcome.value == expected
    assert outcome.repairs == repairs


def test_lenient_records_no_repair_for_a_clean_completion() -> None:
    outcome = extract_json(OBJECT_TEXT, strict=False)
    assert outcome.ok
    assert outcome.repairs == ()
    assert not outcome.repaired


def test_lenient_keeps_the_first_of_two_different_objects() -> None:
    outcome = extract_json('{"a": 1} {"a": 2}', strict=False)
    assert outcome.value == {"a": 1}
    assert outcome.repairs == (Repair.DUPLICATE_OBJECT,)


def test_lenient_reports_no_json_when_there_is_no_brace() -> None:
    outcome = extract_json("I am unable to produce that.", strict=False)
    assert not outcome.ok
    assert outcome.violation is not None
    assert outcome.violation.kind is ViolationKind.NO_JSON


def test_lenient_reports_no_json_for_a_top_level_array() -> None:
    """Strict calls this NOT_AN_OBJECT; lenient never saw a `{`, so it says NO_JSON.

    Each answer is correct in its own terms, and the difference is recorded here so a later
    reading of the slice breakdown is not surprised by it.
    """
    outcome = extract_json("[1, 2]", strict=False)
    assert outcome.violation is not None
    assert outcome.violation.kind is ViolationKind.NO_JSON


def test_lenient_does_not_invent_content_for_a_truncation_inside_a_string() -> None:
    """Closing brackets can be added; a missing half of a string value cannot be guessed."""
    outcome = extract_json('{"client_name": "Ad', strict=False)
    assert not outcome.ok
    assert outcome.violation is not None
    assert outcome.violation.kind is ViolationKind.UNPARSEABLE


def test_repairs_are_recorded_even_when_the_parse_fails() -> None:
    outcome = extract_json('{"client_name": "Ad', strict=False)
    assert Repair.CLOSED_UNTERMINATED in outcome.repairs


def test_lenient_gives_up_on_irreparable_syntax() -> None:
    outcome = extract_json('{"a" 1 "b" 2}', strict=False)
    assert not outcome.ok
    assert outcome.violation is not None
    assert outcome.violation.kind is ViolationKind.UNPARSEABLE


# --------------------------------------------------------------------------------------
# String-aware scanning
# --------------------------------------------------------------------------------------


def test_brace_inside_a_string_value_does_not_end_the_object() -> None:
    outcome = extract_json('Here: {"note": "use {curly} braces", "b": 1} ok', strict=False)
    assert outcome.value == {"note": "use {curly} braces", "b": 1}


def test_escaped_quote_inside_a_string_does_not_end_it() -> None:
    text = json.dumps({"note": 'he said "hi" and left {}'})
    outcome = extract_json(f"Answer: {text} end", strict=False)
    assert outcome.value == {"note": 'he said "hi" and left {}'}


def test_apostrophe_in_prose_is_not_a_string_delimiter() -> None:
    """An apostrophe in a preamble must not open a literal that swallows the object."""
    outcome = extract_json(f"Here's the JSON: {OBJECT_TEXT}", strict=False)
    assert outcome.ok
    assert outcome.value == OBJECT


def test_single_quoted_value_may_contain_a_brace() -> None:
    outcome = extract_json("{'a': 'x}y'}", strict=False)
    assert outcome.value == {"a": "x}y"}
    assert outcome.repairs == (Repair.SINGLE_QUOTES,)


def test_single_quoted_value_may_contain_an_escaped_apostrophe() -> None:
    outcome = extract_json(r"{'a': 'it\'s fine'}", strict=False)
    assert outcome.value == {"a": "it's fine"}


def test_single_quoted_value_may_contain_a_double_quote() -> None:
    outcome = extract_json("{'a': 'he said \"hi\"'}", strict=False)
    assert outcome.value == {"a": 'he said "hi"'}


def test_mixed_quoting_converts_only_the_single_quoted_parts() -> None:
    """Models drift between quote styles mid-object; the double-quoted half is left alone."""
    outcome = extract_json("{\"a\": 1, 'b': 'two'}", strict=False)
    assert outcome.value == {"a": 1, "b": "two"}
    assert outcome.repairs == (Repair.SINGLE_QUOTES,)


def test_python_literal_inside_a_string_is_left_alone() -> None:
    outcome = extract_json('Note: {"a": "None means None", "b": None} done', strict=False)
    assert outcome.value == {"a": "None means None", "b": None}


def test_trailing_comma_pattern_inside_a_string_is_left_alone() -> None:
    outcome = extract_json('{"a": "ends with ,}", "b": 1,}', strict=False)
    assert outcome.value == {"a": "ends with ,}", "b": 1}


# --------------------------------------------------------------------------------------
# Fences
# --------------------------------------------------------------------------------------


def test_fence_consumes_the_prose_around_it() -> None:
    """The fence is the stronger signal, so prose outside it is not separately named."""
    outcome = extract_json(
        f"Sure thing!\n```json\n{OBJECT_TEXT}\n```\nAnything else?", strict=False
    )
    assert outcome.repairs == (Repair.CODE_FENCE,)


def test_unterminated_fence_still_yields_its_body() -> None:
    outcome = extract_json(f"```json\n{OBJECT_TEXT}", strict=False)
    assert outcome.ok
    assert outcome.value == OBJECT
    assert outcome.repairs == (Repair.CODE_FENCE,)


def test_fence_without_a_language_tag() -> None:
    outcome = extract_json(f"```\n{OBJECT_TEXT}\n```", strict=False)
    assert outcome.value == OBJECT
    assert outcome.repairs == (Repair.CODE_FENCE,)


def test_fence_on_a_single_line() -> None:
    outcome = extract_json(f"```json{OBJECT_TEXT}```", strict=False)
    assert outcome.value == OBJECT


def test_first_fence_holding_a_brace_wins() -> None:
    text = f"```text\nno object here\n```\n```json\n{OBJECT_TEXT}\n```"
    outcome = extract_json(text, strict=False)
    assert outcome.value == OBJECT


# --------------------------------------------------------------------------------------
# ParseOutcome itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ok": True},
        {"ok": False, "value": {"a": 1}},
        {"ok": True, "value": {"a": 1}, "violation": {"kind": ViolationKind.NO_JSON}},
        {"ok": False},
    ],
)
def test_outcome_rejects_disagreeing_fields(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ParseOutcome(**kwargs)


def test_outcome_is_frozen() -> None:
    outcome = extract_json(OBJECT_TEXT, strict=True)
    with pytest.raises(ValidationError):
        outcome.ok = False  # type: ignore[misc]  # the point of the test is the runtime guard


def test_repaired_property_tracks_the_repair_list() -> None:
    assert not extract_json(OBJECT_TEXT, strict=False).repaired
    assert extract_json(f"```json\n{OBJECT_TEXT}\n```", strict=False).repaired


def test_repairs_are_strings() -> None:
    outcome = extract_json(f"```json\n{OBJECT_TEXT}\n```", strict=False)
    assert all(isinstance(repair, str) for repair in outcome.repairs)
    assert outcome.repairs == ("code_fence",)


def test_every_repair_member_is_reachable() -> None:
    """A repair nobody can trigger is a line in the histogram that will always read zero."""
    triggers = [
        f"```json\n{OBJECT_TEXT}\n```",
        f"prefix {OBJECT_TEXT}",
        f"{OBJECT_TEXT} suffix",
        f"{OBJECT_TEXT} {OBJECT_TEXT}",
        f"{OBJECT_TEXT}}}",
        '{"a": 1',
        '{"a": None}',
        '{"a": 1,}',
        "{'a': 1}",
    ]
    seen: set[str] = set()
    for text in triggers:
        seen.update(extract_json(text, strict=False).repairs)
    assert seen == {repair.value for repair in Repair}


# --------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(_JSON_OBJECTS)
def test_strict_round_trips_any_json_object(obj: dict[str, Any]) -> None:
    outcome = extract_json(json.dumps(obj), strict=True)
    assert outcome.ok
    assert outcome.value == obj


@settings(max_examples=50, deadline=None)
@given(_JSON_OBJECTS, st.sampled_from(["", " ", "\n", "\t\n "]), st.sampled_from(["", "\n", "  "]))
def test_lenient_is_a_superset_of_strict(obj: dict[str, Any], lead: str, trail: str) -> None:
    """Whatever strict accepts, lenient accepts identically and with no repairs.

    This is what makes the strict-minus-lenient gap a difference in model behaviour rather
    than a difference between two code paths that happen to disagree.
    """
    text = lead + json.dumps(obj) + trail
    strict = extract_json(text, strict=True)
    lenient = extract_json(text, strict=False)
    assert strict.ok
    assert lenient.ok
    assert lenient.value == strict.value
    assert lenient.repairs == ()


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=60), st.booleans())
def test_extract_json_is_total(text: str, strict: bool) -> None:
    """Completions are arbitrary text; the verifier may never raise on one."""
    outcome = extract_json(text, strict=strict)
    assert outcome.ok == (outcome.value is not None)


@settings(max_examples=100, deadline=None)
@given(_AWKWARD_TEXT, st.booleans())
def test_extract_json_is_total_on_awkward_text(text: str, strict: bool) -> None:
    outcome = extract_json(text, strict=strict)
    assert outcome.ok == (outcome.violation is None)


@settings(max_examples=50, deadline=None)
@given(_AWKWARD_TEXT)
def test_braces_and_quotes_in_a_value_survive_a_fence(payload: str) -> None:
    obj = {"note": payload}
    text = f"Here is the JSON:\n```json\n{json.dumps(obj)}\n```\nHope that helps."
    outcome = extract_json(text, strict=False)
    assert outcome.ok
    assert outcome.value == obj


@settings(max_examples=50, deadline=None)
@given(st.text(alphabet='{}[]"\\ ab', max_size=25))
def test_string_mask_covers_every_brace_inside_a_string(payload: str) -> None:
    """The only unmasked braces in a dumped object are the two that delimit it."""
    text = json.dumps({"k": payload})
    mask = string_mask(text)
    unmasked = [i for i, ch in enumerate(text) if ch in "{}" and not mask[i]]
    assert unmasked == [0, len(text) - 1]


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=60))
def test_repairs_are_ordered_and_unique(text: str) -> None:
    order = [repair.value for repair in Repair]
    repairs = list(extract_json(text, strict=False).repairs)
    assert len(set(repairs)) == len(repairs)
    assert repairs == sorted(repairs, key=order.index)


@settings(max_examples=50, deadline=None)
@given(_AWKWARD_TEXT)
def test_a_recovered_object_is_itself_strictly_parseable(text: str) -> None:
    """Recovery lands in the strict language, so the two modes agree on the result.

    Stated over `json.dumps` of the recovered value on purpose, and the limit of that is
    worth naming: it checks that recovery produces a value the strict mode accepts when it
    is re-serialised, not that the *original* text was strictly parseable --- which it was
    not, or no repair would have been needed. The property that strict is a subset of lenient
    is the one that carries weight, and it has its own test below.
    """
    lenient = extract_json(text, strict=False)
    if lenient.value is not None:
        again = extract_json(json.dumps(lenient.value), strict=True)
        assert again.ok
        assert again.value == lenient.value
        assert again.repairs == ()


@settings(max_examples=100, deadline=None)
@given(_AWKWARD_TEXT)
def test_strict_success_is_a_subset_of_lenient_success(text: str) -> None:
    """The invariant the reported strict-minus-lenient gap rests on.

    The gap is quoted as the size of the repair step a deployment would still need, and that
    reading is only valid if lenient mode accepts everything strict mode does. If it did not,
    the gap could be negative and the sentence would be meaningless.
    """
    strict = extract_json(text, strict=True)
    lenient = extract_json(text, strict=False)
    if strict.ok:
        assert lenient.ok
        assert lenient.value == strict.value
        assert strict.repairs == ()
