"""Tests for schema validation.

The verifier is the reward signal, so its strictness is pinned here case by case rather than
left to whatever pydantic happens to do. Several of these tests exist to hold a deliberate
choice in place: that a quoted number is a type error, that `12.0` is not an integer, that a
percentage above ten is a unit error, and that a partially valid record is never returned.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any, Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.schemas import AdviceRecord, RiskProfile, ViolationKind
from sftdpo.verify.parse import Repair, extract_json
from sftdpo.verify.schema_check import (
    MAX_ONGOING_FEE_PCT,
    MAX_REVIEW_MONTHS,
    MIN_REVIEW_MONTHS,
    check,
)

GOLD: Final[dict[str, Any]] = {
    "client_name": "Ada Lovelace",
    "record_date": "2026-03-31",
    "risk_profile": "balanced",
    "objectives": ["retire at 60"],
    "recommendations": [{"product": "Growth Fund", "action": "buy", "amount": 25000.0}],
    "fees": {"advice_fee": 3300.0, "ongoing_fee_pct": 0.88},
    "review_months": 12,
    "flags": [],
}

_DELETE: Final = object()


def mutate(**patch: Any) -> dict[str, Any]:
    """A copy of the gold record with fields replaced, or removed via `_DELETE`."""
    value = deepcopy(GOLD)
    for key, replacement in patch.items():
        if replacement is _DELETE:
            del value[key]
        else:
            value[key] = replacement
    return value


def fees(**patch: Any) -> dict[str, Any]:
    return mutate(fees={**GOLD["fees"], **patch})


def first_recommendation(**patch: Any) -> dict[str, Any]:
    return mutate(recommendations=[{**GOLD["recommendations"][0], **patch}])


def kinds(value: dict[str, Any]) -> list[ViolationKind]:
    return [violation.kind for violation in check(value)[1]]


def paths(value: dict[str, Any]) -> list[str]:
    return [violation.path for violation in check(value)[1]]


# --------------------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------------------


def test_gold_record_validates() -> None:
    record, violations = check(GOLD)
    assert violations == ()
    assert record is not None
    assert record.risk_profile is RiskProfile.BALANCED
    assert record.recommendations[0].amount == 25000.0


def test_optional_lists_may_be_omitted() -> None:
    value = mutate(objectives=_DELETE, recommendations=_DELETE, flags=_DELETE)
    record, violations = check(value)
    assert violations == ()
    assert record is not None
    assert record.objectives == []


def test_a_record_is_returned_only_when_nothing_is_wrong() -> None:
    """A partially valid record has no meaning for the reward, so it is never handed back."""
    record, violations = check(mutate(review_months=0))
    assert record is None
    assert violations


# --------------------------------------------------------------------------------------
# The violation table: every kind `check` can produce, with a minimal trigger
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "value", "path"),
    [
        (ViolationKind.MISSING_FIELD, mutate(fees=_DELETE), "fees"),
        (ViolationKind.EXTRA_FIELD, mutate(adviser="Bo"), "adviser"),
        (ViolationKind.WRONG_TYPE, mutate(client_name=7), "client_name"),
        (ViolationKind.BAD_ENUM, mutate(risk_profile="aggressive"), "risk_profile"),
        (ViolationKind.BAD_DATE, mutate(record_date="2026-02-30"), "record_date"),
        (ViolationKind.OUT_OF_RANGE, mutate(review_months=0), "review_months"),
    ],
)
def test_violation_table(kind: ViolationKind, value: dict[str, Any], path: str) -> None:
    record, violations = check(value)
    assert record is None
    assert len(violations) == 1
    assert violations[0].kind is kind
    assert violations[0].path == path
    assert violations[0].detail


def test_every_kind_in_the_table_is_distinct() -> None:
    """The six kinds `check` can produce; the other three belong to parsing."""
    produced = {
        ViolationKind.MISSING_FIELD,
        ViolationKind.EXTRA_FIELD,
        ViolationKind.WRONG_TYPE,
        ViolationKind.BAD_ENUM,
        ViolationKind.BAD_DATE,
        ViolationKind.OUT_OF_RANGE,
    }
    parse_level = {
        ViolationKind.NO_JSON,
        ViolationKind.UNPARSEABLE,
        ViolationKind.NOT_AN_OBJECT,
    }
    assert produced | parse_level == set(ViolationKind)


@pytest.mark.parametrize(
    "value",
    [
        {},
        mutate(client_name=None),
        mutate(recommendations="none"),
        mutate(fees="free"),
        mutate(review_months=[1]),
        {"anything": [1, {"a": None}]},
    ],
)
def test_parse_level_kinds_are_never_produced(value: dict[str, Any]) -> None:
    parse_level = {
        ViolationKind.NO_JSON,
        ViolationKind.UNPARSEABLE,
        ViolationKind.NOT_AN_OBJECT,
    }
    assert not parse_level & set(kinds(value))


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def test_list_index_appears_in_the_path() -> None:
    value = mutate(
        recommendations=[
            {"product": "A", "action": "buy"},
            {"product": "B", "action": "sell"},
            {"product": "C", "action": "teleport"},
        ]
    )
    assert paths(value) == ["recommendations[2].action"]


def test_nested_object_path_is_dotted() -> None:
    assert paths(fees(advice_fee=-1.0)) == ["fees.advice_fee"]


def test_root_level_path_is_the_bare_field_name() -> None:
    assert paths(mutate(client_name=7)) == ["client_name"]


def test_nested_extra_field_is_reported_at_its_own_path() -> None:
    value = first_recommendation(rationale="looked good")
    record, violations = check(value)
    assert record is None
    assert violations[0].kind is ViolationKind.EXTRA_FIELD
    assert violations[0].path == "recommendations[0].rationale"


def test_recommendation_item_that_is_not_an_object() -> None:
    assert paths(mutate(recommendations=["buy the growth fund"])) == ["recommendations[0]"]


def test_recommendations_that_are_not_a_list() -> None:
    assert kinds(mutate(recommendations="none")) == [ViolationKind.WRONG_TYPE]


# --------------------------------------------------------------------------------------
# All violations, ordered
# --------------------------------------------------------------------------------------


def test_every_failing_field_is_reported_not_just_the_first() -> None:
    value = mutate(
        client_name=_DELETE,
        risk_profile="aggressive",
        record_date="2026-02-30",
        review_months=200,
        adviser="Bo",
    )
    assert set(paths(value)) == {
        "client_name",
        "risk_profile",
        "record_date",
        "review_months",
        "adviser",
    }


def test_violations_are_ordered_most_severe_first() -> None:
    """`Reward.headline_violation` reads the first entry, so the order carries meaning."""
    value = mutate(client_name=_DELETE, review_months=0)
    assert kinds(value) == [ViolationKind.MISSING_FIELD, ViolationKind.OUT_OF_RANGE]


def test_equally_severe_violations_are_ordered_by_path() -> None:
    value = mutate(client_name=_DELETE, record_date=_DELETE, fees=_DELETE)
    assert paths(value) == ["client_name", "fees", "record_date"]


@pytest.mark.parametrize("months", ["12", 12.5, 12.0, True])
def test_a_path_is_reported_once_even_when_two_rules_object(months: object) -> None:
    """Pydantic and the numeric rules overlap; the model still made one mistake."""
    assert paths(mutate(review_months=months)) == ["review_months"]
    assert kinds(mutate(review_months=months)) == [ViolationKind.WRONG_TYPE]


def test_empty_object_reports_every_required_field() -> None:
    record, violations = check({})
    assert record is None
    assert [violation.kind for violation in violations] == [ViolationKind.MISSING_FIELD] * 5
    assert paths({}) == [
        "client_name",
        "fees",
        "record_date",
        "review_months",
        "risk_profile",
    ]


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "day", ["2026-03-31", "2024-02-29", "2000-02-29", "1999-12-31", "0001-01-01"]
)
def test_real_calendar_dates_are_accepted(day: str) -> None:
    assert check(mutate(record_date=day))[1] == ()


@pytest.mark.parametrize(
    "day",
    [
        "2026-02-30",
        "2026-02-29",
        "2026-13-01",
        "2026-00-10",
        "2026-01-32",
        "2026-1-1",
        "20260101",
        "2026-W01-1",
        "2026-01-01T00:00",
        "31/03/2026",
        "0000-01-01",
        "not a date",
        "",
    ],
)
def test_a_string_that_is_not_a_calendar_date_is_bad_date(day: str) -> None:
    """`2026-02-30` type-checks as a string and is still not a date."""
    assert kinds(mutate(record_date=day)) == [ViolationKind.BAD_DATE]


def test_a_date_of_the_wrong_type_is_a_type_error_not_a_date_error() -> None:
    assert kinds(mutate(record_date=20260101)) == [ViolationKind.WRONG_TYPE]


# --------------------------------------------------------------------------------------
# review_months
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("months", [MIN_REVIEW_MONTHS, 6, 12, 60, MAX_REVIEW_MONTHS])
def test_review_months_inside_the_range(months: int) -> None:
    assert check(mutate(review_months=months))[1] == ()


@pytest.mark.parametrize("months", [0, -1, MAX_REVIEW_MONTHS + 1, 1000])
def test_review_months_outside_the_range(months: int) -> None:
    assert kinds(mutate(review_months=months)) == [ViolationKind.OUT_OF_RANGE]


@pytest.mark.parametrize("months", ["12", 12.0, 12.5, True, [12], None])
def test_review_months_of_the_wrong_type(months: object) -> None:
    """`"12"` is a quoted number and `12.0` is not an integer; both missed the schema.

    Pydantic in lax mode would accept both. The rule is deliberately "an integer field wants
    a JSON integer", so it does not turn on whether the fraction happens to be zero.
    """
    assert kinds(mutate(review_months=months)) == [ViolationKind.WRONG_TYPE]


# --------------------------------------------------------------------------------------
# Fees
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fee", [0, 0.0, 1.0, 3300.0, 100, 1_000_000.0])
def test_advice_fee_is_accepted_when_non_negative(fee: float) -> None:
    assert check(fees(advice_fee=fee))[1] == ()


@pytest.mark.parametrize("fee", [-0.01, -1, -1000.0])
def test_advice_fee_may_not_be_negative(fee: float) -> None:
    assert kinds(fees(advice_fee=fee)) == [ViolationKind.OUT_OF_RANGE]


@pytest.mark.parametrize("pct", [0, 0.0, 0.88, 5.5, MAX_ONGOING_FEE_PCT])
def test_ongoing_fee_pct_inside_the_percentage_range(pct: float) -> None:
    assert check(fees(ongoing_fee_pct=pct))[1] == ()


@pytest.mark.parametrize("pct", [-0.1, 10.01, 85.0, 100.0])
def test_ongoing_fee_pct_outside_the_percentage_range(pct: float) -> None:
    """An unbounded percentage would let the costliest mistake in this field score as valid.

    A model that writes 88 for a 0.88 % ongoing fee has produced a number a downstream
    system would act on. Ten per cent sits above any ongoing advice fee in the Australian
    market, so a larger value is a unit error rather than a large fee.
    """
    assert kinds(fees(ongoing_fee_pct=pct)) == [ViolationKind.OUT_OF_RANGE]


@pytest.mark.parametrize("fee", ["3300", "3300.0", True, [1], {"amount": 1}])
def test_a_quoted_or_boolean_fee_is_a_type_error(fee: object) -> None:
    assert kinds(fees(advice_fee=fee)) == [ViolationKind.WRONG_TYPE]


@pytest.mark.parametrize("fee", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_fee_is_out_of_range(fee: float) -> None:
    """`json.loads` accepts NaN and Infinity, and every range test against NaN is False.

    Without an explicit finiteness check a NaN fee would pass both bounds and validate.
    """
    assert not math.isfinite(fee)
    assert kinds(fees(advice_fee=fee)) == [ViolationKind.OUT_OF_RANGE]


def test_a_missing_fee_is_a_missing_field_at_the_nested_path() -> None:
    record, violations = check(mutate(fees={"advice_fee": 1.0}))
    assert record is None
    assert violations[0].kind is ViolationKind.MISSING_FIELD
    assert violations[0].path == "fees.ongoing_fee_pct"


def test_a_null_fee_is_a_type_error_reported_once() -> None:
    assert paths(fees(advice_fee=None)) == ["fees.advice_fee"]


# --------------------------------------------------------------------------------------
# Recommendation amounts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("amount", [0, 0.0, 25000.0, 1_000_000, -500.0])
def test_amount_is_deliberately_unbounded(amount: float) -> None:
    """No sign convention for a switch is fixed by the task, so none is enforced here."""
    assert check(first_recommendation(amount=amount))[1] == ()


def test_amount_may_be_null_or_absent() -> None:
    assert check(first_recommendation(amount=None))[1] == ()
    assert check(mutate(recommendations=[{"product": "A", "action": "hold"}]))[1] == ()


@pytest.mark.parametrize("amount", ["25000", True])
def test_a_quoted_or_boolean_amount_is_a_type_error(amount: object) -> None:
    assert kinds(first_recommendation(amount=amount)) == [ViolationKind.WRONG_TYPE]


@pytest.mark.parametrize("amount", [float("nan"), float("inf")])
def test_a_non_finite_amount_is_out_of_range(amount: float) -> None:
    assert kinds(first_recommendation(amount=amount)) == [ViolationKind.OUT_OF_RANGE]


def test_amount_violation_carries_the_indexed_path() -> None:
    value = mutate(
        recommendations=[
            {"product": "A", "action": "buy", "amount": 1.0},
            {"product": "B", "action": "sell", "amount": "2"},
        ]
    )
    assert paths(value) == ["recommendations[1].amount"]


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("risk", [profile.value for profile in RiskProfile])
def test_every_risk_profile_member_is_accepted(risk: str) -> None:
    assert check(mutate(risk_profile=risk))[1] == ()


@pytest.mark.parametrize("risk", ["aggressive", "Balanced", "BALANCED", "", 3])
def test_a_value_outside_the_enum_is_bad_enum(risk: object) -> None:
    assert kinds(mutate(risk_profile=risk)) == [ViolationKind.BAD_ENUM]


def test_enum_detail_names_the_permitted_values() -> None:
    violations = check(mutate(risk_profile="aggressive"))[1]
    assert "conservative" in violations[0].detail


# --------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------

_JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**9), max_value=10**9)
    | st.floats()
    | st.text(max_size=5),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=5), children, max_size=3)
    ),
    max_leaves=8,
)
_JSON_DICTS = st.dictionaries(st.text(max_size=6), _JSON_VALUES, max_size=5)

_VALID_RECORDS = st.builds(
    lambda name, day, risk, months, fee, pct: {
        "client_name": name,
        "record_date": day.isoformat(),
        "risk_profile": risk,
        "objectives": [],
        "recommendations": [],
        "fees": {"advice_fee": fee, "ongoing_fee_pct": pct},
        "review_months": months,
        "flags": [],
    },
    name=st.text(min_size=1, max_size=12),
    day=st.dates(),
    risk=st.sampled_from([profile.value for profile in RiskProfile]),
    months=st.integers(min_value=MIN_REVIEW_MONTHS, max_value=MAX_REVIEW_MONTHS),
    fee=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    pct=st.floats(
        min_value=0, max_value=MAX_ONGOING_FEE_PCT, allow_nan=False, allow_infinity=False
    ),
)


@settings(max_examples=100, deadline=None)
@given(_JSON_DICTS)
def test_check_never_raises(value: dict[str, Any]) -> None:
    """Parsed model output is arbitrary; the verifier may never crash on it."""
    record, violations = check(value)
    assert (record is None) == bool(violations)


@settings(max_examples=100, deadline=None)
@given(_JSON_DICTS)
def test_every_violation_names_a_kind_and_a_path(value: dict[str, Any]) -> None:
    for violation in check(value)[1]:
        assert violation.kind in set(ViolationKind)
        assert violation.path or violation.detail


@settings(max_examples=50, deadline=None)
@given(_VALID_RECORDS)
def test_a_valid_record_always_validates(value: dict[str, Any]) -> None:
    record, violations = check(value)
    assert violations == ()
    assert record is not None


@settings(max_examples=50, deadline=None)
@given(_VALID_RECORDS)
def test_an_accepted_record_survives_a_json_round_trip(value: dict[str, Any]) -> None:
    """The verifier's answer cannot depend on the record having come from Python."""
    record, _ = check(json.loads(json.dumps(value)))
    assert record is not None
    assert record == AdviceRecord.model_validate(value)


@settings(max_examples=100, deadline=None)
@given(_JSON_DICTS)
def test_acceptance_implies_pydantic_would_accept(value: dict[str, Any]) -> None:
    """`check` is stricter than the model, never looser."""
    record, _ = check(value)
    if record is not None:
        assert AdviceRecord.model_validate(value) == record


# --------------------------------------------------------------------------------------
# The two halves together
# --------------------------------------------------------------------------------------


def test_a_messy_completion_is_recovered_and_then_validates() -> None:
    """The whole point of lenient mode: the schema was right, the packaging was not."""
    dumped = json.dumps(GOLD)
    completion = f"Here's the record:\n```json\n{dumped[:-1]},}}\n```\nLet me know."

    assert not extract_json(completion, strict=True).ok

    outcome = extract_json(completion, strict=False)
    assert outcome.repairs == (Repair.CODE_FENCE, Repair.TRAILING_COMMA)
    assert outcome.value is not None
    record, violations = check(outcome.value)
    assert violations == ()
    assert record is not None
    assert record.client_name == GOLD["client_name"]


def test_a_recovered_object_can_still_fail_the_schema() -> None:
    """Recovering the packaging does not forgive the content."""
    outcome = extract_json("Here you go: {'client_name': 'Ada'}", strict=False)
    assert outcome.ok
    assert outcome.value is not None
    record, violations = check(outcome.value)
    assert record is None
    assert all(violation.kind is ViolationKind.MISSING_FIELD for violation in violations)
