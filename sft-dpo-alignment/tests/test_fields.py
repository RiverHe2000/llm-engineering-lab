"""Tests for the field comparison.

Most of these exist to hold a normalisation decision in place, because each one changes what
the trained model is rewarded for: that a reordered recommendation list is one mistake and not
six, that `"Balanced"` is still the wrong enum member even though the string comparison
elsewhere folds case, that a null amount claims nothing, and that an invented field costs
precision even when every real field is right.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Any, Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.schemas import (
    Action,
    AdviceRecord,
    Fees,
    FieldScore,
    Recommendation,
    RiskProfile,
)
from sftdpo.task.dataset import build_dataset
from sftdpo.verify.fields import (
    ABSENT,
    FieldKind,
    f1_from_scores,
    field_f1,
    field_kind,
    field_scores,
    field_stem,
    flatten,
)

GOLD_JSON: Final[dict[str, Any]] = {
    "client_name": "Ada Lovelace",
    "record_date": "2026-03-31",
    "risk_profile": "balanced",
    "objectives": ["retire at 60", "fund school fees"],
    "recommendations": [
        {"product": "Growth Fund", "action": "buy", "amount": 25000.0},
        {"product": "Cash Option", "action": "sell", "amount": 5000.0},
        {"product": "Bond Series", "action": "hold", "amount": None},
    ],
    "fees": {"advice_fee": 3300.0, "ongoing_fee_pct": 0.88},
    "review_months": 12,
    "flags": ["tfn missing"],
}

GOLD: Final = AdviceRecord.model_validate(GOLD_JSON)

GOLD_PATHS: Final = frozenset(
    {
        "client_name",
        "record_date",
        "risk_profile",
        "objectives{retire at 60}",
        "objectives{fund school fees}",
        "recommendations[0].product",
        "recommendations[0].action",
        "recommendations[0].amount",
        "recommendations[1].product",
        "recommendations[1].action",
        "recommendations[1].amount",
        "recommendations[2].product",
        "recommendations[2].action",
        "fees.advice_fee",
        "fees.ongoing_fee_pct",
        "review_months",
        "flags{tfn missing}",
    }
)

N_GOLD: Final = len(GOLD_PATHS)


def mutate(**patch: Any) -> dict[str, Any]:
    """A copy of the gold object with top-level fields replaced."""
    value = deepcopy(GOLD_JSON)
    value.update(patch)
    return value


def fees(**patch: Any) -> dict[str, Any]:
    return mutate(fees={**GOLD_JSON["fees"], **patch})


def recommendation(index: int, **patch: Any) -> dict[str, Any]:
    items = deepcopy(GOLD_JSON["recommendations"])
    items[index] = {**items[index], **patch}
    return mutate(recommendations=items)


def incorrect(candidate: dict[str, Any]) -> list[str]:
    """The paths that did not match, in report order."""
    return [score.path for score in field_scores(candidate, GOLD) if not score.correct]


def score_at(candidate: dict[str, Any], path: str) -> FieldScore:
    matches = [score for score in field_scores(candidate, GOLD) if score.path == path]
    assert len(matches) == 1, f"{path} appears {len(matches)} times"
    return matches[0]


def f1(candidate: dict[str, Any]) -> float:
    return field_f1(candidate, GOLD)


# --------------------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------------------


def test_flatten_produces_the_documented_paths() -> None:
    assert set(flatten(GOLD)) == GOLD_PATHS


def test_flattening_a_record_and_its_json_are_the_same_operation() -> None:
    """The comparison must not be able to tell where the object came from."""
    assert flatten(GOLD) == flatten(GOLD_JSON)


def test_a_null_amount_carries_no_claim() -> None:
    """`amount` is optional, so a null is the schema declining to state one, not a value."""
    assert "recommendations[2].amount" not in flatten(GOLD)
    assert "recommendations[0].amount" in flatten(GOLD)


def test_a_null_elsewhere_is_a_value_the_model_chose() -> None:
    assert flatten(mutate(client_name=None))["client_name"] is None


def test_set_fields_are_keyed_by_normalised_member() -> None:
    flat = flatten(mutate(flags=["  TFN   Missing "]))
    assert "flags{tfn missing}" in flat
    assert flat["flags{tfn missing}"] == "  TFN   Missing "


def test_a_repeated_objective_collapses_to_one_path() -> None:
    """Saying it twice is not two objectives."""
    flat = flatten(mutate(objectives=["retire at 60", "RETIRE AT 60", "fund school fees"]))
    assert sum(path.startswith("objectives{") for path in flat) == 2


def test_flatten_keeps_an_invented_top_level_key() -> None:
    assert flatten(mutate(adviser="Bo"))["adviser"] == "Bo"


def test_flatten_keeps_an_invented_nested_key() -> None:
    flat = flatten(recommendation(0, rationale="looked good"))
    assert flat["recommendations[0].rationale"] == "looked good"
    assert flatten(fees(gst=330.0))["fees.gst"] == 330.0


@pytest.mark.parametrize(
    ("patch", "path"),
    [
        ({"objectives": "retire at 60"}, "objectives"),
        ({"recommendations": "none"}, "recommendations"),
        ({"fees": "free"}, "fees"),
        ({"recommendations": ["buy the growth fund"]}, "recommendations[0]"),
        ({"flags": None}, "flags"),
    ],
)
def test_flatten_is_defensive_about_shape(patch: dict[str, Any], path: str) -> None:
    """The input is model output, so a field of the wrong shape becomes one path, not a crash."""
    assert path in flatten(mutate(**patch))


# --------------------------------------------------------------------------------------
# The comparison contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("client_name", FieldKind.STRING),
        ("record_date", FieldKind.DATE),
        ("risk_profile", FieldKind.ENUM),
        ("objectives{retire at 60}", FieldKind.STRING),
        ("flags{tfn missing}", FieldKind.STRING),
        ("recommendations[0].product", FieldKind.STRING),
        ("recommendations[11].action", FieldKind.ENUM),
        ("recommendations[7].amount", FieldKind.MONEY),
        ("fees.advice_fee", FieldKind.MONEY),
        ("fees.ongoing_fee_pct", FieldKind.PERCENT),
        ("review_months", FieldKind.INTEGER),
    ],
)
def test_the_kind_of_a_path_is_a_property_of_the_path(path: str, kind: FieldKind) -> None:
    assert field_kind(path) is kind


@pytest.mark.parametrize(
    "path", ["adviser", "fees.gst", "recommendations[0].rationale", "", "objectives"]
)
def test_a_path_the_schema_never_asked_for_is_opaque(path: str) -> None:
    assert field_kind(path) is FieldKind.OPAQUE


def test_every_gold_path_has_a_declared_kind() -> None:
    """Nothing the schema asks for may fall through to the opaque comparison."""
    assert all(field_kind(path) is not FieldKind.OPAQUE for path in GOLD_PATHS)


# --------------------------------------------------------------------------------------
# Normalisation, kind by kind
# --------------------------------------------------------------------------------------


def test_gold_against_itself_is_perfect() -> None:
    assert incorrect(GOLD_JSON) == []
    assert f1(GOLD_JSON) == 1.0


@pytest.mark.parametrize("fee", [3300.0, 3300, 3300.004, 3300.0000001, 3299.9999])
def test_money_is_compared_to_the_cent(fee: float) -> None:
    assert incorrect(fees(advice_fee=fee)) == []


@pytest.mark.parametrize("fee", [3300.01, 3299.99, 3300.005, 33000.0, 330.0, 0.0])
def test_a_difference_of_a_cent_is_a_difference(fee: float) -> None:
    assert incorrect(fees(advice_fee=fee)) == ["fees.advice_fee"]


def test_binary_float_noise_is_not_a_modelling_error() -> None:
    """`0.1 + 0.2 != 0.3` is arithmetic the model has no way to avoid."""
    gold = AdviceRecord.model_validate(fees(advice_fee=0.3))
    assert field_f1(fees(advice_fee=0.1 + 0.2), gold) == 1.0


@pytest.mark.parametrize("pct", [0.88, 0.880004, 0.8799999])
def test_a_percentage_is_compared_to_two_decimals(pct: float) -> None:
    assert incorrect(fees(ongoing_fee_pct=pct)) == []


@pytest.mark.parametrize("pct", [0.885, 0.89, 0.87, 88.0])
def test_a_percentage_that_differs_in_the_second_decimal(pct: float) -> None:
    assert incorrect(fees(ongoing_fee_pct=pct)) == ["fees.ongoing_fee_pct"]


@pytest.mark.parametrize("fee", ["3300.0", "3300", True, None, [3300.0], float("nan")])
def test_a_number_that_is_not_a_number_is_wrong(fee: object) -> None:
    assert incorrect(fees(advice_fee=fee)) == ["fees.advice_fee"]


@pytest.mark.parametrize(
    "name",
    ["Ada Lovelace", "ada lovelace", "ADA LOVELACE", "  Ada   Lovelace  ", "Ada\tLovelace"],
)
def test_case_and_spacing_are_not_modelling_errors(name: str) -> None:
    assert incorrect(mutate(client_name=name)) == []


def test_compatibility_characters_are_folded() -> None:
    """A full-width letter is the same client name, however the tokeniser spelt it."""
    assert incorrect(mutate(client_name="Ａda Lovelace")) == []


@pytest.mark.parametrize(
    "name", ["Ada Lovelace.", "AdaLovelace", "Ada-Lovelace", "Ada", "Ada Byron", ""]
)
def test_punctuation_is_deliberately_not_folded(name: str) -> None:
    """Eating the hyphen would stop the reward telling "Fund A" from "Fund-A"."""
    assert incorrect(mutate(client_name=name)) == ["client_name"]


@pytest.mark.parametrize("risk", ["Balanced", "BALANCED", " balanced", "growth"])
def test_an_enum_is_compared_exactly(risk: str) -> None:
    """The schema layer already charged for the casing; forgiving it here would refund it."""
    assert incorrect(mutate(risk_profile=risk)) == ["risk_profile"]


def test_the_action_enum_is_compared_exactly() -> None:
    assert incorrect(recommendation(0, action="BUY")) == ["recommendations[0].action"]
    assert incorrect(recommendation(0, action="buy")) == []


@pytest.mark.parametrize("day", ["2026-03-31", "20260331", "2026-3-31"])
def test_a_date_is_compared_as_the_day_it_denotes(day: str) -> None:
    assert incorrect(mutate(record_date=day)) == []


def test_an_iso_week_date_names_the_same_day() -> None:
    iso = date(2026, 3, 31).isocalendar()
    assert incorrect(mutate(record_date=f"{iso.year}-W{iso.week:02d}-{iso.weekday}")) == []


@pytest.mark.parametrize(
    "day", ["2026-04-01", "2026-03-30", "31/03/2026", "2026-03-31T00:00", "31 March 2026", ""]
)
def test_an_ambiguous_or_different_day_is_wrong(day: str) -> None:
    """`31/03/2026` is read differently on either side of the Pacific, so it is not parsed."""
    assert incorrect(mutate(record_date=day)) == ["record_date"]


@pytest.mark.parametrize("months", [12.0, "12", True, None, 13])
def test_an_integer_field_wants_a_json_integer(months: object) -> None:
    assert incorrect(mutate(review_months=months)) == ["review_months"]


def test_two_values_that_fail_to_normalise_can_still_match_each_other() -> None:
    """The opaque fallback keeps the comparison total, and can never match a real value."""
    assert field_f1({"review_months": "12"}, {"review_months": "12"}) == 1.0
    assert field_f1({"review_months": "12"}, {"review_months": 12}) == 0.0


def test_two_records_that_invented_the_same_field_agree_on_it() -> None:
    """An unknown path has to be comparable, or the report would crash on model output."""
    assert field_f1({"adviser": "Bo"}, {"adviser": "Bo"}) == 1.0
    assert field_f1({"adviser": "Bo"}, {"adviser": "Jo"}) == 0.0


def test_a_magnitude_too_large_to_round_falls_back_to_exact_comparison() -> None:
    """`Decimal` will not quantise 1e30 to the cent, and a fee that size is not a fee."""
    assert field_f1({"fees": {"advice_fee": 1e30}}, {"fees": {"advice_fee": 1e30}}) == 1.0
    assert field_f1({"fees": {"advice_fee": 1e30}}, {"fees": {"advice_fee": 3300.0}}) == 0.0


def test_a_date_that_is_not_a_string_is_wrong() -> None:
    assert incorrect(mutate(record_date=20260331)) == ["record_date"]


@pytest.mark.parametrize("day", ["2026-2-30", "2026-13-1", "2026-0-1", "2026-1-99"])
def test_a_date_shaped_string_that_names_no_day_is_wrong(day: str) -> None:
    assert incorrect(mutate(record_date=day)) == ["record_date"]


# --------------------------------------------------------------------------------------
# Sets: objectives and flags
# --------------------------------------------------------------------------------------


def test_reordering_objectives_costs_nothing() -> None:
    assert f1(mutate(objectives=list(reversed(GOLD_JSON["objectives"])))) == 1.0


def test_objective_membership_is_normalised() -> None:
    assert f1(mutate(objectives=["  RETIRE   AT 60", "Fund School Fees"])) == 1.0


def test_repeating_an_objective_is_not_a_second_objective() -> None:
    assert f1(mutate(objectives=["retire at 60", "retire at 60", "fund school fees"])) == 1.0


def test_a_missing_objective_costs_recall_only() -> None:
    candidate = mutate(objectives=["retire at 60"])
    score = score_at(candidate, "objectives{fund school fees}")
    assert not score.correct
    assert score.actual == ABSENT
    assert score.expected == '"fund school fees"'
    assert f1(candidate) == 2 * (N_GOLD - 1) / ((N_GOLD - 1) + N_GOLD)


def test_an_invented_objective_costs_precision_only() -> None:
    candidate = mutate(objectives=[*GOLD_JSON["objectives"], "buy a boat"])
    score = score_at(candidate, "objectives{buy a boat}")
    assert not score.correct
    assert score.expected == ABSENT
    assert f1(candidate) == 2 * N_GOLD / ((N_GOLD + 1) + N_GOLD)


def test_dropping_the_only_flag_is_one_error() -> None:
    assert incorrect(mutate(flags=[])) == ["flags{tfn missing}"]


# --------------------------------------------------------------------------------------
# Recommendations: an ordered list, matched on product name
# --------------------------------------------------------------------------------------


def test_reordering_recommendations_is_one_mistake_not_six() -> None:
    """The headline case for the greedy match: the same advice, listed the other way up."""
    assert incorrect(mutate(recommendations=list(reversed(GOLD_JSON["recommendations"])))) == []


def test_a_wrong_action_in_a_reordered_list_is_charged_at_its_gold_slot() -> None:
    items = list(reversed(deepcopy(GOLD_JSON["recommendations"])))
    items[1]["action"] = "buy"
    assert incorrect(mutate(recommendations=items)) == ["recommendations[1].action"]


def test_the_product_name_is_matched_after_normalisation() -> None:
    assert incorrect(recommendation(0, product="  growth   FUND ")) == []


def test_an_unmatched_product_is_still_compared_positionally() -> None:
    """A renamed product must not drag its correct action and amount down with it."""
    assert incorrect(recommendation(0, product="Growth Fund II")) == ["recommendations[0].product"]


def test_a_surplus_recommendation_lands_past_the_end_of_gold() -> None:
    extra = {"product": "Mystery Trust", "action": "buy", "amount": 1.0}
    candidate = mutate(recommendations=[*GOLD_JSON["recommendations"], extra])
    score = score_at(candidate, "recommendations[3].product")
    assert score.expected == ABSENT
    assert f1(candidate) == 2 * N_GOLD / ((N_GOLD + 3) + N_GOLD)


def test_a_missing_recommendation_costs_every_path_it_held() -> None:
    candidate = mutate(recommendations=GOLD_JSON["recommendations"][:2])
    assert incorrect(candidate) == ["recommendations[2].action", "recommendations[2].product"]
    assert f1(candidate) == 2 * (N_GOLD - 2) / ((N_GOLD - 2) + N_GOLD)


def test_two_recommendations_with_the_same_product_are_paired_in_order() -> None:
    """With identical names, position is the only thing left to match on, and order matters."""
    twice = [
        {"product": "Growth Fund", "action": "buy", "amount": 1.0},
        {"product": "Growth Fund", "action": "sell", "amount": 2.0},
    ]
    gold = AdviceRecord.model_validate(mutate(recommendations=twice))
    swapped = mutate(recommendations=list(reversed(twice)))
    wrong = [score.path for score in field_scores(swapped, gold) if not score.correct]
    assert wrong == [
        "recommendations[0].action",
        "recommendations[0].amount",
        "recommendations[1].action",
        "recommendations[1].amount",
    ]


def test_a_recommendation_with_no_product_name_takes_no_part_in_the_matching() -> None:
    """Nothing to match on, so it fills whichever slot the named items left over."""
    gold: dict[str, Any] = {
        "recommendations": [{"action": "buy"}, {"product": "Growth Fund", "action": "sell"}]
    }
    swapped: dict[str, Any] = {
        "recommendations": [{"product": "Growth Fund", "action": "sell"}, {"action": "buy"}]
    }
    assert field_f1(swapped, gold) == 1.0

    wrong: dict[str, Any] = {
        "recommendations": [{"product": "Growth Fund", "action": "sell"}, {"action": "hold"}]
    }
    assert [score.path for score in field_scores(wrong, gold) if not score.correct] == [
        "recommendations[0].action"
    ]


def test_an_item_that_is_not_an_object_is_scored_at_its_index() -> None:
    candidate = mutate(recommendations=["buy the growth fund"])
    assert score_at(candidate, "recommendations[0]").expected == ABSENT


# --------------------------------------------------------------------------------------
# The union, the report and the ordering
# --------------------------------------------------------------------------------------


def test_a_missing_field_is_reported_with_an_absent_actual() -> None:
    candidate = {key: value for key, value in GOLD_JSON.items() if key != "client_name"}
    score = score_at(candidate, "client_name")
    assert score.expected == '"Ada Lovelace"'
    assert score.actual == ABSENT
    assert not score.correct


def test_an_invented_field_is_reported_with_an_absent_expected() -> None:
    score = score_at(mutate(adviser="Bo"), "adviser")
    assert score.expected == ABSENT
    assert score.actual == '"Bo"'
    assert not score.correct


def test_the_absent_marker_cannot_be_confused_with_a_value() -> None:
    """A model that literally wrote `<absent>` has filled the field in, wrongly."""
    score = score_at(mutate(client_name=ABSENT), "client_name")
    assert score.actual != ABSENT
    assert score.actual == '"<absent>"'
    assert f1(mutate(client_name=ABSENT)) == 2 * (N_GOLD - 1) / (N_GOLD + N_GOLD)


def test_paths_are_ordered_by_the_schema_then_by_index() -> None:
    many = mutate(
        recommendations=[
            {"product": f"Fund {i}", "action": "buy", "amount": float(i)} for i in range(12)
        ]
    )
    record = AdviceRecord.model_validate(many)
    paths = [score.path for score in field_scores(record, record)]
    assert paths.index("client_name") < paths.index("record_date") < paths.index("risk_profile")
    assert paths.index("risk_profile") < paths.index("recommendations[0].product")
    assert paths.index("recommendations[2].product") < paths.index("recommendations[10].product")
    assert paths.index("fees.advice_fee") < paths.index("review_months")
    assert paths.index("review_months") < paths.index("flags{tfn missing}")


def test_an_invented_field_sorts_after_everything_the_schema_asked_for() -> None:
    paths = [score.path for score in field_scores(mutate(adviser="Bo"), GOLD)]
    assert paths[-1] == "adviser"


def test_the_report_covers_the_union_of_both_records() -> None:
    candidate = mutate(adviser="Bo")
    paths = {score.path for score in field_scores(candidate, GOLD)}
    assert paths == GOLD_PATHS | {"adviser"}


# --------------------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------------------


def test_a_completion_that_did_not_parse_scores_zero() -> None:
    scores = field_scores(None, GOLD)
    assert len(scores) == N_GOLD
    assert all(score.actual == ABSENT for score in scores)
    assert field_f1(None, GOLD) == 0.0


def test_an_empty_object_scores_zero() -> None:
    assert field_f1({}, GOLD) == 0.0


def test_two_empty_records_are_vacuously_perfect() -> None:
    """Unreachable for a real record, which always carries its required fields."""
    assert field_f1({}, {}) == 1.0
    assert f1_from_scores(()) == 1.0


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
_JSON_DICTS = st.dictionaries(st.text(max_size=6), _JSON_VALUES, max_size=6)

# One damage operation per top-level field, so a set of them composes without interfering.
# Each either corrupts a value or removes a correct one and invents a wrong one, which is what
# makes "fewer damages" mean "more correct fields and no new errors".
DAMAGE: Final[dict[str, Any]] = {
    "client_name": "Someone Else",
    "record_date": "2020-01-01",
    "risk_profile": "growth",
    "objectives": ["retire at 60", "buy a boat"],
    "recommendations": [
        {"product": "Growth Fund", "action": "sell", "amount": 25000.0},
        {"product": "Mystery Trust", "action": "buy", "amount": 1.0},
    ],
    "fees": {"advice_fee": 1.0, "ongoing_fee_pct": 2.0},
    "review_months": 24,
    "flags": [],
}
DAMAGE_NAMES: Final = sorted(DAMAGE)


def damaged(names: frozenset[str]) -> dict[str, Any]:
    return mutate(**{name: DAMAGE[name] for name in sorted(names)})


@settings(max_examples=100, deadline=None)
@given(_JSON_DICTS)
def test_flatten_never_raises(value: dict[str, Any]) -> None:
    """The input is parsed model output, which is arbitrary."""
    assert all(isinstance(path, str) for path in flatten(value))


@settings(max_examples=100, deadline=None)
@given(_JSON_DICTS)
def test_scoring_arbitrary_output_stays_in_range(value: dict[str, Any]) -> None:
    scores = field_scores(value, GOLD)
    paths = [score.path for score in scores]
    assert len(set(paths)) == len(paths), "a path may only be judged once"
    assert all(score.expected != ABSENT or score.actual != ABSENT for score in scores)
    assert 0.0 <= field_f1(value, GOLD) <= 1.0


@settings(max_examples=100, deadline=None)
@given(st.sets(st.sampled_from(DAMAGE_NAMES)))
def test_the_score_is_one_exactly_when_every_field_agrees(names: set[str]) -> None:
    candidate = damaged(frozenset(names))
    scores = field_scores(candidate, GOLD)
    assert (field_f1(candidate, GOLD) == 1.0) == all(score.correct for score in scores)


@settings(max_examples=200, deadline=None)
@given(st.sets(st.sampled_from(DAMAGE_NAMES)), st.sets(st.sampled_from(DAMAGE_NAMES)))
def test_repairing_a_field_never_lowers_the_score(kept: set[str], extra: set[str]) -> None:
    """Monotone in correctness: the reward may not teach the model to un-learn a field.

    The damaged sets are nested, so the smaller one is correct wherever the larger one is and
    wrong nowhere the larger one is right — which is the precondition the guarantee is stated
    under. It is not a claim that adding an arbitrary field cannot lower the score; adding a
    wrong one must.
    """
    repaired = field_f1(damaged(frozenset(kept)), GOLD)
    worse = field_f1(damaged(frozenset(kept | extra)), GOLD)
    assert repaired >= worse


@settings(max_examples=50, deadline=None)
@given(st.sets(st.sampled_from(DAMAGE_NAMES)))
def test_deleting_an_invented_field_never_lowers_the_score(names: set[str]) -> None:
    candidate = damaged(frozenset(names))
    with_junk = {**candidate, "adviser": "Bo", "notes": "see file"}
    assert field_f1(candidate, GOLD) >= field_f1(with_junk, GOLD)


@settings(max_examples=50, deadline=None)
@given(st.permutations(GOLD_JSON["recommendations"]), st.permutations(GOLD_JSON["objectives"]))
def test_order_carries_no_meaning_where_the_comparison_says_it_does_not(
    items: list[dict[str, Any]], objectives: list[str]
) -> None:
    assert f1(mutate(recommendations=items, objectives=objectives)) == 1.0


def test_duplicate_gold_products_can_invert_a_repair() -> None:
    """The one documented break in monotonicity, pinned rather than hidden.

    Recommendations are aligned to gold slots by product name, so two gold items sharing a
    name make the alignment ambiguous: repairing one candidate's product name towards gold
    re-aligns both items and can lose more matched paths than the repair gains. The corpus
    never generates a duplicate product name -- `test_the_corpus_never_produces_duplicate_
    gold_products` checks that -- so the reward is monotone on every example this project
    scores. This test exists so that the day the generator changes, it fails here rather than
    silently putting the more-correct completion on the rejected side of a preference pair.
    """
    gold = AdviceRecord(
        client_name="A",
        record_date="2026-01-01",
        risk_profile=RiskProfile.BALANCED,
        objectives=[],
        recommendations=[
            Recommendation(product="Growth Fund", action=Action.BUY, amount=10000.0),
            Recommendation(product="Growth Fund", action=Action.SELL, amount=5000.0),
        ],
        fees=Fees(advice_fee=1000.0, ongoing_fee_pct=0.5),
        review_months=12,
        flags=[],
    )
    both_wrong = gold.model_copy(
        update={
            "recommendations": [
                gold.recommendations[0].model_copy(update={"product": "Cash Option"}),
                gold.recommendations[1].model_copy(update={"product": "Cash Option"}),
            ]
        }
    )
    one_repaired = both_wrong.model_copy(
        update={
            "recommendations": [
                both_wrong.recommendations[0],
                both_wrong.recommendations[1].model_copy(update={"product": "Growth Fund"}),
            ]
        }
    )

    assert field_f1(one_repaired, gold) < field_f1(both_wrong, gold)


def test_the_corpus_never_produces_duplicate_gold_products() -> None:
    """Why the break above is unreachable in practice, checked rather than assumed."""
    dataset = build_dataset(seed=1, n_train=60, n_val=12, n_test=24)
    for split in ("train", "val", "test"):
        for example in getattr(dataset, split):
            products = [rec.product for rec in example.gold.recommendations]
            assert len(products) == len(set(products)), example.example_id


# --------------------------------------------------------------------------------------
# Path stems
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("client_name", "client_name"),
        ("fees.advice_fee", "fees"),
        ("recommendations[0].amount", "recommendations"),
        ("flags{preservation_age_not_reached}", "flags"),
        # The earliest separator wins, so a dot inside a set member does not split it.
        ("objectives{review in 2028. then again}", "objectives"),
        # As does a bracket inside one.
        ("objectives{buy a house [eventually]}", "objectives"),
        ("", ""),
    ],
)
def test_field_stem_reads_the_top_level_field_off_the_path(path: str, expected: str) -> None:
    assert field_stem(path) == expected


def test_every_scored_path_stems_to_a_field_the_schema_declares() -> None:
    """The gate groups by stem, so a stem the schema does not have would be an invented row."""
    stems = {field_stem(score.path) for score in field_scores(None, GOLD)}
    assert stems <= set(AdviceRecord.model_fields)
    assert stems


def test_an_invented_key_stems_to_itself_rather_than_being_hidden() -> None:
    """A field the model made up must show up as its own row, not be folded into a real one."""
    candidate = mutate(adviser={"name": "Bo"})
    stems = {field_stem(score.path) for score in field_scores(candidate, GOLD)}
    assert "adviser" in stems
