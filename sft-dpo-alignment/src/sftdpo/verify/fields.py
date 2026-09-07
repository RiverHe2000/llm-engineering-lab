"""Comparing a candidate record with gold, field by field, under a normalisation this task
can defend.

The reward this package trains against is only as good as its notion of "the same". Every
choice below either forgives a difference that is not a modelling error, or charges for one
that is, and each is a decision that changes what the trained model learns to do:

*Money to the cent, percentages to two decimals.* An advice fee is a dollar figure and an
ongoing fee is quoted to the basis point; comparing either to full binary-float precision
would score `0.1 + 0.2` against `0.3` as an error the model has no way to avoid. Rounding is
done through `Decimal(str(value))` so the comparison depends on the number the model wrote
rather than on the shortest float that happens to hold it.

*Strings case-folded and whitespace-collapsed, and nothing else.* `"  JANE   SMITH "` and
`"Jane Smith"` are the same client, and a source note is often shouting. Punctuation is left
alone deliberately: "Fund A" and "Fund-A" are different products, and a normaliser that ate
the hyphen would hand the trainer a reward that cannot tell them apart.

*Enums exact.* The enum layer has already been enforced: `schema_check` reports `"Balanced"`
as `BAD_ENUM`. Folding the case here would forgive at the field layer what the schema layer
charged for, which is the mirror image of the double-charging this module avoids elsewhere.

*Dates as calendar dates.* `2026-03-31` and `20260331` denote the same day, and the schema
layer has already charged for the spelling. Only unambiguous spellings are accepted: an
Australian model writing `31/03/2026` and an American one writing `03/31/2026` cannot be
told apart, so neither is parsed and both count as wrong.

*`objectives` and `flags` as SETS, keyed by member.* Their order carries no meaning, so a
reordering must cost nothing. Comparing each list as a single value would cost everything —
three objectives out of four right would score zero, and the reward would be flat across a
range of behaviour the training needs to distinguish. Keying each member as its own path
gives graded credit and makes a missing member cost recall and an invented one cost
precision. A repeated objective collapses to one path: saying it twice is not two objectives.

*`recommendations` as an ORDERED list, matched greedily on product name.* Order does carry
meaning here — a statement of advice reads top down — so the paths stay indexed. But a model
that emitted the right three recommendations in a different order has made one mistake, not
six, so each gold slot first claims the candidate item with the same product name; whatever
is left over is paired positionally, and any surplus candidate item lands on an index past
the end of gold where it costs precision as an invented recommendation.

Scores are computed over the union of the paths present in either record, so a field the
model left out and a field it invented both cost, and `field_f1` is the harmonic mean of
precision over the candidate's paths and recall over gold's.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Final

from sftdpo.schemas import AdviceRecord, FieldScore

__all__ = [
    "ABSENT",
    "FieldKind",
    "RecordLike",
    "f1_from_scores",
    "field_f1",
    "field_kind",
    "field_scores",
    "field_stem",
    "flatten",
]

type RecordLike = AdviceRecord | Mapping[str, Any]
"""Either a validated record or the raw object a completion parsed to.

The raw form matters: `schema_check.check` hands back a record only when the object is
completely valid, and a completion that got seven fields right and invented an eighth key
still has to be scored on the seven. Comparing the raw object keeps the reward graded across
the schema failures that dominate early training.
"""

ABSENT: Final = "<absent>"
"""Marks a path one side does not have, in `FieldScore.expected` / `.actual`.

Values are rendered with `json.dumps`, so a string whose text is `<absent>` renders as
`"<absent>"` — quotes included — and cannot be mistaken for the marker.
"""


class FieldKind(StrEnum):
    """How one path is compared. The kind is a property of the path, not of the value.

    Reading the kind off the path rather than off the value is what makes a type error
    visible: `review_months: "12"` is compared as an integer, fails to normalise as one, and
    is scored wrong, instead of quietly being compared as the string it happens to be.
    """

    STRING = "string"
    ENUM = "enum"
    MONEY = "money"
    PERCENT = "percent"
    INTEGER = "integer"
    DATE = "date"
    OPAQUE = "opaque"


_SET_FIELDS: Final = frozenset({"objectives", "flags"})
_CENT: Final = Decimal("0.01")

_WHITESPACE_RE: Final = re.compile(r"\s+")
_INDEX_RE: Final = re.compile(r"\[(\d+)\]")
_MEMBER_RE: Final = re.compile(r"\{.*\}$", re.DOTALL)
_HEAD_RE: Final = re.compile(r"[.\[{]")
_REC_PATH_RE: Final = re.compile(r"recommendations\[(?P<index>\d+)\](?:\.(?P<field>.+))?")
_LOOSE_DATE_RE: Final = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")

# The comparison contract, keyed by path shape: indices collapsed to `[]` and set members to
# `{}`. A path that is not in the table is something the model invented, and an invented path
# is never correct, so comparing it opaquely costs nothing but keeps the code total.
_KINDS: Final[dict[str, FieldKind]] = {
    "client_name": FieldKind.STRING,
    "record_date": FieldKind.DATE,
    "risk_profile": FieldKind.ENUM,
    "objectives{}": FieldKind.STRING,
    "recommendations[].product": FieldKind.STRING,
    "recommendations[].action": FieldKind.ENUM,
    "recommendations[].amount": FieldKind.MONEY,
    "fees.advice_fee": FieldKind.MONEY,
    "fees.ongoing_fee_pct": FieldKind.PERCENT,
    "review_months": FieldKind.INTEGER,
    "flags{}": FieldKind.STRING,
}

# Reports read best in the order the schema declares, so paths sort by their top-level field
# first. Anything the model invented sorts after everything the schema asked for.
_FIELD_ORDER: Final[dict[str, int]] = {
    name: index for index, name in enumerate(AdviceRecord.model_fields)
}


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


def _normalise_text(value: str) -> str:
    """Case-fold, collapse whitespace runs, strip, and fold compatibility characters.

    NFKC first, so a full-width space or a ligature the tokeniser emitted is not a different
    client name. `casefold` rather than `lower` because it is the operation defined for
    caseless matching.
    """
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _hundredths(value: Any) -> int | None:
    """A number rounded to two decimals, as an integer count of hundredths.

    `Decimal(str(value))` reads the decimal number the model wrote; going through the binary
    float would round `0.145` by whichever side of the value the float landed on. Returns
    `None` for anything that is not a finite JSON number, and for a magnitude too large to
    quantise, both of which then fall back to exact comparison.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    try:
        return int(Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP) * 100)
    except InvalidOperation:
        return None


def _as_date(value: Any) -> date | None:
    """The calendar date a string denotes, or `None` if it denotes no single day.

    ISO spellings are accepted in full (`2026-03-31`, `20260331`, `2026-W14-2`) plus the
    unpadded `2026-3-31`, because each names exactly one day. Slashed forms are refused: they
    are read differently on either side of the Pacific and a verifier that guessed would put
    a bias into the reward.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    match = _LOOSE_DATE_RE.fullmatch(text)
    if match is None:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None


def _render(value: Any) -> str:
    """One-line rendering for the report, kept distinguishable from `ABSENT`."""
    return json.dumps(value, sort_keys=True, default=str)


def _opaque(value: Any) -> tuple[str, str]:
    return ("raw", _render(value))


def _comparison_key(kind: FieldKind, value: Any) -> tuple[str, Any]:
    """The canonical key for a value: two values are the same iff their keys are equal.

    A value that cannot be read as its path's kind falls back to an opaque key, so it can
    still equal an identical piece of junk on the other side but never a well-formed value.
    """
    if kind is FieldKind.STRING:
        return ("text", _normalise_text(value)) if isinstance(value, str) else _opaque(value)
    if kind is FieldKind.ENUM:
        return ("enum", value) if isinstance(value, str) else _opaque(value)
    if kind in (FieldKind.MONEY, FieldKind.PERCENT):
        hundredths = _hundredths(value)
        return _opaque(value) if hundredths is None else (kind.value, hundredths)
    if kind is FieldKind.INTEGER:
        integral = not isinstance(value, bool) and isinstance(value, int)
        return ("integer", value) if integral else _opaque(value)
    if kind is FieldKind.DATE:
        day = _as_date(value)
        return _opaque(value) if day is None else ("date", day.isoformat())
    return _opaque(value)


def _member_key(member: Any) -> str:
    """The set-member key for `objectives` / `flags`.

    Normalised text for a string so that spacing and case cannot make one objective look like
    two; the rendered value for anything else, which keeps the function total on model output
    that put a number or an object in the list.
    """
    return _normalise_text(member) if isinstance(member, str) else _render(member)


# --------------------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------------------


def _flatten_set(field: str, members: Sequence[Any], out: dict[str, Any]) -> None:
    for member in members:
        out[f"{field}{{{_member_key(member)}}}"] = member


def _flatten_recommendations(items: Sequence[Any], out: dict[str, Any]) -> None:
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            out[f"recommendations[{index}]"] = item
            continue
        for key, value in item.items():
            # A null amount is the schema's way of declining to state one, so it carries no
            # claim and is treated as absent. Null anywhere else is a value the model chose
            # for a field that requires one, and is kept so that it costs precision as well
            # as recall.
            if key == "amount" and value is None:
                continue
            out[f"recommendations[{index}].{key}"] = value


def flatten(record: RecordLike) -> dict[str, Any]:
    """Flatten a record to JSON paths, in the shape the comparison is defined over.

    Args:
        record: A validated `AdviceRecord`, or the raw object a completion parsed to. The raw
            form is flattened defensively: a field of the wrong shape becomes one path holding
            the whole value rather than raising, because the caller is model output.

    Returns:
        A mapping from path to the value the record holds there. Scalars and `fees` use dotted
        paths (`fees.advice_fee`), recommendations are indexed (`recommendations[0].product`),
        and the set-valued fields are keyed by normalised member (`flags{tfn missing}`) so
        that membership rather than position is what is compared.
    """
    value = record.model_dump(mode="json") if isinstance(record, AdviceRecord) else record
    flat: dict[str, Any] = {}
    for key, item in value.items():
        if key in _SET_FIELDS and isinstance(item, list):
            _flatten_set(key, item, flat)
        elif key == "recommendations" and isinstance(item, list):
            _flatten_recommendations(item, flat)
        elif key == "fees" and isinstance(item, Mapping):
            for sub_key, sub_value in item.items():
                flat[f"fees.{sub_key}"] = sub_value
        else:
            flat[key] = item
    return flat


def _shape(path: str) -> str:
    """The path with its indices and set members collapsed, for the kind lookup."""
    return _MEMBER_RE.sub("{}", _INDEX_RE.sub("[]", path))


def field_kind(path: str) -> FieldKind:
    """How the path is compared. Unknown paths are `OPAQUE`; only the model invents those."""
    return _KINDS.get(_shape(path), FieldKind.OPAQUE)


def _ordering_key(path: str) -> tuple[int, str]:
    """Sort paths by schema declaration order, then by path with indices zero-padded.

    The padding is for ordering only, and exists so that `recommendations[10]` sorts after
    `recommendations[2]` instead of between `[1]` and `[2]`.
    """
    head = _HEAD_RE.split(path, maxsplit=1)[0]
    padded = _INDEX_RE.sub(lambda match: f"[{int(match[1]):06d}]", path)
    return (_FIELD_ORDER.get(head, len(_FIELD_ORDER)), padded)


# --------------------------------------------------------------------------------------
# Aligning the recommendation list
# --------------------------------------------------------------------------------------


def _recommendation_index(path: str) -> int | None:
    match = _REC_PATH_RE.fullmatch(path)
    return None if match is None else int(match["index"])


def _products(flat: Mapping[str, Any]) -> dict[int, str | None]:
    """Every recommendation index in the record, mapped to its normalised product name.

    An index whose item has no usable product name maps to `None` and so takes no part in the
    name matching; it is paired positionally instead.
    """
    products: dict[int, str | None] = {}
    for path, value in flat.items():
        index = _recommendation_index(path)
        if index is None:
            continue
        products.setdefault(index, None)
        if path.endswith(".product") and isinstance(value, str):
            products[index] = _normalise_text(value)
    return products


def _alignment(candidate: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[int, int]:
    """Map each candidate recommendation index onto the gold slot it is judged against.

    Gold slots claim candidate items in order, each taking the first unclaimed item with the
    same product name; then the unclaimed items fill the remaining slots in order; then any
    surplus lands past the end of gold, where it is an invented recommendation. The result is
    injective, so no two candidate items can be judged against the same slot.
    """
    candidate_products = _products(candidate)
    gold_products = _products(gold)
    gold_indices = sorted(gold_products)
    candidate_indices = sorted(candidate_products)

    mapping: dict[int, int] = {}
    claimed: set[int] = set()
    for gold_index in gold_indices:
        name = gold_products[gold_index]
        if name is None:
            continue
        for candidate_index in candidate_indices:
            if candidate_index not in claimed and candidate_products[candidate_index] == name:
                mapping[candidate_index] = gold_index
                claimed.add(candidate_index)
                break

    filled = set(mapping.values())
    spare_gold = [index for index in gold_indices if index not in filled]
    spare_candidates = [index for index in candidate_indices if index not in claimed]
    for candidate_index, gold_index in zip(spare_candidates, spare_gold, strict=False):
        mapping[candidate_index] = gold_index
        claimed.add(candidate_index)

    next_index = max(gold_indices, default=-1) + 1
    for candidate_index in candidate_indices:
        if candidate_index not in claimed:
            mapping[candidate_index] = next_index
            next_index += 1
    return mapping


def _reindexed(flat: Mapping[str, Any], mapping: Mapping[int, int]) -> dict[str, Any]:
    """Rewrite the candidate's recommendation paths onto the gold slots they were matched to."""
    if all(source == target for source, target in mapping.items()):
        return dict(flat)
    out: dict[str, Any] = {}
    for path, value in flat.items():
        match = _REC_PATH_RE.fullmatch(path)
        if match is None:
            out[path] = value
            continue
        field = match["field"]
        index = mapping[int(match["index"])]
        out[f"recommendations[{index}]" + (f".{field}" if field else "")] = value
    return out


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def field_scores(candidate: RecordLike | None, gold: RecordLike) -> tuple[FieldScore, ...]:
    """Compare a candidate against gold over the union of the paths either one has.

    Args:
        candidate: The candidate record or raw object, or `None` when the completion did not
            parse at all — which scores as every gold field missing, so an unparseable
            completion and one that emitted `{}` are graded the same way.
        gold: The reference record.

    Returns:
        One `FieldScore` per path, ordered by schema declaration order and then by path. A
        path only gold has is reported with `actual == ABSENT`, a path only the candidate has
        with `expected == ABSENT`, and neither can be correct — which is what makes an
        omission and an invention both cost.
    """
    gold_flat = flatten(gold)
    candidate_flat: dict[str, Any] = {}
    if candidate is not None:
        raw = flatten(candidate)
        candidate_flat = _reindexed(raw, _alignment(raw, gold_flat))

    scores: list[FieldScore] = []
    for path in sorted(set(gold_flat) | set(candidate_flat), key=_ordering_key):
        in_gold = path in gold_flat
        in_candidate = path in candidate_flat
        kind = field_kind(path)
        correct = (
            in_gold
            and in_candidate
            and _comparison_key(kind, gold_flat[path])
            == _comparison_key(kind, candidate_flat[path])
        )
        scores.append(
            FieldScore(
                path=path,
                correct=correct,
                expected=_render(gold_flat[path]) if in_gold else ABSENT,
                actual=_render(candidate_flat[path]) if in_candidate else ABSENT,
            )
        )
    return tuple(scores)


def field_stem(path: str) -> str:
    """The top-level field a scored path belongs to.

    `recommendations[0].amount`, `fees.advice_fee` and `flags{preservation_age_not_reached}`
    stem to `recommendations`, `fees` and `flags`. The three separators are checked together
    and the earliest wins, so a value containing a dot inside a set membership marker --
    `objectives{review in 2028. then again}` -- still stems on the brace that opened it.

    Callers group `FieldScore`s by this to ask what an aggregate F1 cannot: not "how much of
    the record was right" but "which field stopped being produced at all".
    """
    cut = min((index for index in (path.find(c) for c in ".[{") if index >= 0), default=-1)
    return path if cut < 0 else path[:cut]


def f1_from_scores(scores: Sequence[FieldScore]) -> float:
    """The F1 of a comparison, recomputed from the scores it produced.

    Taking the number from the reported scores rather than from a parallel calculation is
    what guarantees that the tuple on a `Reward` explains the number next to it: there is no
    second code path that could disagree.

    Precision is over the paths the candidate filled and recall over the paths gold has, so
    both an omission and an invention lower the result. Two empty records score 1.0 — there
    was nothing to get wrong — which is unreachable for a real `AdviceRecord` because it
    always carries at least its required fields.
    """
    correct = sum(1 for score in scores if score.correct)
    in_candidate = sum(1 for score in scores if score.actual != ABSENT)
    in_gold = sum(1 for score in scores if score.expected != ABSENT)
    total = in_candidate + in_gold
    if total == 0:
        return 1.0
    # 2 * correct <= total because a correct path is present on both sides, so the result is
    # in [0, 1]; both operands are exact small integers, so that bound survives the division.
    return 2 * correct / total


def field_f1(candidate: RecordLike | None, gold: RecordLike) -> float:
    """Harmonic mean of field precision and recall, in [0, 1].

    Monotone in correctness in the sense the reward relies on: repairing a field towards gold,
    or deleting a field the record invented, never lowers the result. It is not monotone under
    adding a *wrong* field, and must not be — that is precision doing its job.

    One exception, found by exhaustive search and stated rather than hidden. Recommendations
    are aligned to gold slots by product name (see :func:`_alignment`), and the alignment is
    re-derived on every call. When gold holds two recommendations with the **same** product
    name, repairing one candidate's product name towards gold can re-align both items and
    drag their other fields to different slots, losing more matched paths than the repair
    gains. Constructed example: gold has two "Growth Fund" recommendations; a completion with
    both product names wrong and everything else right scores 0.833, and repairing the second
    name to "Growth Fund" drops it to 0.583.

    The alternative is an optimal assignment over the whole list rather than a greedy match
    on names, which is the right fix and a larger one. It is not reached in this corpus --
    the generator draws recommendation products without replacement, so gold never holds a
    duplicate name -- and `test_duplicate_gold_products_can_invert_a_repair` pins the
    behaviour so that the day the corpus changes, the test fails rather than the reward
    quietly inverting a preference pair.
    """
    return f1_from_scores(field_scores(candidate, gold))
