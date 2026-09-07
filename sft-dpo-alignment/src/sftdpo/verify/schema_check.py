"""Validating a parsed object against `AdviceRecord`, in this project's own vocabulary.

Pydantic is the structural engine — it already reports missing fields, forbidden extras,
wrong types and bad enum members, all of them at once and each with a location. What it does
not do is answer in `ViolationKind`, and the taxonomy is what the reward is built from, so
this module translates rather than re-implements.

Two deliberate departures from stock pydantic validation, both of which change scores:

*Lax mode, with the numeric coercions taken back.* `strict=True` cannot be used at all here:
it demands a `RiskProfile` instance and rejects the string `"balanced"`, and a JSON document
only ever carries strings. Lax mode reads enums correctly but will also accept `"12"` for an
integer, `true` for a number, and `12.0` where an integer was asked for. A model that quoted
a number has not produced the requested schema, so those coercions are undone here as
`WRONG_TYPE` at the four numeric paths. `12.0` is rejected alongside `12.5` so that the rule
is "an integer field wants a JSON integer" rather than "…unless the fraction happens to be
zero".

*Semantic rules the type alone cannot carry.* A date is a `str` in the contract, so
`2026-02-30` type-checks; it is still not a date. Ranges are the same kind of rule. These
live here rather than in `schemas.py` because `schemas.py` is the shape shown to the model
and this is the standard it is marked against.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any, Final

from pydantic import ValidationError

from sftdpo.schemas import AdviceRecord, SchemaViolation, ViolationKind

__all__ = ["MAX_ONGOING_FEE_PCT", "MAX_REVIEW_MONTHS", "MIN_REVIEW_MONTHS", "check"]

MIN_REVIEW_MONTHS: Final = 1
MAX_REVIEW_MONTHS: Final = 120
MAX_ONGOING_FEE_PCT: Final = 10.0

_ISO_DATE_RE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")

# Everything pydantic can report for this schema, mapped to the project's taxonomy. The
# fallback is WRONG_TYPE rather than a catch-all kind: the taxonomy has no "other", and for
# a schema this small anything not listed here is a type problem.
_ERROR_KINDS: Final[dict[str, ViolationKind]] = {
    "missing": ViolationKind.MISSING_FIELD,
    "extra_forbidden": ViolationKind.EXTRA_FIELD,
    "enum": ViolationKind.BAD_ENUM,
    "greater_than": ViolationKind.OUT_OF_RANGE,
    "greater_than_equal": ViolationKind.OUT_OF_RANGE,
    "less_than": ViolationKind.OUT_OF_RANGE,
    "less_than_equal": ViolationKind.OUT_OF_RANGE,
    "too_short": ViolationKind.OUT_OF_RANGE,
    "too_long": ViolationKind.OUT_OF_RANGE,
}

# `ViolationKind` is declared most severe first and `Reward.headline_violation` reads the
# first entry, so the returned tuple is ordered by severity and then by path.
_SEVERITY: Final[dict[ViolationKind, int]] = {
    kind: index for index, kind in enumerate(ViolationKind)
}


def _render_path(loc: tuple[int | str, ...]) -> str:
    """Render a pydantic location as a JSON path such as `recommendations[2].action`.

    The root is the empty string; there is no `$` prefix, so a path can be pasted straight
    into a report next to the field name a reader is looking for.
    """
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(f".{item}" if parts else item)
    return "".join(parts)


def _structural_violations(exc: ValidationError) -> list[SchemaViolation]:
    """Translate every pydantic error, not just the first."""
    return [
        SchemaViolation(
            kind=_ERROR_KINDS.get(error["type"], ViolationKind.WRONG_TYPE),
            path=_render_path(error["loc"]),
            detail=error["msg"],
        )
        for error in exc.errors()
    ]


def _number_violation(
    raw: object,
    path: str,
    *,
    integral: bool,
    low: float | None,
    high: float | None,
) -> SchemaViolation | None:
    """Check one number as JSON presented it, before pydantic had a chance to coerce it.

    `bool` is excluded explicitly because it is a subclass of `int` in Python: without that
    guard `"review_months": true` would validate as a review in one month's time.
    """
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return SchemaViolation(
            kind=ViolationKind.WRONG_TYPE, path=path, detail="expected a JSON number"
        )
    if integral and not isinstance(raw, int):
        return SchemaViolation(
            kind=ViolationKind.WRONG_TYPE,
            path=path,
            detail="expected a JSON integer, not a fractional number",
        )
    if not math.isfinite(raw):
        # json.loads accepts NaN and Infinity; every range test against them is False, so
        # without this they would slip through as valid fees.
        return SchemaViolation(
            kind=ViolationKind.OUT_OF_RANGE, path=path, detail="not a finite number"
        )
    if low is not None and raw < low:
        return SchemaViolation(
            kind=ViolationKind.OUT_OF_RANGE, path=path, detail=f"must be at least {low}"
        )
    if high is not None and raw > high:
        return SchemaViolation(
            kind=ViolationKind.OUT_OF_RANGE, path=path, detail=f"must be at most {high}"
        )
    return None


def _extend(out: list[SchemaViolation], violation: SchemaViolation | None) -> None:
    if violation is not None:
        out.append(violation)


def _is_calendar_date(text: str) -> bool:
    """`YYYY-MM-DD` and a day that exists.

    The regex is not redundant with `date.fromisoformat`, which since 3.11 also accepts
    `20260101` and `2026-W01-1`. The prompt asks for one spelling and the verifier marks
    that spelling.
    """
    if not _ISO_DATE_RE.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _semantic_violations(value: dict[str, Any]) -> list[SchemaViolation]:
    """Rules the type annotations cannot express, checked against the raw JSON.

    Each check is guarded on the JSON type it needs, so a field pydantic has already
    rejected is simply skipped here rather than reported twice.
    """
    out: list[SchemaViolation] = []

    record_date = value.get("record_date")
    if isinstance(record_date, str) and not _is_calendar_date(record_date):
        out.append(
            SchemaViolation(
                kind=ViolationKind.BAD_DATE,
                path="record_date",
                detail="expected a real calendar date in YYYY-MM-DD",
            )
        )

    review_months = value.get("review_months")
    if review_months is not None:
        _extend(
            out,
            _number_violation(
                review_months,
                "review_months",
                integral=True,
                low=MIN_REVIEW_MONTHS,
                high=MAX_REVIEW_MONTHS,
            ),
        )

    fees = value.get("fees")
    if isinstance(fees, dict):
        # A fee cannot be negative, and `ongoing_fee_pct` is a percentage rather than a
        # fraction. Leaving the percentage unbounded would make the schema silent about the
        # one error in this field that costs a client money: a model that writes 80 for a
        # 0.8 % fee has produced a number the downstream system would act on, and an
        # unbounded float validator would score it as a success. Ten per cent is above any
        # ongoing advice fee in the Australian market, so a larger value is a unit error,
        # not a large fee. The bound cannot catch the same confusion in the other direction
        # (0.008 for 0.8) — that residue is left to the field comparison against gold.
        for key, low, high in (
            ("advice_fee", 0.0, None),
            ("ongoing_fee_pct", 0.0, MAX_ONGOING_FEE_PCT),
        ):
            raw = fees.get(key)
            if raw is not None:
                _extend(
                    out, _number_violation(raw, f"fees.{key}", integral=False, low=low, high=high)
                )

    recommendations = value.get("recommendations")
    if isinstance(recommendations, list):
        for index, item in enumerate(recommendations):
            if not isinstance(item, dict):
                continue
            amount = item.get("amount")
            if amount is not None:
                # Deliberately unbounded apart from finiteness: the task fixes no sign
                # convention for a switch, so a bound here would encode a rule the prompt
                # never states.
                _extend(
                    out,
                    _number_violation(
                        amount,
                        f"recommendations[{index}].amount",
                        integral=False,
                        low=None,
                        high=None,
                    ),
                )

    return out


def check(value: dict[str, Any]) -> tuple[AdviceRecord | None, tuple[SchemaViolation, ...]]:
    """Validate a parsed object against `AdviceRecord`.

    Args:
        value: The object produced by `parse.extract_json`.

    Returns:
        `(record, violations)`. The record is returned only when the object is completely
        valid, so `record is not None` and `violations == ()` always agree — a partially
        valid record has no meaning for the reward and returning one would invite callers to
        use it. Violations are ordered most severe first and every failing path is reported,
        never only the first.

        The three parse-level kinds (`NO_JSON`, `UNPARSEABLE`, `NOT_AN_OBJECT`) are never
        produced here; by the time this runs there is an object.
    """
    record: AdviceRecord | None = None
    violations: list[SchemaViolation] = []
    try:
        record = AdviceRecord.model_validate(value)
    except ValidationError as exc:
        violations.extend(_structural_violations(exc))

    reported = {violation.path for violation in violations}
    violations.extend(
        violation for violation in _semantic_violations(value) if violation.path not in reported
    )

    if violations:
        ordered = sorted(violations, key=lambda v: (_SEVERITY[v.kind], v.path))
        return None, tuple(ordered)
    return record, ()
