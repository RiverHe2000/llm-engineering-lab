"""Recovering one JSON object from a model completion, with every repair named.

There are two modes and the distance between them is the measurement. STRICT is the
behaviour the training is trying to produce: the completion is one JSON object and nothing
else. LENIENT recovers the object from what a small instruct model actually emits — a fenced
code block, a sentence of preamble, single quotes, Python literals, a trailing comma, a
second copy of the object, a run of surplus closing braces, an answer cut off by the token
budget.

Scoring the same completions both ways separates two different failures: not knowing the
schema, and not knowing that prose is unwanted. Only the second is cheap to fix by other
means, so the strict-minus-lenient gap is a headline number for this project. Every repair
is therefore a named member of `Repair` rather than one "we tidied it up" flag, and it is
carried on `ParseOutcome.repairs` for a caller that wants to know *which* tidying was needed.

What the evaluator aggregates today is the count, not the histogram: `ParseGap.repaired` is
one integer, so a run whose base model needed sixteen code-fence repairs reports the same
number as one that needed sixteen single-quote repairs. That is a gap in the reporting, not
in the taxonomy, and it is written down here rather than dressed up --- an earlier version of
this paragraph claimed the histogram was reported, which would have had a reader quoting a
diagnostic the project does not produce.

The scanner underneath is deliberately string-aware. A brace inside a string value must not
end the object, an apostrophe in "Here's the JSON:" must not open one, and an escaped quote
must not close one — a verifier that got any of those wrong would hand the trainer a reward
signal with a bias in it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, model_validator

from sftdpo.schemas import SchemaViolation, ViolationKind

__all__ = ["ParseOutcome", "Repair", "extract_json"]


class Repair(StrEnum):
    """One named, countable edit made while recovering the object.

    Declared in the order the pipeline applies them, so a recorded `repairs` tuple is always
    sorted and two outcomes can be compared directly.
    """

    CODE_FENCE = "code_fence"
    LEADING_PROSE = "leading_prose"
    TRAILING_PROSE = "trailing_prose"
    DUPLICATE_OBJECT = "duplicate_object"
    UNBALANCED_BRACES = "unbalanced_braces"
    CLOSED_UNTERMINATED = "closed_unterminated"
    PYTHON_LITERALS = "python_literals"
    TRAILING_COMMA = "trailing_comma"
    SINGLE_QUOTES = "single_quotes"


class ParseOutcome(BaseModel):
    """The result of trying to get one JSON object out of a completion.

    `repairs` records every edit that was applied, whether or not the parse ultimately
    succeeded, because a completion that needed four repairs and still failed is a different
    diagnosis from one that failed immediately. The evaluator counts repairs on successful
    outcomes only.
    """

    model_config = ConfigDict(frozen=True)

    ok: bool
    value: dict[str, Any] | None = None
    repairs: tuple[str, ...] = ()
    violation: SchemaViolation | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ParseOutcome:
        """Reject an outcome whose three answers disagree.

        Downstream code branches on `ok` and then dereferences `value` or `violation`. If
        those could ever disagree the failure would surface far from here, as a reward
        quietly computed from `None`.
        """
        if self.ok != (self.value is not None) or self.ok != (self.violation is None):
            msg = "ok, value and violation must agree"
            raise ValueError(msg)
        return self

    @property
    def repaired(self) -> bool:
        return bool(self.repairs)


_QUOTE_OPENS_AFTER: Final = frozenset("{[,:")
_OPENERS: Final = frozenset("{[")
_CLOSERS: Final = frozenset("}]")
_CLOSE_FOR: Final = {"{": "}", "[": "]"}
_PY_LITERALS: Final = {"None": "null", "True": "true", "False": "false"}
_PY_LITERAL_RE: Final = re.compile(r"\b(?:None|True|False)\b")
_TRAILING_COMMA_RE: Final = re.compile(r",(\s*[}\]])")
_FENCE_RE: Final = re.compile(
    r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*(?:\r?\n)?(?P<body>.*?)(?:```|\Z)", re.DOTALL
)


def _end_of_string(text: str, start: int, quote: str) -> int:
    """Index just past the closing quote of the literal opening at `start`.

    Returns `len(text)` for a literal the completion never closed, which is the normal shape
    of a completion truncated by the token budget.
    """
    i = start + 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == quote:
            return i + 1
        i += 1
    return n


def _string_spans(text: str) -> list[tuple[int, int, str]]:
    """Half-open spans of every string literal, as (start, end, quote character).

    A double quote always opens a literal. A single quote only does so where a JSON key or
    value may begin — directly after `{`, `[`, `,` or `:` — which is what keeps the
    apostrophe in "Here's the JSON:" from swallowing the object that follows it while still
    recognising the `'a'` in `{'a': 1}`.
    """
    spans: list[tuple[int, int, str]] = []
    previous = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"' or (ch == "'" and previous in _QUOTE_OPENS_AFTER):
            end = _end_of_string(text, i, ch)
            spans.append((i, end, ch))
            previous = ch
            i = end
            continue
        if not ch.isspace():
            previous = ch
        i += 1
    return spans


def _string_mask(text: str) -> list[bool]:
    """`True` at every index that lies within a string literal, delimiters included."""
    mask = [False] * len(text)
    for start, end, _ in _string_spans(text):
        for i in range(start, end):
            mask[i] = True
    return mask


def _find_object(text: str) -> tuple[int, int, str] | None:
    """Locate the first brace-delimited object, respecting strings and escapes.

    Returns `(start, end, closers)` with `end` exclusive, or `None` when the text holds no
    `{` outside a string. `closers` is the bracket run needed to terminate an object the
    completion never closed, and is empty when the object closed on its own.
    """
    mask = _string_mask(text)
    start = next((i for i, ch in enumerate(text) if ch == "{" and not mask[i]), None)
    if start is None:
        return None
    stack: list[str] = []
    for i in range(start, len(text)):
        if mask[i]:
            continue
        ch = text[i]
        if ch in _OPENERS:
            stack.append(ch)
        elif ch in _CLOSERS and stack:
            stack.pop()
            if not stack:
                return start, i + 1, ""
    return start, len(text), "".join(_CLOSE_FOR[c] for c in reversed(stack))


def _fence_body(text: str) -> str | None:
    """The body of the first fenced block that contains a brace, or `None` for no fence.

    An unterminated fence still yields its body: a model that opened ```json and ran out of
    tokens has told us where the object starts, which is all this step needs.
    """
    for match in _FENCE_RE.finditer(text):
        body = match.group("body")
        if "{" in body:
            return body
    return None


def _classify_tail(tail: str) -> Repair | None:
    """Name what followed the object, so trailing junk is diagnosed rather than dropped.

    The tail gets exactly one label — the most specific that fits — because a model that
    emitted the object twice has made a different mistake from one that added a sign-off.
    """
    stripped = tail.strip()
    if not stripped:
        return None
    if all(ch in _CLOSERS or ch.isspace() for ch in stripped):
        return Repair.UNBALANCED_BRACES
    if _find_object(tail) is not None:
        return Repair.DUPLICATE_OBJECT
    return Repair.TRAILING_PROSE


def _replace_python_literals(text: str) -> str:
    """`None`/`True`/`False` outside strings become their JSON spellings."""
    mask = _string_mask(text)

    def _sub(match: re.Match[str]) -> str:
        if mask[match.start()]:
            return match.group()
        return _PY_LITERALS[match.group()]

    return _PY_LITERAL_RE.sub(_sub, text)


def _drop_trailing_commas(text: str) -> str:
    """Remove a comma that sits immediately before a closing bracket."""
    mask = _string_mask(text)

    def _sub(match: re.Match[str]) -> str:
        if mask[match.start()]:
            return match.group()
        return match.group(1)

    return _TRAILING_COMMA_RE.sub(_sub, text)


def _as_json_string(literal: str) -> str:
    """Re-spell one single-quoted literal as a JSON string.

    `\\'` is not a JSON escape, so it collapses to a bare apostrophe; a bare `"` has to gain
    one. Every other escape is passed through untouched rather than interpreted, so this
    cannot invent a character the model did not write.
    """
    terminated = len(literal) >= 2 and literal.endswith("'")
    inner = literal[1:-1] if terminated else literal[1:]
    pieces = ['"']
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and i + 1 < len(inner):
            following = inner[i + 1]
            pieces.append(following if following == "'" else ch + following)
            i += 2
            continue
        pieces.append('\\"' if ch == '"' else ch)
        i += 1
    pieces.append('"')
    return "".join(pieces)


def _double_quote_strings(text: str) -> str:
    """Convert every single-quoted literal to a double-quoted one."""
    spans = _string_spans(text)
    if not any(quote == "'" for _, _, quote in spans):
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end, quote in spans:
        if quote != "'":
            continue
        pieces.append(text[cursor:start])
        pieces.append(_as_json_string(text[start:end]))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


_LADDER: Final[tuple[tuple[Repair, Callable[[str], str]], ...]] = (
    (Repair.PYTHON_LITERALS, _replace_python_literals),
    (Repair.TRAILING_COMMA, _drop_trailing_commas),
    (Repair.SINGLE_QUOTES, _double_quote_strings),
)


def _try_load(text: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(text)
    except ValueError:
        return None
    # Callers only pass a body anchored on `{`, and by the JSON grammar such a text either
    # fails to parse or parses to an object; there is no third case to guard against.
    return cast("dict[str, Any]", loaded)


def _repair_and_load(body: str) -> tuple[dict[str, Any] | None, list[Repair]]:
    """Walk the syntax ladder until the body parses, recording what was applied.

    Each rung is applied cumulatively and only recorded when it actually changed the text,
    so the result is the prefix of the ladder the completion needed. It is not the minimal
    set: a rung that changed the text without being the fix is still counted, because
    searching for a minimal subset would cost an exponential number of parses to answer a
    question no metric asks.
    """
    applied: list[Repair] = []
    attempt = body
    value = _try_load(attempt)
    for name, transform in _LADDER:
        if value is not None:
            break
        repaired = transform(attempt)
        if repaired == attempt:
            continue
        attempt = repaired
        applied.append(name)
        value = _try_load(attempt)
    return value, applied


def _failed(kind: ViolationKind, detail: str, repairs: tuple[Repair, ...] = ()) -> ParseOutcome:
    return ParseOutcome(
        ok=False,
        repairs=tuple(repairs),
        violation=SchemaViolation(kind=kind, detail=detail),
    )


def _extract_strict(text: str) -> ParseOutcome:
    """The whole completion must be one JSON object.

    Surrounding whitespace is tolerated and nothing else is: a fence, a preamble or a second
    object all fail here, which is the point of having this mode at all.
    """
    stripped = text.strip()
    if not stripped:
        return _failed(ViolationKind.NO_JSON, "empty completion")
    try:
        value = json.loads(stripped)
    except ValueError as exc:
        kind = ViolationKind.UNPARSEABLE if "{" in stripped else ViolationKind.NO_JSON
        return _failed(kind, str(exc))
    if not isinstance(value, dict):
        return _failed(ViolationKind.NOT_AN_OBJECT, f"top-level value is {type(value).__name__}")
    return ParseOutcome(ok=True, value=value)


def _extract_lenient(text: str) -> ParseOutcome:
    """Recover the object and name every edit it took.

    A completion that already satisfies STRICT returns here unchanged with no repairs, which
    makes lenient scoring a true superset of strict scoring and the gap between them a
    difference in behaviour rather than in code path.

    Because the scan anchors on `{`, anything that loads is an object; `NOT_AN_OBJECT` is
    reachable only in strict mode.
    """
    direct = _extract_strict(text)
    if direct.ok:
        return direct

    repairs: list[Repair] = []
    candidate = text
    body = _fence_body(candidate)
    if body is not None:
        candidate = body
        repairs.append(Repair.CODE_FENCE)

    found = _find_object(candidate)
    if found is None:
        return _failed(ViolationKind.NO_JSON, "no '{' outside a string literal", tuple(repairs))
    start, end, closers = found

    if candidate[:start].strip():
        repairs.append(Repair.LEADING_PROSE)
    tail = _classify_tail(candidate[end:])
    if tail is not None:
        repairs.append(tail)

    object_text = candidate[start:end]
    if closers:
        object_text += closers
        repairs.append(Repair.CLOSED_UNTERMINATED)

    value, ladder = _repair_and_load(object_text)
    repairs.extend(ladder)
    if value is None:
        return _failed(ViolationKind.UNPARSEABLE, "no repair produced valid JSON", tuple(repairs))
    return ParseOutcome(ok=True, value=value, repairs=tuple(repairs))


def extract_json(text: str, *, strict: bool) -> ParseOutcome:
    """Extract one JSON object from a completion.

    Args:
        text: The raw model completion.
        strict: `True` to require the completion to be exactly one JSON object, `False` to
            recover the object from surrounding prose and common syntax slips. The mode has
            no default because the score depends on it and a caller must say which one it
            means.

    Returns:
        A `ParseOutcome`. On failure the violation is one of `NO_JSON` (nothing that could
        be an object), `UNPARSEABLE` (something object-shaped that will not load) or
        `NOT_AN_OBJECT` (valid JSON that is not an object, strict mode only).
    """
    return _extract_strict(text) if strict else _extract_lenient(text)
