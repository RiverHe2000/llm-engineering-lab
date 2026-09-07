"""What one model scored on one test split, at the level of detail a decision needs.

Four numbers describe the task -- did a JSON object come out, did it satisfy the schema, how
much of the record was right, and was the whole record right -- and a fifth, the mean
verifier reward, is the scalar the preference miner ranks by. All five are reported over the
whole split and again per `Slice` and per `ViolationKind`, because an average is the one view
that can hide a regression: a model can lift its overall score while getting worse on absent
fields, and only the breakdown makes that visible.

The strict-versus-lenient parse gap is reported as a metric in its own right rather than left
as a footnote. "The model emits JSON" and "the model emits JSON if you allow a repair" are
different claims, and only the first one describes a system that can be deployed without a
repair step in front of it. The gap is the size of the repair step, and because lenient
parsing is a superset of strict parsing it can never be negative.

`EvalReport` also carries a per-example row for every sample. That is not redundancy: the
comparison gate pairs the two models over example ids, and pairing is what removes the
variance due to example difficulty from the difference it is trying to measure. Aggregates
alone would force the gate back onto an unpaired test with several times the variance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sftdpo.schemas import Example, FieldScore, Reward, Sample, Slice, ViolationKind
from sftdpo.verify.fields import ABSENT, field_stem
from sftdpo.verify.reward import Verifier, lenient_verifier, strict_verifier

__all__ = [
    "EvalReport",
    "ExampleOutcome",
    "FieldCoverage",
    "MetricBlock",
    "ParseGap",
    "evaluate",
]


class ExampleOutcome(BaseModel):
    """How one model did on one example, kept so the comparison can pair over ids."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    slice: Slice
    json_valid: bool
    schema_valid: bool
    field_f1: float
    exact_match: bool
    reward: float
    lenient_json_valid: bool
    violation: ViolationKind | None = None

    @property
    def success(self) -> bool:
        """The binary outcome the gate tests.

        Schema validity rather than exact match, because a schema-valid record with one field
        wrong is something a downstream system can consume and correct, while an object that
        fails validation is not. Exact match is reported alongside it, not instead of it.
        """
        return self.schema_valid

    @property
    def repaired_only(self) -> bool:
        """True when a lenient parse recovered an object that strict parsing rejected."""
        return self.lenient_json_valid and not self.json_valid


class MetricBlock(BaseModel):
    """The five headline numbers over some set of outcomes.

    An empty block reports zero for every rate rather than being omitted, so a table never
    grows a hole; `n` is what says the group was empty.
    """

    model_config = ConfigDict(frozen=True)

    n: int = 0
    json_valid_rate: float = 0.0
    schema_valid_rate: float = 0.0
    mean_field_f1: float = 0.0
    exact_match_rate: float = 0.0
    mean_reward: float = 0.0

    @classmethod
    def over(cls, outcomes: Sequence[ExampleOutcome]) -> MetricBlock:
        """Summarise a group of outcomes."""
        total = len(outcomes)
        if total == 0:
            return cls()
        return cls(
            n=total,
            json_valid_rate=sum(o.json_valid for o in outcomes) / total,
            schema_valid_rate=sum(o.schema_valid for o in outcomes) / total,
            mean_field_f1=sum(o.field_f1 for o in outcomes) / total,
            exact_match_rate=sum(o.exact_match for o in outcomes) / total,
            mean_reward=sum(o.reward for o in outcomes) / total,
        )

    def as_dict(self) -> dict[str, float]:
        """Manifest-friendly view, with `n` widened to a float for a uniform value type."""
        return {
            "n": float(self.n),
            "json_valid_rate": self.json_valid_rate,
            "schema_valid_rate": self.schema_valid_rate,
            "mean_field_f1": self.mean_field_f1,
            "exact_match_rate": self.exact_match_rate,
            "mean_reward": self.mean_reward,
        }


class ParseGap(BaseModel):
    """How much of the JSON validity rate is owed to a repair.

    Reported because the two verifiers answer different questions about the same
    completions, and quoting only the lenient number would describe a system that does not
    exist unless a repair step ships with it.
    """

    model_config = ConfigDict(frozen=True)

    n: int = 0
    strict_rate: float = 0.0
    lenient_rate: float = 0.0
    repaired: int = 0

    @property
    def gap(self) -> float:
        """Lenient minus strict; never negative, because lenient parsing is a superset."""
        return self.lenient_rate - self.strict_rate

    @classmethod
    def over(cls, outcomes: Sequence[ExampleOutcome]) -> ParseGap:
        """Summarise the two parse rates over a group of outcomes."""
        total = len(outcomes)
        if total == 0:
            return cls()
        return cls(
            n=total,
            strict_rate=sum(o.json_valid for o in outcomes) / total,
            lenient_rate=sum(o.lenient_json_valid for o in outcomes) / total,
            repaired=sum(o.repaired_only for o in outcomes),
        )


class FieldCoverage(BaseModel):
    """How much of one top-level field the model produced, across the whole split.

    The mean field F1 is an average over records, and an average over records cannot say
    *which* field moved. A model that stops emitting one optional field entirely loses only
    that field's share of every record's F1, and buys back more than it lost if the field was
    one it used to get wrong -- so the headline metric rises while a whole column of the
    output disappears. This block is the view that makes that visible, and
    `Floors.max_field_recall_drop` is the rule that acts on it.

    Attributes:
        field: The top-level field name, as `field_stem` derives it.
        expected: Paths under this field that gold has, summed over every example.
        emitted: Paths the model produced under it, right or wrong.
        correct: Paths it produced that match gold.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    expected: int = 0
    emitted: int = 0
    correct: int = 0

    @property
    def recall(self) -> float:
        """Share of gold paths the model produced correctly; 1.0 when gold had none."""
        return 1.0 if self.expected == 0 else self.correct / self.expected

    @property
    def precision(self) -> float:
        """Share of produced paths that were right; 1.0 when the model produced none.

        One rather than zero, so that a field gold never asks for and the model never invents
        does not read as a precision failure. Recall is the number the gate watches.
        """
        return 1.0 if self.emitted == 0 else self.correct / self.emitted

    def as_dict(self) -> dict[str, float | int | str]:
        """JSON-serialisable view, with the two derived rates spelled out."""
        return {
            "field": self.field,
            "expected": self.expected,
            "emitted": self.emitted,
            "correct": self.correct,
            "recall": self.recall,
            "precision": self.precision,
        }


def _coverage(scored: Sequence[Sequence[FieldScore]]) -> dict[str, FieldCoverage]:
    """Aggregate per-example field scores into one row per top-level field.

    The rows come back in first-appearance order, which is schema declaration order because
    that is the order `field_scores` produces, so two runs render identically.
    """
    expected: dict[str, int] = {}
    emitted: dict[str, int] = {}
    correct: dict[str, int] = {}
    for scores in scored:
        for score in scores:
            stem = field_stem(score.path)
            expected.setdefault(stem, 0)
            emitted.setdefault(stem, 0)
            correct.setdefault(stem, 0)
            if score.expected != ABSENT:
                expected[stem] += 1
            if score.actual != ABSENT:
                emitted[stem] += 1
            if score.correct:
                correct[stem] += 1
    return {
        stem: FieldCoverage(
            field=stem, expected=expected[stem], emitted=emitted[stem], correct=correct[stem]
        )
        for stem in expected
    }


class EvalReport(BaseModel):
    """Everything one evaluation run of one model variant produced.

    Frozen, and carrying the per-example rows rather than only the aggregates, so that a
    report can be written to disk and later compared against another without re-running
    either model.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    overall: MetricBlock = Field(default_factory=MetricBlock)
    per_slice: dict[Slice, MetricBlock] = Field(default_factory=dict)
    per_violation: dict[ViolationKind, MetricBlock] = Field(default_factory=dict)
    per_field: dict[str, FieldCoverage] = Field(default_factory=dict)
    parse_gap: ParseGap = Field(default_factory=ParseGap)
    outcomes: tuple[ExampleOutcome, ...] = ()

    @property
    def n(self) -> int:
        """How many examples the report covers."""
        return len(self.outcomes)

    @property
    def example_ids(self) -> tuple[str, ...]:
        """The example ids, in the order the samples were scored."""
        return tuple(outcome.example_id for outcome in self.outcomes)

    def outcome_for(self, example_id: str) -> ExampleOutcome:
        """One example's row.

        Raises:
            KeyError: If the report does not cover that example.
        """
        for outcome in self.outcomes:
            if outcome.example_id == example_id:
                return outcome
        raise KeyError(f"no outcome for example {example_id!r} in report for {self.model!r}")

    def success_by_example(self) -> dict[str, float]:
        """Per-example success as 0.0/1.0, the shape the paired statistics consume."""
        return {outcome.example_id: float(outcome.success) for outcome in self.outcomes}

    def field_f1_by_example(self) -> dict[str, float]:
        """Per-example field F1, the continuous companion to the success indicator."""
        return {outcome.example_id: outcome.field_f1 for outcome in self.outcomes}

    def slice_by_example(self) -> dict[str, Slice]:
        """Which slice each example came from."""
        return {outcome.example_id: outcome.slice for outcome in self.outcomes}

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary, without the per-example rows."""
        return {
            "model": self.model,
            "n": self.n,
            "overall": self.overall.as_dict(),
            "per_slice": {name.value: block.as_dict() for name, block in self.per_slice.items()},
            "per_violation": {
                kind.value: block.as_dict() for kind, block in self.per_violation.items()
            },
            "per_field": {name: block.as_dict() for name, block in self.per_field.items()},
            "parse_gap": {
                "strict_rate": self.parse_gap.strict_rate,
                "lenient_rate": self.parse_gap.lenient_rate,
                "gap": self.parse_gap.gap,
                "repaired": self.parse_gap.repaired,
            },
        }


def _index_examples(examples: Sequence[Example]) -> dict[str, Example]:
    index: dict[str, Example] = {}
    for example in examples:
        if example.example_id in index:
            raise ValueError(f"duplicate example_id {example.example_id!r} in the test split")
        index[example.example_id] = example
    return index


def _resolve_model(samples: Sequence[Sample], model: str | None) -> str:
    if model is not None:
        return model
    names = {sample.model for sample in samples}
    if len(names) != 1:
        raise ValueError(
            "cannot infer the model name from "
            f"{sorted(names)!r}; pass model= explicitly. A report mixing variants would be "
            "compared against itself"
        )
    return names.pop()


def _outcome(
    sample: Sample,
    example: Example,
    strict_reward: Reward,
    lenient_reward: Reward,
) -> ExampleOutcome:
    return ExampleOutcome(
        example_id=sample.example_id,
        slice=example.slice,
        json_valid=strict_reward.parsed,
        schema_valid=strict_reward.schema_valid,
        field_f1=strict_reward.field_f1,
        exact_match=strict_reward.exact_match,
        reward=strict_reward.value,
        lenient_json_valid=lenient_reward.parsed,
        violation=strict_reward.headline_violation,
    )


def evaluate(
    samples: Sequence[Sample],
    examples: Sequence[Example],
    *,
    model: str | None = None,
    strict: Verifier | None = None,
    lenient: Verifier | None = None,
) -> EvalReport:
    """Score one completion per example and summarise the result.

    Args:
        samples: Exactly one completion per example. Duplicates are rejected rather than
            averaged: this report feeds a paired comparison, and two rows for one id would
            weight that example twice in a statistic whose whole purpose is that every
            example counts once on both sides.
        examples: The examples the samples came from, in any order.
        model: The variant label. Inferred from the samples when they agree.
        strict: Verifier for the headline numbers; `strict_verifier()` by default.
        lenient: Verifier used only for the parse gap; `lenient_verifier()` by default.

    Returns:
        The report, with the per-slice and per-violation blocks in the enum's own order so
        that two runs render identically.

    Raises:
        ValueError: If a sample refers to an unknown example, if two samples share an
            example id, if the examples themselves contain a duplicate id, or if the model
            name is neither given nor inferable.
    """
    known = _index_examples(examples)
    resolved_model = _resolve_model(samples, model)
    strict_verify = strict_verifier() if strict is None else strict
    lenient_verify = lenient_verifier() if lenient is None else lenient

    outcomes: list[ExampleOutcome] = []
    scored: list[tuple[FieldScore, ...]] = []
    seen: set[str] = set()
    for sample in samples:
        if sample.example_id in seen:
            raise ValueError(
                f"two samples for example {sample.example_id!r}; evaluation expects one "
                "completion per example"
            )
        example = known.get(sample.example_id)
        if example is None:
            raise ValueError(f"no example for sample {sample.example_id!r}")
        seen.add(sample.example_id)
        strict_reward = strict_verify.score(sample.text, example.gold)
        scored.append(strict_reward.fields)
        outcomes.append(
            _outcome(
                sample,
                example,
                strict_reward,
                lenient_verify.score(sample.text, example.gold),
            )
        )

    by_slice: dict[Slice, list[ExampleOutcome]] = {}
    by_violation: dict[ViolationKind, list[ExampleOutcome]] = {}
    for outcome in outcomes:
        by_slice.setdefault(outcome.slice, []).append(outcome)
        if outcome.violation is not None:
            by_violation.setdefault(outcome.violation, []).append(outcome)

    return EvalReport(
        model=resolved_model,
        overall=MetricBlock.over(outcomes),
        # Enum order rather than insertion order: the rendered table must not depend on
        # which slice the first sample happened to come from.
        per_slice={name: MetricBlock.over(by_slice[name]) for name in Slice if name in by_slice},
        per_violation={
            kind: MetricBlock.over(by_violation[kind])
            for kind in ViolationKind
            if kind in by_violation
        },
        per_field=_coverage(scored),
        parse_gap=ParseGap.over(outcomes),
        outcomes=tuple(outcomes),
    )
