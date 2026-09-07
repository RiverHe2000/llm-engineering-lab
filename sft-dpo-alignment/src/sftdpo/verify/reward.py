"""The scalar that stands in for a human preference label.

DPO needs a `(chosen, rejected)` ordering over completions. This project buys that ordering
with a verifier rather than with annotators, which is only honest if the number the verifier
produces has the properties a preference label is assumed to have. Three of them are enforced
here rather than hoped for:

*Bounded.* `Reward.value` is a convex combination of three components that are each in
[0, 1], so it is in [0, 1] for every completion and every legal set of weights. The weights
are normalised by their own sum, and the numerator and denominator are summed in the same
order, so IEEE rounding — which is monotone — cannot push the result past either end.

*Monotone in correctness.* Repairing a field towards gold, or deleting one the model invented,
never lowers the reward. Without that, mining a preference pair could teach the model to
un-learn a field it had right. Adding a *wrong* field does lower it, because precision is part
of the signal, so the guarantee is stated over completions that introduce no new errors.

*Graded.* Parse and schema validity are worth 0.2 each and the field F1 the remaining 0.6,
and the field comparison runs even when the schema check failed. Most completions fail the
schema early in training; if the reward collapsed to a constant for all of them there would
be nothing to rank inside the failing majority and the preference miner would produce pairs
with no signal in them. The weights say what the task is: packaging is necessary, cheap, and
worth a fifth each; the content is the point.

The two configured verifiers exist to separate two failures a single number hides. A model
that emits the right object wrapped in prose does not know that prose is unwanted; one that
emits clean JSON with the wrong fields does not know the schema. `strict_verifier()` charges
for both, `lenient_verifier()` only for the second, and the gap between the two scores on the
same completions is a headline number for the project. Because lenient parsing is a superset
of strict parsing, the lenient score is never the lower of the two.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from sftdpo.schemas import (
    AdviceRecord,
    Example,
    FieldScore,
    Reward,
    Sample,
    SchemaViolation,
    ViolationKind,
)
from sftdpo.verify.fields import RecordLike, f1_from_scores, field_scores
from sftdpo.verify.parse import extract_json
from sftdpo.verify.schema_check import check

__all__ = [
    "DEFAULT_FIELD_WEIGHT",
    "DEFAULT_PARSE_WEIGHT",
    "DEFAULT_SCHEMA_WEIGHT",
    "RewardBreakdown",
    "Verifier",
    "lenient_verifier",
    "strict_verifier",
]

DEFAULT_PARSE_WEIGHT: Final = 0.2
DEFAULT_SCHEMA_WEIGHT: Final = 0.2
DEFAULT_FIELD_WEIGHT: Final = 0.6


class Verifier:
    """Parse, then schema check, then field comparison, combined into one `Reward`.

    The three stages are kept separate on the way out — `Reward` carries `parsed`,
    `schema_valid`, `field_f1` and the per-field scores alongside the scalar — so that a
    change in the headline number can always be attributed to a stage rather than argued
    about.
    """

    def __init__(
        self,
        *,
        strict: bool,
        parse_weight: float = DEFAULT_PARSE_WEIGHT,
        schema_weight: float = DEFAULT_SCHEMA_WEIGHT,
        field_weight: float = DEFAULT_FIELD_WEIGHT,
    ) -> None:
        """Configure a verifier.

        Args:
            strict: `True` to require the completion to be exactly one JSON object, `False`
                to allow the parser to recover it from prose and syntax slips. There is no
                default because the score depends on it.
            parse_weight: Credit for producing a parseable object at all.
            schema_weight: Credit for that object satisfying `AdviceRecord`.
            field_weight: Credit, pro rata, for the fields matching gold.

        The weights are relative: they are divided by their own sum, so `(1, 1, 3)` and
        `(0.2, 0.2, 0.6)` are the same verifier. That is what keeps the bound on `value`
        true for weights a caller invented for an ablation.

        Raises:
            ValueError: If any weight is negative or not finite, or if they sum to zero,
                which would leave the reward undefined.
        """
        weights = (parse_weight, schema_weight, field_weight)
        if not all(math.isfinite(weight) for weight in weights):
            msg = f"weights must be finite, got {weights}"
            raise ValueError(msg)
        if any(weight < 0 for weight in weights):
            msg = f"weights must be non-negative, got {weights}"
            raise ValueError(msg)
        # Summed in exactly the order the numerator uses, so the numerator can never exceed
        # the denominator by a rounding step.
        total = parse_weight + schema_weight + field_weight
        if total == 0:
            msg = "at least one weight must be positive"
            raise ValueError(msg)

        self.strict = strict
        self.parse_weight = parse_weight
        self.schema_weight = schema_weight
        self.field_weight = field_weight
        self._total = total

    @property
    def weights(self) -> tuple[float, float, float]:
        """The parse, schema and field weights as the fractions of the reward they carry."""
        return (
            self.parse_weight / self._total,
            self.schema_weight / self._total,
            self.field_weight / self._total,
        )

    def score(self, text: str, gold: AdviceRecord) -> Reward:
        """Judge one completion against one gold record.

        Args:
            text: The raw model completion, exactly as generated.
            gold: The reference record.

        Returns:
            A `Reward` whose components explain its `value`. A completion that did not parse
            is still given a full set of field scores — every gold path missing — so that
            downstream reporting can treat every reward the same way.
        """
        outcome = extract_json(text, strict=self.strict)
        if outcome.value is None:
            # `ParseOutcome` guarantees that a missing value comes with a violation; the
            # conditional keeps the type checker satisfied without an assertion.
            parse_violation = () if outcome.violation is None else (outcome.violation,)
            return self._reward(
                parsed=False,
                schema_valid=False,
                scores=field_scores(None, gold),
                violations=parse_violation,
            )

        record, violations = check(outcome.value)
        # The raw object is scored when validation failed, because a completion that got
        # seven fields right and invented an eighth key must not score the same as one that
        # got nothing right.
        candidate: RecordLike = outcome.value if record is None else record
        return self._reward(
            parsed=True,
            schema_valid=record is not None,
            scores=field_scores(candidate, gold),
            violations=violations,
        )

    def score_many(self, samples: Sequence[Sample], examples: Iterable[Example]) -> list[Reward]:
        """Score a batch of completions against the examples they came from.

        Args:
            samples: The completions to score.
            examples: The examples they refer to, in any order.

        Returns:
            One `Reward` per sample, positionally aligned with `samples`, so a caller can zip
            the two together. The ordering is the caller's own and is never rearranged: a
            reordering here would silently mislabel a preference pair.

        Raises:
            ValueError: If two examples share an `example_id`, or if a sample refers to an
                example that was not supplied. Both are dataset faults, and dropping the
                sample instead would misalign every reward after it.
        """
        golds: dict[str, AdviceRecord] = {}
        for example in examples:
            if example.example_id in golds:
                msg = f"duplicate example_id {example.example_id!r}"
                raise ValueError(msg)
            golds[example.example_id] = example.gold

        rewards: list[Reward] = []
        for sample in samples:
            gold = golds.get(sample.example_id)
            if gold is None:
                msg = f"no example for sample {sample.example_id!r}"
                raise ValueError(msg)
            rewards.append(self.score(sample.text, gold))
        return rewards

    def _reward(
        self,
        *,
        parsed: bool,
        schema_valid: bool,
        scores: tuple[FieldScore, ...],
        violations: tuple[SchemaViolation, ...],
    ) -> Reward:
        """Assemble the reward, and compute the one number everything else is judged by."""
        field_f1 = f1_from_scores(scores)
        numerator = (
            self.parse_weight * float(parsed)
            + self.schema_weight * float(schema_valid)
            + self.field_weight * field_f1
        )
        return Reward(
            parsed=parsed,
            schema_valid=schema_valid,
            field_f1=field_f1,
            # An exact match is the whole record, not a good enough one: every path on both
            # sides agrees after normalisation, and the object was schema-valid. Getting every
            # field right while inventing a ninth is not an exact match, because the invented
            # path is a path that does not agree.
            exact_match=schema_valid and bool(scores) and all(score.correct for score in scores),
            value=numerator / self._total,
            violations=violations,
            fields=scores,
        )


def strict_verifier() -> Verifier:
    """The verifier the training targets: the completion must be one JSON object, alone.

    This is the number to quote. Anything a repair recovered is behaviour the deployed system
    would still have to work around.
    """
    return Verifier(strict=True)


def lenient_verifier() -> Verifier:
    """The same standard applied to the object, whatever packaging it arrived in.

    Scoring the same completions with both verifiers separates a model that does not know the
    schema from one that only does not know that prose is unwanted.
    """
    return Verifier(strict=False)


class RewardBreakdown(BaseModel):
    """What a set of rewards says about a run.

    Reported instead of a single mean because the mean is the one number that cannot be acted
    on: a run whose parse rate is 0.4 and a run whose field F1 is 0.4 need opposite work.
    """

    model_config = ConfigDict(frozen=True)

    count: int = 0
    parse_rate: float = 0.0
    schema_valid_rate: float = 0.0
    mean_field_f1: float = 0.0
    exact_match_rate: float = 0.0
    mean_value: float = 0.0
    violations: dict[ViolationKind, int] = Field(default_factory=dict)

    @classmethod
    def over(cls, rewards: Iterable[Reward]) -> RewardBreakdown:
        """Summarise many rewards.

        Args:
            rewards: The rewards to summarise, in any order.

        Returns:
            The rates and means, plus the distribution over `ViolationKind`. Each failing
            completion contributes exactly one vote, for its headline violation, so the
            distribution sums to the number of failures and one completion with six missing
            fields cannot outvote six completions that each missed one. Kinds nobody hit are
            left out, and the rest appear most severe first.

            An empty run reports zero for every rate; `count` is what says the run was empty.
        """
        scored = list(rewards)
        counted = Counter(
            reward.headline_violation for reward in scored if reward.headline_violation is not None
        )
        distribution = {kind: counted[kind] for kind in ViolationKind if counted[kind]}
        if not scored:
            return cls(violations=distribution)

        total = len(scored)
        return cls(
            count=total,
            parse_rate=sum(reward.parsed for reward in scored) / total,
            schema_valid_rate=sum(reward.schema_valid for reward in scored) / total,
            mean_field_f1=sum(reward.field_f1 for reward in scored) / total,
            exact_match_rate=sum(reward.exact_match for reward in scored) / total,
            mean_value=sum(reward.value for reward in scored) / total,
            violations=distribution,
        )

    @property
    def failures(self) -> int:
        """How many completions carried at least one violation."""
        return sum(self.violations.values())
