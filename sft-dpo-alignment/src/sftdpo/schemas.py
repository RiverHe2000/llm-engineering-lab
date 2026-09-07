"""Shared vocabulary for the whole package.

Nothing here imports anything else from `sftdpo`: this is the bottom of the dependency
graph, so the data generator, the verifier, the trainers and the evaluator can be built
and tested independently.

The task is deliberately narrow — turn an adviser's free-text note into one JSON object
that satisfies a fixed schema — because that is where a small instruct model measurably
fails. A companion project of mine (`llm-app-ops-loop`) has a promotion gate that refused
to promote either of two candidate prompts because both scored 0/2 on the cases requiring
JSON output. Prompt engineering did not move it. This package asks whether training does,
and holds the answer to the same paired statistics.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Action",
    "AdviceRecord",
    "Example",
    "Fees",
    "FieldScore",
    "GenerationConfig",
    "PreferencePair",
    "Recommendation",
    "Reward",
    "RiskProfile",
    "Sample",
    "SchemaViolation",
    "Slice",
    "Split",
    "ViolationKind",
    "advice_record_json_schema",
]


# --------------------------------------------------------------------------------------
# The target structure
# --------------------------------------------------------------------------------------


class RiskProfile(StrEnum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    BALANCED = "balanced"
    GROWTH = "growth"
    HIGH_GROWTH = "high_growth"


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    SWITCH = "switch"


class Recommendation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    product: str
    action: Action
    amount: float | None = None


class Fees(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    advice_fee: float
    ongoing_fee_pct: float


class AdviceRecord(BaseModel):
    """The object the model must emit.

    Eight top-level fields, three of them compound: two lists and a nested object. Counting
    the fields inside them there are thirteen distinct names, and the deepest path is three
    levels (`recommendations[i].product`). That shape is the point -- a flat object of five
    strings would be solved by pattern matching, and the errors worth measuring only appear
    once a model has to keep a list of objects consistent while it writes.

    `extra="forbid"` matters: a model that invents a plausible extra key has not followed
    the schema, and a permissive validator would score that as success.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_name: str
    record_date: str = Field(description="ISO-8601 date, YYYY-MM-DD")
    risk_profile: RiskProfile
    objectives: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    fees: Fees
    review_months: int
    flags: list[str] = Field(default_factory=list)


def advice_record_json_schema() -> dict[str, Any]:
    """The JSON Schema shown to the model in the prompt.

    Generated from the pydantic model rather than written by hand, so the schema the
    model is asked to follow and the schema the verifier enforces cannot drift apart.
    """
    return AdviceRecord.model_json_schema()


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


class Slice(StrEnum):
    """Difficulty slices. Reported separately so an average cannot hide a regression.

    A model can lift its overall score while getting worse on, say, absent fields; the
    slice breakdown is what makes that visible.
    """

    CLEAN = "clean"
    DISTRACTOR = "distractor"
    MIXED_FORMATS = "mixed_formats"
    ABSENT_FIELDS = "absent_fields"
    LONG_CONTEXT = "long_context"
    MANY_ITEMS = "many_items"


Split = Literal["train", "val", "test"]


class Example(BaseModel):
    """One (note, gold record) pair."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    split: Split
    slice: Slice
    note: str
    gold: AdviceRecord

    @property
    def gold_json(self) -> str:
        return self.gold.model_dump_json(indent=None)


class GenerationConfig(BaseModel):
    """Decoding settings recorded alongside every sample, so a run is reproducible."""

    model_config = ConfigDict(frozen=True)

    max_new_tokens: int = 320
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


class Sample(BaseModel):
    """One completion produced by one model for one example."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    model: str
    text: str
    sample_index: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


# --------------------------------------------------------------------------------------
# Verification and reward
# --------------------------------------------------------------------------------------


class ViolationKind(StrEnum):
    """Why a completion failed, most severe first.

    The order matters: `reward.py` reports the first violation as the headline cause, and
    a model that cannot produce parseable JSON has a different problem from one that
    parses but mislabels an enum.
    """

    NO_JSON = "no_json"
    UNPARSEABLE = "unparseable"
    NOT_AN_OBJECT = "not_an_object"
    MISSING_FIELD = "missing_field"
    EXTRA_FIELD = "extra_field"
    WRONG_TYPE = "wrong_type"
    BAD_ENUM = "bad_enum"
    BAD_DATE = "bad_date"
    OUT_OF_RANGE = "out_of_range"


class SchemaViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ViolationKind
    path: str = ""
    detail: str = ""


class FieldScore(BaseModel):
    """Per-field comparison against gold, after normalisation."""

    model_config = ConfigDict(frozen=True)

    path: str
    correct: bool
    expected: str
    actual: str


class Reward(BaseModel):
    """The verifier's judgement of one completion.

    `value` is the scalar used to rank completions when mining preference pairs. The
    components are kept so a change in the reward can always be attributed.
    """

    model_config = ConfigDict(frozen=True)

    parsed: bool
    schema_valid: bool
    field_f1: float
    exact_match: bool
    value: float
    violations: tuple[SchemaViolation, ...] = ()
    fields: tuple[FieldScore, ...] = ()

    @property
    def headline_violation(self) -> ViolationKind | None:
        return self.violations[0].kind if self.violations else None


class PreferencePair(BaseModel):
    """A (chosen, rejected) pair mined from sampled completions.

    The label comes from the verifier, not from a human or a judge model, so the
    preference signal is exactly as trustworthy as the schema — which is to say, exact.
    """

    model_config = ConfigDict(frozen=True)

    example_id: str
    slice: Slice
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float

    @property
    def margin(self) -> float:
        return self.chosen_reward - self.rejected_reward
