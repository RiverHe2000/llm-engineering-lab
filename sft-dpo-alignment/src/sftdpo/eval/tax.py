"""The alignment-tax probe: what the fine-tune cost outside the task it was trained on.

Training a small instruct model hard on one output format is a good way to damage the thing
that made it useful in the first place. A model taught that every reply is a JSON advice
record will happily answer "name a colour" with a JSON advice record. That is not a bug in
this code and it is not a bug in the trainer -- it is the known cost of the method, and the
only dishonest thing to do with it is to leave it unmeasured while quoting the lift on the
task.

So the probe is a small held-out set of generic instructions with nothing to do with
extraction, each with a property that can be checked without a judge model and without a
network call: answer in one word, obey a stated constraint, produce a list of exactly N
items. The same model is measured before and after alignment on the same probes, and the
result is reported as a paired comparison using the same statistics as the promotion gate,
because a two-point drop over twelve probes is noise and should be reported as noise.

Two decisions about what is scored are worth stating.

*Format, not knowledge.* The one-word probes ask questions with an obvious answer, but only
the format is checked. Scoring correctness as well would confound the tax with whatever the
base model happened to know, and the quantity of interest here is whether the model still
does what it is told.

*The checks are strict and the instructions say so.* A probe that asks for a list says "one
per line", so the checker can count lines. A checker that tried to be generous about format
would be measuring its own leniency rather than the model's compliance, which is the same
mistake the strict-versus-lenient parse gap exists to expose on the main task.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from sftdpo.eval.stats import (
    DEFAULT_RESAMPLES,
    Interval,
    NonInferiority,
    discordant_counts,
    mcnemar_exact,
    non_inferiority,
    paired_bootstrap_diff,
)

__all__ = [
    "PROBES",
    "AlignmentTax",
    "Probe",
    "ProbeKind",
    "ProbeOutcome",
    "ProbeReport",
    "alignment_tax",
    "list_items",
    "run_probes",
    "score_probes",
    "word_count",
]

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_PUNCTUATION = re.compile(r"[^\w'-]+")


class ProbeKind(StrEnum):
    """What kind of instruction-following the probe tests."""

    ONE_WORD = "one_word"
    CONSTRAINT = "constraint"
    LIST_LENGTH = "list_length"


def word_count(text: str) -> int:
    """How many whitespace-separated words a reply contains."""
    return len(text.split())


def list_items(text: str) -> list[str]:
    """The list items in a reply, one per non-empty line, with bullets stripped.

    Line-based on purpose: every list probe asks for one item per line, so a reply that
    crams the items into a sentence has not followed the instruction. Being generous here
    would measure the parser rather than the model.
    """
    items: list[str] = []
    for line in text.splitlines():
        stripped = _BULLET.sub("", line).strip()
        if stripped:
            items.append(stripped)
    return items


def _is_single_word(text: str) -> bool:
    """One word, ignoring surrounding punctuation such as a full stop or quotes."""
    cleaned = _PUNCTUATION.sub(" ", text).strip()
    return word_count(cleaned) == 1


def _exactly_items(count: int) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        return len(list_items(text)) == count

    return check


def _one_of(*options: str) -> Callable[[str], bool]:
    allowed = {option.lower() for option in options}

    def check(text: str) -> bool:
        return _PUNCTUATION.sub(" ", text).strip().lower() in allowed

    return check


def _starts_with_within(prefix: str, limit: int) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        stripped = text.strip()
        return stripped.lower().startswith(prefix.lower()) and word_count(stripped) <= limit

    return check


def _within_words(limit: int) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        return 0 < word_count(text) <= limit

    return check


def _avoids_letter(letter: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        return bool(text.strip()) and letter.lower() not in text.lower()

    return check


def _exactly(expected: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        return text.strip() == expected

    return check


@dataclass(frozen=True, slots=True)
class Probe:
    """One generic instruction and the property its reply must have.

    The check is a plain callable rather than a pattern or a rule table because the
    properties have nothing in common with each other, and a rule language expressive enough
    to cover all three would be harder to read than the three functions it replaced.
    """

    probe_id: str
    kind: ProbeKind
    instruction: str
    check: Callable[[str], bool]

    def passed(self, response: str) -> bool:
        """Whether one reply satisfies this probe."""
        return self.check(response)


PROBES: tuple[Probe, ...] = (
    Probe(
        probe_id="one_word_capital",
        kind=ProbeKind.ONE_WORD,
        instruction="What is the capital city of Australia? Answer with one word only.",
        check=_is_single_word,
    ),
    Probe(
        probe_id="one_word_larger",
        kind=ProbeKind.ONE_WORD,
        instruction="Which is larger, seven or twelve? Answer with one word only.",
        check=_is_single_word,
    ),
    Probe(
        probe_id="one_word_colour",
        kind=ProbeKind.ONE_WORD,
        instruction="Name a primary colour. Reply with exactly one word and nothing else.",
        check=_is_single_word,
    ),
    Probe(
        probe_id="one_word_season",
        kind=ProbeKind.ONE_WORD,
        instruction="Which season follows winter? Reply with one word.",
        check=_is_single_word,
    ),
    Probe(
        probe_id="constraint_yes_no",
        kind=ProbeKind.CONSTRAINT,
        instruction="Reply with the single word yes or no: is the sea salty?",
        check=_one_of("yes", "no"),
    ),
    Probe(
        probe_id="constraint_prefix",
        kind=ProbeKind.CONSTRAINT,
        instruction=(
            "Begin your reply with the word Because, and use no more than twenty words in "
            "total. Why do people carry an umbrella?"
        ),
        check=_starts_with_within("because", 20),
    ),
    Probe(
        probe_id="constraint_five_words",
        kind=ProbeKind.CONSTRAINT,
        instruction="In at most five words, say what a library is for.",
        check=_within_words(5),
    ),
    Probe(
        probe_id="constraint_no_letter_e",
        kind=ProbeKind.CONSTRAINT,
        instruction=(
            "Write one short sentence about a cat without using the letter e anywhere in it."
        ),
        check=_avoids_letter("e"),
    ),
    Probe(
        probe_id="constraint_echo_number",
        kind=ProbeKind.CONSTRAINT,
        instruction="Reply with the number 42 and nothing else.",
        check=_exactly("42"),
    ),
    Probe(
        probe_id="list_three_colours",
        kind=ProbeKind.LIST_LENGTH,
        instruction="List exactly three colours, one per line, with no other text.",
        check=_exactly_items(3),
    ),
    Probe(
        probe_id="list_five_cities",
        kind=ProbeKind.LIST_LENGTH,
        instruction="List exactly five Australian cities, one per line, and nothing else.",
        check=_exactly_items(5),
    ),
    Probe(
        probe_id="list_two_habits",
        kind=ProbeKind.LIST_LENGTH,
        instruction="Write exactly two tips for saving money, one per line, no other text.",
        check=_exactly_items(2),
    ),
)
"""The held-out probes. Twelve is small on purpose: the probe set is a cost measurement, not
a benchmark, and the paired statistics report honestly how little twelve examples can
resolve."""


class ProbeOutcome(BaseModel):
    """Whether one probe was satisfied by one model."""

    model_config = ConfigDict(frozen=True)

    probe_id: str
    kind: ProbeKind
    passed: bool


class ProbeReport(BaseModel):
    """One model's behaviour on the whole probe set."""

    model_config = ConfigDict(frozen=True)

    model: str
    outcomes: tuple[ProbeOutcome, ...] = ()

    @property
    def n(self) -> int:
        """How many probes were run."""
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        """Fraction of probes satisfied; zero for an empty report."""
        if not self.outcomes:
            return 0.0
        return sum(outcome.passed for outcome in self.outcomes) / len(self.outcomes)

    def passed_by_probe(self) -> dict[str, float]:
        """Per-probe outcome as 0.0/1.0, the shape the paired statistics consume."""
        return {outcome.probe_id: float(outcome.passed) for outcome in self.outcomes}

    def per_kind(self) -> dict[ProbeKind, float]:
        """Pass rate per probe kind, in the enum's order so a table is stable."""
        grouped: dict[ProbeKind, list[ProbeOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.kind, []).append(outcome)
        return {
            kind: sum(o.passed for o in grouped[kind]) / len(grouped[kind])
            for kind in ProbeKind
            if kind in grouped
        }


def score_probes(
    responses: Mapping[str, str],
    *,
    model: str,
    probes: Sequence[Probe] = PROBES,
) -> ProbeReport:
    """Check a set of replies against the probes that produced them.

    Args:
        responses: One reply per probe, keyed by probe id.
        model: The variant label recorded on the report.
        probes: The probe set; the module's own by default.

    Returns:
        The report, with one outcome per probe in probe order.

    Raises:
        ValueError: If a probe has no reply, or a reply refers to an unknown probe. Silently
            skipping either would change the denominator of the pass rate without saying so.
    """
    known = {probe.probe_id for probe in probes}
    unknown = sorted(set(responses) - known)
    if unknown:
        raise ValueError(f"responses refer to unknown probes: {unknown}")
    missing = sorted(known - set(responses))
    if missing:
        raise ValueError(f"no response for probes: {missing}")
    return ProbeReport(
        model=model,
        outcomes=tuple(
            ProbeOutcome(
                probe_id=probe.probe_id,
                kind=probe.kind,
                passed=probe.passed(responses[probe.probe_id]),
            )
            for probe in probes
        ),
    )


def run_probes(
    respond: Callable[[Sequence[str]], Sequence[str]],
    *,
    model: str,
    probes: Sequence[Probe] = PROBES,
) -> ProbeReport:
    """Run the probes through a responder and score the replies.

    The responder is passed in rather than built here because the probes are not extraction
    examples and have no gold record, so they cannot travel through the task's own generation
    path. A caller wires in whatever decoding it uses -- in this project, a batched greedy
    call on the same model -- and this function stays free of torch.

    Args:
        respond: Takes the instructions and returns one reply each, in the same order.
        model: The variant label recorded on the report.
        probes: The probe set; the module's own by default.

    Returns:
        The scored report.

    Raises:
        ValueError: If the responder returns a different number of replies than it was given,
            which would silently misalign every reply with the wrong probe.
    """
    instructions = [probe.instruction for probe in probes]
    replies = list(respond(instructions))
    if len(replies) != len(instructions):
        raise ValueError(
            f"responder returned {len(replies)} replies for {len(instructions)} probes"
        )
    return score_probes(
        dict(zip((probe.probe_id for probe in probes), replies, strict=True)),
        model=model,
        probes=probes,
    )


class AlignmentTax(BaseModel):
    """The paired before-and-after comparison on the probe set.

    A negative `delta.point` is the alignment tax. It is a cost of the method, not a defect
    in this code: fine-tuning towards one rigid output format is expected to reduce general
    instruction-following, and the reason to measure it is that a result quoting only the
    task lift is an incomplete account of what the training did.
    """

    model_config = ConfigDict(frozen=True)

    before_model: str
    after_model: str
    n: int
    before_rate: float
    after_rate: float
    delta: Interval
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_p: float
    verdict: NonInferiority

    @property
    def tax(self) -> float:
        """How much pass rate was lost, floored at zero; zero when nothing was lost."""
        return max(0.0, self.before_rate - self.after_rate)

    @property
    def significant(self) -> bool:
        """Whether the probe set can actually resolve the change it measured."""
        return self.delta.excludes_zero

    def to_markdown(self) -> str:
        """Render the probe comparison, byte-stably, with the caveat it needs.

        The closing note is part of the report rather than something a reader is expected to
        remember: a table showing a drop invites the conclusion that something is broken, and
        the correct reading is that the method has a price.
        """
        cleaned = 0.0 if self.delta.point == 0 else self.delta.point
        return "\n".join(
            [
                f"# Alignment tax: {self.after_model} vs {self.before_model}",
                "",
                "| Quantity | Value |",
                "| --- | --- |",
                f"| Probes | {self.n} |",
                f"| Pass rate before | {self.before_rate:.4f} |",
                f"| Pass rate after | {self.after_rate:.4f} |",
                f"| Difference | {cleaned:+.4f} |",
                (
                    f"| Difference CI ({self.delta.confidence:.0%}) | "
                    f"[{self.delta.low:+.4f}, {self.delta.high:+.4f}] |"
                ),
                f"| Probes lost / gained | {self.mcnemar_b} / {self.mcnemar_c} |",
                f"| McNemar p (exact) | {self.mcnemar_p:.4f} |",
                f"| Within margin {self.verdict.margin:.4f} | "
                f"{'yes' if self.verdict.passed else 'no'} |",
                "",
                "A drop here is a cost of the alignment method, not a fault in the harness: "
                "training a small model hard on one output format is expected to reduce "
                "general instruction-following. It is reported so the lift on the task is "
                "quoted with its price attached.",
                "",
            ]
        )


def alignment_tax(
    before: ProbeReport,
    after: ProbeReport,
    *,
    margin: float = 0.05,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> AlignmentTax:
    """Compare a model's probe behaviour before and after alignment.

    Args:
        before: The report for the model as it was.
        after: The report for the aligned model, over exactly the same probes.
        margin: How much general instruction-following may be given up before the tax is
            called a regression. Non-zero by default, because some tax is expected and a
            gate that treated any loss as a failure would never promote anything.
        confidence: Coverage of the difference interval.
        n_resamples: Bootstrap resamples.
        seed: Seed for the bootstrap.

    Returns:
        The paired comparison.

    Raises:
        ValueError: If the two reports do not cover the same probes.
    """
    before_map = before.passed_by_probe()
    after_map = after.passed_by_probe()
    delta = paired_bootstrap_diff(
        before_map,
        after_map,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    lost, gained = discordant_counts(
        {key: value > 0.0 for key, value in before_map.items()},
        {key: value > 0.0 for key, value in after_map.items()},
    )
    return AlignmentTax(
        before_model=before.model,
        after_model=after.model,
        n=before.n,
        before_rate=before.pass_rate,
        after_rate=after.pass_rate,
        delta=delta,
        mcnemar_b=lost,
        mcnemar_c=gained,
        mcnemar_p=mcnemar_exact(lost, gained),
        verdict=non_inferiority(delta, margin),
    )
