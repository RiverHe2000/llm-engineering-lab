"""Turning sampled completions into (chosen, rejected) pairs, labelled by the verifier.

The usual way to build a DPO dataset is to pay people, or to ask a larger model to judge.
Both buy a preference signal whose reliability is unknown and, in the second case,
correlated with the errors of the model being trained. This project buys it from
`sftdpo.verify.reward` instead: the task has a checkable answer, so "which of these two
completions is better" is a fact rather than an opinion, and the label is exactly as
trustworthy as the schema.

The rules below all exist to stop a preference dataset that is technically well formed from
teaching the model something other than the task.

*Best against worst, then inwards.* Completions are sorted by reward and paired from the
outside in, each used at most once. The first pair therefore carries the largest margin
available for that prompt, and no completion appears as `chosen` in one pair and `rejected`
in another, which would ask the optimiser to push the same text in both directions.

*A minimum margin.* Two completions that score 0.61 and 0.60 differ by a rounding error in
one field. Training on that is training on noise, and noise in a preference dataset is worse
than absence: DPO will happily raise the log-probability gap between two texts that were
never meaningfully different.

*Never pair two identical strings.* With a deterministic verifier this is implied by the
margin rule, since identical text scores identically. It is enforced anyway and tested
against a deliberately inconsistent scorer, because "the verifier is a pure function" is an
assumption about another module, and a pair whose two sides are the same string contributes
a loss term of exactly `ln 2` with a gradient that cancels -- a silent no-op costing a slot
in every batch it lands in.

*A cap per prompt.* One prompt on which the sampler happened to produce four good and four
terrible completions can otherwise contribute eight pairs, while the prompts the model
actually finds hard contribute none. The cap keeps a single easy prompt from setting the
gradient.

*Count what could not be paired.* A prompt whose completions all scored the same is skipped,
and skipping it silently would throw away the most informative number this stage produces.
There are two very different reasons for it, so they are counted separately. `FLAT` means
the sampler found no variation at a low score: k is too small, or the temperature too cold,
or the model is uniformly bad here and needs supervised data rather than preferences.
`SATURATED` means every sample was already perfect.

*The degenerate case is the goal.* Read that last one again: as the policy improves, more
and more prompts saturate and the yield of this stage falls towards zero. A model good
enough that almost every sample is perfect produces almost no pairs. That is not a bug to be
worked around by lowering `min_margin` until pairs appear -- it is what success looks like,
and the honest response is to report the saturation rate, stop DPO on this slice, and go
find harder prompts. `MiningStats` is shaped to make that reading obvious rather than
something a reader has to infer from an empty output file.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from sftdpo.schemas import (
    Example,
    PreferencePair,
    Reward,
    Sample,
    Slice,
    ViolationKind,
)
from sftdpo.task.generate import render_prompt

__all__ = [
    "DEFAULT_MAX_PAIRS_PER_PROMPT",
    "DEFAULT_MIN_MARGIN",
    "SATURATION_TOLERANCE",
    "MiningResult",
    "MiningStats",
    "PromptOutcome",
    "RewardScorer",
    "SliceYield",
    "mine_pairs",
]

DEFAULT_MIN_MARGIN: Final = 0.015
"""Just under what a single wrong field costs on the widest record this corpus generates.

The cost of one wrong field is not a constant. The field weight is 0.6 and the F1 is taken
over leaf paths, so one wrong field is worth `0.6 / n`, and `n` depends on how many
objectives and recommendations the gold record happens to hold. Measured over the 640
examples of the shipped configuration (`build_dataset(seed=1, n_train=400, n_val=80,
n_test=160)`), `n` runs from 13 to 35 with a median of 18, which puts the cost of one wrong
field between **0.0171 and 0.0462**.

This was 0.05, and that number came from a ten-path record the generator never produces:
`_draw_facts` always draws two objectives and one flag, so the smallest real record has 13
paths. The consequence was not a slightly conservative filter but a silent floor above the
entire distribution --- a completion differing from gold by exactly one field cleared the
margin on **none** of the 640 examples, so the preference data could contain no
single-field pair at all and the DPO stage never saw the residue the supervised stage leaves
behind. A default calibrated against a fixture rather than against the data is worse than an
arbitrary one, because it looks principled.

0.015 sits under the tightest real one-field gap (0.0171) and far above float noise. It is
still a default and not a law, which is why it is an argument: a corpus with longer lists
needs a smaller one, and `test_the_default_margin_admits_one_wrong_field` measures the figure
it is calibrated against rather than restating it."""

DEFAULT_MAX_PAIRS_PER_PROMPT: Final = 2

SATURATION_TOLERANCE: Final = 1e-9
"""How close to a perfect 1.0 an all-equal group must be to count as saturated rather than
flat. A tolerance rather than equality because the reward is a weighted sum and the weights
are normalised by their own sum, so a perfect completion can land a bit under one."""


class RewardScorer(Protocol):
    """The part of `sftdpo.verify.reward.Verifier` this module actually depends on.

    Declared as a protocol rather than importing the class so that a test can inject a
    scorer which breaks the assumptions -- most usefully one that gives two identical
    strings different rewards, which is what proves the duplicate-text guard is a real
    check and not an accident of the verifier being deterministic.
    """

    def score_many(self, samples: Sequence[Sample], examples: Iterable[Example]) -> list[Reward]:
        """Score each sample against the example it names, positionally aligned."""


class PromptOutcome(StrEnum):
    """What mining managed to do with one prompt's completions.

    Every prompt lands in exactly one of these, so the counts partition the prompts seen and
    a yield of zero always has a stated cause.
    """

    PAIRED = "paired"
    NO_PAIR = "no_pair"
    SATURATED = "saturated"
    FLAT = "flat"
    SINGLE_SAMPLE = "single_sample"


class SliceYield(BaseModel):
    """What one difficulty slice contributed.

    Broken out because the slices are the reason the corpus is built the way it is: a mining
    run that draws every pair from `many_items` is about to train a model that is better at
    long lists and no better at anything else, and the mean over all prompts cannot say so.
    """

    model_config = ConfigDict(frozen=True)

    prompts: int = 0
    prompts_with_pairs: int = 0
    pairs: int = 0
    mean_margin: float = 0.0

    @property
    def pairs_per_prompt(self) -> float:
        """Pairs mined per prompt sampled in this slice."""
        return self.pairs / self.prompts if self.prompts else 0.0

    @property
    def prompt_yield(self) -> float:
        """Fraction of this slice's prompts that produced at least one pair."""
        return self.prompts_with_pairs / self.prompts if self.prompts else 0.0


class MiningStats(BaseModel):
    """The result of a mining run, in the shape a run record should quote it.

    Every field here is either a number someone will ask about or a number that explains a
    disappointing one. In particular `saturated_prompts` and `flat_prompts` are the
    difference between "the policy has outgrown this data" and "the sampler found no
    signal", which are opposite situations that a bare pair count cannot distinguish.
    """

    model_config = ConfigDict(frozen=True)

    prompts: int = 0
    prompts_with_pairs: int = 0
    pairs: int = 0
    mean_margin: float = 0.0
    saturated_prompts: int = 0
    flat_prompts: int = 0
    no_pair_prompts: int = 0
    single_sample_prompts: int = 0
    identical_text_rejections: int = 0
    pairs_dropped_by_cap: int = 0
    rejected_violations: dict[ViolationKind, int] = Field(default_factory=dict)
    rejected_without_violation: int = 0
    slices: dict[Slice, SliceYield] = Field(default_factory=dict)

    @property
    def prompt_yield(self) -> float:
        """Fraction of prompts that produced at least one pair; zero for an empty run."""
        return self.prompts_with_pairs / self.prompts if self.prompts else 0.0

    @property
    def pairs_per_prompt(self) -> float:
        """Pairs mined per prompt sampled; zero for an empty run."""
        return self.pairs / self.prompts if self.prompts else 0.0

    @property
    def saturation_rate(self) -> float:
        """Fraction of prompts on which every sample was already perfect.

        The headline number for deciding whether another round of DPO on this data is worth
        the GPU time. A high value here with a low pair count is a finished pipeline, not a
        broken one.
        """
        return self.saturated_prompts / self.prompts if self.prompts else 0.0

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view, for writing straight into a run manifest."""
        payload = self.model_dump(mode="json")
        payload["prompt_yield"] = self.prompt_yield
        payload["pairs_per_prompt"] = self.pairs_per_prompt
        payload["saturation_rate"] = self.saturation_rate
        return payload


@dataclass(frozen=True, slots=True)
class MiningResult:
    """The mined pairs and everything needed to explain how many there are.

    The counters are collected during mining rather than recomputed from `pairs`, because
    the interesting numbers are all about completions that did *not* become a pair and are
    therefore not in the output at all.

    Attributes:
        pairs: The mined pairs, grouped by prompt in first-seen order and, within a prompt,
            in descending margin.
        outcomes: How many prompts fell into each `PromptOutcome`.
        prompts_by_slice: How many prompts were seen per difficulty slice.
        rejected_violations: Headline `ViolationKind` of each pair's rejected completion.
            This is the distribution that says what the model is being taught to stop doing.
        rejected_without_violation: Pairs whose rejected side broke no schema rule and was
            simply less correct. They carry no violation to count, and leaving them out of
            the distribution silently would make it look as though every rejected completion
            was malformed.
        identical_text_rejections: Candidate pairs discarded because both sides were the
            same string.
        pairs_dropped_by_cap: Candidate pairs that cleared the margin but exceeded
            `max_pairs_per_prompt`. Reported so it is visible when the cap is what is
            limiting the dataset.
    """

    pairs: list[PreferencePair] = field(default_factory=list)
    outcomes: dict[PromptOutcome, int] = field(default_factory=dict)
    prompts_by_slice: dict[Slice, int] = field(default_factory=dict)
    rejected_violations: dict[ViolationKind, int] = field(default_factory=dict)
    rejected_without_violation: int = 0
    identical_text_rejections: int = 0
    pairs_dropped_by_cap: int = 0

    def __post_init__(self) -> None:
        """Check the two accounting invariants that make the statistics trustworthy.

        Raises:
            ValueError: If the slice counts do not add up to the prompts seen, or if the
                rejected-violation votes do not add up to the pairs. Either would mean a
                reported percentage has a different denominator from the one its name
                implies, which is the kind of error that survives review.
        """
        if sum(self.prompts_by_slice.values()) != self.prompts_seen:
            raise ValueError(
                f"slice counts sum to {sum(self.prompts_by_slice.values())} but "
                f"{self.prompts_seen} prompts were seen"
            )
        votes = sum(self.rejected_violations.values()) + self.rejected_without_violation
        if votes != len(self.pairs):
            raise ValueError(
                f"rejected-side votes sum to {votes} but there are {len(self.pairs)} pairs"
            )

    @property
    def prompts_seen(self) -> int:
        """How many distinct prompts had at least one completion to consider."""
        return sum(self.outcomes.values())

    @property
    def prompts_with_pairs(self) -> int:
        """How many prompts produced at least one pair."""
        return self.outcomes.get(PromptOutcome.PAIRED, 0)

    @property
    def mean_margin(self) -> float:
        """Mean reward gap across the mined pairs; zero when there are none."""
        return sum(pair.margin for pair in self.pairs) / len(self.pairs) if self.pairs else 0.0

    def stats(self) -> MiningStats:
        """Summarise the run.

        Returns:
            The counts, the mean margin, the distribution over the rejected completions'
            headline violations, and the per-slice yield. Slices with no prompts are left
            out; slices with prompts but no pairs are kept, because a slice that yielded
            nothing is the most interesting row in the table.
        """
        pairs_by_slice: Counter[Slice] = Counter()
        margins: dict[Slice, float] = {}
        paired_prompts: dict[Slice, set[str]] = {}
        for pair in self.pairs:
            pairs_by_slice[pair.slice] += 1
            margins[pair.slice] = margins.get(pair.slice, 0.0) + pair.margin
            paired_prompts.setdefault(pair.slice, set()).add(pair.example_id)

        slices: dict[Slice, SliceYield] = {}
        for name in Slice:
            prompts = self.prompts_by_slice.get(name, 0)
            if not prompts:
                continue
            count = pairs_by_slice[name]
            slices[name] = SliceYield(
                prompts=prompts,
                prompts_with_pairs=len(paired_prompts.get(name, ())),
                pairs=count,
                mean_margin=margins.get(name, 0.0) / count if count else 0.0,
            )

        return MiningStats(
            prompts=self.prompts_seen,
            prompts_with_pairs=self.prompts_with_pairs,
            pairs=len(self.pairs),
            mean_margin=self.mean_margin,
            saturated_prompts=self.outcomes.get(PromptOutcome.SATURATED, 0),
            flat_prompts=self.outcomes.get(PromptOutcome.FLAT, 0),
            no_pair_prompts=self.outcomes.get(PromptOutcome.NO_PAIR, 0),
            single_sample_prompts=self.outcomes.get(PromptOutcome.SINGLE_SAMPLE, 0),
            identical_text_rejections=self.identical_text_rejections,
            pairs_dropped_by_cap=self.pairs_dropped_by_cap,
            rejected_violations=dict(self.rejected_violations),
            rejected_without_violation=self.rejected_without_violation,
            slices=slices,
        )


def _validate_settings(*, min_margin: float, max_pairs_per_prompt: int) -> None:
    """Reject settings that would produce a dataset with nothing in it to learn.

    Raises:
        ValueError: If `min_margin` is not a finite positive number -- zero would permit
            pairing two equally scored completions, which is the one thing this module
            exists to avoid -- or if the cap is below one.
    """
    if not math.isfinite(min_margin) or min_margin <= 0.0:
        raise ValueError(f"min_margin must be a finite positive number, got {min_margin}")
    if max_pairs_per_prompt < 1:
        raise ValueError(f"max_pairs_per_prompt must be at least 1, got {max_pairs_per_prompt}")


def _index_examples(examples: Iterable[Example]) -> dict[str, Example]:
    """Index examples by id.

    Raises:
        ValueError: On a duplicate id, which would silently score half the samples against
            the wrong gold record.
    """
    index: dict[str, Example] = {}
    for example in examples:
        if example.example_id in index:
            raise ValueError(f"duplicate example_id {example.example_id!r}")
        index[example.example_id] = example
    return index


def _group_by_example(
    samples: Sequence[Sample], rewards: Sequence[Reward]
) -> dict[str, list[tuple[Sample, Reward]]]:
    """Group scored samples by prompt, keeping first-appearance order.

    Insertion order rather than sorted order so the output is a stable function of the input
    without imposing an ordering the caller did not ask for.
    """
    grouped: dict[str, list[tuple[Sample, Reward]]] = {}
    for sample, reward in zip(samples, rewards, strict=True):
        grouped.setdefault(sample.example_id, []).append((sample, reward))
    return grouped


def _ranked(
    group: Sequence[tuple[Sample, Reward]], *, seed: int, example_id: str
) -> list[tuple[Sample, Reward]]:
    """Order one prompt's completions best first, breaking ties at random but reproducibly.

    The shuffle happens before a stable sort, so equal rewards end up in a random order and
    unequal ones do not move. Without it the tie would always be broken by arrival order,
    which is `sample_index` order, and low sample indices would be over-represented as the
    chosen side for no reason other than having been generated first.

    The list is canonicalised before the shuffle rather than shuffled as it arrived, so the
    mined pairs are a function of the *set* of samples and not of the order they were handed
    over in. That is what lets a caller concatenate shards, or re-mine after a re-sort, and
    get the same dataset.

    The stream is seeded from the run seed and the example id together, so a prompt's tie
    break does not depend on how many prompts preceded it -- mining a subset of a run
    reproduces exactly the pairs that subset produced in the full run.
    """
    # A string seed is hashed with SHA-512 by `random.Random`, unlike `hash()`, whose salt
    # changes between processes.
    rng = random.Random(f"{seed}:{example_id}")
    order = sorted(group, key=lambda item: (item[0].sample_index, item[0].text))
    rng.shuffle(order)
    order.sort(key=lambda item: -item[1].value)
    return order


@dataclass(slots=True)
class _Tally:
    """Mutable counters filled in during mining and frozen into a `MiningResult` after."""

    outcomes: Counter[PromptOutcome] = field(default_factory=Counter)
    slices: Counter[Slice] = field(default_factory=Counter)
    violations: Counter[ViolationKind] = field(default_factory=Counter)
    clean_rejections: int = 0
    identical: int = 0
    capped: int = 0


def _pairs_for_prompt(
    order: Sequence[tuple[Sample, Reward]],
    example: Example,
    *,
    min_margin: float,
    max_pairs_per_prompt: int,
    tally: _Tally,
) -> list[PreferencePair]:
    """Pair one prompt's ranked completions from the outside in.

    Args:
        order: The completions, best reward first.
        example: The example they answer, for the prompt text and the slice.
        min_margin: Smallest reward gap worth training on.
        max_pairs_per_prompt: Cap on emitted pairs.
        tally: Counters to update for rejections and capped candidates.

    Returns:
        The pairs, in descending margin.
    """
    prompt = render_prompt(example.note)
    pairs: list[PreferencePair] = []
    head, tail = 0, len(order) - 1
    while head < tail:
        chosen, chosen_reward = order[head]
        rejected, rejected_reward = order[tail]
        # Moving inwards can only shrink the gap, because the list is sorted, so the first
        # candidate under the threshold means every remaining candidate is too.
        if chosen_reward.value - rejected_reward.value < min_margin:
            break
        head, tail = head + 1, tail - 1
        if chosen.text == rejected.text:
            tally.identical += 1
            continue
        if len(pairs) >= max_pairs_per_prompt:
            tally.capped += 1
            continue
        kind = rejected_reward.headline_violation
        if kind is None:
            tally.clean_rejections += 1
        else:
            tally.violations[kind] += 1
        pairs.append(
            PreferencePair(
                example_id=example.example_id,
                slice=example.slice,
                prompt=prompt,
                chosen=chosen.text,
                rejected=rejected.text,
                chosen_reward=chosen_reward.value,
                rejected_reward=rejected_reward.value,
            )
        )
    return pairs


def _classify_flat(group: Sequence[tuple[Sample, Reward]]) -> PromptOutcome:
    """Decide why a prompt whose completions all scored the same could not be paired."""
    value = group[0][1].value
    return PromptOutcome.SATURATED if value >= 1.0 - SATURATION_TOLERANCE else PromptOutcome.FLAT


def mine_pairs(
    samples: Sequence[Sample],
    examples: Iterable[Example],
    verifier: RewardScorer,
    *,
    min_margin: float = DEFAULT_MIN_MARGIN,
    max_pairs_per_prompt: int = DEFAULT_MAX_PAIRS_PER_PROMPT,
    seed: int = 0,
) -> MiningResult:
    """Score sampled completions and mine preference pairs from them.

    Args:
        samples: Completions to mine, in any order and from any number of prompts.
        examples: The examples they answer. Extra examples with no samples are ignored;
            they are not counted as prompts, because a prompt nobody sampled has no yield.
        verifier: The scorer that supplies the preference label.
        min_margin: Smallest reward gap worth a pair. See the module docstring for why a
            gap of nearly zero is worse than no pair at all.
        max_pairs_per_prompt: Cap on pairs from a single prompt.
        seed: Seed for tie-breaking among equally scored completions.

    Returns:
        A `MiningResult` holding the pairs and the counts that explain how many there are,
        including the prompts that could not be paired.

    Raises:
        ValueError: If `min_margin` is not a finite positive number, if
            `max_pairs_per_prompt` is below one, if two examples share an id, or if a sample
            names an example that was not supplied.
    """
    _validate_settings(min_margin=min_margin, max_pairs_per_prompt=max_pairs_per_prompt)
    index = _index_examples(examples)
    # Checked here rather than left to the verifier, which happens to raise on this too: the
    # protocol above promises a reward per sample and nothing about validation, so relying on
    # someone else's check would turn a stated error into a `KeyError` from the middle of the
    # loop the moment a caller supplies a different scorer. It also fails before scoring.
    unknown = sorted({sample.example_id for sample in samples} - index.keys())
    if unknown:
        raise ValueError(f"samples refer to examples that were not supplied: {unknown}")
    rewards = verifier.score_many(samples, index.values())
    grouped = _group_by_example(samples, rewards)

    tally = _Tally()
    mined: list[PreferencePair] = []
    for example_id, group in grouped.items():
        example = index[example_id]
        tally.slices[example.slice] += 1
        if len(group) < 2:
            # A prompt sampled once is all-equal by definition. Counted apart from the
            # genuinely flat ones so that a k=1 run cannot be mistaken for a saturated model.
            tally.outcomes[PromptOutcome.SINGLE_SAMPLE] += 1
            continue

        order = _ranked(group, seed=seed, example_id=example_id)
        if order[0][1].value - order[-1][1].value <= 0.0:
            tally.outcomes[_classify_flat(order)] += 1
            continue

        pairs = _pairs_for_prompt(
            order,
            example,
            min_margin=min_margin,
            max_pairs_per_prompt=max_pairs_per_prompt,
            tally=tally,
        )
        tally.outcomes[PromptOutcome.PAIRED if pairs else PromptOutcome.NO_PAIR] += 1
        mined.extend(pairs)

    return MiningResult(
        pairs=mined,
        outcomes=dict(tally.outcomes),
        prompts_by_slice=dict(tally.slices),
        rejected_violations={
            kind: tally.violations[kind] for kind in ViolationKind if tally.violations[kind]
        },
        rejected_without_violation=tally.clean_rejections,
        identical_text_rejections=tally.identical,
        pairs_dropped_by_cap=tally.capped,
    )
