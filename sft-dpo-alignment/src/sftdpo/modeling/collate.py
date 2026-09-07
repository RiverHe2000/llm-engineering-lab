"""Hand-written causal-LM collators with completion-only supervision.

Written out rather than delegated to a library because prompt masking is where a
fine-tuning run goes wrong invisibly: if the mask is off by one the loss still falls, the
tokens still look fluent, and the model has been taught to reproduce the prompt. The three
invariants that matter are asserted in the tests -- nothing before the completion is
supervised, the number of supervised positions equals the completion length, and padding is
never supervised.

Labels are aligned with `input_ids`, not shifted: every causal-LM head in `transformers`
does the shift internally, so masking position `i` means "do not ask the model to predict
token `i`". Masking the prompt therefore makes the first supervised prediction the first
token of the answer, conditioned on the whole prompt.

Truncation is deliberately unforgiving. An example longer than `max_length` has its tail
cut, which costs the end of a JSON object; an example whose *prompt* alone exceeds
`max_length` is dropped and counted, because the alternative -- cutting the note the answer
depends on -- teaches the model to invent the missing fields.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from sftdpo.modeling.chat import ChatFormatter
from sftdpo.schemas import PreferencePair

__all__ = [
    "DEFAULT_MAX_SEQ_LENGTH",
    "IGNORE_INDEX",
    "CollationReport",
    "CompletionOnlyCollator",
    "PairFeature",
    "PreferenceCollator",
    "SFTFeature",
    "encode_preference_pair",
    "encode_sft_example",
]

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

DEFAULT_MAX_SEQ_LENGTH = 2048
"""Token budget per sequence, chosen by measuring the corpus rather than by habit.

The prompt carries the whole JSON Schema, which is most of a thousand tokens before the
adviser's note is added, so nothing this project generates is short. Measured with the
Qwen2.5 tokenizer over a 320-example corpus, prompt lengths run from a median of 1 072
tokens on the shortest slice to a maximum of 1 726 on `long_context`, and prompt plus gold
answer reaches 1 964. A 1 024-token budget — the value that looks unremarkable in a
fine-tuning script — therefore truncates *every single example*, and on a preference pair
it truncates both sides away and leaves nothing to prefer.

That failure is loud here only because the collators count what they drop. Set the budget
too low in a pipeline that pads silently and the run still completes, still reports a
falling loss, and has trained on prompts with their answers cut off. The number is 2 048
because that is the next power of two above the measured maximum; `test_collate.py` pins
the relationship, and the corpus-fitting check lives in `test_dataset.py` behind the
`network` marker, since CI has no real tokenizer offline.
"""


@dataclass(slots=True)
class CollationReport:
    """Running counts of what collation did to the data.

    Kept on the collator instead of returned, because a `transformers` data collator must
    return exactly the model's keyword arguments. A run that silently dropped a third of its
    training set is a result worth reading off at the end, so the counters are cumulative.
    """

    seen: int = 0
    truncated: int = 0
    dropped: int = 0

    @property
    def drop_rate(self) -> float:
        """Fraction of examples discarded, zero when nothing has been seen."""
        return self.dropped / self.seen if self.seen else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Manifest-friendly view of the counters."""
        return {
            "seen": self.seen,
            "truncated": self.truncated,
            "dropped": self.dropped,
            "drop_rate": self.drop_rate,
        }

    def __str__(self) -> str:
        return (
            f"{self.seen} seen, {self.truncated} truncated, "
            f"{self.dropped} dropped ({self.drop_rate:.2%})"
        )


@dataclass(frozen=True, slots=True)
class SFTFeature:
    """One tokenised example: the full sequence plus where the answer starts.

    Storing the boundary as a token count rather than storing prompt and completion
    separately keeps the feature a single contiguous sequence, which is what the model
    consumes, and makes the masking rule a slice.
    """

    input_ids: tuple[int, ...]
    prompt_len: int

    def __post_init__(self) -> None:
        if not self.input_ids:
            raise ValueError("input_ids must not be empty")
        if self.prompt_len < 0:
            raise ValueError(f"prompt_len must not be negative, got {self.prompt_len}")
        if self.prompt_len >= len(self.input_ids):
            raise ValueError(
                f"prompt_len {self.prompt_len} leaves no completion tokens in a sequence of "
                f"{len(self.input_ids)}"
            )

    @property
    def total_len(self) -> int:
        return len(self.input_ids)

    @property
    def completion_len(self) -> int:
        return len(self.input_ids) - self.prompt_len

    @property
    def prompt_ids(self) -> tuple[int, ...]:
        return self.input_ids[: self.prompt_len]

    @property
    def completion_ids(self) -> tuple[int, ...]:
        return self.input_ids[self.prompt_len :]

    @classmethod
    def from_ids(cls, input_ids: Sequence[int], prompt_len: int) -> SFTFeature:
        """Build from any integer sequence, e.g. the list a tokenizer returns."""
        return cls(tuple(int(token) for token in input_ids), int(prompt_len))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SFTFeature:
        """Build from a dataset row, the shape a `datasets.Dataset` yields."""
        missing = sorted({"input_ids", "prompt_len"} - set(mapping))
        if missing:
            raise ValueError(f"feature mapping is missing {missing}")
        return cls.from_ids(mapping["input_ids"], mapping["prompt_len"])


@dataclass(frozen=True, slots=True)
class PairFeature:
    """A DPO training pair: two completions of one prompt.

    The two sides are kept as full `SFTFeature`s rather than as a shared prompt plus two
    tails, because a tokenizer may merge the boundary differently on each side; the shared
    prefix is then a derived quantity (`shared_prompt_len`) rather than an assumption.
    """

    chosen: SFTFeature
    rejected: SFTFeature

    def __post_init__(self) -> None:
        shared = self.shared_prompt_len
        if self.chosen.prompt_ids[:shared] != self.rejected.prompt_ids[:shared]:
            raise ValueError(
                "chosen and rejected must be completions of the same prompt, but their "
                "prompt tokens differ"
            )

    @property
    def shared_prompt_len(self) -> int:
        return min(self.chosen.prompt_len, self.rejected.prompt_len)

    @classmethod
    def from_shared(
        cls,
        prompt_ids: Sequence[int],
        chosen_ids: Sequence[int],
        rejected_ids: Sequence[int],
    ) -> PairFeature:
        """Build from one prompt and two completion token sequences."""
        prompt = tuple(int(token) for token in prompt_ids)
        return cls(
            chosen=SFTFeature(prompt + tuple(int(t) for t in chosen_ids), len(prompt)),
            rejected=SFTFeature(prompt + tuple(int(t) for t in rejected_ids), len(prompt)),
        )


def encode_sft_example(
    formatter: ChatFormatter,
    tokenizer: Any,
    user_prompt: str,
    completion: str,
) -> SFTFeature:
    """Tokenise one (prompt, answer) pair into a supervised feature."""
    full_ids, prompt_len = formatter.encoded_split(tokenizer, user_prompt, completion)
    return SFTFeature(tuple(full_ids), prompt_len)


def encode_preference_pair(
    formatter: ChatFormatter,
    tokenizer: Any,
    pair: PreferencePair,
) -> PairFeature:
    """Tokenise a mined preference pair into the two sequences DPO compares."""
    return PairFeature(
        chosen=encode_sft_example(formatter, tokenizer, pair.prompt, pair.chosen),
        rejected=encode_sft_example(formatter, tokenizer, pair.prompt, pair.rejected),
    )


def _validate_settings(
    *,
    pad_token_id: int,
    max_length: int,
    label_pad_token_id: int,
    pad_to_multiple_of: int | None,
) -> None:
    if pad_token_id < 0:
        raise ValueError(f"pad_token_id must be a real token id, got {pad_token_id}")
    if max_length < 2:
        raise ValueError(
            f"max_length must leave room for at least one prompt and one completion token, "
            f"got {max_length}"
        )
    if label_pad_token_id >= 0:
        raise ValueError(
            "label_pad_token_id must be negative so cross-entropy ignores it; "
            f"got {label_pad_token_id}"
        )
    if pad_to_multiple_of is not None and pad_to_multiple_of < 1:
        raise ValueError(f"pad_to_multiple_of must be positive, got {pad_to_multiple_of}")


def _as_feature(feature: SFTFeature | Mapping[str, Any]) -> SFTFeature:
    if isinstance(feature, SFTFeature):
        return feature
    if isinstance(feature, Mapping):
        return SFTFeature.from_mapping(feature)
    raise TypeError(f"expected an SFTFeature or a mapping, got {type(feature).__name__}")


def _truncate(feature: SFTFeature, max_length: int, report: CollationReport) -> SFTFeature | None:
    """Cut a feature to `max_length`, or return None if that would leave nothing to learn.

    Only the `truncated` counter is touched here; the caller owns `dropped`, because one
    dropped preference pair is one drop even when both of its sides are unusable.
    """
    if feature.total_len <= max_length:
        return feature
    if feature.prompt_len >= max_length:
        logger.warning(
            "dropping example: prompt is %d tokens, max_length is %d, so no completion "
            "token would survive truncation",
            feature.prompt_len,
            max_length,
        )
        return None
    report.truncated += 1
    return SFTFeature(feature.input_ids[:max_length], feature.prompt_len)


def _pad_width(lengths: Sequence[int], pad_to_multiple_of: int | None) -> int:
    width = max(lengths)
    if pad_to_multiple_of is None:
        return width
    # Rounding the batch width up keeps every matrix multiplication on a tensor-core-
    # friendly shape; the extra columns are pure padding and never supervised.
    remainder = width % pad_to_multiple_of
    return width if remainder == 0 else width + pad_to_multiple_of - remainder


def _stack(
    features: Sequence[SFTFeature],
    *,
    width: int,
    pad_token_id: int,
    label_pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Right-pad features to `width` and build the masked label matrix."""
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for feature in features:
        pad = width - feature.total_len
        input_rows.append([*feature.input_ids, *([pad_token_id] * pad)])
        mask_rows.append([1] * feature.total_len + [0] * pad)
        label_rows.append(
            [label_pad_token_id] * feature.prompt_len
            + list(feature.completion_ids)
            + [label_pad_token_id] * pad
        )
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


@dataclass(slots=True)
class CompletionOnlyCollator:
    """Collate supervised examples, supervising the answer and nothing else.

    Right padding is used because the loss is computed from labels rather than from the
    last hidden state, so pad tokens on the right are simply ignored; left padding would
    only be needed for batched generation.

    Attributes:
        pad_token_id: Filler id for `input_ids`. Attention masks it out, so its value only
            has to be inside the vocabulary.
        max_length: Sequence budget per example.
        label_pad_token_id: Ignore index; -100 is what `CrossEntropyLoss` skips.
        pad_to_multiple_of: Optional batch-width rounding for tensor cores.
        report: Cumulative truncation and drop counters.
    """

    pad_token_id: int
    max_length: int = DEFAULT_MAX_SEQ_LENGTH
    label_pad_token_id: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None
    report: CollationReport = field(default_factory=CollationReport)

    def __post_init__(self) -> None:
        _validate_settings(
            pad_token_id=self.pad_token_id,
            max_length=self.max_length,
            label_pad_token_id=self.label_pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def _prepare(self, features: Sequence[SFTFeature | Mapping[str, Any]]) -> list[SFTFeature]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        kept: list[SFTFeature] = []
        for raw in features:
            self.report.seen += 1
            fitted = _truncate(_as_feature(raw), self.max_length, self.report)
            if fitted is None:
                self.report.dropped += 1
                continue
            kept.append(fitted)
        if not kept:
            raise ValueError(
                f"every example in this batch of {len(features)} had a prompt at least "
                f"max_length ({self.max_length}) tokens long"
            )
        return kept

    def __call__(
        self,
        features: Sequence[SFTFeature | Mapping[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Build a padded batch of `input_ids`, `attention_mask` and `labels`."""
        kept = self._prepare(features)
        width = _pad_width([f.total_len for f in kept], self.pad_to_multiple_of)
        return _stack(
            kept,
            width=width,
            pad_token_id=self.pad_token_id,
            label_pad_token_id=self.label_pad_token_id,
        )


@dataclass(slots=True)
class PreferenceCollator:
    """Collate preference pairs into the chosen/rejected batch DPO needs.

    Both halves are padded to one common width so a trainer can concatenate them into a
    single forward pass; two passes over a half-width batch would cost the same FLOPs but
    twice the Python and kernel-launch overhead, and the reference log-probabilities have to
    be computed for both halves as well.

    A pair is kept only if both of its sides survive truncation. Dropping one side would
    leave a preference with nothing to prefer, so `report.seen` and `report.dropped` count
    pairs while `report.truncated` counts individual sequences.
    """

    pad_token_id: int
    max_length: int = DEFAULT_MAX_SEQ_LENGTH
    label_pad_token_id: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None
    report: CollationReport = field(default_factory=CollationReport)

    def __post_init__(self) -> None:
        _validate_settings(
            pad_token_id=self.pad_token_id,
            max_length=self.max_length,
            label_pad_token_id=self.label_pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def _prepare(self, pairs: Sequence[PairFeature]) -> list[tuple[SFTFeature, SFTFeature]]:
        if not pairs:
            raise ValueError("cannot collate an empty batch")
        kept: list[tuple[SFTFeature, SFTFeature]] = []
        for pair in pairs:
            self.report.seen += 1
            chosen = _truncate(pair.chosen, self.max_length, self.report)
            rejected = _truncate(pair.rejected, self.max_length, self.report)
            if chosen is None or rejected is None:
                self.report.dropped += 1
                continue
            kept.append((chosen, rejected))
        if not kept:
            raise ValueError(
                f"every pair in this batch of {len(pairs)} lost a side to truncation at "
                f"max_length {self.max_length}"
            )
        return kept

    def __call__(self, pairs: Sequence[PairFeature]) -> dict[str, torch.Tensor]:
        """Build the six tensors a DPO step consumes."""
        kept = self._prepare(pairs)
        chosen = [pair[0] for pair in kept]
        rejected = [pair[1] for pair in kept]
        width = _pad_width(
            [f.total_len for f in chosen] + [f.total_len for f in rejected],
            self.pad_to_multiple_of,
        )
        batch: dict[str, torch.Tensor] = {}
        for prefix, side in (("chosen", chosen), ("rejected", rejected)):
            stacked = _stack(
                side,
                width=width,
                pad_token_id=self.pad_token_id,
                label_pad_token_id=self.label_pad_token_id,
            )
            for key, value in stacked.items():
                batch[f"{prefix}_{key}"] = value
        return batch
