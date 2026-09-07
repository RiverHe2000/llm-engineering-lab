"""Tests for chat formatting and the causal-LM collators.

Everything here runs against tiny deterministic stand-in tokenizers defined in this file.
CI has no Qwen tokenizer available offline, and more importantly a fake tokenizer lets the
awkward cases be constructed on purpose: a template that merges the prompt/completion
boundary into one token, a template that raises, a template that is not prefix-consistent.
Those are the cases where masking silently goes wrong on a real run.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.modeling.chat import (
    DEFAULT_SYSTEM_PROMPT,
    IM_END,
    IM_START,
    ChatFormatter,
    ChatTemplateError,
    common_prefix_length,
    encode_text,
    render_chatml,
)
from sftdpo.modeling.collate import (
    IGNORE_INDEX,
    CollationReport,
    CompletionOnlyCollator,
    PairFeature,
    PreferenceCollator,
    SFTFeature,
    encode_preference_pair,
    encode_sft_example,
)
from sftdpo.schemas import PreferencePair, Slice

USER_PROMPT = "Note: client Ada wants growth. Emit JSON."
COMPLETION = '{"client_name":"Ada","risk_profile":"growth"}'

FAKE_TEMPLATE = "[{role}]\n{content}\n"


def _stable_id(token: str, vocab_size: int) -> int:
    """Deterministic id for a token string.

    CRC32 rather than `hash()`, whose salt changes between processes: a test that depends on
    tokenisation must give the same ids on every run and every machine.
    """
    return 2 + zlib.crc32(token.encode("utf-8")) % (vocab_size - 2)


class WordTokenizer:
    """Whitespace tokeniser. Never merges across a newline, so splits land where expected."""

    def __init__(self, *, chat_template: str | None = None, vocab_size: int = 4096) -> None:
        self.chat_template = chat_template
        self.vocab_size = vocab_size
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        return [_stable_id(token, self.vocab_size) for token in text.split()]

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        if tokenize:
            raise ValueError("this stand-in only renders text")
        parts = [FAKE_TEMPLATE.format(role=m["role"], content=m["content"]) for m in messages]
        if add_generation_prompt:
            parts.append("[assistant]\n")
        return "".join(parts)


class ChunkTokenizer(WordTokenizer):
    """Fixed three-character chunks, so the prompt/completion boundary merges into a token.

    This is the pathological case the split logic has to survive: `prompt_ids` is not a
    prefix of `full_ids`, because the chunk straddling the boundary differs.
    """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        chunks = [text[i : i + 3] for i in range(0, len(text), 3)]
        return [_stable_id(chunk, self.vocab_size) for chunk in chunks]


class RaisingTemplateTokenizer(WordTokenizer):
    """A tokenizer whose template is present but broken."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        raise TypeError(
            f"jinja blew up on {len(messages)} messages "
            f"(tokenize={tokenize}, gen={add_generation_prompt})"
        )


class NonStringTemplateTokenizer(WordTokenizer):
    """A tokenizer that ignores `tokenize=False` and returns ids anyway."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> Any:
        return [len(messages), int(add_generation_prompt), int(tokenize)]


class UnrenderableTokenizer:
    """Carries a template string but has no renderer for it, as older tokenizers did."""

    chat_template = FAKE_TEMPLATE

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.seen_add_special_tokens = add_special_tokens
        return [_stable_id(token, 4096) for token in text.split()]


class NonPrefixTemplateTokenizer(WordTokenizer):
    """A template that rewrites earlier turns once an answer is appended."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        if tokenize:
            raise ValueError("this stand-in only renders text")
        if add_generation_prompt:
            return "PROMPT-ONLY\n"
        return "REWRITTEN\n" + "".join(m["content"] for m in messages)


@pytest.fixture
def tokenizer() -> WordTokenizer:
    return WordTokenizer()


@pytest.fixture
def formatter() -> ChatFormatter:
    return ChatFormatter(system="SYS")


# --------------------------------------------------------------------------------------
# chat.py: rendering
# --------------------------------------------------------------------------------------


def test_render_chatml_prompt_is_exact() -> None:
    text = render_chatml(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        add_generation_prompt=True,
    )
    assert text == (
        f"{IM_START}system\nS{IM_END}\n{IM_START}user\nU{IM_END}\n{IM_START}assistant\n"
    )


def test_render_chatml_full_closes_the_assistant_turn() -> None:
    text = render_chatml(
        [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}],
        add_generation_prompt=False,
    )
    assert text.endswith(f"{IM_START}assistant\nA{IM_END}\n")


def test_render_chatml_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown chat role"):
        render_chatml([{"role": "tool", "content": "x"}], add_generation_prompt=False)


def test_render_chatml_rejects_empty_conversation() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        render_chatml([], add_generation_prompt=True)


def test_default_system_prompt_is_used_when_none_given() -> None:
    assert DEFAULT_SYSTEM_PROMPT in ChatFormatter().prompt_text(USER_PROMPT)


def test_messages_carry_the_answer_only_when_given(formatter: ChatFormatter) -> None:
    assert [m["role"] for m in formatter.messages(USER_PROMPT)] == ["system", "user"]
    assert [m["role"] for m in formatter.messages(USER_PROMPT, COMPLETION)] == [
        "system",
        "user",
        "assistant",
    ]


def test_prompt_is_a_prefix_of_full_text_on_the_fallback(formatter: ChatFormatter) -> None:
    prompt = formatter.prompt_text(USER_PROMPT)
    assert formatter.full_text(USER_PROMPT, COMPLETION).startswith(prompt)


def test_prompt_is_a_prefix_of_full_text_on_the_template_path(
    formatter: ChatFormatter, tokenizer: WordTokenizer
) -> None:
    tokenizer.chat_template = FAKE_TEMPLATE
    prompt = formatter.prompt_text(USER_PROMPT, tokenizer=tokenizer)
    full = formatter.full_text(USER_PROMPT, COMPLETION, tokenizer=tokenizer)
    assert full.startswith(prompt)
    assert full[len(prompt) :].startswith(COMPLETION)


def test_tokenizer_template_is_preferred_over_the_fallback(
    formatter: ChatFormatter, tokenizer: WordTokenizer
) -> None:
    tokenizer.chat_template = FAKE_TEMPLATE
    text = formatter.prompt_text(USER_PROMPT, tokenizer=tokenizer)
    assert text.startswith("[system]\n")
    assert IM_START not in text


def test_tokenizer_without_a_template_falls_back(
    formatter: ChatFormatter, tokenizer: WordTokenizer
) -> None:
    assert formatter.uses_chat_template(tokenizer) is False
    assert formatter.prompt_text(USER_PROMPT, tokenizer=tokenizer).startswith(IM_START)


def test_tokenizer_with_a_template_but_no_renderer_falls_back(
    formatter: ChatFormatter,
) -> None:
    stub = UnrenderableTokenizer()
    assert formatter.uses_chat_template(stub) is False
    assert formatter.prompt_text(USER_PROMPT, tokenizer=stub).startswith(IM_START)


def test_bound_tokenizer_is_used_when_no_call_argument_is_given() -> None:
    bound = WordTokenizer(chat_template=FAKE_TEMPLATE)
    assert ChatFormatter(system="SYS", tokenizer=bound).uses_chat_template() is True


def test_call_argument_overrides_the_bound_tokenizer() -> None:
    bound = WordTokenizer(chat_template=FAKE_TEMPLATE)
    formatter = ChatFormatter(system="SYS", tokenizer=bound)
    assert formatter.prompt_text(USER_PROMPT, tokenizer=WordTokenizer()).startswith(IM_START)


def test_broken_template_raises_rather_than_silently_falling_back(
    formatter: ChatFormatter,
) -> None:
    broken = RaisingTemplateTokenizer(chat_template=FAKE_TEMPLATE)
    with pytest.raises(ChatTemplateError, match="chat template failed"):
        formatter.prompt_text(USER_PROMPT, tokenizer=broken)


def test_template_returning_non_text_raises(formatter: ChatFormatter) -> None:
    odd = NonStringTemplateTokenizer(chat_template=FAKE_TEMPLATE)
    with pytest.raises(ChatTemplateError, match="expected str"):
        formatter.full_text(USER_PROMPT, COMPLETION, tokenizer=odd)


def test_non_prefix_template_is_rejected_by_the_split(formatter: ChatFormatter) -> None:
    odd = NonPrefixTemplateTokenizer(chat_template=FAKE_TEMPLATE)
    with pytest.raises(ValueError, match="not a prefix"):
        formatter.split_lengths(odd, USER_PROMPT, COMPLETION)


# --------------------------------------------------------------------------------------
# chat.py: the prompt/completion split
# --------------------------------------------------------------------------------------


def test_encode_text_disables_extra_special_tokens(tokenizer: WordTokenizer) -> None:
    encode_text(tokenizer, "a b c")
    assert tokenizer.encode_calls == [("a b c", False)]


def test_split_lengths_partitions_the_full_sequence(
    formatter: ChatFormatter, tokenizer: WordTokenizer
) -> None:
    prompt_len, completion_len = formatter.split_lengths(tokenizer, USER_PROMPT, COMPLETION)
    full_ids = encode_text(tokenizer, formatter.full_text(USER_PROMPT, COMPLETION))
    assert prompt_len + completion_len == len(full_ids)
    assert prompt_len > 0
    assert completion_len > 0


def test_split_lengths_matches_the_encoded_prompt_when_nothing_merges(
    formatter: ChatFormatter, tokenizer: WordTokenizer
) -> None:
    prompt_len, _ = formatter.split_lengths(tokenizer, USER_PROMPT, COMPLETION)
    prompt_ids = encode_text(tokenizer, formatter.prompt_text(USER_PROMPT))
    assert prompt_len == len(prompt_ids)


@pytest.mark.parametrize("system", ["SYS", "SYSA", "SYSAB"])
def test_split_is_conservative_when_the_boundary_token_merges(system: str) -> None:
    """A chunk straddling the boundary belongs to the completion, not to the prompt.

    The three system strings shift the prompt length by one character each, so at least two
    of the three runs put the boundary inside a chunk -- the case where `prompt_ids` is not
    a prefix of `full_ids` and a naive `len(prompt_ids)` would mis-align every label.
    """
    formatter = ChatFormatter(system=system)
    chunky = ChunkTokenizer()
    prompt_text = formatter.prompt_text(USER_PROMPT)
    prompt_ids = encode_text(chunky, prompt_text)
    full_ids, prompt_len = formatter.encoded_split(chunky, USER_PROMPT, COMPLETION)
    merged = len(prompt_text) % 3 != 0
    assert prompt_len == len(prompt_text) // 3
    assert (prompt_len < len(prompt_ids)) is merged
    assert full_ids[:prompt_len] == prompt_ids[:prompt_len]
    assert len(full_ids) > prompt_len


def test_split_lengths_rejects_an_empty_completion(formatter: ChatFormatter) -> None:
    # The ChatML fallback always appends a stop token, so an answer can only tokenise to
    # nothing under a template that adds none: exactly what the stand-in template does.
    templated = WordTokenizer(chat_template=FAKE_TEMPLATE)
    with pytest.raises(ValueError, match="no tokens"):
        formatter.split_lengths(templated, USER_PROMPT, "   ")


def test_chatml_fallback_always_supervises_a_stop_token(
    formatter: ChatFormatter, tokenizer: WordTokenizer
) -> None:
    """Without a stop token in the labels the model never learns to end its answer."""
    _, completion_len = formatter.split_lengths(tokenizer, USER_PROMPT, "")
    assert completion_len == 1


@given(
    left=st.lists(st.integers(0, 5), max_size=8),
    right=st.lists(st.integers(0, 5), max_size=8),
)
def test_common_prefix_length_is_the_longest_agreeing_prefix(
    left: list[int], right: list[int]
) -> None:
    n = common_prefix_length(left, right)
    assert 0 <= n <= min(len(left), len(right))
    assert left[:n] == right[:n]
    if n < min(len(left), len(right)):
        assert left[n] != right[n]


# --------------------------------------------------------------------------------------
# collate.py: features
# --------------------------------------------------------------------------------------


def test_feature_exposes_its_two_halves() -> None:
    feature = SFTFeature((10, 11, 12, 13), 2)
    assert feature.total_len == 4
    assert feature.completion_len == 2
    assert feature.prompt_ids == (10, 11)
    assert feature.completion_ids == (12, 13)


@pytest.mark.parametrize(
    ("input_ids", "prompt_len", "message"),
    [
        ((), 0, "must not be empty"),
        ((1, 2), -1, "must not be negative"),
        ((1, 2), 2, "no completion tokens"),
        ((1, 2), 5, "no completion tokens"),
    ],
)
def test_feature_rejects_impossible_boundaries(
    input_ids: tuple[int, ...], prompt_len: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SFTFeature(input_ids, prompt_len)


def test_feature_from_ids_accepts_a_plain_list() -> None:
    assert SFTFeature.from_ids([7, 8, 9], 1) == SFTFeature((7, 8, 9), 1)


def test_feature_from_mapping_reads_a_dataset_row() -> None:
    row: dict[str, Any] = {"input_ids": [7, 8, 9], "prompt_len": 2, "extra": "ignored"}
    assert SFTFeature.from_mapping(row) == SFTFeature((7, 8, 9), 2)


def test_feature_from_mapping_names_the_missing_keys() -> None:
    with pytest.raises(ValueError, match=r"missing \['prompt_len'\]"):
        SFTFeature.from_mapping({"input_ids": [1, 2]})


def test_pair_from_shared_builds_two_sequences() -> None:
    pair = PairFeature.from_shared([1, 2], [3], [4, 5])
    assert pair.chosen == SFTFeature((1, 2, 3), 2)
    assert pair.rejected == SFTFeature((1, 2, 4, 5), 2)
    assert pair.shared_prompt_len == 2


def test_pair_rejects_two_different_prompts() -> None:
    with pytest.raises(ValueError, match="same prompt"):
        PairFeature(chosen=SFTFeature((1, 2, 3), 2), rejected=SFTFeature((1, 9, 4), 2))


def test_pair_tolerates_a_boundary_that_merged_on_one_side_only() -> None:
    # One side split a token later than the other; the shared prefix is the shorter prompt.
    pair = PairFeature(chosen=SFTFeature((1, 2, 3), 2), rejected=SFTFeature((1, 2, 3, 4), 1))
    assert pair.shared_prompt_len == 1


def test_encode_sft_example_round_trips_through_the_formatter(
    formatter: ChatFormatter, tokenizer: WordTokenizer
) -> None:
    feature = encode_sft_example(formatter, tokenizer, USER_PROMPT, COMPLETION)
    full_ids = encode_text(tokenizer, formatter.full_text(USER_PROMPT, COMPLETION))
    prompt_ids = encode_text(tokenizer, formatter.prompt_text(USER_PROMPT))
    assert list(feature.input_ids) == full_ids
    assert list(feature.prompt_ids) == prompt_ids
    assert feature.completion_len == len(full_ids) - len(prompt_ids)


def test_encode_preference_pair_shares_the_prompt(
    formatter: ChatFormatter, tokenizer: WordTokenizer
) -> None:
    pair = PreferencePair(
        example_id="ex-1",
        slice=Slice.CLEAN,
        prompt=USER_PROMPT,
        chosen=COMPLETION,
        rejected="not json at all",
        chosen_reward=1.0,
        rejected_reward=0.0,
    )
    feature = encode_preference_pair(formatter, tokenizer, pair)
    assert feature.chosen.prompt_ids == feature.rejected.prompt_ids
    assert feature.chosen.completion_ids != feature.rejected.completion_ids


# --------------------------------------------------------------------------------------
# collate.py: the masking invariants
# --------------------------------------------------------------------------------------


def _collator(**kwargs: Any) -> CompletionOnlyCollator:
    return CompletionOnlyCollator(pad_token_id=0, max_length=16, **kwargs)


def test_batch_has_exactly_the_three_model_arguments() -> None:
    batch = _collator()([SFTFeature((5, 6, 7), 2)])
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert all(t.dtype == torch.long for t in batch.values())


def test_nothing_before_the_completion_is_supervised() -> None:
    features = [SFTFeature((5, 6, 7, 8), 3), SFTFeature((9, 10, 11), 1)]
    labels = _collator()(features)["labels"]
    for row, feature in zip(labels.tolist(), features, strict=True):
        assert row[: feature.prompt_len] == [IGNORE_INDEX] * feature.prompt_len


def test_supervised_token_count_equals_the_completion_length() -> None:
    features = [SFTFeature((5, 6, 7, 8), 3), SFTFeature((9, 10, 11), 1)]
    labels = _collator()(features)["labels"]
    for row, feature in zip(labels.tolist(), features, strict=True):
        supervised = [label for label in row if label != IGNORE_INDEX]
        assert supervised == list(feature.completion_ids)


def test_padding_is_never_supervised() -> None:
    batch = _collator()([SFTFeature((5, 6, 7, 8, 9), 1), SFTFeature((3, 4), 1)])
    labels = batch["labels"]
    mask = batch["attention_mask"]
    assert torch.equal(labels[mask == 0], torch.full_like(labels[mask == 0], IGNORE_INDEX))


def test_padding_is_on_the_right_and_uses_the_pad_id() -> None:
    batch = CompletionOnlyCollator(pad_token_id=7, max_length=16)(
        [SFTFeature((1, 2, 3, 4), 2), SFTFeature((5, 6), 1)]
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4], [5, 6, 7, 7]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


def test_mapping_features_are_accepted() -> None:
    batch = _collator()([{"input_ids": [4, 5, 6], "prompt_len": 1}])
    assert batch["input_ids"].tolist() == [[4, 5, 6]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 5, 6]]


def test_unknown_feature_type_is_refused() -> None:
    bogus: Any = (1, 2, 3)
    with pytest.raises(TypeError, match="expected an SFTFeature"):
        _collator()([bogus])


def test_pad_to_multiple_of_rounds_the_batch_width_up() -> None:
    batch = _collator(pad_to_multiple_of=8)([SFTFeature((1, 2, 3), 1)])
    assert batch["input_ids"].shape[1] == 8
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 2, 3] + [IGNORE_INDEX] * 5]


def test_pad_to_multiple_of_leaves_an_exact_width_alone() -> None:
    batch = _collator(pad_to_multiple_of=2)([SFTFeature((1, 2, 3, 4), 1)])
    assert batch["input_ids"].shape[1] == 4


# --------------------------------------------------------------------------------------
# collate.py: truncation and reporting
# --------------------------------------------------------------------------------------


def test_long_example_is_truncated_and_counted() -> None:
    collator = CompletionOnlyCollator(pad_token_id=0, max_length=4)
    batch = collator([SFTFeature((1, 2, 3, 4, 5, 6), 2)])
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 3, 4]]
    assert collator.report.truncated == 1
    assert collator.report.dropped == 0


def test_example_whose_prompt_fills_the_budget_is_dropped_not_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    collator = CompletionOnlyCollator(pad_token_id=0, max_length=4)
    with caplog.at_level("WARNING"):
        batch = collator([SFTFeature((1, 2, 3, 4, 5), 4), SFTFeature((6, 7), 1)])
    assert batch["input_ids"].tolist() == [[6, 7]]
    assert collator.report.dropped == 1
    assert collator.report.seen == 2
    assert "no completion token" in caplog.text


def test_a_batch_of_only_droppable_examples_raises() -> None:
    collator = CompletionOnlyCollator(pad_token_id=0, max_length=3)
    with pytest.raises(ValueError, match="every example in this batch"):
        collator([SFTFeature((1, 2, 3, 4), 3)])
    assert collator.report.dropped == 1


def test_empty_batches_are_refused() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        _collator()([])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pad_token_id": -1}, "real token id"),
        ({"pad_token_id": 0, "max_length": 1}, "at least one prompt"),
        ({"pad_token_id": 0, "label_pad_token_id": 0}, "must be negative"),
        ({"pad_token_id": 0, "pad_to_multiple_of": 0}, "must be positive"),
    ],
)
def test_collator_settings_are_validated(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CompletionOnlyCollator(**kwargs)
    with pytest.raises(ValueError, match=message):
        PreferenceCollator(**kwargs)


def test_report_summarises_a_run() -> None:
    report = CollationReport(seen=4, truncated=1, dropped=1)
    assert report.drop_rate == pytest.approx(0.25)
    assert report.as_dict()["dropped"] == 1
    assert "4 seen" in str(report)


def test_report_drop_rate_is_zero_before_anything_is_seen() -> None:
    assert CollationReport().drop_rate == 0.0


# --------------------------------------------------------------------------------------
# collate.py: preference batches
# --------------------------------------------------------------------------------------


def _pair_collator(**kwargs: Any) -> PreferenceCollator:
    return PreferenceCollator(pad_token_id=0, max_length=16, **kwargs)


def test_preference_batch_has_both_halves() -> None:
    batch = _pair_collator()([PairFeature.from_shared([1, 2], [3], [4, 5])])
    assert set(batch) == {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_labels",
    }


def test_preference_halves_share_one_width_for_a_single_forward_pass() -> None:
    batch = _pair_collator()(
        [
            PairFeature.from_shared([1, 2], [3], [4, 5, 6, 7]),
            PairFeature.from_shared([8], [9, 10], [11]),
        ]
    )
    assert batch["chosen_input_ids"].shape == batch["rejected_input_ids"].shape
    assert batch["chosen_input_ids"].shape[1] == 6


def test_preference_masking_follows_the_same_rule() -> None:
    pair = PairFeature.from_shared([1, 2], [3], [4, 5])
    batch = _pair_collator()([pair])
    assert batch["chosen_labels"].tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 3, IGNORE_INDEX]]
    assert batch["rejected_labels"].tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 4, 5]]


def test_a_pair_is_dropped_whole_when_one_side_cannot_survive() -> None:
    collator = PreferenceCollator(pad_token_id=0, max_length=4)
    good = PairFeature.from_shared([1, 2], [3], [4])
    bad = PairFeature.from_shared([1, 2, 3, 4], [5], [6])
    batch = collator([good, bad])
    assert batch["chosen_input_ids"].shape[0] == 1
    assert collator.report.seen == 2
    assert collator.report.dropped == 1


def test_preference_batch_of_only_droppable_pairs_raises() -> None:
    collator = PreferenceCollator(pad_token_id=0, max_length=4)
    with pytest.raises(ValueError, match="lost a side to truncation"):
        collator([PairFeature.from_shared([1, 2, 3, 4], [5], [6])])


def test_preference_empty_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        _pair_collator()([])


def test_preference_truncation_counts_sequences() -> None:
    collator = PreferenceCollator(pad_token_id=0, max_length=4)
    collator([PairFeature.from_shared([1, 2], [3, 4, 5], [6, 7, 8])])
    assert collator.report.truncated == 2
    assert collator.report.seen == 1


# --------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------


@st.composite
def sft_features(draw: st.DrawFn) -> SFTFeature:
    prompt_len = draw(st.integers(min_value=1, max_value=5))
    completion_len = draw(st.integers(min_value=1, max_value=5))
    total = prompt_len + completion_len
    ids = draw(st.lists(st.integers(min_value=1, max_value=999), min_size=total, max_size=total))
    return SFTFeature(tuple(ids), prompt_len)


PROPERTY_SETTINGS = settings(max_examples=60, deadline=None)


@given(features=st.lists(sft_features(), min_size=1, max_size=5))
@PROPERTY_SETTINGS
def test_a_row_does_not_depend_on_what_it_was_batched_with(features: list[SFTFeature]) -> None:
    """Padding invariance: batching must not change any example's supervised content."""
    collator = CompletionOnlyCollator(pad_token_id=0, max_length=32)
    batched = collator(features)
    for index, feature in enumerate(features):
        alone = collator([feature])
        length = feature.total_len
        assert batched["input_ids"][index][:length].tolist() == alone["input_ids"][0].tolist()
        assert batched["labels"][index][:length].tolist() == alone["labels"][0].tolist()
        assert batched["attention_mask"][index][:length].tolist() == [1] * length


@given(features=st.lists(sft_features(), min_size=1, max_size=5))
@PROPERTY_SETTINGS
def test_supervision_is_conserved_across_any_batch(features: list[SFTFeature]) -> None:
    """Every batch supervises exactly the completion tokens, whatever the shapes."""
    batch = CompletionOnlyCollator(pad_token_id=0, max_length=32)(features)
    labels = batch["labels"]
    assert int((labels != IGNORE_INDEX).sum()) == sum(f.completion_len for f in features)
    for index, feature in enumerate(features):
        row = labels[index].tolist()
        assert [label for label in row if label != IGNORE_INDEX] == list(feature.completion_ids)


@given(features=st.lists(sft_features(), min_size=1, max_size=4), multiple=st.integers(1, 8))
@PROPERTY_SETTINGS
def test_batch_width_is_a_multiple_and_covers_the_longest_example(
    features: list[SFTFeature], multiple: int
) -> None:
    batch = CompletionOnlyCollator(pad_token_id=0, max_length=32, pad_to_multiple_of=multiple)(
        features
    )
    width = int(batch["input_ids"].shape[1])
    assert width % multiple == 0
    assert width >= max(f.total_len for f in features)
    assert width < max(f.total_len for f in features) + multiple


@given(prompt=st.text(min_size=1, max_size=40), completion=st.text(min_size=1, max_size=40))
@PROPERTY_SETTINGS
def test_split_lengths_always_partition_the_encoded_full_text(prompt: str, completion: str) -> None:
    """Whatever the text, the two lengths add up and the answer is never empty."""
    formatter = ChatFormatter(system="SYS")
    tokenizer = WordTokenizer()
    full_ids = encode_text(tokenizer, formatter.full_text(prompt, completion))
    prompt_len, completion_len = formatter.split_lengths(tokenizer, prompt, completion)
    assert prompt_len + completion_len == len(full_ids)
    assert completion_len >= 1
    assert full_ids[:prompt_len] == encode_text(tokenizer, formatter.prompt_text(prompt))
