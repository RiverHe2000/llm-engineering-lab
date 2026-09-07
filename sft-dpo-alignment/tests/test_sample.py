"""Tests for batched, replayable sampling of completions.

The headline test is `test_batched_matches_unbatched`, which asserts the one property that
the classic decoder-only padding bug destroys: greedy decoding of a batch must give exactly
what greedy decoding of each row alone gives. `test_right_padding_breaks_the_equivalence`
sits next to it and shows, on the same model, that the property really is sensitive to the
mistake -- an equivalence test that would pass either way is worse than no test.

Everything runs on CPU against the two-layer Qwen2 from `build_tiny_model`, or against a
scripted stand-in model whose continuations are fixed. The stand-in is not a shortcut: it is
the only way to test what happens *around* generation -- how batches are cut, that padding
really arrives left-aligned at `generate`, that a stop token truncates a row, that the
caller's training mode is put back -- without those assertions depending on what a randomly
initialised network happens to emit.

CI has no Qwen tokenizer offline, so a character tokenizer stands in. It is exactly
invertible, which makes decoded text a faithful function of the ids and keeps the
equivalence assertions about generation rather than about tokenisation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.modeling.chat import ChatFormatter
from sftdpo.modeling.loader import build_tiny_model
from sftdpo.prefs.sample import (
    Prompt,
    SamplingStats,
    generation_kwargs,
    left_pad,
    prompts_for_examples,
    resolve_eos_ids,
    resolve_pad_token_id,
    rng_devices,
    sample_completions,
    stream_seed,
    trim_completion,
)
from sftdpo.schemas import (
    AdviceRecord,
    Example,
    Fees,
    GenerationConfig,
    Recommendation,
    RiskProfile,
    Sample,
    Slice,
)

PAD_ID = 0
EOS_ID = 1

# Different lengths on purpose: a batch of equal-length prompts needs no padding at all and
# would pass the equivalence test with the padding side wired up backwards.
PROMPT_TEXTS: tuple[str, ...] = (
    "note one: client Ada wants growth",
    "note two: short",
    "note three: a considerably longer adviser note about pension phase and fees",
    "note four: medium length note",
)


class CharTokenizer:
    """Character-level stand-in: `ord` in, `chr` out, so encoding is exactly invertible.

    The tiny model's vocabulary is the full Qwen 151936, which covers every code point a
    character in these tests can produce, so ids from here are always in range.
    """

    pad_token_id = PAD_ID
    # Optional so a test can take the stop token away, as a base checkpoint would.
    eos_token_id: int | None = EOS_ID

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002
        # The keyword is part of the interface `encode_text` calls with; a character
        # tokenizer has no special tokens to add or withhold.
        return [ord(char) for char in text]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(int(token)) for token in ids if not skip_special_tokens or token > 31)


class ScriptedModel:
    """A model whose `generate` returns fixed continuations and records what it was given.

    Written so that assertions about batching and padding can be exact. A real model would
    answer "did the rows arrive left-padded?" only indirectly, through text that a change of
    random seed would alter.
    """

    def __init__(self, script: Sequence[Sequence[int]], *, name: str = "scripted-model") -> None:
        widths = {len(row) for row in script}
        if len(widths) != 1:
            raise ValueError(f"scripted rows must share one width, got {sorted(widths)}")
        self.script = [list(row) for row in script]
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(name_or_path=name)
        self.generation_config = SimpleNamespace(eos_token_id=None, pad_token_id=None)
        self.training = True
        self._cursor = 0
        self._parameter = torch.zeros(1)

    def parameters(self) -> Iterator[torch.Tensor]:
        yield self._parameter

    def eval(self) -> ScriptedModel:
        self.training = False
        return self

    def train(self, mode: bool = True) -> ScriptedModel:
        self.training = mode
        return self

    def generate(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "kwargs": dict(kwargs),
            }
        )
        rows = int(input_ids.shape[0])
        continuation = [self.script[(self._cursor + i) % len(self.script)] for i in range(rows)]
        self._cursor += rows
        return torch.cat([input_ids, torch.tensor(continuation, dtype=torch.long)], dim=1)


def make_prompts(texts: Sequence[str] = PROMPT_TEXTS) -> list[Prompt]:
    return [Prompt(example_id=f"ex-{index}", text=text) for index, text in enumerate(texts)]


def make_example(example_id: str = "ex-0") -> Example:
    return Example(
        example_id=example_id,
        split="train",
        slice=Slice.CLEAN,
        note="Client Ada. Risk profile growth.",
        gold=AdviceRecord(
            client_name="Ada",
            record_date="2026-03-04",
            risk_profile=RiskProfile.GROWTH,
            objectives=["retire at 62"],
            recommendations=[Recommendation(product="A200", action="buy", amount=1000.0)],
            fees=Fees(advice_fee=3300.0, ongoing_fee_pct=0.55),
            review_months=12,
            flags=[],
        ),
    )


def tiny_model() -> Any:
    model = build_tiny_model(seed=11)
    model.eval()
    return model


def generate_rows(
    model: Any, rows: Sequence[Sequence[int]], *, max_new_tokens: int
) -> list[list[int]]:
    """Greedy-generate with explicit left padding, returning only the new tokens."""
    input_ids, attention_mask = left_pad(rows, PAD_ID)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=PAD_ID,
        )
    return [row[int(input_ids.shape[1]) :] for row in out.tolist()]


# --------------------------------------------------------------------------------------
# left_pad
# --------------------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    rows=st.lists(
        st.lists(st.integers(min_value=2, max_value=5000), min_size=1, max_size=9),
        min_size=1,
        max_size=6,
    )
)
def test_left_pad_keeps_every_row_flush_right(rows: list[list[int]]) -> None:
    """The defining property: real tokens end at the last column, marked by the mask."""
    input_ids, attention_mask = left_pad(rows, PAD_ID)
    width = max(len(row) for row in rows)
    assert input_ids.shape == (len(rows), width)
    assert attention_mask.shape == input_ids.shape
    for index, row in enumerate(rows):
        assert input_ids[index, width - len(row) :].tolist() == row
        assert attention_mask[index].tolist() == [0] * (width - len(row)) + [1] * len(row)


@settings(max_examples=50, deadline=None)
@given(
    rows=st.lists(
        st.lists(st.integers(min_value=2, max_value=5000), min_size=1, max_size=9),
        min_size=1,
        max_size=6,
    )
)
def test_left_pad_mask_selects_exactly_the_original_tokens(rows: list[list[int]]) -> None:
    """The mask is a lossless index back to the input, which is what generation relies on."""
    input_ids, attention_mask = left_pad(rows, PAD_ID)
    for index, row in enumerate(rows):
        selected = input_ids[index][attention_mask[index].bool()]
        assert selected.tolist() == row


def test_left_pad_of_equal_length_rows_adds_nothing() -> None:
    input_ids, attention_mask = left_pad([[5, 6, 7], [8, 9, 10]], PAD_ID)
    assert input_ids.tolist() == [[5, 6, 7], [8, 9, 10]]
    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_left_pad_uses_the_given_filler() -> None:
    input_ids, _ = left_pad([[5, 6, 7], [9]], 42)
    assert input_ids.tolist() == [[5, 6, 7], [42, 42, 9]]


def test_left_pad_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        left_pad([], PAD_ID)


def test_left_pad_rejects_an_empty_row() -> None:
    """An empty prompt would leave the model conditioning on padding alone."""
    with pytest.raises(ValueError, match="empty prompt"):
        left_pad([[1, 2], []], PAD_ID)


# --------------------------------------------------------------------------------------
# trim_completion
# --------------------------------------------------------------------------------------


def test_trim_completion_cuts_at_the_first_stop_token() -> None:
    assert trim_completion([7, 8, EOS_ID, PAD_ID, PAD_ID], frozenset({EOS_ID})) == [7, 8]


def test_trim_completion_without_stop_tokens_keeps_everything() -> None:
    assert trim_completion([7, 8, 9], frozenset()) == [7, 8, 9]


def test_trim_completion_of_an_immediate_stop_is_empty() -> None:
    assert trim_completion([EOS_ID, 5, 6], frozenset({EOS_ID})) == []


def test_trim_completion_honours_several_stop_ids() -> None:
    assert trim_completion([7, 3, 9], frozenset({EOS_ID, 3})) == [7]


@settings(max_examples=50, deadline=None)
@given(
    ids=st.lists(st.integers(min_value=0, max_value=20), max_size=12),
    stops=st.sets(st.integers(min_value=0, max_value=20), max_size=4),
)
def test_trim_completion_returns_a_stop_free_prefix(ids: list[int], stops: set[int]) -> None:
    """Two properties at once: the result is a prefix, and it contains no stop token."""
    stop_ids = frozenset(stops)
    trimmed = trim_completion(ids, stop_ids)
    assert trimmed == ids[: len(trimmed)]
    assert not stop_ids.intersection(trimmed)


# --------------------------------------------------------------------------------------
# Tokenizer and model interrogation
# --------------------------------------------------------------------------------------


def test_resolve_pad_token_id_prefers_the_tokenizer() -> None:
    tokenizer = SimpleNamespace(pad_token_id=17, eos_token_id=99)
    assert resolve_pad_token_id(tokenizer, ScriptedModel([[1]])) == 17


def test_resolve_pad_token_id_falls_back_to_eos() -> None:
    """The common shape for instruct checkpoints that ship without a pad token."""
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=99)
    assert resolve_pad_token_id(tokenizer, ScriptedModel([[1]])) == 99


def test_resolve_pad_token_id_falls_back_to_the_model() -> None:
    model = ScriptedModel([[1]])
    model.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=151643)
    assert resolve_pad_token_id(SimpleNamespace(), model) == 151643


def test_resolve_pad_token_id_raises_when_nothing_is_available() -> None:
    """Defaulting to zero would push a real token into the batch and change the prompt."""
    model = ScriptedModel([[1]])
    model.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=None)
    with pytest.raises(ValueError, match="neither a pad nor an eos"):
        resolve_pad_token_id(SimpleNamespace(), model)


def test_resolve_eos_ids_unions_both_sources() -> None:
    model = ScriptedModel([[1]])
    model.generation_config = SimpleNamespace(eos_token_id=151643)
    assert resolve_eos_ids(SimpleNamespace(eos_token_id=151645), model) == frozenset(
        {151643, 151645}
    )


def test_resolve_eos_ids_accepts_a_list() -> None:
    model = ScriptedModel([[1]])
    model.generation_config = SimpleNamespace(eos_token_id=[7, 8])
    assert resolve_eos_ids(SimpleNamespace(eos_token_id=None), model) == frozenset({7, 8})


def test_resolve_eos_ids_is_empty_when_neither_source_has_one() -> None:
    model = ScriptedModel([[1]])
    model.generation_config = SimpleNamespace(eos_token_id=None)
    assert resolve_eos_ids(SimpleNamespace(eos_token_id=None), model) == frozenset()


# --------------------------------------------------------------------------------------
# Seeding and decoding settings
# --------------------------------------------------------------------------------------


def test_stream_seed_is_stable_across_processes() -> None:
    """Pinned to literals, because a salted hash here would silently break replay.

    These are the values `blake2b` gives; `hash()` would give different ones in every
    process, which is exactly the bug the implementation avoids.
    """
    assert stream_seed(0, 0, 0) == 3649034661406903291
    assert stream_seed(7, 2, 3) == 8718843434107328879


@settings(max_examples=50, deadline=None)
@given(
    base=st.integers(min_value=0, max_value=2**31),
    index=st.integers(min_value=0, max_value=64),
    chunk=st.integers(min_value=0, max_value=64),
)
def test_stream_seed_is_in_range_and_repeatable(base: int, index: int, chunk: int) -> None:
    seed = stream_seed(base, index, chunk)
    assert 0 <= seed < 2**63
    assert seed == stream_seed(base, index, chunk)


def test_stream_seed_separates_sample_indices_and_chunks() -> None:
    """Every (index, chunk) draws from its own stream, so two sweeps cannot coincide."""
    seeds = {stream_seed(0, index, chunk) for index in range(6) for chunk in range(6)}
    assert len(seeds) == 36


def test_rng_devices_forks_cuda_only() -> None:
    """CPU generation needs no fork; CUDA does, or seeding would disturb the caller."""
    assert rng_devices(torch.device("cpu")) == []
    assert rng_devices(torch.device("cuda:0")) == [torch.device("cuda:0")]


def test_generation_kwargs_greedy_omits_sampling_settings() -> None:
    """`generate` warns about settings it ignores, and an unactionable warning is noise."""
    kwargs = generation_kwargs(GenerationConfig(temperature=0.0, max_new_tokens=8), pad_token_id=3)
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 8
    assert kwargs["pad_token_id"] == 3
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_generation_kwargs_sampling_disables_the_hidden_top_k() -> None:
    """top_p=1.0 must mean the whole distribution, not the fifty tokens HF defaults to."""
    kwargs = generation_kwargs(GenerationConfig(temperature=0.9, top_p=1.0), pad_token_id=3)
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.9
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_new_tokens", 0),
        ("temperature", -0.5),
        ("top_p", 0.0),
        ("top_p", 1.5),
    ],
)
def test_generation_kwargs_rejects_impossible_settings(field: str, value: float) -> None:
    config = GenerationConfig(**{field: value})
    with pytest.raises(ValueError, match=field):
        generation_kwargs(config, pad_token_id=3)


# --------------------------------------------------------------------------------------
# The padding equivalence
# --------------------------------------------------------------------------------------


def test_batched_matches_unbatched() -> None:
    """Greedy decoding of a batch equals greedy decoding of each prompt alone.

    This is the assertion the left-padding bug breaks, and it is the reason the batching in
    `sample_completions` can be trusted to be a pure speed-up rather than a change of
    behaviour. Greedy on purpose: sampled decoding draws the whole batch's tokens from one
    `multinomial` call, so its randomness is a function of the batch, not of the row.
    """
    model = tiny_model()
    tokenizer = CharTokenizer()
    prompts = make_prompts()
    config = GenerationConfig(max_new_tokens=12, temperature=0.0)

    batched = sample_completions(
        model, tokenizer, prompts, k=1, config=config, batch_size=len(prompts)
    )
    unbatched = sample_completions(model, tokenizer, prompts, k=1, config=config, batch_size=1)

    assert [sample.text for sample in batched] == [sample.text for sample in unbatched]
    assert [sample.completion_tokens for sample in batched] == [
        sample.completion_tokens for sample in unbatched
    ]


def test_right_padding_breaks_the_equivalence() -> None:
    """The same model, padded on the right, gives different tokens.

    Without this the equivalence test above would be reassuring but unfalsifiable: it must
    be shown that the property is actually sensitive to getting the padding side wrong.
    """
    model = tiny_model()
    rows = [[ord(char) for char in text] for text in PROMPT_TEXTS]
    width = max(len(row) for row in rows)
    right_ids = torch.tensor([row + [PAD_ID] * (width - len(row)) for row in rows])
    right_mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])
    with torch.inference_mode():
        wrong = model.generate(
            input_ids=right_ids,
            attention_mask=right_mask,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=PAD_ID,
        )
    wrong_new = [row[width:] for row in wrong.tolist()]
    alone = [generate_rows(model, [row], max_new_tokens=12)[0] for row in rows]

    assert generate_rows(model, rows, max_new_tokens=12) == alone
    assert wrong_new != alone


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_greedy_results_do_not_depend_on_batch_size(batch_size: int) -> None:
    """Batch size is a throughput knob under greedy decoding and nothing more."""
    model = tiny_model()
    tokenizer = CharTokenizer()
    prompts = make_prompts()
    config = GenerationConfig(max_new_tokens=10, temperature=0.0)
    reference = sample_completions(model, tokenizer, prompts, k=1, config=config, batch_size=1)
    other = sample_completions(model, tokenizer, prompts, k=1, config=config, batch_size=batch_size)
    assert [sample.text for sample in other] == [sample.text for sample in reference]


# --------------------------------------------------------------------------------------
# Determinism and shape of the returned samples
# --------------------------------------------------------------------------------------


def test_sampling_is_deterministic_under_a_seed() -> None:
    model = tiny_model()
    tokenizer = CharTokenizer()
    prompts = make_prompts()
    config = GenerationConfig(max_new_tokens=8, temperature=0.9, seed=5)
    first = sample_completions(model, tokenizer, prompts, k=2, config=config, batch_size=2)
    second = sample_completions(model, tokenizer, prompts, k=2, config=config, batch_size=2)
    assert [sample.text for sample in first] == [sample.text for sample in second]


def test_a_different_seed_gives_different_completions() -> None:
    """Determinism must not have been bought by ignoring the seed entirely."""
    model = tiny_model()
    tokenizer = CharTokenizer()
    prompts = make_prompts()
    first = sample_completions(
        model,
        tokenizer,
        prompts,
        k=2,
        config=GenerationConfig(max_new_tokens=8, temperature=0.9, seed=5),
        batch_size=2,
    )
    second = sample_completions(
        model,
        tokenizer,
        prompts,
        k=2,
        config=GenerationConfig(max_new_tokens=8, temperature=0.9, seed=6),
        batch_size=2,
    )
    assert [s.text for s in first] != [s.text for s in second]


def test_sampling_leaves_the_global_rng_undisturbed() -> None:
    """A trainer refreshing preference data mid-run must not have its stream shifted."""
    model = tiny_model()
    tokenizer = CharTokenizer()
    torch.manual_seed(1234)
    expected = torch.randint(0, 1000, (4,))
    torch.manual_seed(1234)
    sample_completions(
        model,
        tokenizer,
        make_prompts(),
        k=2,
        config=GenerationConfig(max_new_tokens=6, temperature=1.0, seed=3),
        batch_size=2,
    )
    assert torch.equal(torch.randint(0, 1000, (4,)), expected)


def test_samples_are_grouped_by_prompt_and_ordered_by_index() -> None:
    model = tiny_model()
    prompts = make_prompts()
    samples = sample_completions(
        model,
        CharTokenizer(),
        prompts,
        k=3,
        config=GenerationConfig(max_new_tokens=6, temperature=0.8),
        batch_size=2,
    )
    assert len(samples) == len(prompts) * 3
    assert [sample.example_id for sample in samples] == [
        prompt.example_id for prompt in prompts for _ in range(3)
    ]
    assert [sample.sample_index for sample in samples] == [0, 1, 2] * len(prompts)


def test_token_counts_are_recorded() -> None:
    model = tiny_model()
    tokenizer = CharTokenizer()
    prompts = make_prompts()
    samples = sample_completions(
        model,
        tokenizer,
        prompts,
        k=1,
        config=GenerationConfig(max_new_tokens=7, temperature=0.0),
        batch_size=4,
    )
    expected = {prompt.example_id: len(prompt.text) for prompt in prompts}
    for sample in samples:
        assert sample.prompt_tokens == expected[sample.example_id]
        assert 0 < sample.completion_tokens <= 7
        assert sample.latency_ms > 0.0


def test_model_name_defaults_to_the_checkpoint_and_can_be_overridden() -> None:
    model = ScriptedModel([[5, 6]], name="Qwen/Qwen2.5-0.5B-Instruct")
    config = GenerationConfig(max_new_tokens=2, temperature=0.0)
    default = sample_completions(
        model, CharTokenizer(), make_prompts(PROMPT_TEXTS[:2]), k=1, config=config, batch_size=2
    )
    named = sample_completions(
        model,
        CharTokenizer(),
        make_prompts(PROMPT_TEXTS[:2]),
        k=1,
        config=config,
        batch_size=2,
        model_name="policy-after-sft",
    )
    assert {sample.model for sample in default} == {"Qwen/Qwen2.5-0.5B-Instruct"}
    assert {sample.model for sample in named} == {"policy-after-sft"}


# --------------------------------------------------------------------------------------
# Batching mechanics, asserted exactly against a scripted model
# --------------------------------------------------------------------------------------


def test_batches_are_cut_to_the_requested_size() -> None:
    model = ScriptedModel([[5, 6]])
    sample_completions(
        model,
        CharTokenizer(),
        make_prompts(),
        k=2,
        config=GenerationConfig(max_new_tokens=2, temperature=0.7),
        batch_size=3,
    )
    assert [int(call["input_ids"].shape[0]) for call in model.calls] == [3, 1, 3, 1]


def test_generate_receives_left_padded_rows() -> None:
    """The padding side is asserted at the boundary, not inferred from the output text."""
    model = ScriptedModel([[5, 6]])
    prompts = make_prompts(("abc", "z"))
    sample_completions(
        model,
        CharTokenizer(),
        prompts,
        k=1,
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
        batch_size=2,
    )
    call = model.calls[0]
    assert call["input_ids"].tolist() == [
        [ord("a"), ord("b"), ord("c")],
        [PAD_ID, PAD_ID, ord("z")],
    ]
    assert call["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]


def test_a_stop_token_truncates_the_completion() -> None:
    """Padding after an early stop must not be decoded or counted as generated tokens."""
    model = ScriptedModel([[ord("x"), EOS_ID, PAD_ID, PAD_ID]])
    samples = sample_completions(
        model,
        CharTokenizer(),
        make_prompts(("abc",)),
        k=1,
        config=GenerationConfig(max_new_tokens=4, temperature=0.0),
        batch_size=1,
    )
    assert samples[0].text == "x"
    assert samples[0].completion_tokens == 1


def test_stop_ids_are_passed_to_generate() -> None:
    """So that generation halts where trimming cuts instead of burning the token budget."""
    model = ScriptedModel([[5, 6]])
    sample_completions(
        model,
        CharTokenizer(),
        make_prompts(("abc",)),
        k=1,
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
        batch_size=1,
    )
    assert model.calls[0]["kwargs"]["eos_token_id"] == [EOS_ID]


def test_a_model_without_a_stop_token_runs_to_the_budget() -> None:
    """A base checkpoint may offer no end-of-sequence id at all. Nothing is then passed to
    `generate`, which is not the same as passing an empty list: `generate` treats an empty
    `eos_token_id` as a value to validate rather than as an absence."""
    tokenizer = CharTokenizer()
    tokenizer.eos_token_id = None
    model = ScriptedModel([[ord("x"), ord("y")]])

    samples = sample_completions(
        model,
        tokenizer,
        make_prompts(("abc",)),
        k=1,
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
        batch_size=1,
    )

    assert "eos_token_id" not in model.calls[0]["kwargs"]
    assert samples[0].text == "xy"
    assert samples[0].completion_tokens == 2


def test_training_mode_is_restored() -> None:
    """A trainer must not find its model quietly switched to eval by a sampling call."""
    model = ScriptedModel([[5, 6]])
    model.train()
    sample_completions(
        model,
        CharTokenizer(),
        make_prompts(("abc",)),
        k=1,
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
        batch_size=1,
    )
    assert model.training is True

    model.eval()
    sample_completions(
        model,
        CharTokenizer(),
        make_prompts(("abc",)),
        k=1,
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
        batch_size=1,
    )
    assert model.training is False


def test_prompts_are_encoded_once_per_run_not_once_per_sample() -> None:
    """k sweeps over a prompt carrying the whole JSON schema must not re-tokenise it k times."""
    model = ScriptedModel([[5, 6]])
    tokenizer = CharTokenizer()
    calls: list[str] = []
    original = tokenizer.encode

    def counting_encode(text: str, add_special_tokens: bool = True) -> list[int]:
        calls.append(text)
        return original(text, add_special_tokens)

    tokenizer.encode = counting_encode  # type: ignore[method-assign]
    sample_completions(
        model,
        tokenizer,
        make_prompts(),
        k=4,
        config=GenerationConfig(max_new_tokens=2, temperature=0.7),
        batch_size=2,
    )
    assert len(calls) == len(PROMPT_TEXTS)


# --------------------------------------------------------------------------------------
# Rejected requests
# --------------------------------------------------------------------------------------


def test_greedy_with_k_above_one_is_refused() -> None:
    """k identical completions can never yield a pair, and cost k times the GPU time."""
    with pytest.raises(ValueError, match="identical completions"):
        sample_completions(
            tiny_model(),
            CharTokenizer(),
            make_prompts(),
            k=4,
            config=GenerationConfig(max_new_tokens=4, temperature=0.0),
            batch_size=2,
        )


@pytest.mark.parametrize(("k", "batch_size"), [(0, 2), (-1, 2), (1, 0), (1, -3)])
def test_non_positive_k_or_batch_size_is_refused(k: int, batch_size: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        sample_completions(
            ScriptedModel([[5, 6]]),
            CharTokenizer(),
            make_prompts(),
            k=k,
            config=GenerationConfig(max_new_tokens=2, temperature=0.7),
            batch_size=batch_size,
        )


def test_duplicate_example_ids_are_refused() -> None:
    """Two prompts under one id would merge into a single mining group and mislabel it."""
    prompts = [Prompt(example_id="ex-0", text="one"), Prompt(example_id="ex-0", text="two")]
    with pytest.raises(ValueError, match="duplicate example_id"):
        sample_completions(
            ScriptedModel([[5, 6]]),
            CharTokenizer(),
            prompts,
            k=1,
            config=GenerationConfig(max_new_tokens=2, temperature=0.0),
            batch_size=2,
        )


def test_an_empty_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="is empty"):
        sample_completions(
            ScriptedModel([[5, 6]]),
            CharTokenizer(),
            [Prompt(example_id="ex-0", text="")],
            k=1,
            config=GenerationConfig(max_new_tokens=2, temperature=0.0),
            batch_size=1,
        )


def test_no_prompts_gives_no_samples() -> None:
    model = ScriptedModel([[5, 6]])
    assert (
        sample_completions(
            model,
            CharTokenizer(),
            [],
            k=2,
            config=GenerationConfig(max_new_tokens=2, temperature=0.7),
            batch_size=2,
        )
        == []
    )
    assert model.calls == []


# --------------------------------------------------------------------------------------
# Prompt rendering and throughput accounting
# --------------------------------------------------------------------------------------


def test_prompts_for_examples_uses_the_shared_formatter_and_instruction() -> None:
    """The sampling prompt must be the training prompt, character for character."""
    example = make_example("ex-7")
    prompts = prompts_for_examples([example], formatter=ChatFormatter())
    assert len(prompts) == 1
    assert prompts[0].example_id == "ex-7"
    assert example.note in prompts[0].text
    assert "JSON object:" in prompts[0].text
    assert prompts[0].text.endswith("<|im_start|>assistant\n")


def test_sampling_stats_totals_the_samples() -> None:
    samples = [
        Sample(
            example_id=f"ex-{index}",
            model="m",
            text="{}",
            prompt_tokens=100,
            completion_tokens=20,
            latency_ms=250.0,
        )
        for index in range(4)
    ]
    stats = SamplingStats.over(samples)
    assert stats.completions == 4
    assert stats.prompt_tokens == 400
    assert stats.completion_tokens == 80
    assert stats.wall_ms == pytest.approx(1000.0)
    assert stats.tokens_per_second == pytest.approx(80.0)
    assert stats.ms_per_completion == pytest.approx(250.0)
    assert stats.as_dict()["tokens_per_second"] == pytest.approx(80.0)


def test_sampling_stats_of_an_empty_run_are_zero() -> None:
    """Zero rather than a division by zero: an empty run is a result, not an error."""
    stats = SamplingStats.over([])
    assert stats.completions == 0
    assert stats.tokens_per_second == 0.0
    assert stats.ms_per_completion == 0.0


def test_sampling_stats_reconstruct_the_wall_clock_of_a_real_run() -> None:
    """Per-sample latency is the batch's elapsed time shared out, so the total is the run."""
    model = tiny_model()
    samples = sample_completions(
        model,
        CharTokenizer(),
        make_prompts(),
        k=2,
        config=GenerationConfig(max_new_tokens=6, temperature=0.8),
        batch_size=2,
    )
    stats = SamplingStats.over(samples)
    assert stats.completions == 8
    assert stats.wall_ms > 0.0
    assert stats.tokens_per_second > 0.0
    assert stats.completion_tokens == sum(sample.completion_tokens for sample in samples)
