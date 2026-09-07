"""Batched greedy decoding of one completion per test example.

Evaluation is the one place in this project where decoding has to be boring. The promotion
gate reads a difference between two models, so any variance the decoder introduces is
variance the statistics must pay for before they can see the effect. Everything here is
therefore greedy, and `generate_samples` refuses a `GenerationConfig` that asks for sampling
rather than quietly ignoring the temperature it was handed.

Two details are load-bearing.

*Left padding.* A decoder-only model continues from the last position of its input. Padding
on the right leaves the model generating from a pad token, so every short prompt in a batch
produces junk -- a harness failure that looks exactly like a bad model. Padding on the left
keeps the real final token last on every row, and the attention mask keeps the filler out of
the attention.

*Length bucketing.* Prompts are batched with prompts of a similar length, because a
long-context note is several times the length of a clean one and mixing them spends most of
a batch on padding. The completions are restored to the caller's order before they are
returned: a `Sample` list that silently reordered itself would attach every completion to the
wrong gold record, and the verifier would happily score the result.

Latency is reported per example as the batch wall time divided by the batch size. That is
amortised throughput, not single-stream latency, and the two differ by a large factor on a
GPU -- 22.8 tok/s single-stream on the card this was developed on, far more in a batch. It is
recorded because it is the honest cost of the evaluation run, not as a serving benchmark.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from sftdpo.modeling.chat import ChatFormatter, encode_text
from sftdpo.schemas import Example, GenerationConfig, Sample
from sftdpo.task.generate import render_prompt

__all__ = [
    "completion_length",
    "generate_samples",
    "left_pad",
    "length_sorted_batches",
    "resolve_pad_token_id",
]


def resolve_pad_token_id(tokenizer: Any) -> int:
    """Find an id to pad with, falling back to the end-of-sequence token.

    Many instruct tokenizers ship without a dedicated pad token because nothing in
    pre-training needed one. Reusing the EOS id is the standard remedy and is safe for
    generation: the pad positions are masked out of the attention, so the id only has to be
    inside the vocabulary.

    Args:
        tokenizer: Any tokenizer exposing `pad_token_id` or `eos_token_id`.

    Returns:
        The id to use as filler.

    Raises:
        ValueError: If the tokenizer offers neither. Padding with a guessed id would corrupt
            the prompt of every short row in the batch.
    """
    for attribute in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            return int(value)
    raise ValueError(
        "tokenizer has neither pad_token_id nor eos_token_id; batched generation needs a filler id"
    )


def left_pad(
    sequences: Sequence[Sequence[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack token sequences into a batch, padding on the left.

    Args:
        sequences: One token-id sequence per row, in the order the rows should appear.
        pad_token_id: Filler id, masked out by the returned attention mask.

    Returns:
        A pair of `(input_ids, attention_mask)`, both `[batch, width]` and `torch.long`,
        where `width` is the longest sequence. Every row ends with its own final real token,
        which is the property batched decoder-only generation depends on.

    Raises:
        ValueError: If there are no sequences, or one of them is empty. An empty prompt gives
            the model no position to continue from, and the failure is much clearer here than
            inside a kernel.
    """
    if not sequences:
        raise ValueError("cannot pad an empty batch")
    lengths = [len(sequence) for sequence in sequences]
    if min(lengths) == 0:
        raise ValueError("cannot generate from an empty prompt")

    width = max(lengths)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for sequence, length in zip(sequences, lengths, strict=True):
        pad = width - length
        input_rows.append([pad_token_id] * pad + [int(token) for token in sequence])
        mask_rows.append([0] * pad + [1] * length)
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.long),
    )


def length_sorted_batches(lengths: Sequence[int], batch_size: int) -> list[tuple[int, ...]]:
    """Group example indices into batches of similar prompt length.

    Args:
        lengths: Prompt length of each example, indexed as the caller indexes them.
        batch_size: Maximum rows per batch.

    Returns:
        A partition of `range(len(lengths))`: every index appears exactly once, batches are
        at most `batch_size` long, and the order is a pure function of the lengths. Ties are
        broken by index so that two runs over the same corpus build identical batches, which
        is what makes a re-run reproduce a completion byte for byte.

    Raises:
        ValueError: If `batch_size` is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    return [tuple(order[start : start + batch_size]) for start in range(0, len(order), batch_size)]


def completion_length(
    token_ids: Sequence[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> int:
    """How many tokens the model actually produced in one generated row.

    `generate` squares the batch off by filling finished rows with the pad id, so the raw
    row length counts other rows' work. Two rules separate the model's output from the
    filler, and their order matters: the end-of-sequence token is checked first, so that when
    a tokenizer reuses EOS as its pad token -- the common case for instruct models -- the
    stop token the model chose is counted once and everything after it is filler.

    Args:
        token_ids: The generated ids for one row, with the prompt already removed.
        eos_token_id: The stop token, or None if the tokenizer has none.
        pad_token_id: The filler token, or None.

    Returns:
        The number of tokens attributable to the model, counting the stop token it emitted.
    """
    for index, token in enumerate(token_ids):
        if eos_token_id is not None and token == eos_token_id:
            return index + 1
        if pad_token_id is not None and token == pad_token_id:
            return index
    return len(token_ids)


def _model_device(model: Any) -> Any:
    """Where to put the batch, asked of the model rather than guessed from the environment."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            return parameter.device
    return torch.device("cpu")


def _check_config(config: GenerationConfig) -> None:
    if not config.greedy:
        raise ValueError(
            f"evaluation decodes greedily, but temperature is {config.temperature}; sample "
            "with the preference miner instead, where the variance is the point"
        )
    if config.max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be positive, got {config.max_new_tokens}")


def generate_samples(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    *,
    model_name: str,
    formatter: ChatFormatter | None = None,
    config: GenerationConfig | None = None,
    batch_size: int = 8,
    device: Any = None,
) -> list[Sample]:
    """Decode one greedy completion per example.

    Args:
        model: Anything with a `generate` method of the `transformers` shape. Typed loosely
            because the tests drive it with a stub: the Qwen tokenizer is not available
            offline in CI, so the real path cannot be exercised there and the harness logic
            -- padding, bucketing, token accounting -- is what the tests are for.
        tokenizer: Encoder and decoder for the prompts and completions.
        examples: The examples to run, in the order the samples should come back.
        model_name: The variant label recorded on every `Sample`. This is what the comparison
            gate later reads as "baseline" or "candidate", so it belongs to the caller.
        formatter: Chat formatter; the default matches the one used for training, which is
            the point of sharing it rather than re-deriving the prompt here.
        config: Decoding settings. Greedy only.
        batch_size: Rows per generate call.
        device: Override for where the batch is placed; taken from the model by default.

    Returns:
        One `Sample` per example, positionally aligned with `examples`.

    Raises:
        ValueError: If the config asks for sampling, if `batch_size` is not positive, if the
            tokenizer offers no filler id, or if `generate` returns a tensor that does not
            begin with the prompt it was given.
    """
    if not examples:
        return []
    settings = GenerationConfig() if config is None else config
    _check_config(settings)
    chat = ChatFormatter() if formatter is None else formatter

    prompt_ids = [
        encode_text(tokenizer, chat.prompt_text(render_prompt(example.note), tokenizer=tokenizer))
        for example in examples
    ]
    pad_token_id = resolve_pad_token_id(tokenizer)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    target = _model_device(model) if device is None else device

    # Greedy decoding is only deterministic with dropout off, and a model handed straight
    # from a trainer is still in train mode.
    if callable(getattr(model, "eval", None)):
        model.eval()

    samples: list[Sample | None] = [None] * len(examples)
    for batch in length_sorted_batches([len(ids) for ids in prompt_ids], batch_size):
        input_ids, attention_mask = left_pad([prompt_ids[index] for index in batch], pad_token_id)
        input_ids = input_ids.to(target)
        attention_mask = attention_mask.to(target)

        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_example_ms = elapsed_ms / len(batch)

        width = int(input_ids.shape[1])
        if int(generated.shape[1]) <= width:
            raise ValueError(
                f"generate returned {int(generated.shape[1])} columns for a {width}-column "
                "prompt; this code expects the prompt to be echoed back before the completion"
            )
        for row, index in zip(generated[:, width:], batch, strict=True):
            token_ids = [int(token) for token in row.tolist()]
            produced = completion_length(
                token_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id
            )
            samples[index] = Sample(
                example_id=examples[index].example_id,
                model=model_name,
                text=tokenizer.decode(token_ids[:produced], skip_special_tokens=True),
                prompt_tokens=len(prompt_ids[index]),
                completion_tokens=produced,
                latency_ms=per_example_ms,
            )

    # `length_sorted_batches` returns a partition, so every slot is filled; the comprehension
    # narrows the type rather than trusting that.
    return [sample for sample in samples if sample is not None]
