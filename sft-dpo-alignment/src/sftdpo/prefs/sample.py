"""Drawing k completions per prompt from the policy, batched and replayable.

This is the stage that pays for the preference data. Nothing here judges a completion; it
only produces them, in the quantity and with the reproducibility that the mining stage in
`sftdpo.prefs.pairs` needs in order to say anything defensible.

*Why batched.* Measured on this project's GPU (RTX 4070, bf16), Qwen2.5-0.5B-Instruct
decodes at 22.8 tokens per second single-stream. A preference run over a few hundred
prompts at k=8 and a few hundred new tokens each is millions of tokens; single-stream that
is hours, and hours of GPU time is exactly the cost that makes people quietly cut k and
then wonder why the mined pairs are so thin. Decoding is memory-bandwidth bound at batch
size one -- the weights are re-read for every single token -- so putting rows beside each
other is close to free until the batch saturates the arithmetic units. Batching is
therefore not an optimisation to add later; it is what makes the stage affordable at all.

*Why left padding, in detail.* Batching prompts of different lengths means padding, and for
a decoder-only model the padding has to go on the *left*. The model reads the last position
of the sequence to produce the next token; pad the right and that last position is a pad
token, so the model continues from padding rather than from the prompt. The attention mask
does not save you, because the mask says which positions may be *attended to*, not which
position is last. The failure is quiet: generation still returns fluent text, just text
conditioned on the wrong thing, and it only shows up as a mysterious few points of accuracy
lost between a batch size of one and a batch size of eight. `test_batched_matches_unbatched`
in the test module pins this down by asserting the equivalence that the bug destroys, and
the same test file demonstrates that right padding really does break it on this model.

*What determinism is on offer.* Two runs with the same weights, prompts, `k`, `batch_size`
and `GenerationConfig.seed` produce byte-identical text. `batch_size` is in that list for
sampled decoding, and only for sampled decoding: `generate` draws the whole batch's next
tokens from one `torch.multinomial` call, so which random numbers a given row receives
depends on how many rows sit beside it. Greedy decoding consumes no randomness at all and
is therefore invariant to batching, which is why the padding equivalence is asserted
greedily. Rather than pretend otherwise, the per-call seed is derived from
`(seed, sample_index, chunk_index)` with a hash that is stable across processes, so a run
is replayable from its recorded configuration and every `sample_index` draws from its own
stream instead of all of them sharing one.

*Where that equivalence stops, measured rather than assumed.* Greedy decoding consumes no
randomness, but it is not thereby invariant to batching on real hardware. Padding changes
the shapes the kernels are called with, and bf16 addition is not associative, so the same
logit computed in a batch of twelve and alone can differ in the last bits. Where two tokens
are near-tied, argmax then tips the other way and the rows diverge from that point on.
Measured here on Qwen2.5-0.5B-Instruct in bf16 on the RTX 4070: six prompts, greedy, batch
of six against batch of one, **five of six completions byte-identical and one different**
(`"amount": 8500` versus `"amount": "$8,500"` -- a real difference in the scored answer, not
a cosmetic one). The equivalence asserted in the tests holds because they run the two-layer
CI model in float32 on CPU, where the arithmetic is reproducible and the logit gaps are
wide; it is a test of *this module's padding logic*, which is what it is for, and it is not
a claim about bf16 GPU decoding. The invariant that does survive contact with the GPU, and
the one an experiment should rely on, is the one stated above: same weights, same prompts,
same `k`, same `batch_size`, same seed, byte-identical output. That is why `batch_size` is
recorded in the run manifest alongside the seed. Comparing two model variants means holding
it fixed between them, not hoping it does not matter.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sftdpo.modeling.chat import ChatFormatter, encode_text
from sftdpo.schemas import Example, GenerationConfig, Sample
from sftdpo.task.generate import render_prompt

__all__ = [
    "Prompt",
    "SamplingStats",
    "generation_kwargs",
    "left_pad",
    "prompts_for_examples",
    "resolve_eos_ids",
    "resolve_pad_token_id",
    "rng_devices",
    "sample_completions",
    "stream_seed",
    "trim_completion",
]


@dataclass(frozen=True, slots=True)
class Prompt:
    """One prompt to sample from, carrying the id its completions will be filed under.

    A bare string would be shorter, but every completion has to find its way back to the
    example whose gold record the verifier will score it against. Pairing the text with the
    id here makes that link a value rather than a positional convention that a later
    `zip` could silently get wrong.

    Attributes:
        example_id: The `Example.example_id` these completions belong to.
        text: The fully rendered prompt, chat template included, exactly as the model sees
            it. Rendering happens before this module so that sampling, training and
            evaluation can be shown to use one formatter.
    """

    example_id: str
    text: str


def prompts_for_examples(
    examples: Iterable[Example],
    *,
    formatter: ChatFormatter,
    tokenizer: Any = None,
) -> list[Prompt]:
    """Render the sampling prompt for each example.

    The task instruction comes from `render_prompt`, and the chat wrapper from the shared
    `ChatFormatter`. Both are imported rather than reimplemented, because a preference run
    whose prompt differs by even one character from the supervised run is measuring a
    prompt change as if it were a training effect.

    Args:
        examples: The examples to sample completions for.
        formatter: The chat formatter, which decides between a tokenizer's own template and
            the explicit ChatML fallback.
        tokenizer: Tokenizer to render with, when the formatter has none bound.

    Returns:
        One `Prompt` per example, in the order given.
    """
    return [
        Prompt(
            example_id=example.example_id,
            text=formatter.prompt_text(render_prompt(example.note), tokenizer=tokenizer),
        )
        for example in examples
    ]


@dataclass(frozen=True, slots=True)
class SamplingStats:
    """Throughput accounting for a sampling run.

    Reported because the honest answer to "why is k only 4?" is a number of tokens per
    second, and because a run that quietly fell back to a batch size of one shows up here
    as a throughput collapse long before it shows up as a schedule overrun.
    """

    completions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: float = 0.0

    @classmethod
    def over(cls, samples: Iterable[Sample]) -> SamplingStats:
        """Total up a list of samples.

        `wall_ms` is the sum of the per-sample `latency_ms`, which reconstructs the wall
        clock of the run because each batch's elapsed time is shared out evenly across its
        rows. That is the honest per-completion cost when the entire point of batching is
        to amortise a weight read across rows.
        """
        collected = list(samples)
        return cls(
            completions=len(collected),
            prompt_tokens=sum(sample.prompt_tokens for sample in collected),
            completion_tokens=sum(sample.completion_tokens for sample in collected),
            wall_ms=sum(sample.latency_ms for sample in collected),
        )

    @property
    def tokens_per_second(self) -> float:
        """Generated tokens per second of wall clock; zero when nothing was generated."""
        return 0.0 if self.wall_ms <= 0.0 else 1000.0 * self.completion_tokens / self.wall_ms

    @property
    def ms_per_completion(self) -> float:
        """Amortised wall clock per completion; zero for an empty run."""
        return 0.0 if not self.completions else self.wall_ms / self.completions

    def as_dict(self) -> dict[str, float | int]:
        """Manifest-friendly view, so a run record can quote the throughput it achieved."""
        return {
            "completions": self.completions,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_ms": self.wall_ms,
            "tokens_per_second": self.tokens_per_second,
            "ms_per_completion": self.ms_per_completion,
        }


def left_pad(rows: Sequence[Sequence[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad token id rows on the left into one batch, with the matching attention mask.

    Left rather than right, because a decoder-only model continues from the final position
    of the sequence. This is the whole padding story in four lines, kept as a pure function
    so it can be tested without a model: the invariant that matters is that every row's real
    tokens end at the last column and that the mask marks exactly those columns.

    Args:
        rows: One sequence of token ids per prompt. All must be non-empty.
        pad_token_id: Filler id. Attention masks it out, so any in-vocabulary id works.

    Returns:
        A pair of `(input_ids, attention_mask)` long tensors of shape (rows, width), where
        width is the longest row.

    Raises:
        ValueError: If there are no rows, or any row is empty. An empty prompt would leave
            the model nothing to condition on, and silently generating from padding alone is
            precisely the failure this function exists to prevent.
    """
    if not rows:
        raise ValueError("cannot pad an empty batch")
    if any(len(row) == 0 for row in rows):
        raise ValueError("cannot generate from an empty prompt")
    width = max(len(row) for row in rows)
    input_rows = [[pad_token_id] * (width - len(row)) + list(row) for row in rows]
    mask_rows = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.long),
    )


def trim_completion(ids: Sequence[int], stop_ids: frozenset[int]) -> list[int]:
    """Cut a generated row at its first stop token.

    `generate` returns a rectangular tensor: a row that finished early is filled to the
    batch's width with the pad token. Counting those filler tokens would inflate the
    throughput figure and, worse, decoding them would append junk to a JSON object that had
    already been closed properly. Cutting at the stop token handles both, and it does so
    identically whether the row was padded or simply ran to the token budget -- which is
    what lets a batched result equal an unbatched one.

    Args:
        ids: The generated token ids for one row, prompt excluded.
        stop_ids: End-of-sequence ids. Empty means the row runs to its full length.

    Returns:
        The ids before the first stop token, the stop token itself excluded.
    """
    trimmed: list[int] = []
    for token in ids:
        if token in stop_ids:
            break
        trimmed.append(int(token))
    return trimmed


def resolve_pad_token_id(tokenizer: Any, model: Any) -> int:
    """Find an id to pad with, preferring the tokenizer's own.

    Falls back to the end-of-sequence id, which is the convention for the many instruct
    checkpoints that ship without a distinct pad token. Padding is masked out of attention
    either way, so the choice only has to be a real id.

    Raises:
        ValueError: If neither a pad nor an eos id can be found. Guessing zero here would
            put a real token into the batch and change the prompt.
    """
    for candidate in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if isinstance(candidate, int):
            return candidate
    raise ValueError("tokenizer and model provide neither a pad nor an eos token id to pad with")


def resolve_eos_ids(tokenizer: Any, model: Any) -> frozenset[int]:
    """Collect every id that should end a completion.

    Both sources are unioned rather than one preferred, because a chat checkpoint commonly
    stops on a template token the tokenizer knows about and on the base end-of-text token
    the generation config knows about, and honouring only one leaves trailing rubbish on
    half the completions.

    Returns:
        The stop ids, possibly empty, in which case every row runs to the token budget.
    """
    collected: set[int] = set()
    for source in (tokenizer, getattr(model, "generation_config", None)):
        raw = getattr(source, "eos_token_id", None)
        if isinstance(raw, int):
            collected.add(raw)
        elif isinstance(raw, list | tuple):
            collected.update(int(item) for item in raw if isinstance(item, int))
    return frozenset(collected)


def stream_seed(base_seed: int, sample_index: int, chunk_index: int) -> int:
    """Derive the RNG seed for one `generate` call.

    Hashed with `blake2b` rather than Python's `hash()`, whose salt changes between
    processes: a seed that differs from run to run would make the whole reproducibility
    claim of this module false in the one place nobody looks.

    Args:
        base_seed: `GenerationConfig.seed`, the only thing a caller sets.
        sample_index: Which of the k completions is being drawn.
        chunk_index: Which batch within that sweep.

    Returns:
        A non-negative seed inside the range `torch.manual_seed` accepts.
    """
    key = f"{base_seed}:{sample_index}:{chunk_index}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**63)


def rng_devices(device: torch.device) -> list[torch.device]:
    """Which devices' RNG state a sampling call must save and restore.

    CUDA sampling draws from the device generator, so seeding without forking that state
    would perturb whatever else the process is doing -- most obviously a training loop that
    calls this mid-run to refresh its preference data.
    """
    return [device] if device.type == "cuda" else []


def generation_kwargs(config: GenerationConfig, *, pad_token_id: int) -> dict[str, Any]:
    """Translate a `GenerationConfig` into `generate` keyword arguments.

    Split out as a pure function because two of the choices here are judgement calls worth
    testing rather than trusting.

    The first is `top_k=0` under sampling. `transformers` defaults to `top_k=50`, so a
    caller who asked for `top_p=1.0` -- meaning "do not truncate the distribution" -- would
    silently get the top fifty tokens anyway. For preference mining that matters more than
    usual: the tail is where the bad completions live, and pairs need bad completions.

    The second is that temperature and top-p are omitted entirely when decoding greedily,
    because `generate` warns about settings it is about to ignore and a warning nobody can
    act on trains people to ignore warnings.

    Raises:
        ValueError: If the config asks for something `generate` cannot honour.
    """
    if config.max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {config.max_new_tokens}")
    if config.temperature < 0.0:
        raise ValueError(f"temperature must not be negative, got {config.temperature}")
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {config.top_p}")

    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": not config.greedy,
        "pad_token_id": pad_token_id,
        "num_beams": 1,
    }
    if not config.greedy:
        kwargs["temperature"] = config.temperature
        kwargs["top_p"] = config.top_p
        kwargs["top_k"] = 0
    return kwargs


def _validate_request(prompts: Sequence[Prompt], *, k: int, batch_size: int) -> None:
    """Reject a request that cannot produce usable preference data.

    Raises:
        ValueError: On a non-positive `k` or `batch_size`, on a duplicate `example_id` --
            which would merge two prompts' completions into one mining group and mislabel
            every pair drawn from it -- or on an empty prompt string.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    seen: set[str] = set()
    for prompt in prompts:
        if not prompt.text:
            raise ValueError(f"prompt {prompt.example_id!r} is empty")
        if prompt.example_id in seen:
            raise ValueError(f"duplicate example_id {prompt.example_id!r} in prompts")
        seen.add(prompt.example_id)


def _resolve_model_name(model: Any, model_name: str | None) -> str:
    if model_name is not None:
        return model_name
    configured = getattr(getattr(model, "config", None), "name_or_path", "")
    return str(configured) if configured else type(model).__name__


def _generate_batch(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    kwargs: dict[str, Any],
    eos_ids: frozenset[int],
    seed: int | None,
) -> torch.Tensor:
    """Run one `generate` call and return only the newly generated columns.

    `seed` is None for greedy decoding, and the global RNG is then left completely alone
    rather than reseeded to a value that would not be used. Under sampling the RNG is forked
    so that seeding here cannot disturb a caller's own stream.
    """
    call: dict[str, Any] = dict(kwargs)
    if eos_ids:
        # Passed explicitly so that generation stops exactly where `trim_completion` cuts;
        # otherwise a tokenizer-only stop token would be generated and then silently
        # trimmed, wasting the rest of the token budget on every row.
        call["eos_token_id"] = sorted(eos_ids)

    prompt_width = int(input_ids.shape[1])
    with torch.inference_mode():
        if seed is None:
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, **call)
        else:
            with torch.random.fork_rng(devices=rng_devices(input_ids.device)):
                torch.manual_seed(seed)
                generated = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **call
                )
    sequences: torch.Tensor = generated
    return sequences[:, prompt_width:]


def sample_completions(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Prompt],
    *,
    k: int,
    config: GenerationConfig,
    batch_size: int,
    model_name: str | None = None,
) -> list[Sample]:
    """Draw `k` completions for every prompt, in batches.

    The loop is sample-index-major -- all prompts at index 0, then all prompts at index 1 --
    so that a single `generate` call never mixes two sample indices. That keeps the seed
    derivation honest (one stream per index) and means an interrupted run has a complete
    lower-k dataset rather than a complete set for a prefix of the prompts.

    Args:
        model: A causal LM with a `generate` method. Left in whatever training mode it
            arrived in: the mode is switched to eval for the duration and restored
            afterwards, because a trainer refreshing its preference data mid-run must not
            find dropout silently disabled from then on.
        tokenizer: Used to encode the prompts and decode the completions. Padding is done
            here rather than by the tokenizer, so `padding_side` is never mutated.
        prompts: The prompts, with distinct `example_id`s.
        k: Completions per prompt.
        config: Decoding settings, recorded on nothing but consulted for everything.
        batch_size: Rows per `generate` call.
        model_name: Value for `Sample.model`. Defaults to the checkpoint name the model
            carries, falling back to its class name.

    Returns:
        The samples, grouped by prompt in the order given and ascending in `sample_index`
        within each group. An empty prompt list yields an empty list.

    Raises:
        ValueError: On a non-positive `k` or `batch_size`, a duplicate `example_id`, an
            empty prompt, a decoding setting `generate` cannot honour, or `k > 1` with
            greedy decoding -- which would return k identical completions, from which no
            preference pair can ever be mined, after paying k times the GPU cost.
    """
    _validate_request(prompts, k=k, batch_size=batch_size)
    if k > 1 and config.greedy:
        raise ValueError(
            f"k={k} with temperature 0 would return {k} identical completions; "
            "raise the temperature or set k=1"
        )
    if not prompts:
        return []

    pad_token_id = resolve_pad_token_id(tokenizer, model)
    eos_ids = resolve_eos_ids(tokenizer, model)
    kwargs = generation_kwargs(config, pad_token_id=pad_token_id)
    name = _resolve_model_name(model, model_name)
    device = next(model.parameters()).device
    # Encoded once and reused across the k sweeps: tokenising a prompt that carries the
    # whole JSON schema is not free, and doing it k times is k-1 times too many.
    encoded = [encode_text(tokenizer, prompt.text) for prompt in prompts]

    grouped: list[list[Sample]] = [[] for _ in prompts]
    was_training = bool(model.training)
    model.eval()
    try:
        for sample_index in range(k):
            for chunk_index, start in enumerate(range(0, len(prompts), batch_size)):
                rows = encoded[start : start + batch_size]
                input_ids, attention_mask = left_pad(rows, pad_token_id)
                started = time.perf_counter()
                generated = _generate_batch(
                    model,
                    input_ids.to(device),
                    attention_mask.to(device),
                    kwargs=kwargs,
                    eos_ids=eos_ids,
                    seed=None
                    if config.greedy
                    else stream_seed(config.seed, sample_index, chunk_index),
                )
                elapsed_ms = 1000.0 * (time.perf_counter() - started)
                share_ms = elapsed_ms / len(rows)
                for offset, row in enumerate(generated.tolist()):
                    completion = trim_completion(row, eos_ids)
                    grouped[start + offset].append(
                        Sample(
                            example_id=prompts[start + offset].example_id,
                            model=name,
                            text=tokenizer.decode(completion, skip_special_tokens=True),
                            sample_index=sample_index,
                            prompt_tokens=len(rows[offset]),
                            completion_tokens=len(completion),
                            latency_ms=share_ms,
                        )
                    )
    finally:
        if was_training:
            model.train()

    return [sample for group in grouped for sample in group]
