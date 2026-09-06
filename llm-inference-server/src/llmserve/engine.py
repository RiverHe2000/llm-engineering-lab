"""Batched, KV-cached generation for any Hugging Face causal LM.

The engine is synchronous and stateless between calls; the scheduler decides *what* goes
into a batch, the engine decides *how* to run it:

* prompts of different lengths are **left-padded** and given explicit ``position_ids`` so
  each sequence sees the same positions it would see alone (tested for equality);
* the prompt is **prefilled** once, then one token per step is fed with the KV cache;
* rows that finish (EOS or their own ``max_new_tokens``) are **dropped from the batch**
  (``DynamicCache.batch_select_indices``) so a long request does not pay for short ones;
* sampling parameters are per request; identical parameters take a vectorised path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

import torch
from torch import Tensor, nn

from llmserve.config import Settings
from llmserve.sampling import SamplingParams, apply_repetition_penalty, sample_token

log = logging.getLogger(__name__)

FinishReason = Literal["stop", "length"]


class ContextLengthError(ValueError):
    """prompt tokens + max_new_tokens would exceed the model context."""


class TokenizerLike(Protocol):
    padding_side: str
    eos_token_id: int | None
    pad_token_id: int | None

    def __call__(
        self,
        text: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> Any: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int
    sampling: SamplingParams = field(default_factory=SamplingParams)
    stop_on_eos: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")


@dataclass
class GenerationResult:
    request_id: str
    text: str
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason
    latency_ms: float


class GenerationEngine:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerLike,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_context: int = 1024,
        eos_token_id: int | None = None,
        model_name: str = "custom",
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_context = max_context
        self.model_name = model_name
        self.model = model.to(self.device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # GPT-2-style tokenizers have no pad token; EOS is the standard stand-in.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)
        self.batches_run = 0

    # ----- construction ---------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> GenerationEngine:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from llmserve.quantize import maybe_compile, quantize_dynamic_int8

        device = settings.resolve_device()
        dtype = settings.resolve_dtype(device)
        log.info("loading %s on %s (%s)", settings.model_name, device, dtype)
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        model = AutoModelForCausalLM.from_pretrained(settings.model_name, dtype=dtype)
        if settings.quantize_int8:
            if device.type != "cpu":
                raise ValueError("INT8 dynamic quantization is a CPU-only path")
            model = quantize_dynamic_int8(model)
        model = maybe_compile(model, settings.torch_compile)
        return cls(
            model,
            cast(TokenizerLike, tokenizer),
            device=device,
            dtype=dtype,
            max_context=settings.max_context,
            model_name=settings.model_name,
            seed=settings.seed,
        )

    # ----- generation -----------------------------------------------------------------
    @torch.inference_mode()
    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            raise ValueError("empty batch")
        t0 = time.perf_counter()
        bsz = len(requests)

        enc = self.tokenizer(
            [r.prompt for r in requests],
            padding=True,
            truncation=True,
            max_length=self.max_context - 1,
            return_tensors="pt",
        )
        input_ids: Tensor = enc["input_ids"].to(self.device)
        attn: Tensor = enc["attention_mask"].to(self.device)
        prompt_lens = attn.sum(dim=1).tolist()
        for r, n_prompt in zip(requests, prompt_lens, strict=True):
            if n_prompt + r.max_new_tokens > self.max_context:
                raise ContextLengthError(
                    f"request {r.request_id}: {n_prompt} prompt tokens + "
                    f"{r.max_new_tokens} new tokens exceeds max_context={self.max_context}"
                )

        # Left padding: positions must start at 0 at the first real token of each row.
        position_ids = (attn.cumsum(dim=-1) - 1).clamp(min=0)
        out = self.model(
            input_ids=input_ids, attention_mask=attn, position_ids=position_ids, use_cache=True
        )
        cache = out.past_key_values
        logits: Tensor = out.logits[:, -1, :].float()

        prompt_rows = [input_ids[i][attn[i].bool()] for i in range(bsz)]
        generated: list[list[int]] = [[] for _ in range(bsz)]
        finish: list[FinishReason | None] = [None] * bsz
        active = list(range(bsz))
        cur_attn = attn
        cur_pos = position_ids[:, -1]
        generators = [self._generator_for(r) for r in requests]

        for _ in range(max(r.max_new_tokens for r in requests)):
            next_tokens = self._sample_rows(
                logits,
                [requests[i] for i in active],
                [prompt_rows[i] for i in active],
                [generated[i] for i in active],
                [generators[i] for i in active],
            )
            for j, i in enumerate(active):
                tok = int(next_tokens[j])
                generated[i].append(tok)
                if requests[i].stop_on_eos and tok == self.eos_token_id:
                    finish[i] = "stop"
                elif len(generated[i]) >= requests[i].max_new_tokens:
                    finish[i] = "length"

            keep = [j for j, i in enumerate(active) if finish[i] is None]
            if not keep:
                break
            if len(keep) < len(active):
                idx = torch.tensor(keep, device=self.device)
                cache.batch_select_indices(idx)
                next_tokens = next_tokens[idx]
                cur_attn = cur_attn[idx]
                cur_pos = cur_pos[idx]
                active = [active[j] for j in keep]

            cur_attn = torch.cat(
                [cur_attn, torch.ones(len(active), 1, dtype=cur_attn.dtype, device=self.device)],
                dim=1,
            )
            cur_pos = cur_pos + 1
            out = self.model(
                input_ids=next_tokens[:, None],
                attention_mask=cur_attn,
                position_ids=cur_pos[:, None],
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :].float()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.batches_run += 1
        results: list[GenerationResult] = []
        for i, r in enumerate(requests):
            ids = generated[i]
            reason: FinishReason = finish[i] or "length"
            visible = ids[:-1] if reason == "stop" else ids
            results.append(
                GenerationResult(
                    request_id=r.request_id,
                    text=self.tokenizer.decode(visible, skip_special_tokens=True),
                    token_ids=ids,
                    prompt_tokens=int(prompt_lens[i]),
                    completion_tokens=len(ids),
                    finish_reason=reason,
                    latency_ms=latency_ms,
                )
            )
        return results

    # ----- helpers --------------------------------------------------------------------
    def _generator_for(self, request: GenerationRequest) -> torch.Generator:
        if request.sampling.seed is None:
            return self.generator
        return torch.Generator(device=self.device).manual_seed(request.sampling.seed)

    def _sample_rows(
        self,
        logits: Tensor,
        requests: list[GenerationRequest],
        prompt_rows: list[Tensor],
        generated_rows: list[list[int]],
        generators: list[torch.Generator],
    ) -> Tensor:
        params = [r.sampling for r in requests]
        uniform = all(p == params[0] for p in params)
        seeded = any(p.seed is not None for p in params)
        if uniform and params[0].repetition_penalty == 1.0 and not seeded:
            # Fast path: one vectorised sampling call for the whole batch. Seeded requests
            # take the per-row path so a request's output never depends on its batch-mates.
            return sample_token(logits, params[0], generators[0])
        tokens: list[Tensor] = []
        for j, p in enumerate(params):
            row = logits[j]
            if p.repetition_penalty > 1.0:
                seen = torch.cat(
                    [
                        prompt_rows[j],
                        torch.tensor(generated_rows[j], device=self.device, dtype=torch.long),
                    ]
                )
                row = apply_repetition_penalty(row, seen, p.repetition_penalty)
            tokens.append(sample_token(row[None], p, generators[j])[0])
        return torch.stack(tokens)

    def count_tokens(self, text: str) -> int:
        enc = self.tokenizer(
            [text],
            padding=False,
            truncation=False,
            max_length=self.max_context,
            return_tensors="pt",
        )
        return int(enc["input_ids"].shape[1])
