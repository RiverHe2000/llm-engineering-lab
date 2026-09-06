"""Sampling parameters and the pure tensor transforms that implement them.

Kept free of model code so every rule (top-k, nucleus, repetition penalty, greedy) can be
unit-tested on hand-built logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    """0 = greedy (argmax); higher = flatter distribution."""
    top_k: int = 0
    """Keep only the k most likely tokens (0 = disabled)."""
    top_p: float = 1.0
    """Nucleus sampling: keep the smallest set with cumulative probability >= p."""
    repetition_penalty: float = 1.0
    """> 1 discourages tokens already present in prompt + generation (CTRL, Keskar 2019)."""
    seed: int | None = None
    """Per-request seed; makes a sampled completion reproducible."""

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the ``k`` largest logits per row; the rest become ``-inf``. ``k <= 0`` = off."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filter (Holtzman et al., 2020); the top-1 token is always kept."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits.float(), dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = (cumulative - probs) > p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def apply_repetition_penalty(logits: Tensor, seen_ids: Tensor, penalty: float) -> Tensor:
    """For every token in ``seen_ids``: divide a positive logit by ``penalty``, multiply a
    negative one — i.e. always push it toward "less likely" (the HF/CTRL rule).
    ``logits`` is ``[V]`` and ``seen_ids`` a 1-D tensor of token ids."""
    if penalty == 1.0 or seen_ids.numel() == 0:
        return logits
    out = logits.clone()
    ids = torch.unique(seen_ids)
    vals = out[ids]
    out[ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return out


def sample_token(
    logits: Tensor, params: SamplingParams, generator: torch.Generator | None = None
) -> Tensor:
    """Turn ``[B, V]`` logits into ``[B]`` token ids under ``params``."""
    if logits.dim() != 2:
        raise ValueError("logits must be [batch, vocab]")
    if params.is_greedy:
        return logits.argmax(dim=-1)
    scaled = logits.float() / params.temperature
    scaled = top_k_filter(scaled, params.top_k)
    scaled = top_p_filter(scaled, params.top_p)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
