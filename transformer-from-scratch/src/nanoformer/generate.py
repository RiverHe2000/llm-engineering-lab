"""Autoregressive decoding with a KV cache and the usual sampling controls."""

from __future__ import annotations

import torch
from torch import Tensor

from nanoformer.cache import KVCache
from nanoformer.model import Transformer


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the ``k`` largest logits per row; set the rest to ``-inf``. ``k <= 0`` = off."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus sampling (Holtzman et al., 2020): keep the smallest set of tokens whose
    cumulative probability reaches ``p``. The top-1 token is always kept.
    """
    if p >= 1.0:
        return logits
    if p <= 0.0:
        raise ValueError("top_p must be in (0, 1]")
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits.float(), dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Remove tokens once the mass *before* them already exceeds p.
    remove = (cumulative - probs) > p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Turn ``[B, V]`` logits into ``[B]`` token ids. ``temperature == 0`` means greedy."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    scaled = logits.float() / temperature
    scaled = top_k_filter(scaled, top_k)
    scaled = top_p_filter(scaled, top_p)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: Transformer,
    idx: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    use_cache: bool = True,
) -> Tensor:
    """Extend every row of ``idx`` (``[B, T]``) by up to ``max_new_tokens`` tokens.

    With ``use_cache`` the prompt is *prefilled* once and each further step feeds a
    single token, so the cost per step is O(T) instead of O(T^2). ``use_cache=False``
    recomputes the full sequence each step and exists so the two paths can be
    tested for equality.

    Rows that emit ``eos_id`` are frozen: subsequent positions are filled with ``eos_id``
    so the returned tensor stays rectangular.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    bsz, t0 = idx.shape
    if t0 + max_new_tokens > model.cfg.max_seq_len:
        raise ValueError(
            f"prompt ({t0}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"max_seq_len={model.cfg.max_seq_len}"
        )
    was_training = model.training
    model.eval()
    try:
        out = idx
        finished = torch.zeros(bsz, dtype=torch.bool, device=idx.device)
        cache: KVCache | None = None
        if use_cache:
            cache = KVCache(model.cfg, bsz, device=idx.device, dtype=model.dtype)
            logits = model(idx, cache=cache).logits[:, -1]
        for _ in range(max_new_tokens):
            if not use_cache:
                logits = model(out).logits[:, -1]
            next_tok = sample_next_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
            )
            if eos_id is not None:
                next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
                finished |= next_tok == eos_id
            out = torch.cat([out, next_tok[:, None]], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
            if use_cache:
                assert cache is not None
                logits = model(next_tok[:, None], cache=cache).logits[:, -1]
        return out
    finally:
        model.train(was_training)
