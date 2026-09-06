"""Building blocks of the Transformer.

Every class here is small, documented, and has an isolated unit test. The attention
module ships with a *reference* implementation (plain matmul + softmax) that the fast
path (``F.scaled_dot_product_attention``) is tested against.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanoformer.cache import KVCache
from nanoformer.config import ModelConfig

# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square normalisation (Zhang & Sennrich, 2019), as used in LLaMA.

    ``y = x / sqrt(mean(x^2) + eps) * g``

    Compared with LayerNorm it drops mean-centring and the bias, which is cheaper and
    empirically equivalent at scale. Statistics are computed in float32 even under
    bf16/fp16 autocast — squaring low-precision activations loses accuracy fast.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


# --------------------------------------------------------------------------------------
# Rotary position embeddings (Su et al., 2021)
# --------------------------------------------------------------------------------------


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[Tensor, Tensor]:
    """Return ``(cos, sin)`` tables of shape ``[max_seq_len, head_dim // 2]``.

    Frequency ``i`` rotates at ``theta ** (-2i / head_dim)`` radians per position, so low
    dimensions encode fine-grained (local) offsets and high dimensions encode long-range
    ones.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta**exponents)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` (``[B, H, T, D]``) by the angles in ``cos``/``sin`` (``[T, D/2]``).

    Uses the "rotate-half" pairing ``(x[:D/2], x[D/2:])`` — the same convention as
    HF LLaMA — so checkpoints could be converted without re-permuting weights.
    Because rotation is orthogonal, ``<RoPE(q, m), RoPE(k, n)>`` depends only on ``m - n``;
    that relative-position property is verified in ``tests/test_layers.py``.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c = cos.to(x.dtype)[None, None, :, :]
    s = sin.to(x.dtype)[None, None, :, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# --------------------------------------------------------------------------------------
# Feed-forward
# --------------------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated MLP ``w_down(silu(w_gate(x)) * w_up(x))`` (Shazeer, 2020).

    Three matrices instead of two, so LLaMA shrinks the hidden size to 2/3 * 4d to keep
    the parameter count of a classic 4d GELU MLP.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
        return out


# --------------------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------------------


def causal_mask(t_query: int, t_key: int, device: torch.device | str) -> Tensor:
    """Boolean ``[t_query, t_key]`` mask; ``True`` = may attend.

    The queries are assumed to be the *last* ``t_query`` positions of a sequence of
    length ``t_key`` (this is exactly the incremental-decoding situation), so query ``i``
    sits at absolute position ``t_key - t_query + i`` and may see keys ``<=`` that.
    """
    if t_query > t_key:
        raise ValueError("t_query cannot exceed t_key")
    offset = t_key - t_query
    q_pos = torch.arange(t_query, device=device)[:, None] + offset
    k_pos = torch.arange(t_key, device=device)[None, :]
    return k_pos <= q_pos


def attention_reference(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor, dropout_p: float = 0.0, training: bool = False
) -> Tensor:
    """Textbook attention, kept as the oracle for the fused kernel.

    ``softmax(QK^T / sqrt(d) + mask) V`` with the softmax evaluated in float32.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    probs = F.dropout(probs, p=dropout_p, training=training)
    return probs @ v


class CausalSelfAttention(nn.Module):
    """Multi-head / grouped-query causal self-attention with optional KV cache.

    GQA (Ainslie et al., 2023): ``kv_heads < n_heads`` key/value heads are shared by
    groups of ``n_rep`` query heads. It cuts the KV-cache size by ``n_rep`` — the main
    memory cost at inference — with a negligible quality drop.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_heads = cfg.kv_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.layer_idx = layer_idx
        self.dropout_p = cfg.dropout
        self.use_sdpa = True

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None = None) -> Tensor:
        bsz, t, _ = x.shape
        q = self.wq(x).view(bsz, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, t, self.kv_heads, self.head_dim).transpose(1, 2)

        # Positions are encoded *before* caching so cached keys never need re-rotation.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        t_key = k.shape[2]
        dropout_p = self.dropout_p if self.training else 0.0
        if self.use_sdpa:
            if t == t_key:
                # Full causal square: let the kernel build the mask (FlashAttention path).
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            else:
                mask = causal_mask(t, t_key, x.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        else:
            mask = causal_mask(t, t_key, x.device)
            y = attention_reference(q, k, v, mask, dropout_p=dropout_p, training=self.training)

        y = y.transpose(1, 2).contiguous().view(bsz, t, self.n_heads * self.head_dim)
        out: Tensor = self.resid_dropout(self.wo(y))
        return out


# --------------------------------------------------------------------------------------
# Block
# --------------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm residual block: ``x + Attn(Norm(x))`` then ``x + FFN(Norm(x))``.

    Pre-norm (as opposed to the original post-norm) keeps the residual stream an identity
    path, which is why deep Transformers train stably without warm-up tricks.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ff_dim, cfg.dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        out: Tensor = x + self.ffn(self.ffn_norm(x))
        return out
