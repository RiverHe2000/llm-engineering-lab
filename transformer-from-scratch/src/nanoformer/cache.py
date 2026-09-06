"""Pre-allocated key/value cache for incremental decoding.

Why pre-allocate?
    Growing the cache with ``torch.cat`` every step copies the whole cache each step,
    i.e. O(T^2) memory traffic over a generation of length T. Writing into a fixed
    buffer is O(T) and produces no allocator churn, which is what production engines
    do (vLLM's paged KV blocks, HF's ``StaticCache``).

Layout: one ``[batch, kv_heads, max_seq_len, head_dim]`` tensor per layer for K and V.
"""

from __future__ import annotations

import torch

from nanoformer.config import ModelConfig


class KVCache:
    """Stores past keys/values for every layer and tracks the current sequence length."""

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        *,
        max_seq_len: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.max_seq_len = cfg.max_seq_len if max_seq_len is None else max_seq_len
        if self.max_seq_len <= 0 or self.max_seq_len > cfg.max_seq_len:
            raise ValueError(
                f"max_seq_len must be in (0, {cfg.max_seq_len}], got {self.max_seq_len}"
            )
        shape = (batch_size, cfg.kv_heads, self.max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.batch_size = batch_size
        self.seq_len = 0

    def update(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``k_new``/``v_new`` (``[B, kv_heads, T, D]``) at the current position.

        Returns views over *all* cached positions ``[0, seq_len + T)`` for this layer.
        The global position is advanced by :meth:`advance` once every layer has written,
        so all layers write to the same slot range within one forward pass.
        """
        t = k_new.shape[2]
        start, end = self.seq_len, self.seq_len + t
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: writing positions [{start}, {end}) "
                f"but max_seq_len={self.max_seq_len}"
            )
        self.k[layer][:, :, start:end] = k_new.to(self.k[layer].dtype)
        self.v[layer][:, :, start:end] = v_new.to(self.v[layer].dtype)
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, t: int) -> None:
        if self.seq_len + t > self.max_seq_len:
            raise ValueError("cannot advance past max_seq_len")
        self.seq_len += t

    def reset(self) -> None:
        self.seq_len = 0

    def select(self, indices: torch.Tensor) -> None:
        """Keep only the batch rows in ``indices`` (used to drop finished sequences)."""
        self.k = [k.index_select(0, indices) for k in self.k]
        self.v = [v.index_select(0, indices) for v in self.v]
        self.batch_size = int(indices.numel())

    @property
    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.k, *self.v))
