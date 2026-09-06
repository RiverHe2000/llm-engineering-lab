"""Model hyper-parameters as an immutable, validated dataclass.

Design notes
------------
* ``frozen=True`` so a config can be hashed and safely shared between the model,
  the checkpoint and the KV cache without accidental mutation.
* Derived quantities (``head_dim``, ``ff_dim``) are properties rather than fields so
  they can never drift out of sync with the primary fields.
* Validation happens once, in ``__post_init__``; every consumer can trust the object.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters of a LLaMA-style decoder-only Transformer."""

    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None
    """Number of key/value heads. ``None`` = multi-head attention (== n_heads);
    ``1`` = multi-query attention; anything in between = grouped-query attention."""
    d_ff: int | None = None
    """Hidden size of the SwiGLU MLP. ``None`` = LLaMA rule: 2/3 * 4 * d_model, rounded up
    to a multiple of ``ff_multiple_of``."""
    ff_multiple_of: int = 64
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for rotary embeddings")
        kv = self.kv_heads
        if kv <= 0 or kv > self.n_heads or self.n_heads % kv != 0:
            raise ValueError(
                f"n_kv_heads={kv} must be a positive divisor of n_heads={self.n_heads}"
            )
        if self.d_ff is not None and self.d_ff <= 0:
            raise ValueError("d_ff must be positive when given")
        if self.ff_multiple_of <= 0:
            raise ValueError("ff_multiple_of must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0 or self.norm_eps <= 0 or self.init_std <= 0:
            raise ValueError("rope_theta, norm_eps and init_std must be positive")

    # ----- derived quantities -------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share one KV head (1 for plain MHA)."""
        return self.n_heads // self.kv_heads

    @property
    def ff_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        hidden = int(2 * (4 * self.d_model) / 3)
        m = self.ff_multiple_of
        return m * ((hidden + m - 1) // m)

    # ----- (de)serialisation --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)
