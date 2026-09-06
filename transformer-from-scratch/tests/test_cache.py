from __future__ import annotations

import pytest
import torch

from nanoformer.cache import KVCache
from nanoformer.config import ModelConfig
from nanoformer.model import Transformer


def test_incremental_decoding_matches_full_forward(tiny_model: Transformer) -> None:
    """Prefill 5 tokens, then feed one token at a time: every step's logits must equal
    the corresponding slice of a single full-sequence forward pass."""
    cfg = tiny_model.cfg
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        full = tiny_model(x).logits
        cache = KVCache(cfg, batch_size=2)
        step_logits = [tiny_model(x[:, :5], cache=cache).logits]
        for t in range(5, 12):
            step_logits.append(tiny_model(x[:, t : t + 1], cache=cache).logits)
    incremental = torch.cat(step_logits, dim=1)
    assert cache.seq_len == 12
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)


def test_chunked_prefill_matches_full_forward(tiny_model: Transformer) -> None:
    cfg = tiny_model.cfg
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        full = tiny_model(x).logits
        cache = KVCache(cfg, batch_size=1)
        a = tiny_model(x[:, :4], cache=cache).logits
        b = tiny_model(x[:, 4:], cache=cache).logits
    torch.testing.assert_close(torch.cat([a, b], dim=1), full, atol=1e-5, rtol=1e-5)


def test_overflow_is_rejected(tiny_cfg: ModelConfig) -> None:
    cache = KVCache(tiny_cfg, batch_size=1, max_seq_len=4)
    k = torch.zeros(1, tiny_cfg.kv_heads, 3, tiny_cfg.head_dim)
    cache.update(0, k, k)
    cache.advance(3)
    with pytest.raises(ValueError, match="overflow"):
        cache.update(0, k, k)
    with pytest.raises(ValueError):
        cache.advance(2)


def test_model_rejects_sequences_beyond_cache(tiny_model: Transformer) -> None:
    cache = KVCache(tiny_model.cfg, batch_size=1)
    x = torch.zeros(1, tiny_model.cfg.max_seq_len, dtype=torch.long)
    tiny_model(x, cache=cache)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        tiny_model(x[:, :1], cache=cache)


@pytest.mark.parametrize(("batch_size", "max_seq_len"), [(0, None), (1, 0), (1, 1000)])
def test_invalid_construction(
    tiny_cfg: ModelConfig, batch_size: int, max_seq_len: int | None
) -> None:
    with pytest.raises(ValueError):
        KVCache(tiny_cfg, batch_size=batch_size, max_seq_len=max_seq_len)


def test_select_keeps_only_requested_rows(tiny_cfg: ModelConfig) -> None:
    cache = KVCache(tiny_cfg, batch_size=3)
    k = (
        torch.arange(3, dtype=torch.float32)
        .view(3, 1, 1, 1)
        .expand(3, tiny_cfg.kv_heads, 2, tiny_cfg.head_dim)
    )
    cache.update(0, k, k)
    cache.advance(2)
    cache.select(torch.tensor([2, 0]))
    assert cache.batch_size == 2
    assert cache.k[0][0, 0, 0, 0].item() == 2.0
    assert cache.k[0][1, 0, 0, 0].item() == 0.0
    assert cache.seq_len == 2


def test_memory_accounting_and_reset(tiny_cfg: ModelConfig) -> None:
    cache = KVCache(tiny_cfg, batch_size=2, dtype=torch.float16)
    per_tensor = 2 * tiny_cfg.kv_heads * tiny_cfg.max_seq_len * tiny_cfg.head_dim * 2
    assert cache.memory_bytes == per_tensor * 2 * tiny_cfg.n_layers
    cache.advance(3)
    cache.reset()
    assert cache.seq_len == 0


def test_cache_casts_to_its_own_dtype(tiny_cfg: ModelConfig) -> None:
    cache = KVCache(tiny_cfg, batch_size=1, dtype=torch.bfloat16)
    k = torch.randn(1, tiny_cfg.kv_heads, 2, tiny_cfg.head_dim)
    k_out, _ = cache.update(0, k, k)
    assert k_out.dtype == torch.bfloat16
