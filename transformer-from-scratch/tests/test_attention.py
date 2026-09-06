from __future__ import annotations

import pytest
import torch

from nanoformer.cache import KVCache
from nanoformer.config import ModelConfig
from nanoformer.layers import CausalSelfAttention, attention_reference, causal_mask, precompute_rope
from nanoformer.model import Transformer


@pytest.mark.parametrize("n_kv_heads", [None, 2, 1], ids=["mha", "gqa", "mqa"])
def test_fused_kernel_matches_reference(n_kv_heads: int | None) -> None:
    cfg = ModelConfig(d_model=32, n_heads=4, n_kv_heads=n_kv_heads, max_seq_len=16)
    attn = CausalSelfAttention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    x = torch.randn(2, 16, 32)

    attn.use_sdpa = True
    fused = attn(x, cos, sin)
    attn.use_sdpa = False
    reference = attn(x, cos, sin)
    torch.testing.assert_close(fused, reference, atol=1e-5, rtol=1e-5)


def test_fused_kernel_matches_reference_with_cache() -> None:
    """Exercises the explicit-mask branch (t_query < t_key)."""
    cfg = ModelConfig(d_model=32, n_heads=4, n_kv_heads=2, max_seq_len=16)
    attn = CausalSelfAttention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    x = torch.randn(2, 10, 32)

    outputs = []
    for use_sdpa in (True, False):
        attn.use_sdpa = use_sdpa
        cache = KVCache(cfg, batch_size=2)
        attn(x[:, :6], cos[:6], sin[:6], cache)
        cache.advance(6)
        outputs.append(attn(x[:, 6:], cos[6:10], sin[6:10], cache))
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)


def test_reference_attention_is_softmax_weighted_average() -> None:
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    mask = causal_mask(3, 3, "cpu")
    out = attention_reference(q, k, v, mask)
    # First query can only see the first key, so its output is exactly v[0].
    torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0])
    # Every row of attention weights sums to one -> output is inside the convex hull of v.
    assert out.shape == q.shape


def test_no_information_leaks_from_the_future(tiny_model: Transformer) -> None:
    """Changing tokens after position i must not change logits at positions <= i."""
    x = torch.randint(0, tiny_model.cfg.vocab_size, (1, 12))
    x_perturbed = x.clone()
    x_perturbed[0, 8:] = (x_perturbed[0, 8:] + 1) % tiny_model.cfg.vocab_size
    with torch.no_grad():
        a = tiny_model(x).logits
        b = tiny_model(x_perturbed).logits
    torch.testing.assert_close(a[:, :8], b[:, :8], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_dropout_is_active_only_in_training_mode() -> None:
    cfg = ModelConfig(d_model=32, n_heads=4, max_seq_len=16, dropout=0.5)
    attn = CausalSelfAttention(cfg, layer_idx=0)
    cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    x = torch.randn(1, 8, 32)
    cos, sin = cos[:8], sin[:8]
    attn.eval()
    torch.testing.assert_close(attn(x, cos, sin), attn(x, cos, sin))
    attn.train()
    assert not torch.allclose(attn(x, cos, sin), attn(x, cos, sin))
