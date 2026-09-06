from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nanoformer.layers import RMSNorm, SwiGLU, apply_rope, causal_mask, precompute_rope


class TestRMSNorm:
    def test_matches_formula(self) -> None:
        norm = RMSNorm(8, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
        x = torch.randn(3, 5, 8)
        expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
        torch.testing.assert_close(norm(x), expected)

    def test_unit_rms_output_with_unit_gain(self) -> None:
        x = torch.randn(4, 16) * 10
        y = RMSNorm(16)(x)
        torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(4), atol=1e-4, rtol=1e-4)

    def test_half_precision_input_does_not_overflow(self) -> None:
        # 300**2 = 90000 overflows fp16 (max 65504) if the statistic were computed in fp16.
        x = torch.full((2, 8), 300.0, dtype=torch.float16)
        y = RMSNorm(8)(x)
        assert torch.isfinite(y).all()


class TestRoPE:
    def test_table_shapes_and_first_frequency(self) -> None:
        cos, sin = precompute_rope(head_dim=8, max_seq_len=10, theta=10_000.0)
        assert cos.shape == (10, 4) and sin.shape == (10, 4)
        # Dimension 0 rotates at 1 rad/position: angle at position 3 is 3.
        torch.testing.assert_close(cos[3, 0], torch.tensor(3.0).cos())
        torch.testing.assert_close(sin[3, 0], torch.tensor(3.0).sin())

    def test_odd_head_dim_rejected(self) -> None:
        with pytest.raises(ValueError):
            precompute_rope(7, 4, 10_000.0)

    def test_rotation_preserves_norm(self) -> None:
        cos, sin = precompute_rope(16, 32, 10_000.0)
        x = torch.randn(2, 3, 32, 16)
        y = apply_rope(x, cos, sin)
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))

    def test_position_zero_is_identity(self) -> None:
        cos, sin = precompute_rope(16, 4, 10_000.0)
        x = torch.randn(1, 1, 1, 16)
        torch.testing.assert_close(apply_rope(x, cos[:1], sin[:1]), x)

    def test_relative_position_property(self) -> None:
        """<RoPE(q, m), RoPE(k, n)> must depend only on m - n, not on the absolute shift."""
        cos, sin = precompute_rope(32, 64, 10_000.0)
        q = torch.randn(1, 1, 1, 32)
        k = torch.randn(1, 1, 1, 32)

        def score(m: int, n: int) -> torch.Tensor:
            qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
            kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
            return (qm * kn).sum()

        torch.testing.assert_close(score(5, 2), score(25, 22), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(score(7, 7), score(40, 40), atol=1e-4, rtol=1e-4)
        assert not torch.isclose(score(5, 2), score(5, 4), atol=1e-3)


class TestSwiGLU:
    def test_matches_formula_and_shape(self) -> None:
        ffn = SwiGLU(8, 16).eval()
        x = torch.randn(2, 4, 8)
        expected = ffn.w_down(F.silu(ffn.w_gate(x)) * ffn.w_up(x))
        out = ffn(x)
        assert out.shape == (2, 4, 8)
        torch.testing.assert_close(out, expected)


class TestCausalMask:
    def test_square_is_lower_triangular(self) -> None:
        m = causal_mask(4, 4, "cpu")
        assert torch.equal(m, torch.ones(4, 4, dtype=torch.bool).tril())

    def test_decoding_offset(self) -> None:
        # 2 new queries over 5 keys: query 0 is absolute position 3, query 1 is position 4.
        m = causal_mask(2, 5, "cpu")
        expected = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        assert torch.equal(m, expected)

    def test_more_queries_than_keys_rejected(self) -> None:
        with pytest.raises(ValueError):
            causal_mask(5, 4, "cpu")
