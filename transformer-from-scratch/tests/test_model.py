from __future__ import annotations

import math

import pytest
import torch

from nanoformer.config import ModelConfig
from nanoformer.layers import TransformerBlock
from nanoformer.model import Transformer


def expected_param_count(cfg: ModelConfig, exclude_embeddings: bool) -> int:
    d, hd = cfg.d_model, cfg.head_dim
    attn = d * cfg.n_heads * hd + 2 * d * cfg.kv_heads * hd + cfg.n_heads * hd * d
    ffn = 3 * d * cfg.ff_dim
    norms = 2 * d
    per_layer = attn + ffn + norms
    total = cfg.n_layers * per_layer + d  # + final norm
    if not exclude_embeddings:
        total += cfg.vocab_size * d * (1 if cfg.tie_embeddings else 2)
    return total


@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("n_kv_heads", [None, 2])
def test_param_count_matches_closed_form(tie: bool, n_kv_heads: int | None) -> None:
    cfg = ModelConfig(
        vocab_size=100, d_model=64, n_layers=3, n_heads=4, n_kv_heads=n_kv_heads, tie_embeddings=tie
    )
    model = Transformer(cfg)
    assert model.num_params(exclude_embeddings=True) == expected_param_count(cfg, True)
    assert model.num_params(exclude_embeddings=False) == expected_param_count(cfg, False)


def test_output_shapes_and_loss(tiny_model: Transformer) -> None:
    cfg = tiny_model.cfg
    x = torch.randint(0, cfg.vocab_size, (3, 7))
    y = torch.randint(0, cfg.vocab_size, (3, 7))
    out = tiny_model(x, y)
    assert out.logits.shape == (3, 7, cfg.vocab_size)
    assert out.loss is not None and out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert tiny_model(x).loss is None


def test_initial_loss_is_close_to_uniform(tiny_cfg: ModelConfig) -> None:
    """With N(0, 0.02) init the logits are ~0, so the loss must start near ln(vocab)."""
    model = Transformer(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (8, 16))
    y = torch.randint(0, tiny_cfg.vocab_size, (8, 16))  # independent of x: no tied-weight boost
    loss = model(x, y).loss
    assert loss is not None
    assert abs(float(loss.detach()) - math.log(tiny_cfg.vocab_size)) < 0.15


def test_ignored_targets_do_not_contribute(tiny_model: Transformer) -> None:
    x = torch.randint(0, 64, (1, 6))
    y = x.clone()
    y[0, 3:] = -1
    partial = tiny_model(x, y).loss
    # Compute the same loss manually over the kept positions.
    logits = tiny_model(x).logits[0, :3]
    manual = torch.nn.functional.cross_entropy(logits, x[0, :3])
    assert partial is not None
    torch.testing.assert_close(partial, manual)


def test_weight_tying(tiny_cfg: ModelConfig) -> None:
    tied = Transformer(tiny_cfg)
    assert tied.lm_head.weight.data_ptr() == tied.tok_emb.weight.data_ptr()
    untied = Transformer(ModelConfig(**{**tiny_cfg.to_dict(), "tie_embeddings": False}))
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()


def test_residual_projections_use_scaled_init() -> None:
    cfg = ModelConfig(vocab_size=10, d_model=256, n_layers=8, n_heads=4, max_seq_len=8)
    model = Transformer(cfg)
    expected = cfg.init_std / math.sqrt(2 * cfg.n_layers)
    block = model.blocks[0]
    assert isinstance(block, TransformerBlock)
    std_wo = float(block.attn.wo.weight.detach().std())
    std_wq = float(block.attn.wq.weight.detach().std())
    assert abs(std_wo - expected) < 0.1 * expected
    assert abs(std_wq - cfg.init_std) < 0.1 * cfg.init_std


def test_gradients_reach_every_parameter(tiny_cfg: ModelConfig) -> None:
    model = Transformer(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    loss = model(x, x).loss
    assert loss is not None
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_input_validation(tiny_model: Transformer) -> None:
    with pytest.raises(ValueError, match="batch, seq"):
        tiny_model(torch.zeros(5, dtype=torch.long))
    too_long = torch.zeros(1, tiny_model.cfg.max_seq_len + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds"):
        tiny_model(too_long)


def test_attention_impl_switch_gives_same_logits(tiny_model: Transformer) -> None:
    x = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        tiny_model.set_attention_impl(use_sdpa=True)
        a = tiny_model(x).logits
        tiny_model.set_attention_impl(use_sdpa=False)
        b = tiny_model(x).logits
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_device_dtype_and_flops(tiny_model: Transformer) -> None:
    assert tiny_model.device.type == "cpu"
    assert tiny_model.dtype == torch.float32
    assert tiny_model.estimate_flops_per_token() > 2 * tiny_model.num_params()
