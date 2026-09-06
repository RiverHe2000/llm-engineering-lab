from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nanoformer.config import ModelConfig


def test_defaults_are_valid() -> None:
    cfg = ModelConfig()
    assert cfg.head_dim == cfg.d_model // cfg.n_heads
    assert cfg.kv_heads == cfg.n_heads
    assert cfg.n_rep == 1


def test_ff_dim_follows_llama_rule() -> None:
    # 2/3 * 4 * 128 = 341.33 -> 341 -> rounded up to a multiple of 64 = 384
    assert ModelConfig(d_model=128).ff_dim == 384
    assert ModelConfig(d_model=128, d_ff=200).ff_dim == 200
    assert ModelConfig(d_model=128, ff_multiple_of=1).ff_dim == 341


def test_gqa_derived_values() -> None:
    cfg = ModelConfig(n_heads=8, n_kv_heads=2)
    assert cfg.kv_heads == 2
    assert cfg.n_rep == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 0},
        {"d_model": 0},
        {"n_layers": 0},
        {"n_heads": 0},
        {"d_model": 30, "n_heads": 4},  # not divisible
        {"d_model": 12, "n_heads": 4},  # head_dim 3 is odd
        {"n_heads": 4, "n_kv_heads": 3},  # not a divisor
        {"n_heads": 4, "n_kv_heads": 8},  # more kv than q heads
        {"d_ff": -1},
        {"ff_multiple_of": 0},
        {"max_seq_len": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"rope_theta": 0.0},
        {"norm_eps": 0.0},
        {"init_std": 0.0},
    ],
)
def test_invalid_configs_raise(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_dict_round_trip_and_unknown_keys(tmp_path: Path) -> None:
    cfg = ModelConfig(vocab_size=100, d_model=64, n_heads=8, n_kv_heads=4)
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown"):
        ModelConfig.from_dict({"bogus": 1})
    path = tmp_path / "cfg.json"
    cfg.save(path)
    assert ModelConfig.load(path) == cfg


def test_config_is_frozen_and_hashable() -> None:
    cfg = ModelConfig()
    with pytest.raises(AttributeError):
        cfg.d_model = 1  # type: ignore[misc]
    assert hash(cfg) == hash(ModelConfig())
