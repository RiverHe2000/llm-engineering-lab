from __future__ import annotations

import numpy as np
import pytest
import torch

from nanoformer.config import ModelConfig
from nanoformer.data import TokenDataset
from nanoformer.model import Transformer


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32
    )


@pytest.fixture
def tiny_model(tiny_cfg: ModelConfig) -> Transformer:
    torch.manual_seed(0)
    return Transformer(tiny_cfg).eval()


@pytest.fixture
def cyclic_tokens() -> np.ndarray:
    """A perfectly predictable token stream (0,1,...,15,0,1,...) for overfitting tests."""
    return np.tile(np.arange(16, dtype=np.uint16), 200)


@pytest.fixture
def cyclic_dataset(cyclic_tokens: np.ndarray) -> TokenDataset:
    return TokenDataset(cyclic_tokens, block_size=16)
