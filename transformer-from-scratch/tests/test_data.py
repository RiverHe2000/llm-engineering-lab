from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nanoformer.data import TokenDataset, encode_to_bin, prepare_corpus
from nanoformer.tokenizer import BPETokenizer


def test_batches_are_shifted_windows(cyclic_dataset: TokenDataset) -> None:
    rng = torch.Generator().manual_seed(0)
    x, y = cyclic_dataset.get_batch(4, rng)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.dtype == torch.long
    torch.testing.assert_close(y[:, :-1], x[:, 1:])
    assert len(cyclic_dataset) == len(cyclic_dataset.tokens) - 16


def test_batches_are_reproducible_from_generator_seed(cyclic_dataset: TokenDataset) -> None:
    a = cyclic_dataset.get_batch(3, torch.Generator().manual_seed(7))
    b = cyclic_dataset.get_batch(3, torch.Generator().manual_seed(7))
    c = cyclic_dataset.get_batch(3, torch.Generator().manual_seed(8))
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], c[0])


@pytest.mark.parametrize(
    ("tokens", "block"),
    [
        (np.zeros((2, 2), dtype=np.uint16), 1),
        (np.zeros(10, dtype=np.uint16), 0),
        (np.zeros(10, dtype=np.uint16), 10),
    ],
)
def test_dataset_validation(tokens: np.ndarray, block: int) -> None:
    with pytest.raises(ValueError):
        TokenDataset(tokens, block)


def test_encode_to_bin_and_memmap(tmp_path: Path) -> None:
    tok = BPETokenizer([], special_tokens=[])
    n = encode_to_bin(tok, "abcdef" * 10, tmp_path / "x.bin")
    assert n == 60
    ds = TokenDataset.from_bin(tmp_path / "x.bin", block_size=4)
    assert len(ds) == 56
    assert ds.tokens[:3].tolist() == list(b"abc")


def test_prepare_corpus_contiguous_split(tmp_path: Path) -> None:
    text_path = tmp_path / "corpus.txt"
    text_path.write_text("x" * 90 + "y" * 10, encoding="utf-8")
    tok = BPETokenizer([], special_tokens=[])
    stats = prepare_corpus(tok, text_path, tmp_path / "out", val_fraction=0.1)
    assert stats == {"train_tokens": 90, "val_tokens": 10}
    val = np.fromfile(tmp_path / "out" / "val.bin", dtype=np.uint16)
    assert (val == ord("y")).all(), "validation split must be the tail of the corpus"
    with pytest.raises(ValueError):
        prepare_corpus(tok, text_path, tmp_path / "out2", val_fraction=1.5)
