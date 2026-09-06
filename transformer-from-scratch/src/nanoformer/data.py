"""Token datasets: encode a text corpus once to a flat ``uint16`` array, then sample
random context windows from it (the standard nanoGPT-style pipeline).

Random-window sampling avoids any epoch bookkeeping and gives every training step an
i.i.d. batch, which is what the cosine schedule and the loss curves assume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from nanoformer.tokenizer import BPETokenizer

MAX_UINT16_VOCAB = 65_536


class TokenDataset:
    """A flat 1-D token array with random-window batch sampling."""

    def __init__(self, tokens: NDArray[np.uint16], block_size: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1-D array")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if len(tokens) <= block_size:
            raise ValueError(f"need more than block_size={block_size} tokens, got {len(tokens)}")
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        """Number of distinct ``(x, y)`` windows available."""
        return len(self.tokens) - self.block_size

    @classmethod
    def from_bin(cls, path: str | Path, block_size: int) -> TokenDataset:
        """Memory-map a ``.bin`` file written by :func:`encode_to_bin` (no RAM copy)."""
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        return cls(np.asarray(arr), block_size)

    def get_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` windows; returns ``x``, ``y`` of shape ``[B, block_size]``
        where ``y`` is ``x`` shifted right by one token.

        The RNG is an explicit ``torch.Generator`` so that data order is reproducible and
        can be checkpointed independently of global RNG state.
        """
        starts = torch.randint(0, len(self), (batch_size,), generator=generator)
        bs = self.block_size
        x = torch.stack(
            [torch.from_numpy(self.tokens[s : s + bs].astype(np.int64)) for s in starts.tolist()]
        )
        y = torch.stack(
            [
                torch.from_numpy(self.tokens[s + 1 : s + 1 + bs].astype(np.int64))
                for s in starts.tolist()
            ]
        )
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def encode_to_bin(tokenizer: BPETokenizer, text: str, out_path: str | Path) -> int:
    """Encode ``text`` and write it as ``uint16`` tokens. Returns the token count."""
    if tokenizer.vocab_size > MAX_UINT16_VOCAB:
        raise ValueError("vocab too large for uint16 storage")
    ids = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(out_path)
    return int(ids.size)


def prepare_corpus(
    tokenizer: BPETokenizer,
    text_path: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.1,
) -> dict[str, int]:
    """Split a raw text file into contiguous train/val parts and encode both.

    A *contiguous* split (rather than shuffled documents) is used on purpose: with
    random-window sampling a shuffled split would leak validation windows into training.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    text = Path(text_path).read_text(encoding="utf-8")
    cut = int(len(text) * (1.0 - val_fraction))
    out_dir = Path(out_dir)
    n_train = encode_to_bin(tokenizer, text[:cut], out_dir / "train.bin")
    n_val = encode_to_bin(tokenizer, text[cut:], out_dir / "val.bin")
    return {"train_tokens": n_train, "val_tokens": n_val}
