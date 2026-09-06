"""Financial PhraseBank (Malo et al., 2014) loading, stratified splitting and batching.

The dataset ships as a zip of ``sentence@label`` lines inside the HF dataset repo
``takala/financial_phrasebank``. Its old ``datasets`` loading script no longer runs on
``datasets>=3``, so the zip is fetched directly with ``huggingface_hub`` and parsed
here — fewer moving parts and no dependency on a deprecated code path.
"""

from __future__ import annotations

import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LABEL_NAMES: tuple[str, ...] = ("negative", "neutral", "positive")
LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}

HF_REPO = "takala/financial_phrasebank"
HF_ZIP = "data/FinancialPhraseBank-v1.0.zip"

Agreement = Literal["50", "66", "75", "all"]
_AGREEMENT_FILES: dict[str, str] = {
    "50": "Sentences_50Agree.txt",
    "66": "Sentences_66Agree.txt",
    "75": "Sentences_75Agree.txt",
    "all": "Sentences_AllAgree.txt",
}


@dataclass(frozen=True)
class Example:
    text: str
    label: int


def parse_phrasebank(lines: Iterable[str]) -> list[Example]:
    """Parse ``sentence@label`` lines. Fails loudly on malformed input rather than
    silently dropping rows — a silently shrunk test set is a classic evaluation bug."""
    out: list[Example] = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if "@" not in line:
            raise ValueError(f"line {lineno}: missing '@' separator")
        text, label = line.rsplit("@", 1)
        label = label.strip().lower()
        if label not in LABEL_TO_ID:
            raise ValueError(f"line {lineno}: unknown label {label!r}")
        out.append(Example(text=text.strip(), label=LABEL_TO_ID[label]))
    return out


def read_phrasebank_zip(zip_path: str | Path, agreement: Agreement = "all") -> list[Example]:
    if agreement not in _AGREEMENT_FILES:
        raise ValueError(f"agreement must be one of {sorted(_AGREEMENT_FILES)}")
    target = _AGREEMENT_FILES[agreement].lower()
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n for n in zf.namelist() if n.lower().endswith(target) and not n.startswith("__MACOSX")
        ]
        if not members:
            raise FileNotFoundError(f"{target} not found in {zip_path}")
        # The 2014 release is ISO-8859-1 (e.g. 'ñ' at byte 0xF1), not UTF-8.
        text = zf.read(members[0]).decode("latin-1")
    return parse_phrasebank(text.splitlines())


def load_financial_phrasebank(
    agreement: Agreement = "all", cache_dir: str | Path | None = None
) -> list[Example]:
    """Download (once, cached by huggingface_hub) and parse the requested subset.

    ``"all"`` = 2,264 sentences on which all annotators agreed; the cleanest labels,
    hence the standard benchmark subset.
    """
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(HF_REPO, HF_ZIP, repo_type="dataset", cache_dir=cache_dir)
    return read_phrasebank_zip(zip_path, agreement)


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSplit:
    train: list[Example]
    val: list[Example]
    test: list[Example]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def label_distribution(examples: Sequence[Example]) -> dict[str, int]:
    counts = Counter(e.label for e in examples)
    return {LABEL_NAMES[i]: counts.get(i, 0) for i in range(len(LABEL_NAMES))}


def stratified_split(
    examples: Sequence[Example],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> DataSplit:
    """Per-class shuffle-and-cut so every split has the same label mix.

    Stratification matters here because the minority class (negative, 13%) is exactly
    the one a bank cares about; a random 15% test cut could easily under-sample it.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("need val_fraction, test_fraction >= 0 and their sum < 1")
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[Example]] = {}
    for e in examples:
        by_class.setdefault(e.label, []).append(e)

    train: list[Example] = []
    val: list[Example] = []
    test: list[Example] = []
    for label in sorted(by_class):
        items = by_class[label]
        perm = rng.permutation(len(items))
        n_test = round(len(items) * test_fraction)
        n_val = round(len(items) * val_fraction)
        test.extend(items[i] for i in perm[:n_test])
        val.extend(items[i] for i in perm[n_test : n_test + n_val])
        train.extend(items[i] for i in perm[n_test + n_val :])

    def shuffled(xs: list[Example]) -> list[Example]:
        return [xs[i] for i in rng.permutation(len(xs))]

    return DataSplit(train=shuffled(train), val=shuffled(val), test=shuffled(test))


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------


class TokenizerLike(Protocol):
    """The slice of the HF tokenizer interface we rely on (keeps tests offline)."""

    def __call__(
        self,
        text: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> Any: ...


class TextClassificationDataset(Dataset[Example]):
    def __init__(self, examples: Sequence[Example]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


class Collator:
    """Tokenise a batch of examples with dynamic padding (pad to the longest in the
    batch, not to ``max_length``): ~2x fewer padded tokens on this dataset."""

    def __init__(self, tokenizer: TokenizerLike, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[Example]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            [e.text for e in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([e.label for e in batch], dtype=torch.long),
        }


def make_loader(
    examples: Sequence[Example],
    tokenizer: TokenizerLike,
    *,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader[Example]:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        TextClassificationDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collator(tokenizer, max_length),
        generator=generator,
    )
