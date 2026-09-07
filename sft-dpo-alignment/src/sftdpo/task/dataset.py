"""Split construction, the on-disk format, and descriptive statistics for the corpus.

Two properties matter more than anything else here, because without them the headline
result of the project is not defensible.

The first is that the splits are disjoint. A model that has seen a test note during
supervised fine-tuning will score well for the wrong reason, and with synthetic data the
temptation to generate one big pile and slice it afterwards is exactly how leakage gets in.
Instead every example is drawn from a stream derived from (seed, split, slice, index), and
a note that has already been produced anywhere in the corpus is rejected and redrawn. The
splits are therefore disjoint by construction rather than by a filtering step that someone
could later remove without noticing.

The second is that a finished experiment can prove which data it used. `content_hash`
covers every example of every split in order, so an experiment record that cites a hash
pins the corpus exactly; a change of seed, of split size, or of a single character in one
note produces a different hash, and `load` refuses to return a directory whose files no
longer match its manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sftdpo.schemas import Example, Slice, Split
from sftdpo.task.generate import generate_one

__all__ = [
    "Dataset",
    "DatasetStats",
    "LengthQuantiles",
    "SplitStats",
    "build_dataset",
]


FORMAT_VERSION = "sftdpo-dataset-v1"
"""Bumping this invalidates every stored hash, which is the intended blast radius of a
change to the note text or the on-disk layout."""

SPLIT_NAMES: tuple[Split, ...] = ("train", "val", "test")
SLICES: tuple[Slice, ...] = tuple(Slice)
MANIFEST_NAME = "manifest.json"
PRESENCE_FIELDS: tuple[str, ...] = (
    "objectives",
    "recommendations",
    "flags",
    "recommendation_amount",
)

DEFAULT_TRAIN = 96
DEFAULT_VAL = 24
DEFAULT_TEST = 48

_MAX_DRAW_ATTEMPTS = 8


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------


def _allocate(total: int, buckets: int) -> list[int]:
    """Split `total` across `buckets` as evenly as an integer count allows.

    The remainder goes to the earliest buckets rather than being scattered randomly, so the
    per-slice counts of a split are a deterministic function of its size and can be stated
    in a table without running the generator.

    Raises:
        ValueError: If `total` is negative or `buckets` is not positive.
    """
    if buckets <= 0:
        raise ValueError(f"buckets must be positive, got {buckets}")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _derive_seed(seed: int, split: Split, note_slice: Slice, index: int, attempt: int) -> int:
    """Derive an independent stream seed for one cell of the corpus.

    Deriving per cell rather than drawing sequentially from one stream buys two things.
    Splits cannot interfere with each other, so growing the training set leaves the
    validation and test notes untouched and a comparison across sizes stays honest. And the
    derivation is an explicit SHA-256 rather than `random.Random(str)`, so the corpus does
    not depend on CPython's internal choice of string seeding.
    """
    key = f"{FORMAT_VERSION}|{seed}|{split}|{note_slice.value}|{index}|{attempt}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _draw_unique(
    seed: int,
    split: Split,
    note_slice: Slice,
    index: int,
    seen: set[str],
    attempts: int = _MAX_DRAW_ATTEMPTS,
) -> Example:
    """Draw one example whose note has not been produced anywhere in the corpus yet.

    `seen` is mutated: it is the corpus-wide record of note text, and it is what makes the
    splits disjoint. Collisions are vanishingly rare given the size of the note space, but
    "rare" is not "impossible" and a duplicate across the train/test boundary would quietly
    inflate the result the whole project exists to report.

    Raises:
        RuntimeError: If every attempt collided, which means the vocabulary has shrunk far
            enough that the corpus can no longer be built at the requested size.
    """
    for attempt in range(attempts):
        rng = random.Random(_derive_seed(seed, split, note_slice, index, attempt))
        drawn = generate_one(note_slice, rng)
        if drawn.note not in seen:
            seen.add(drawn.note)
            return Example(
                example_id=f"{split}-{note_slice.value}-{index:04d}",
                split=split,
                slice=note_slice,
                note=drawn.note,
                gold=drawn.gold,
            )
    raise RuntimeError(
        f"could not draw a unique {note_slice.value} note for {split}[{index}] in {attempts} "
        "attempts"
    )


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile, the same convention numpy uses by default.

    Written out rather than imported so that `stats()` has no numpy dependency and returns
    the same numbers on any machine.

    Raises:
        ValueError: If `values` is empty or `q` falls outside the unit interval.
    """
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class LengthQuantiles(BaseModel):
    """Distribution of note lengths in words, summarised."""

    model_config = ConfigDict(frozen=True)

    count: int
    minimum: float
    p25: float
    p50: float
    p75: float
    maximum: float
    mean: float

    @classmethod
    def of(cls, values: Sequence[int]) -> LengthQuantiles:
        """Summarise a non-empty sequence of word counts.

        Raises:
            ValueError: If `values` is empty.
        """
        if not values:
            raise ValueError("cannot summarise an empty sequence of lengths")
        floats = [float(value) for value in values]
        return cls(
            count=len(floats),
            minimum=min(floats),
            p25=_quantile(floats, 0.25),
            p50=_quantile(floats, 0.50),
            p75=_quantile(floats, 0.75),
            maximum=max(floats),
            mean=sum(floats) / len(floats),
        )


class SplitStats(BaseModel):
    """What one split contains, at the level of detail a README table needs."""

    model_config = ConfigDict(frozen=True)

    n: int
    per_slice: dict[str, int]
    note_words: LengthQuantiles | None
    field_presence: dict[str, float]


class DatasetStats(BaseModel):
    """Descriptive statistics for a whole corpus, carrying its hash so a report cannot be
    quoted against the wrong data."""

    model_config = ConfigDict(frozen=True)

    seed: int
    content_hash: str
    total: int
    splits: dict[str, SplitStats]


def _split_stats(examples: Sequence[Example]) -> SplitStats:
    per_slice = dict.fromkeys((s.value for s in SLICES), 0)
    for example in examples:
        per_slice[example.slice.value] += 1

    recommendations = [rec for example in examples for rec in example.gold.recommendations]
    with_amount = sum(1 for rec in recommendations if rec.amount is not None)
    # Presence rates over an empty split are reported as zero rather than omitted, so the
    # shape of the statistics does not depend on the data and a table never grows a hole.
    presence = {
        "objectives": _rate(sum(1 for e in examples if e.gold.objectives), len(examples)),
        "recommendations": _rate(sum(1 for e in examples if e.gold.recommendations), len(examples)),
        "flags": _rate(sum(1 for e in examples if e.gold.flags), len(examples)),
        "recommendation_amount": _rate(with_amount, len(recommendations)),
    }
    lengths = [len(example.note.split()) for example in examples]
    return SplitStats(
        n=len(examples),
        per_slice=per_slice,
        note_words=LengthQuantiles.of(lengths) if lengths else None,
        field_presence=presence,
    )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


# --------------------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------------------


def _jsonl(examples: Sequence[Example]) -> str:
    return "".join(f"{example.model_dump_json()}\n" for example in examples)


def _read_jsonl(path: Path) -> tuple[Example, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing split file {path}")
    text = path.read_text(encoding="utf-8")
    return tuple(Example.model_validate_json(line) for line in text.splitlines())


class Dataset(BaseModel):
    """A stratified corpus of adviser notes, split three ways.

    Frozen, because a corpus whose contents can be edited after its hash has been recorded
    is a corpus whose hash proves nothing.
    """

    model_config = ConfigDict(frozen=True)

    seed: int
    train: tuple[Example, ...] = ()
    val: tuple[Example, ...] = ()
    test: tuple[Example, ...] = ()

    def examples_for(self, split: Split) -> tuple[Example, ...]:
        """The examples of one split."""
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        return self.test

    @property
    def all_examples(self) -> tuple[Example, ...]:
        """Every example, in split order."""
        return self.train + self.val + self.test

    def content_hash(self) -> str:
        """A hash over the whole corpus, split by split, in order.

        The split name is folded in as well as the examples, so moving one example from
        validation to test changes the hash even though the set of notes has not changed.
        That is the intended behaviour: a different assignment of notes to splits is a
        different experiment.
        """
        digest = hashlib.sha256()
        digest.update(FORMAT_VERSION.encode("utf-8"))
        for name in SPLIT_NAMES:
            digest.update(f"\n[{name}]\n".encode())
            digest.update(_jsonl(self.examples_for(name)).encode("utf-8"))
        return digest.hexdigest()

    def save(self, directory: Path) -> None:
        """Write the corpus as one JSONL file per split plus a manifest.

        Newlines are pinned to "\\n" so that the bytes, and therefore the hash of the files,
        are identical on Windows and on the Linux CI runner.

        Args:
            directory: Created if it does not exist; existing split files are overwritten.
        """
        directory.mkdir(parents=True, exist_ok=True)
        for name in SPLIT_NAMES:
            (directory / f"{name}.jsonl").write_text(
                _jsonl(self.examples_for(name)), encoding="utf-8", newline="\n"
            )
        manifest = {
            "format": FORMAT_VERSION,
            "seed": self.seed,
            "content_hash": self.content_hash(),
            "counts": {name: len(self.examples_for(name)) for name in SPLIT_NAMES},
        }
        (directory / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, directory: Path) -> Dataset:
        """Read a corpus back and verify it against its manifest.

        Args:
            directory: A directory previously written by `save`.

        Returns:
            The corpus, guaranteed to hash to the value recorded when it was written.

        Raises:
            FileNotFoundError: If the manifest or a split file is missing.
            ValueError: If the format version is unknown, an example is filed under the
                wrong split, or the content hash no longer matches.
        """
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing {MANIFEST_NAME} in {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported dataset format {manifest.get('format')!r}, "
                f"expected {FORMAT_VERSION!r}"
            )

        columns: dict[str, tuple[Example, ...]] = {}
        for name in SPLIT_NAMES:
            rows = _read_jsonl(directory / f"{name}.jsonl")
            for example in rows:
                if example.split != name:
                    raise ValueError(
                        f"example {example.example_id} is labelled {example.split!r} but was "
                        f"read from {name}.jsonl"
                    )
            columns[name] = rows

        dataset = cls(
            seed=int(manifest["seed"]),
            train=columns["train"],
            val=columns["val"],
            test=columns["test"],
        )
        if dataset.content_hash() != manifest.get("content_hash"):
            raise ValueError(
                f"content hash mismatch in {directory}: the split files do not match the manifest"
            )
        return dataset

    def stats(self) -> DatasetStats:
        """Per-slice counts, note-length quantiles and gold-field presence rates.

        Presence rates are the reason this exists: if the absent-fields slice ever stopped
        producing empty lists, every other check in the suite would still pass and the
        corpus would silently lose the one thing that tests whether a model invents values.
        """
        return DatasetStats(
            seed=self.seed,
            content_hash=self.content_hash(),
            total=len(self.all_examples),
            splits={name: _split_stats(self.examples_for(name)) for name in SPLIT_NAMES},
        )


def build_dataset(
    seed: int = 1,
    n_train: int = DEFAULT_TRAIN,
    n_val: int = DEFAULT_VAL,
    n_test: int = DEFAULT_TEST,
) -> Dataset:
    """Build a slice-stratified corpus with disjoint splits.

    Every slice appears in every split in known proportions: each split's size is divided
    evenly across the six slices, with any remainder going to the earliest slices. Stratify
    rather than shuffle because the whole reporting story is per-slice, and a split that
    happened to draw two long-context notes and no absent-field notes would make its
    breakdown meaningless.

    Args:
        seed: Root seed. Every note is a pure function of this and its position.
        n_train: Training examples.
        n_val: Validation examples, used for early stopping and for choosing a checkpoint.
        n_test: Test examples, touched only once per reported result.

    Returns:
        The corpus, with `train`, `val` and `test` sharing no note text.

    Raises:
        ValueError: If any split size is negative.
    """
    sizes: dict[Split, int] = {"train": n_train, "val": n_val, "test": n_test}
    seen: set[str] = set()
    columns: dict[str, tuple[Example, ...]] = {}
    for name in SPLIT_NAMES:
        rows: list[Example] = []
        for note_slice, count in zip(SLICES, _allocate(sizes[name], len(SLICES)), strict=True):
            rows.extend(_draw_unique(seed, name, note_slice, index, seen) for index in range(count))
        columns[name] = tuple(rows)
    return Dataset(seed=seed, train=columns["train"], val=columns["val"], test=columns["test"])
