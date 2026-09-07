"""Tests for split construction, the on-disk format and the statistics.

The two claims this module has to defend are that the splits share no note text and that a
content hash pins the corpus. Both are tested as properties over a rebuilt corpus rather
than against a stored fixture, because a fixture would freeze whatever the generator did on
the day it was written and would keep passing after the generator broke.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sftdpo.schemas import Example, Slice
from sftdpo.task.dataset import (
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_VAL,
    FORMAT_VERSION,
    MANIFEST_NAME,
    PRESENCE_FIELDS,
    SPLIT_NAMES,
    Dataset,
    LengthQuantiles,
    _allocate,
    _derive_seed,
    _draw_unique,
    _quantile,
    build_dataset,
)

SMALL = {"n_train": 12, "n_val": 6, "n_test": 6}


@pytest.fixture(scope="module")
def corpus() -> Dataset:
    """One small corpus shared by the read-only tests; building it is the slow part."""
    return build_dataset(seed=1, **SMALL)


# --------------------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------------------


def test_allocate_divides_evenly_when_it_can() -> None:
    assert _allocate(12, 6) == [2, 2, 2, 2, 2, 2]


def test_allocate_gives_the_remainder_to_the_earliest_buckets() -> None:
    assert _allocate(14, 6) == [3, 3, 2, 2, 2, 2]


@given(st.integers(min_value=0, max_value=500), st.integers(min_value=1, max_value=12))
def test_allocate_conserves_the_total_and_stays_balanced(total: int, buckets: int) -> None:
    """A conservation law and a balance bound, which together pin the function completely."""
    parts = _allocate(total, buckets)
    assert len(parts) == buckets
    assert sum(parts) == total
    assert max(parts) - min(parts) <= 1
    assert min(parts) >= 0


def test_allocate_rejects_a_non_positive_bucket_count() -> None:
    with pytest.raises(ValueError, match="buckets must be positive"):
        _allocate(10, 0)


def test_allocate_rejects_a_negative_total() -> None:
    with pytest.raises(ValueError, match="total must be non-negative"):
        _allocate(-1, 6)


def test_the_default_sizes_stratify_without_a_remainder() -> None:
    """The advertised defaults divide evenly, so the documented per-slice table is exact."""
    for size in (DEFAULT_TRAIN, DEFAULT_VAL, DEFAULT_TEST):
        assert size % len(Slice) == 0


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------


def test_derived_seeds_differ_across_every_coordinate() -> None:
    base = _derive_seed(1, "train", Slice.CLEAN, 0, 0)
    assert base != _derive_seed(2, "train", Slice.CLEAN, 0, 0)
    assert base != _derive_seed(1, "val", Slice.CLEAN, 0, 0)
    assert base != _derive_seed(1, "train", Slice.MANY_ITEMS, 0, 0)
    assert base != _derive_seed(1, "train", Slice.CLEAN, 1, 0)
    assert base != _derive_seed(1, "train", Slice.CLEAN, 0, 1)


def test_derived_seeds_are_stable_and_in_range() -> None:
    assert _derive_seed(1, "train", Slice.CLEAN, 0, 0) == _derive_seed(
        1, "train", Slice.CLEAN, 0, 0
    )
    assert 0 <= _derive_seed(7, "test", Slice.LONG_CONTEXT, 3, 0) < 2**64


def test_draw_unique_redraws_when_the_note_is_already_in_the_corpus() -> None:
    """The retry path: a note already seen anywhere forces a different draw."""
    seen: set[str] = set()
    first = _draw_unique(1, "train", Slice.CLEAN, 0, seen)
    again = _draw_unique(1, "train", Slice.CLEAN, 0, seen)
    assert again.note != first.note
    assert len(seen) == 2


def test_draw_unique_gives_up_rather_than_looping_forever() -> None:
    with pytest.raises(RuntimeError, match="could not draw a unique"):
        _draw_unique(1, "train", Slice.CLEAN, 0, set(), attempts=0)


def test_example_ids_encode_their_coordinates(corpus: Dataset) -> None:
    example = corpus.train[0]
    assert example.example_id == f"train-{example.slice.value}-0000"


# --------------------------------------------------------------------------------------
# Structure of a built corpus
# --------------------------------------------------------------------------------------


def test_build_dataset_is_reproducible() -> None:
    assert build_dataset(seed=1, **SMALL) == build_dataset(seed=1, **SMALL)


def test_build_dataset_depends_on_the_seed() -> None:
    assert build_dataset(seed=1, **SMALL) != build_dataset(seed=2, **SMALL)


def test_split_sizes_are_exactly_what_was_asked_for(corpus: Dataset) -> None:
    assert (len(corpus.train), len(corpus.val), len(corpus.test)) == (12, 6, 6)
    assert len(corpus.all_examples) == 24


def test_every_slice_appears_in_every_split(corpus: Dataset) -> None:
    for name in SPLIT_NAMES:
        present = {example.slice for example in corpus.examples_for(name)}
        assert present == set(Slice)


def test_slices_are_stratified_in_known_proportions(corpus: Dataset) -> None:
    for name in SPLIT_NAMES:
        rows = corpus.examples_for(name)
        counts = [sum(1 for e in rows if e.slice is s) for s in Slice]
        assert counts == _allocate(len(rows), len(Slice))


def test_splits_share_no_note_text(corpus: Dataset) -> None:
    """The leakage check. A note in two splits would inflate the reported result."""
    notes = [example.note for example in corpus.all_examples]
    assert len(set(notes)) == len(notes)
    train, val, test = ({e.note for e in corpus.examples_for(n)} for n in SPLIT_NAMES)
    assert not train & val
    assert not train & test
    assert not val & test


def test_example_ids_are_unique(corpus: Dataset) -> None:
    ids = [example.example_id for example in corpus.all_examples]
    assert len(set(ids)) == len(ids)


def test_every_example_is_labelled_with_the_split_it_sits_in(corpus: Dataset) -> None:
    for name in SPLIT_NAMES:
        assert all(example.split == name for example in corpus.examples_for(name))


def test_growing_the_training_set_leaves_the_earlier_notes_alone() -> None:
    """Per-cell seed derivation means a bigger corpus extends the smaller one.

    Without this, a scaling curve over training-set size would compare different data at
    every point and the curve would mean nothing.
    """
    small = build_dataset(seed=1, n_train=12, n_val=6, n_test=6)
    large = build_dataset(seed=1, n_train=18, n_val=6, n_test=6)
    assert {e.note for e in small.train} <= {e.note for e in large.train}
    assert small.val == large.val
    assert small.test == large.test


def test_an_empty_split_is_allowed() -> None:
    empty = build_dataset(seed=1, n_train=6, n_val=0, n_test=6)
    assert empty.val == ()
    assert len(empty.all_examples) == 12


def test_build_dataset_rejects_a_negative_split_size() -> None:
    with pytest.raises(ValueError, match="total must be non-negative"):
        build_dataset(seed=1, n_train=-6, n_val=6, n_test=6)


def test_examples_for_covers_all_three_names(corpus: Dataset) -> None:
    assert corpus.examples_for("train") == corpus.train
    assert corpus.examples_for("val") == corpus.val
    assert corpus.examples_for("test") == corpus.test


# --------------------------------------------------------------------------------------
# The content hash
# --------------------------------------------------------------------------------------


def test_content_hash_is_stable_across_rebuilds() -> None:
    assert (
        build_dataset(seed=1, **SMALL).content_hash()
        == build_dataset(seed=1, **SMALL).content_hash()
    )


def test_content_hash_changes_with_the_seed(corpus: Dataset) -> None:
    assert corpus.content_hash() != build_dataset(seed=2, **SMALL).content_hash()


def test_content_hash_changes_with_the_split_sizes(corpus: Dataset) -> None:
    assert (
        corpus.content_hash() != build_dataset(seed=1, n_train=18, n_val=6, n_test=6).content_hash()
    )


def test_content_hash_notices_a_single_edited_character(corpus: Dataset) -> None:
    edited = corpus.train[0].model_copy(update={"note": corpus.train[0].note + "."})
    tampered = corpus.model_copy(update={"train": (edited, *corpus.train[1:])})
    assert tampered.content_hash() != corpus.content_hash()


def test_content_hash_notices_a_move_between_splits(corpus: Dataset) -> None:
    """Reassigning a note from validation to test is a different experiment, not the same
    corpus, so the hash must change even though the set of notes has not."""
    moved = corpus.val[0]
    shuffled = corpus.model_copy(update={"val": corpus.val[1:], "test": (*corpus.test, moved)})
    assert shuffled.content_hash() != corpus.content_hash()


def test_content_hash_is_a_hex_digest(corpus: Dataset) -> None:
    digest = corpus.content_hash()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


# --------------------------------------------------------------------------------------
# The on-disk format
# --------------------------------------------------------------------------------------


def test_save_writes_one_file_per_split_plus_a_manifest(corpus: Dataset, tmp_path: Path) -> None:
    corpus.save(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        MANIFEST_NAME,
        "test.jsonl",
        "train.jsonl",
        "val.jsonl",
    ]


def test_jsonl_has_one_line_per_example(corpus: Dataset, tmp_path: Path) -> None:
    corpus.save(tmp_path)
    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(corpus.train)
    assert Example.model_validate_json(lines[0]) == corpus.train[0]


def test_save_is_byte_stable(corpus: Dataset, tmp_path: Path) -> None:
    """Written twice, the bytes match: the artefact can be committed or diffed safely."""
    first, second = tmp_path / "a", tmp_path / "b"
    corpus.save(first)
    corpus.save(second)
    for name in (MANIFEST_NAME, "train.jsonl", "val.jsonl", "test.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_save_uses_unix_newlines(corpus: Dataset, tmp_path: Path) -> None:
    """Pinned so a Windows author and a Linux runner produce identical files."""
    corpus.save(tmp_path)
    assert b"\r\n" not in (tmp_path / "train.jsonl").read_bytes()


def test_round_trip_restores_the_corpus(corpus: Dataset, tmp_path: Path) -> None:
    corpus.save(tmp_path)
    assert Dataset.load(tmp_path) == corpus


def test_round_trip_is_byte_stable(corpus: Dataset, tmp_path: Path) -> None:
    """save -> load -> save reproduces the same bytes, so the format has no lossy corner."""
    first, second = tmp_path / "a", tmp_path / "b"
    corpus.save(first)
    Dataset.load(first).save(second)
    for name in (MANIFEST_NAME, "train.jsonl", "val.jsonl", "test.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_round_trip_of_an_empty_split(tmp_path: Path) -> None:
    empty = build_dataset(seed=1, n_train=6, n_val=0, n_test=6)
    empty.save(tmp_path)
    assert (tmp_path / "val.jsonl").read_bytes() == b""
    assert Dataset.load(tmp_path) == empty


def test_the_manifest_records_the_seed_the_counts_and_the_hash(
    corpus: Dataset, tmp_path: Path
) -> None:
    corpus.save(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest == {
        "format": FORMAT_VERSION,
        "seed": 1,
        "content_hash": corpus.content_hash(),
        "counts": {"train": 12, "val": 6, "test": 6},
    }


def test_load_rejects_a_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
        Dataset.load(tmp_path)


def test_load_rejects_a_missing_split_file(corpus: Dataset, tmp_path: Path) -> None:
    corpus.save(tmp_path)
    (tmp_path / "test.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="missing split file"):
        Dataset.load(tmp_path)


def test_load_rejects_an_unknown_format_version(corpus: Dataset, tmp_path: Path) -> None:
    corpus.save(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["format"] = "sftdpo-dataset-v0"
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="unsupported dataset format"):
        Dataset.load(tmp_path)


def test_load_rejects_an_example_filed_under_the_wrong_split(
    corpus: Dataset, tmp_path: Path
) -> None:
    corpus.save(tmp_path)
    path = tmp_path / "val.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["split"] = "train"
    lines[0] = json.dumps(row)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match=r"labelled 'train' but was read from val\.jsonl"):
        Dataset.load(tmp_path)


def test_load_rejects_a_tampered_note(corpus: Dataset, tmp_path: Path) -> None:
    """The reason the hash is stored: editing a note by hand is caught on the next load."""
    corpus.save(tmp_path)
    path = tmp_path / "train.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["note"] = row["note"] + " Extra sentence nobody dictated."
    lines[0] = json.dumps(row)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        Dataset.load(tmp_path)


def test_load_rejects_a_manifest_whose_hash_was_edited(corpus: Dataset, tmp_path: Path) -> None:
    corpus.save(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["content_hash"] = "0" * 64
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        Dataset.load(tmp_path)


def test_save_overwrites_an_earlier_corpus_in_the_same_directory(tmp_path: Path) -> None:
    build_dataset(seed=1, n_train=12, n_val=6, n_test=6).save(tmp_path)
    smaller = build_dataset(seed=1, n_train=6, n_val=6, n_test=6)
    smaller.save(tmp_path)
    assert Dataset.load(tmp_path) == smaller


def test_save_creates_missing_parent_directories(corpus: Dataset, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "corpus"
    corpus.save(target)
    assert Dataset.load(target) == corpus


# --------------------------------------------------------------------------------------
# Quantiles
# --------------------------------------------------------------------------------------


def test_quantile_interpolates_between_neighbours() -> None:
    assert _quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _quantile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert _quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_quantile_of_a_constant_sample_is_that_constant() -> None:
    assert _quantile([7.0] * 5, 0.37) == 7.0


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=40,
    ),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_quantile_is_monotonic_in_q(values: list[float], first: float, second: float) -> None:
    """The defining property of a quantile, and the one a hand-rolled version tends to break."""
    low, high = sorted((first, second))
    assert _quantile(values, low) <= _quantile(values, high) + 1e-6


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=40,
    )
)
def test_quantile_endpoints_are_the_extremes(values: list[float]) -> None:
    assert _quantile(values, 0.0) == min(values)
    assert _quantile(values, 1.0) == max(values)


def test_quantile_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        _quantile([], 0.5)


@pytest.mark.parametrize("q", [-0.01, 1.01])
def test_quantile_rejects_a_q_outside_the_unit_interval(q: float) -> None:
    with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
        _quantile([1.0, 2.0], q)


def test_length_quantiles_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sequence of lengths"):
        LengthQuantiles.of([])


def test_length_quantiles_orders_its_summary() -> None:
    summary = LengthQuantiles.of([10, 20, 30, 40, 50])
    assert summary.minimum <= summary.p25 <= summary.p50 <= summary.p75 <= summary.maximum
    assert summary.count == 5
    assert summary.mean == 30.0


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def test_stats_reports_the_corpus_it_was_taken_from(corpus: Dataset) -> None:
    stats = corpus.stats()
    assert stats.seed == corpus.seed
    assert stats.content_hash == corpus.content_hash()
    assert stats.total == len(corpus.all_examples)


def test_stats_per_slice_counts_add_up_to_the_split(corpus: Dataset) -> None:
    for name in SPLIT_NAMES:
        split_stats = corpus.stats().splits[name]
        assert set(split_stats.per_slice) == {s.value for s in Slice}
        assert sum(split_stats.per_slice.values()) == split_stats.n
        assert split_stats.n == len(corpus.examples_for(name))


def test_stats_note_length_quantiles_are_ordered(corpus: Dataset) -> None:
    for name in SPLIT_NAMES:
        words = corpus.stats().splits[name].note_words
        assert words is not None
        assert words.minimum <= words.p25 <= words.p50 <= words.p75 <= words.maximum
        assert words.count == len(corpus.examples_for(name))


def test_stats_length_summary_matches_the_notes(corpus: Dataset) -> None:
    words = corpus.stats().splits["train"].note_words
    assert words is not None
    lengths = [len(example.note.split()) for example in corpus.train]
    assert words.minimum == min(lengths)
    assert words.maximum == max(lengths)


def test_stats_presence_rates_lie_in_the_unit_interval(corpus: Dataset) -> None:
    for name in SPLIT_NAMES:
        presence = corpus.stats().splits[name].field_presence
        assert set(presence) == set(PRESENCE_FIELDS)
        assert all(0.0 <= rate <= 1.0 for rate in presence.values())


def test_stats_show_the_absent_fields_slice_doing_its_job(corpus: Dataset) -> None:
    """Presence below one is the evidence that some gold records really do omit things."""
    presence = corpus.stats().splits["train"].field_presence
    assert presence["recommendations"] == 1.0
    assert presence["objectives"] < 1.0 or presence["flags"] < 1.0
    assert presence["recommendation_amount"] < 1.0


def test_stats_of_an_empty_split_has_no_length_summary() -> None:
    stats = build_dataset(seed=1, n_train=6, n_val=0, n_test=6).stats()
    empty = stats.splits["val"]
    assert empty.n == 0
    assert empty.note_words is None
    assert all(rate == 0.0 for rate in empty.field_presence.values())
    assert set(empty.per_slice.values()) == {0}


def test_stats_serialise_for_an_experiment_record(corpus: Dataset) -> None:
    """The statistics go into a run report, so they have to survive JSON unchanged."""
    stats = corpus.stats()
    assert stats.model_validate_json(stats.model_dump_json()) == stats


# --------------------------------------------------------------------------------------
# The corpus against the token budget it will be trained under
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_the_default_token_budget_fits_the_whole_corpus() -> None:
    """Every example must survive collation under the default `max_seq_length`.

    This project's prompts carry the full JSON Schema, so they are long before the adviser's
    note is added, and an example whose prompt alone exceeds the budget is dropped rather
    than trained on. A budget that silently drops the `long_context` slice would leave the
    hardest slice out of training while the report still showed a score for it.

    That is not hypothetical: the first real run of this pipeline was configured at 1 024
    tokens and dropped every preference pair it built, which is how the measured figure in
    `DEFAULT_MAX_SEQ_LENGTH` was arrived at. Marked `network` because it needs the real Qwen
    tokenizer, which CI does not have offline; run it with `pytest -m network`.
    """
    from transformers import AutoTokenizer

    from sftdpo.modeling.chat import ChatFormatter
    from sftdpo.modeling.collate import DEFAULT_MAX_SEQ_LENGTH
    from sftdpo.task.generate import render_prompt

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    formatter = ChatFormatter()
    dataset = build_dataset(seed=1, n_train=120, n_val=24, n_test=48)

    longest_prompt = 0
    longest_total = 0
    for split in ("train", "val", "test"):
        for example in getattr(dataset, split):
            prompt_len, completion_len = formatter.split_lengths(
                tokenizer, render_prompt(example.note), example.gold_json
            )
            longest_prompt = max(longest_prompt, prompt_len)
            longest_total = max(longest_total, prompt_len + completion_len)

    # The prompt bound is the one that decides whether an example is usable at all: a
    # prompt longer than the budget leaves no completion token to supervise.
    assert longest_prompt < DEFAULT_MAX_SEQ_LENGTH
    assert longest_total <= DEFAULT_MAX_SEQ_LENGTH
