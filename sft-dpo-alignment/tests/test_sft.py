"""Tests for the shared training machinery and the hand-written SFT loop.

Everything runs on CPU in seconds against a two-layer Qwen2 built from config with a 256-token
vocabulary. The small vocabulary is what makes it fast: the logit tensor of a real 151 936-token
vocabulary is 150 MB per forward pass even at these sequence lengths, and a convergence test has
to take dozens of steps.

The three tests that carry the weight of the file are `test_fully_masked_batch_*` (no prompt
token can contribute to the loss), `test_two_runs_with_the_same_seed_are_identical` (the loop is
deterministic) and `test_loss_falls_by_a_large_factor_on_a_memorisation_set` (the loop learns).
The rest exist so that when one of those three breaks, the reason is already isolated.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from sftdpo.modeling.chat import ChatFormatter
from sftdpo.modeling.collate import IGNORE_INDEX, CompletionOnlyCollator, SFTFeature
from sftdpo.modeling.loader import build_tiny_model
from sftdpo.modeling.lora import attach_adapter, lora_config
from sftdpo.schemas import AdviceRecord, Example, Fees, RiskProfile, Slice
from sftdpo.train.common import (
    CUDA_DETERMINISM_NOTE,
    CURVE_METRICS,
    MAX_SEED,
    EvalPoint,
    LogRecord,
    RunManifest,
    TrainConfig,
    TrainLog,
    collect_library_versions,
    cosine_schedule_with_warmup,
    set_determinism,
    token_content_hash,
    warmup_steps_for,
)
from sftdpo.train.sft import (
    LOG_FILENAME,
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    SFTResult,
    accumulation_groups,
    batch_loss,
    encode_examples,
    evaluate_loss,
    resolve_pad_token_id,
    save_adapter,
    supervised_token_count,
    train_sft,
)

VOCAB_SIZE = 256
SEQ_BUDGET = 48

PROPERTY_SETTINGS = settings(max_examples=50, deadline=None)


# --------------------------------------------------------------------------------------
# Fixtures and stand-ins
# --------------------------------------------------------------------------------------


class TinyTokenizer:
    """Whitespace tokeniser with stable ids inside the tiny model's vocabulary.

    CRC32 rather than `hash()`, whose salt changes between processes: a determinism test is
    worthless if the ids themselves move between runs.
    """

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None
    last_add_special_tokens: bool = True

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.last_add_special_tokens = add_special_tokens
        return [2 + zlib.crc32(word.encode("utf-8")) % (VOCAB_SIZE - 2) for word in text.split()]


class PadlessTokenizer:
    """A tokenizer that offers no padding token at all."""

    pad_token_id = None
    eos_token_id = None


class EosOnlyTokenizer:
    """The common instruct-checkpoint case: an EOS token and no pad token."""

    pad_token_id = None
    eos_token_id = 11


class FullyMaskedCollator(CompletionOnlyCollator):
    """Collates normally, then masks every label.

    This is the only way to build a batch with nothing supervised: the real collator refuses
    a feature whose completion is empty, which is exactly the guarantee being relied on here.
    """

    def __call__(self, features: Any) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        batch["labels"] = torch.full_like(batch["labels"], IGNORE_INDEX)
        return batch


def tiny_model(seed: int = 0) -> Any:
    """A two-layer Qwen2 small enough for a forty-step run to cost a fraction of a second."""
    return build_tiny_model(
        seed=seed,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQ_BUDGET * 2,
    )


def make_features(count: int, *, prompt_len: int = 4, completion_len: int = 5) -> list[SFTFeature]:
    """Deterministic features with disjoint token ranges, so they can be memorised."""
    features: list[SFTFeature] = []
    total = prompt_len + completion_len
    for index in range(count):
        start = 3 + index * total
        ids = tuple((start + offset) % (VOCAB_SIZE - 3) + 3 for offset in range(total))
        features.append(SFTFeature(ids, prompt_len))
    return features


def formatter() -> ChatFormatter:
    """A formatter with a one-word system turn, so examples fit the tiny sequence budget."""
    return ChatFormatter(system="SYS")


def make_example(index: int) -> Example:
    """One dataset example, short enough to survive the tiny sequence budget."""
    return Example(
        example_id=f"ex-{index}",
        split="train",
        slice=Slice.CLEAN,
        note=f"note {index} growth",
        gold=AdviceRecord(
            client_name=f"C{index}",
            record_date="2026-01-01",
            risk_profile=RiskProfile.GROWTH,
            fees=Fees(advice_fee=1000.0, ongoing_fee_pct=0.5),
            review_months=12,
        ),
    )


def make_config(tmp_path: Path, **overrides: Any) -> TrainConfig:
    """A config that never writes to the repository and never evaluates unless asked."""
    settings_: dict[str, Any] = {
        "learning_rate": 1e-2,
        "batch_size": 2,
        "epochs": 1,
        "warmup_ratio": 0.0,
        "eval_every_steps": 0,
        "log_every_steps": 1,
        "max_seq_length": SEQ_BUDGET,
        "save_best_adapter": False,
        "output_dir": tmp_path,
    }
    settings_.update(overrides)
    return TrainConfig(**settings_)


@pytest.fixture
def tokenizer() -> TinyTokenizer:
    return TinyTokenizer()


@pytest.fixture
def collator() -> CompletionOnlyCollator:
    return CompletionOnlyCollator(pad_token_id=0, max_length=SEQ_BUDGET)


# --------------------------------------------------------------------------------------
# common.py: TrainConfig
# --------------------------------------------------------------------------------------


def test_config_defaults_describe_a_runnable_loop() -> None:
    config = TrainConfig()
    assert config.effective_batch_size == config.batch_size
    assert config.betas == (0.9, 0.999)
    assert config.max_steps is None


def test_config_is_frozen() -> None:
    config = TrainConfig()
    with pytest.raises(Exception, match="frozen"):
        config.learning_rate = 1.0  # type: ignore[misc]


def test_config_refuses_an_unknown_field() -> None:
    with pytest.raises(Exception, match=r"extra_forbidden|Extra inputs"):
        TrainConfig(lr=1e-4)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"learning_rate": -1e-4},
        {"batch_size": 0},
        {"gradient_accumulation_steps": 0},
        {"epochs": 0},
        {"max_steps": 0},
        {"warmup_ratio": 1.0},
        {"warmup_ratio": -0.1},
        {"weight_decay": -0.1},
        {"max_grad_norm": 0.0},
        {"adam_beta1": 1.0},
        {"adam_beta2": 1.5},
        {"adam_epsilon": 0.0},
        {"seed": -1},
        {"seed": MAX_SEED + 1},
        {"log_every_steps": 0},
        {"eval_every_steps": -1},
        {"max_seq_length": 1},
    ],
)
def test_config_validates_every_numeric_bound(kwargs: dict[str, Any]) -> None:
    with pytest.raises(Exception, match="validation error"):
        TrainConfig(**kwargs)


def test_effective_batch_size_multiplies_accumulation() -> None:
    config = TrainConfig(batch_size=4, gradient_accumulation_steps=8)
    assert config.effective_batch_size == 32


@pytest.mark.parametrize(
    ("examples", "expected"),
    [(1, 1), (4, 1), (5, 2), (8, 2), (9, 3)],
)
def test_steps_per_epoch_rounds_up(examples: int, expected: int) -> None:
    config = TrainConfig(batch_size=2, gradient_accumulation_steps=2)
    assert config.steps_per_epoch(examples) == expected


def test_steps_per_epoch_rejects_an_empty_dataset() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        TrainConfig().steps_per_epoch(0)


def test_total_steps_multiplies_by_epochs() -> None:
    config = TrainConfig(batch_size=2, epochs=3)
    assert config.total_steps(10) == 15


def test_max_steps_overrides_the_epoch_arithmetic() -> None:
    config = TrainConfig(batch_size=2, epochs=100, max_steps=7)
    assert config.total_steps(1000) == 7


def test_warmup_steps_come_from_the_ratio() -> None:
    assert TrainConfig(warmup_ratio=0.1).warmup_steps(200) == 20


def test_lr_at_peaks_at_the_end_of_warmup_and_decays_to_zero() -> None:
    config = TrainConfig(learning_rate=2e-4, warmup_ratio=0.1)
    warmup = config.warmup_steps(100)
    assert config.lr_at(0, 100) == pytest.approx(0.0)
    assert config.lr_at(warmup, 100) == pytest.approx(2e-4)
    assert config.lr_at(100, 100) == pytest.approx(0.0)


def test_config_round_trips_through_json() -> None:
    config = TrainConfig(learning_rate=3e-4, output_dir=Path("runs/x"), bf16=True)
    assert TrainConfig.model_validate_json(config.model_dump_json()) == config


# --------------------------------------------------------------------------------------
# common.py: determinism
# --------------------------------------------------------------------------------------


def test_set_determinism_reproduces_all_three_generators() -> None:
    set_determinism(1234)
    first = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
    set_determinism(1234)
    second = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
    assert first == second


def test_different_seeds_give_different_draws() -> None:
    set_determinism(1)
    first = torch.rand(4).tolist()
    set_determinism(2)
    assert torch.rand(4).tolist() != first


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1])
def test_set_determinism_rejects_a_seed_numpy_cannot_take(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be in"):
        set_determinism(seed)


def test_set_determinism_pins_cudnn() -> None:
    torch.backends.cudnn.benchmark = True
    set_determinism(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_deterministic_algorithms_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    environment = dict(os.environ)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)
    monkeypatch.setattr(os, "environ", environment)

    set_determinism(0)
    assert calls == []
    assert "CUBLAS_WORKSPACE_CONFIG" not in environment

    set_determinism(0, deterministic_algorithms=True)
    assert calls == [True]
    # cuBLAS reuses workspaces unless this is set, which torch refuses to allow under
    # deterministic algorithms.
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_seeding_torch_reaches_the_cuda_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    """`set_determinism` makes no separate CUDA call because `torch.manual_seed` makes it.

    That is a documented delegation rather than an obvious one, so it is pinned here: if a
    future torch stops forwarding, this fails instead of a GPU run quietly losing its seed.
    """
    seeded: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded.append)
    set_determinism(5)
    assert seeded == [5]


def test_cuda_note_names_what_a_seed_cannot_buy() -> None:
    note = CUDA_DETERMINISM_NOTE.lower()
    assert "atomics" in note
    assert "bf16" in note


# --------------------------------------------------------------------------------------
# common.py: the schedule
# --------------------------------------------------------------------------------------


def test_schedule_starts_at_zero_when_there_is_warmup() -> None:
    assert cosine_schedule_with_warmup(0, 100, 10) == 0.0


def test_schedule_reaches_one_at_the_end_of_warmup() -> None:
    assert cosine_schedule_with_warmup(10, 100, 10) == pytest.approx(1.0)


def test_schedule_ends_at_zero() -> None:
    assert cosine_schedule_with_warmup(100, 100, 10) == pytest.approx(0.0, abs=1e-12)


def test_schedule_without_warmup_starts_at_full_rate() -> None:
    assert cosine_schedule_with_warmup(0, 100, 0) == pytest.approx(1.0)


def test_schedule_is_half_way_down_at_the_midpoint() -> None:
    assert cosine_schedule_with_warmup(50, 100, 0) == pytest.approx(0.5)


def test_schedule_clamps_past_the_end_instead_of_reviving() -> None:
    # cos() is periodic; without the clamp an extra step would push the rate back up.
    assert cosine_schedule_with_warmup(150, 100, 10) == pytest.approx(0.0, abs=1e-12)


def test_warmup_covering_the_whole_run_never_decays() -> None:
    assert cosine_schedule_with_warmup(10, 10, 10) == pytest.approx(1.0)


def test_warmup_is_linear() -> None:
    assert cosine_schedule_with_warmup(5, 100, 10) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, 0, 0), "total must be at least 1"),
        ((-1, 10, 0), "step must not be negative"),
        ((0, 10, -1), "warmup must not be negative"),
        ((0, 10, 11), "cannot exceed the run length"),
    ],
)
def test_schedule_rejects_impossible_arguments(args: tuple[int, int, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_schedule_with_warmup(*args)


@pytest.mark.parametrize(("total", "ratio"), [(0, 0.1), (10, 1.0), (10, -0.1)])
def test_warmup_steps_for_rejects_impossible_arguments(total: int, ratio: float) -> None:
    with pytest.raises(ValueError):
        warmup_steps_for(total, ratio)


@given(total=st.integers(min_value=1, max_value=500), ratio=st.floats(0.0, 0.99))
@PROPERTY_SETTINGS
def test_warmup_always_leaves_room_for_the_decay(total: int, ratio: float) -> None:
    """Flooring the warmup is what guarantees the schedule can reach zero at the end."""
    warmup = warmup_steps_for(total, ratio)
    assert 0 <= warmup < total
    assert cosine_schedule_with_warmup(total, total, warmup) == pytest.approx(0.0, abs=1e-12)
    assert cosine_schedule_with_warmup(warmup, total, warmup) == pytest.approx(1.0)


@given(
    total=st.integers(min_value=2, max_value=200),
    ratio=st.floats(0.0, 0.5),
    step=st.integers(min_value=0, max_value=400),
)
@PROPERTY_SETTINGS
def test_schedule_multiplier_stays_inside_the_unit_interval(
    total: int, ratio: float, step: int
) -> None:
    warmup = warmup_steps_for(total, ratio)
    assert 0.0 <= cosine_schedule_with_warmup(step, total, warmup) <= 1.0


@given(total=st.integers(min_value=4, max_value=200), ratio=st.floats(0.0, 0.4))
@PROPERTY_SETTINGS
def test_schedule_never_rises_after_warmup(total: int, ratio: float) -> None:
    warmup = warmup_steps_for(total, ratio)
    values = [cosine_schedule_with_warmup(s, total, warmup) for s in range(warmup, total + 1)]
    assert all(later <= earlier + 1e-12 for earlier, later in itertools.pairwise(values))


# --------------------------------------------------------------------------------------
# common.py: the log
# --------------------------------------------------------------------------------------


def test_log_record_round_trips_through_a_dict() -> None:
    record = LogRecord(step=3, loss=1.5, lr=1e-4, grad_norm=0.7, tokens=120)
    assert LogRecord.from_dict(record.as_dict()) == record


def test_log_record_names_every_missing_field_at_once() -> None:
    with pytest.raises(ValueError, match=r"missing \['grad_norm', 'tokens'\]"):
        LogRecord.from_dict({"step": 1, "loss": 2.0, "lr": 1e-4})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"step": -1}, "step must not be negative"),
        ({"tokens": -5}, "tokens must not be negative"),
        ({"lr": -1e-4}, "lr must not be negative"),
        ({"grad_norm": -0.5}, "grad_norm must not be negative"),
    ],
)
def test_log_record_validates_its_fields(kwargs: dict[str, Any], message: str) -> None:
    base: dict[str, Any] = {"step": 1, "loss": 1.0, "lr": 1e-4, "grad_norm": 1.0, "tokens": 10}
    with pytest.raises(ValueError, match=message):
        LogRecord(**{**base, **kwargs})


def test_a_diverged_loss_is_still_recordable() -> None:
    """A NaN loss must be logged, not rejected: it is the evidence that the run failed."""
    log = TrainLog()
    log.append(step=1, loss=float("nan"), lr=1e-4, grad_norm=float("inf"), tokens=4)
    assert log.diverged is True


def test_a_healthy_log_is_not_diverged() -> None:
    log = TrainLog()
    log.append(step=1, loss=0.5, lr=1e-4, grad_norm=1.0, tokens=4)
    assert log.diverged is False


def test_an_empty_log_has_no_last_record() -> None:
    log = TrainLog()
    assert len(log) == 0
    assert log.last is None


def test_log_grows_and_exposes_its_last_record() -> None:
    log = TrainLog()
    log.append(step=1, loss=2.0, lr=1e-4, grad_norm=1.0, tokens=8)
    log.append(step=2, loss=1.0, lr=1e-4, grad_norm=1.0, tokens=16)
    last = log.last
    assert len(log) == 2
    assert last is not None
    assert last.step == 2
    assert [record.step for record in log] == [1, 2]


def test_log_refuses_a_step_that_goes_backwards() -> None:
    log = TrainLog()
    log.append(step=10, loss=1.0, lr=1e-4, grad_norm=1.0, tokens=4)
    with pytest.raises(ValueError, match="step went backwards"):
        log.append(step=9, loss=1.0, lr=1e-4, grad_norm=1.0, tokens=8)


def test_total_tokens_reads_the_last_cumulative_count() -> None:
    log = TrainLog()
    log.append(step=1, loss=1.0, lr=1e-4, grad_norm=1.0, tokens=10)
    log.append(step=5, loss=1.0, lr=1e-4, grad_norm=1.0, tokens=50)
    assert log.total_tokens == 50
    assert TrainLog().total_tokens == 0


@pytest.mark.parametrize("metric", CURVE_METRICS)
def test_curve_returns_a_step_value_series_for_every_metric(metric: str) -> None:
    log = TrainLog()
    log.append(step=1, loss=2.0, lr=1e-4, grad_norm=3.0, tokens=8)
    curve = log.curve(metric)
    assert curve[0][0] == 1
    assert isinstance(curve[0][1], float)


def test_curve_defaults_to_the_loss() -> None:
    log = TrainLog()
    log.append(step=1, loss=2.0, lr=1e-4, grad_norm=3.0, tokens=8)
    assert log.curve() == [(1, 2.0)]


def test_curve_refuses_an_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        TrainLog().curve("perplexity")


def test_log_round_trips_through_jsonl(tmp_path: Path) -> None:
    log = TrainLog()
    log.append(step=1, loss=2.0, lr=1e-4, grad_norm=1.0, tokens=8)
    log.append(step=2, loss=1.0, lr=9e-5, grad_norm=0.5, tokens=16)
    path = log.to_jsonl(tmp_path / "nested" / LOG_FILENAME)
    assert TrainLog.from_jsonl(path).records == log.records


def test_jsonl_uses_unix_newlines_even_on_windows(tmp_path: Path) -> None:
    """A log written here and hashed on Linux must be the same bytes."""
    log = TrainLog()
    log.append(step=1, loss=2.0, lr=1e-4, grad_norm=1.0, tokens=8)
    path = log.to_jsonl(tmp_path / LOG_FILENAME)
    assert b"\r\n" not in path.read_bytes()


def test_jsonl_reader_tolerates_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / LOG_FILENAME
    row = json.dumps({"step": 1, "loss": 1.0, "lr": 1e-4, "grad_norm": 1.0, "tokens": 4})
    path.write_text(f"{row}\n\n{row}\n", encoding="utf-8")
    assert len(TrainLog.from_jsonl(path)) == 2


def test_jsonl_reader_names_the_broken_line(tmp_path: Path) -> None:
    path = tmp_path / LOG_FILENAME
    row = json.dumps({"step": 1, "loss": 1.0, "lr": 1e-4, "grad_norm": 1.0, "tokens": 4})
    path.write_text(f"{row}\n{{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        TrainLog.from_jsonl(path)


def test_jsonl_reader_rejects_a_bare_array(tmp_path: Path) -> None:
    path = tmp_path / LOG_FILENAME
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        TrainLog.from_jsonl(path)


def test_eval_point_serialises_and_validates() -> None:
    assert EvalPoint(step=4, loss=0.5).as_dict() == {"step": 4, "loss": 0.5}
    with pytest.raises(ValueError, match="step must not be negative"):
        EvalPoint(step=-1, loss=0.5)


# --------------------------------------------------------------------------------------
# common.py: provenance
# --------------------------------------------------------------------------------------


def test_content_hash_is_stable_across_calls() -> None:
    assert token_content_hash([[1, 2, 3]]) == token_content_hash([[1, 2, 3]])


def test_content_hash_separates_the_sequences() -> None:
    """Without a length prefix these two would digest to the same bytes."""
    assert token_content_hash([[1, 2], [3]]) != token_content_hash([[1], [2, 3]])


def test_content_hash_depends_on_order() -> None:
    assert token_content_hash([[1], [2]]) != token_content_hash([[2], [1]])


@given(
    left=st.lists(st.lists(st.integers(0, 50), max_size=5), max_size=5),
    right=st.lists(st.lists(st.integers(0, 50), max_size=5), max_size=5),
)
@PROPERTY_SETTINGS
def test_content_hash_agrees_exactly_when_the_content_agrees(
    left: list[list[int]], right: list[list[int]]
) -> None:
    assert (token_content_hash(left) == token_content_hash(right)) == (left == right)


def test_library_versions_cover_the_stack_that_moves_results() -> None:
    versions = collect_library_versions()
    assert versions["torch"] == torch.__version__
    assert versions["python"].count(".") == 2
    assert set(versions) >= {"python", "torch", "transformers", "peft", "numpy", "pydantic"}


def test_a_missing_library_is_recorded_rather_than_raised() -> None:
    versions = collect_library_versions(["definitely-not-a-real-package"])
    assert versions["definitely-not-a-real-package"] == "not installed"


def test_manifest_records_what_produced_a_result(tmp_path: Path) -> None:
    model = tiny_model()
    manifest = RunManifest.build(
        stage="sft",
        model=model,
        model_name="tiny",
        data_content_hash="abc123",
        config=make_config(tmp_path),
    )
    assert manifest.stage == "sft"
    assert manifest.total_parameters == sum(p.numel() for p in model.parameters())
    assert manifest.trainable_parameters == manifest.total_parameters
    assert manifest.trainable_pct == pytest.approx(100.0)


def test_manifest_counts_only_the_adapter_when_one_is_attached(tmp_path: Path) -> None:
    model = attach_adapter(tiny_model(), lora_config(r=4))
    manifest = RunManifest.build(
        stage="sft",
        model=model,
        model_name="tiny+lora",
        data_content_hash="abc123",
        config=make_config(tmp_path),
    )
    assert 0 < manifest.trainable_parameters < manifest.total_parameters
    # The share only looks large because this model's vocabulary is 256 tokens; the same
    # adapter on Qwen2.5-0.5B is 0.167 % of the weights.
    assert manifest.trainable_pct < 15.0
    assert manifest.trainable_pct == pytest.approx(
        100.0 * manifest.trainable_parameters / manifest.total_parameters
    )


def test_manifest_refuses_an_empty_data_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data_content_hash"):
        RunManifest(
            stage="sft",
            model_name="tiny",
            data_content_hash="  ",
            trainable_parameters=1,
            total_parameters=2,
            config=make_config(tmp_path),
        )


def test_manifest_refuses_more_trainable_than_total(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        RunManifest(
            stage="sft",
            model_name="tiny",
            data_content_hash="abc",
            trainable_parameters=10,
            total_parameters=5,
            config=make_config(tmp_path),
        )


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = RunManifest(
        stage="dpo",
        model_name="tiny",
        data_content_hash="abc",
        trainable_parameters=5,
        total_parameters=10,
        config=make_config(tmp_path, bf16=True),
        library_versions={"torch": "2.11.0"},
    )
    restored = RunManifest.from_json(manifest.to_json(tmp_path / MANIFEST_FILENAME))
    assert restored == manifest
    assert restored.config.bf16 is True


# --------------------------------------------------------------------------------------
# sft.py: the pieces around the loop
# --------------------------------------------------------------------------------------


def test_pad_token_is_preferred_when_the_tokenizer_has_one(tokenizer: TinyTokenizer) -> None:
    assert resolve_pad_token_id(tokenizer) == 0


def test_eos_stands_in_for_a_missing_pad_token() -> None:
    assert resolve_pad_token_id(EosOnlyTokenizer()) == 11


def test_a_tokenizer_with_neither_is_refused() -> None:
    with pytest.raises(ValueError, match="neither pad_token_id nor eos_token_id"):
        resolve_pad_token_id(PadlessTokenizer())


def test_features_pass_through_encoding_untouched(tokenizer: TinyTokenizer) -> None:
    features = make_features(2)
    assert encode_examples(features, formatter=formatter(), tokenizer=tokenizer) == features


def test_examples_are_encoded_with_the_gold_json_as_the_completion(
    tokenizer: TinyTokenizer,
) -> None:
    example = make_example(0)
    chat = formatter()
    [feature] = encode_examples([example], formatter=chat, tokenizer=tokenizer)
    full = chat.full_text(example.note, example.gold_json, tokenizer=tokenizer)
    assert list(feature.input_ids) == tokenizer.encode(full)
    assert feature.prompt_len == len(
        tokenizer.encode(chat.prompt_text(example.note, tokenizer=tokenizer))
    )
    assert feature.completion_len >= 1


def test_encoding_never_asks_for_extra_special_tokens(tokenizer: TinyTokenizer) -> None:
    """The chat template already carries every special token the model expects; a tokenizer
    that also prepended BOS would shift the supervision boundary by one."""
    encode_examples([make_example(0)], formatter=formatter(), tokenizer=tokenizer)
    assert tokenizer.last_add_special_tokens is False


def test_encoding_refuses_an_item_it_does_not_understand(tokenizer: TinyTokenizer) -> None:
    with pytest.raises(TypeError, match="expected an Example or an SFTFeature"):
        encode_examples(["a note"], formatter=formatter(), tokenizer=tokenizer)  # type: ignore[list-item]


def test_supervised_token_count_ignores_the_mask() -> None:
    labels = torch.tensor([[IGNORE_INDEX, 5, 6], [IGNORE_INDEX, IGNORE_INDEX, 7]])
    assert supervised_token_count(labels) == 3


def test_accumulation_groups_partition_the_epoch() -> None:
    groups = accumulation_groups(list(range(7)), batch_size=2, accumulation=2)
    assert groups == [[[0, 1], [2, 3]], [[4, 5], [6]]]


@pytest.mark.parametrize(("batch_size", "accumulation"), [(0, 1), (1, 0)])
def test_accumulation_groups_validate_their_shape(batch_size: int, accumulation: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        accumulation_groups([0, 1], batch_size=batch_size, accumulation=accumulation)


@given(
    size=st.integers(min_value=1, max_value=40),
    batch_size=st.integers(min_value=1, max_value=8),
    accumulation=st.integers(min_value=1, max_value=5),
)
@PROPERTY_SETTINGS
def test_grouping_visits_every_example_exactly_once(
    size: int, batch_size: int, accumulation: int
) -> None:
    order = list(range(size))
    groups = accumulation_groups(order, batch_size=batch_size, accumulation=accumulation)
    flat = [index for group in groups for micro in group for index in micro]
    assert flat == order
    assert all(len(micro) <= batch_size for group in groups for micro in group)
    assert all(len(group) <= accumulation for group in groups)


# --------------------------------------------------------------------------------------
# sft.py: where the loss comes from
# --------------------------------------------------------------------------------------


def _manual_completion_loss(model: Any, batch: dict[str, torch.Tensor]) -> float:
    """Cross-entropy over the supervised positions only, computed independently.

    Written out with an explicit shift so that it is a genuine second opinion about which
    logit predicts which token, rather than a rearrangement of the same call.
    """
    logits = model(**batch).logits.float()
    labels = batch["labels"]
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    keep = shifted_labels != IGNORE_INDEX
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    picked = log_probs.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return float(-picked[keep].mean())


def test_batch_loss_equals_a_hand_computed_completion_only_cross_entropy(
    collator: CompletionOnlyCollator,
) -> None:
    """Equivalence between two code paths: the model's loss supervises the completion only."""
    model = tiny_model()
    model.eval()
    batch = collator(make_features(3))
    with torch.no_grad():
        loss, tokens = batch_loss(model, batch)
    assert tokens == supervised_token_count(batch["labels"])
    assert float(loss) == pytest.approx(_manual_completion_loss(model, batch), rel=1e-5)


def test_batch_loss_is_exactly_zero_when_nothing_is_supervised() -> None:
    """The model's own loss is NaN here; the guard turns it into a real, harmless zero."""
    model = tiny_model()
    batch = FullyMaskedCollator(pad_token_id=0, max_length=SEQ_BUDGET)(make_features(2))
    loss, tokens = batch_loss(model, batch)
    assert tokens == 0
    assert float(loss) == 0.0
    assert torch.isnan(model(**batch).loss).item() is True


def test_fully_masked_batch_leaves_every_gradient_at_zero() -> None:
    model = tiny_model()
    batch = FullyMaskedCollator(pad_token_id=0, max_length=SEQ_BUDGET)(make_features(2))
    loss, _ = batch_loss(model, batch)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(float(grad.abs().max()) == 0.0 for grad in grads)


def test_fully_masked_training_leaves_the_weights_untouched(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """Two steps on a batch with no supervised token must not move a single weight.

    This is the strongest available statement that no prompt token contributes to the loss:
    the prompt is still in `input_ids`, the forward pass still runs over it, and nothing
    changes.
    """
    model = tiny_model()
    before = [p.detach().clone() for p in model.parameters()]
    result = train_sft(
        model,
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=2, batch_size=2),
        collator=FullyMaskedCollator(pad_token_id=0, max_length=SEQ_BUDGET),
    )
    assert result.steps == 2
    assert result.supervised_tokens == 0
    assert [record.loss for record in result.log.records] == [0.0, 0.0]
    assert all(
        torch.equal(old, new) for old, new in zip(before, list(model.parameters()), strict=True)
    )


# --------------------------------------------------------------------------------------
# sft.py: evaluation
# --------------------------------------------------------------------------------------


def test_validation_loss_does_not_depend_on_the_batch_size(
    collator: CompletionOnlyCollator,
) -> None:
    """Token-weighted averaging is what makes two batchings of the same data agree."""
    model = tiny_model()
    features = make_features(6, completion_len=4) + make_features(3, completion_len=7)
    at_one = evaluate_loss(model, features, collator, batch_size=1)
    at_four = evaluate_loss(model, features, collator, batch_size=4)
    assert at_one == pytest.approx(at_four, rel=1e-5)


def test_evaluation_restores_the_training_mode(collator: CompletionOnlyCollator) -> None:
    model = tiny_model()
    model.train()
    evaluate_loss(model, make_features(2), collator, batch_size=2)
    assert model.training is True
    model.eval()
    evaluate_loss(model, make_features(2), collator, batch_size=2)
    assert model.training is False


def test_evaluation_refuses_an_empty_validation_set(collator: CompletionOnlyCollator) -> None:
    with pytest.raises(ValueError, match="at least one validation example"):
        evaluate_loss(tiny_model(), [], collator, batch_size=2)


def test_evaluation_of_a_fully_masked_set_is_zero_not_nan() -> None:
    masked = FullyMaskedCollator(pad_token_id=0, max_length=SEQ_BUDGET)
    assert evaluate_loss(tiny_model(), make_features(2), masked, batch_size=2) == 0.0


def test_bf16_autocast_still_produces_a_finite_loss(collator: CompletionOnlyCollator) -> None:
    loss = evaluate_loss(tiny_model(), make_features(2), collator, batch_size=2, bf16=True)
    assert math.isfinite(loss)


# --------------------------------------------------------------------------------------
# sft.py: the loop itself
# --------------------------------------------------------------------------------------


def test_loss_falls_by_a_large_factor_on_a_memorisation_set(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """The test that proves the loop learns at all.

    Four fixed sequences, forty steps, one batch: a model that can memorise this has a
    working forward pass, a working backward pass, an optimiser that is actually stepping
    and a schedule that is not holding the rate at zero. If any of those is broken the loss
    stays near ln(vocab).
    """
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=40, batch_size=4, learning_rate=1e-2),
    )
    first = result.log.records[0].loss
    assert first > math.log(VOCAB_SIZE) / 2
    assert result.final_train_loss < first / 10.0
    assert result.loss_reduction > 10.0
    assert not result.log.diverged


def test_two_runs_with_the_same_seed_are_identical(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """Same seed, same data, same losses -- to the last bit, on CPU."""
    features = make_features(6)

    def run() -> list[float]:
        result = train_sft(
            tiny_model(),
            tokenizer,
            features,
            features[:2],
            make_config(tmp_path, max_steps=6, batch_size=2, eval_every_steps=3),
        )
        return [record.loss for record in result.log.records] + [
            point.loss for point in result.val_curve
        ]

    assert run() == run()


def test_a_different_seed_changes_the_run(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    features = make_features(8)

    def run(seed: int) -> list[float]:
        result = train_sft(
            tiny_model(),
            tokenizer,
            features,
            [],
            make_config(tmp_path, max_steps=4, batch_size=2, seed=seed),
        )
        return [record.loss for record in result.log.records]

    assert run(0) != run(7)


def test_gradient_accumulation_equals_one_larger_batch(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """Four micro-batches of two must produce the gradient of one batch of eight.

    This is the property the usual "divide the mean loss by the number of micro-batches"
    recipe loses as soon as the examples differ in length, and it is why the loop
    accumulates summed token losses and normalises once, by tokens.

    The gradient norm is the assertion that matters, and it agrees exactly. The weights are
    checked far more loosely on purpose: Adam's first update is `g / (|g| + eps)`, so a
    coordinate whose gradient is near zero turns a 1e-8 difference in `g` into a visible
    difference in the step. That is Adam's sensitivity, not an inexact accumulation.
    """
    features = make_features(4, completion_len=3) + make_features(4, completion_len=6)

    def run(batch_size: int, accumulation: int) -> tuple[float, float, list[torch.Tensor]]:
        model = tiny_model()
        result = train_sft(
            model,
            tokenizer,
            features,
            [],
            make_config(
                tmp_path,
                max_steps=1,
                batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
            ),
        )
        record = result.log.records[0]
        return (
            record.loss,
            record.grad_norm,
            [parameter.detach().clone() for parameter in model.parameters()],
        )

    big_loss, big_norm, big_weights = run(8, 1)
    split_loss, split_norm, split_weights = run(2, 4)
    assert split_loss == pytest.approx(big_loss, rel=1e-6)
    assert split_norm == big_norm
    for one, many in zip(big_weights, split_weights, strict=True):
        assert torch.allclose(one, many, atol=1e-3)


def test_the_logged_rate_follows_the_configured_schedule(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    config = make_config(tmp_path, max_steps=10, batch_size=4, warmup_ratio=0.3, learning_rate=1e-3)
    result = train_sft(tiny_model(), tokenizer, make_features(4), [], config)
    for record in result.log.records:
        expected = config.lr_at(record.step - 1, 10)
        assert record.lr == pytest.approx(expected)
    assert result.log.records[0].lr == pytest.approx(0.0)


def test_logging_honours_its_interval_and_always_records_the_last_step(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=7, batch_size=4, log_every_steps=3),
    )
    assert [record.step for record in result.log.records] == [3, 6, 7]


def test_token_counts_accumulate_over_the_run(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    features = make_features(4, completion_len=5)
    result = train_sft(
        tiny_model(),
        tokenizer,
        features,
        [],
        make_config(tmp_path, max_steps=3, batch_size=4),
    )
    per_step = sum(feature.completion_len for feature in features)
    assert result.supervised_tokens == 3 * per_step
    assert result.log.total_tokens == result.supervised_tokens
    assert result.log.curve("tokens") == [(1, per_step), (2, 2 * per_step), (3, 3 * per_step)]


def test_validation_runs_on_the_interval_and_tracks_the_best_step(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    features = make_features(4)
    result = train_sft(
        tiny_model(),
        tokenizer,
        features,
        features[:2],
        make_config(tmp_path, max_steps=6, batch_size=4, eval_every_steps=2),
    )
    assert [point.step for point in result.val_curve] == [2, 4, 6]
    assert result.best_val_loss == pytest.approx(min(p.loss for p in result.val_curve))
    assert result.best_step in {point.step for point in result.val_curve}


def test_validation_is_skipped_entirely_when_the_interval_is_zero(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    features = make_features(4)
    result = train_sft(
        tiny_model(),
        tokenizer,
        features,
        features[:2],
        make_config(tmp_path, max_steps=2, batch_size=4, eval_every_steps=0),
    )
    assert result.val_curve == ()
    assert result.best_val_loss is None
    assert result.best_step is None


def test_epochs_and_max_steps_both_control_the_length(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    features = make_features(8)
    by_epochs = train_sft(
        tiny_model(), tokenizer, features, [], make_config(tmp_path, epochs=3, batch_size=4)
    )
    by_cap = train_sft(
        tiny_model(),
        tokenizer,
        features,
        [],
        make_config(tmp_path, epochs=3, batch_size=4, max_steps=2),
    )
    assert by_epochs.steps == 6
    assert by_cap.steps == 2


def test_more_steps_than_the_data_supports_cycles_epochs(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(2),
        [],
        make_config(tmp_path, max_steps=5, batch_size=2),
    )
    assert result.steps == 5


def test_training_refuses_an_empty_dataset(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one training example"):
        train_sft(tiny_model(), tokenizer, [], [], make_config(tmp_path, max_steps=1))


def test_training_refuses_a_model_with_nothing_to_train(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    model = tiny_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="no parameter has requires_grad"):
        train_sft(model, tokenizer, make_features(2), [], make_config(tmp_path, max_steps=1))


def test_a_trainable_parameter_the_forward_pass_never_uses_is_tolerated(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """A head that this task does not use gets no gradient; the step must still happen."""
    model = tiny_model()
    model.unused_head = torch.nn.Parameter(torch.zeros(4))
    result = train_sft(
        model,
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=2, batch_size=4),
    )
    assert result.steps == 2
    assert model.unused_head.grad is None
    assert torch.equal(model.unused_head.detach(), torch.zeros(4))


def test_a_worse_validation_score_does_not_replace_the_best(
    tokenizer: TinyTokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-checkpoint tracking has to survive a run that starts overfitting.

    The validation curve is scripted rather than trained, because the property under test
    is what the tracker does when a later score is *worse*, and a real eight-step run on a
    randomly initialised tiny model does not reliably produce one: an earlier version of this
    test leaned on a large learning rate to provoke it, passed on one machine and failed in CI
    on both Python versions, where the curve happened to fall monotonically and the best step
    was the last one. A premise the test cannot guarantee is not a premise, it is a hope.
    """
    scripted = [0.9, 0.4, 0.6, 0.5, 0.7, 0.8, 0.55, 0.65]
    served = iter(scripted)
    monkeypatch.setattr("sftdpo.train.sft.evaluate_loss", lambda *_args, **_kw: next(served))

    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(4),
        make_features(2, prompt_len=3, completion_len=6),
        make_config(tmp_path, max_steps=8, batch_size=4, eval_every_steps=1),
    )
    losses = [point.loss for point in result.val_curve]
    assert losses == scripted
    # The minimum came at step 2 and six worse scores followed it, one of them close.
    assert result.best_val_loss == pytest.approx(0.4)
    assert result.best_step == 2
    assert result.best_step < result.steps


def test_only_the_adapter_moves_when_one_is_attached(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """The LoRA promise, checked end to end: the base weights come out bit-identical."""
    model = attach_adapter(tiny_model(), lora_config(r=4))
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    adapters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    train_sft(
        model,
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=5, batch_size=4, learning_rate=1e-2),
    )
    after = dict(model.named_parameters())
    assert all(torch.equal(value, after[name].detach()) for name, value in frozen.items())
    assert any(not torch.equal(value, after[name].detach()) for name, value in adapters.items())


def test_training_with_dataset_examples_uses_the_chat_formatter(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    examples = [make_example(index) for index in range(4)]
    result = train_sft(
        tiny_model(),
        tokenizer,
        examples,
        examples[:2],
        make_config(tmp_path, max_steps=2, batch_size=2, eval_every_steps=2),
        formatter=formatter(),
    )
    assert result.steps == 2
    assert result.supervised_tokens > 0
    assert result.collation.seen > 0


def test_a_tight_sequence_budget_is_reported_not_hidden(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    """Truncation is a fact about the run, so it has to reach the result object."""
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(4, prompt_len=4, completion_len=8),
        [],
        make_config(tmp_path, max_steps=1, batch_size=4, max_seq_length=6),
    )
    assert result.collation.truncated == 4
    assert result.collation.dropped == 0


def test_bf16_training_runs_and_stays_finite(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=3, batch_size=4, bf16=True),
    )
    assert not result.log.diverged
    assert all(math.isfinite(record.grad_norm) for record in result.log.records)


def test_gradient_norms_are_recorded_and_positive(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=3, batch_size=4),
    )
    assert all(record.grad_norm > 0.0 for record in result.log.records)


# --------------------------------------------------------------------------------------
# sft.py: checkpoints and artefacts
# --------------------------------------------------------------------------------------


def test_the_best_checkpoint_is_written_when_validation_improves(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    model = attach_adapter(tiny_model(), lora_config(r=4))
    features = make_features(4)
    result = train_sft(
        model,
        tokenizer,
        features,
        features[:2],
        make_config(
            tmp_path,
            max_steps=2,
            batch_size=4,
            eval_every_steps=1,
            save_best_adapter=True,
            learning_rate=1e-2,
        ),
    )
    assert result.adapter_path == tmp_path / "best"
    assert (tmp_path / "best" / "adapter_config.json").exists()


def test_a_run_without_validation_still_saves_its_final_weights(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    model = attach_adapter(tiny_model(), lora_config(r=4))
    result = train_sft(
        model,
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=1, batch_size=4, save_best_adapter=True),
    )
    assert result.adapter_path == tmp_path / "final"
    assert (tmp_path / "final" / "adapter_config.json").exists()


def test_nothing_is_written_when_saving_is_off(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(4),
        [],
        make_config(tmp_path, max_steps=1, batch_size=4),
    )
    assert result.adapter_path is None
    assert list(tmp_path.iterdir()) == []


def test_saving_refuses_an_object_that_cannot_checkpoint_itself(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="no save_pretrained"):
        save_adapter(object(), tmp_path / "nowhere")


def test_the_manifest_ties_the_result_to_its_inputs(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    features = make_features(4)
    config = make_config(tmp_path, max_steps=2, batch_size=4)
    result = train_sft(tiny_model(), tokenizer, features, [], config)
    assert result.manifest.config == config
    assert result.manifest.stage == "sft"
    assert result.manifest.data_content_hash
    assert result.manifest.trainable_parameters == result.manifest.total_parameters


def test_the_data_hash_changes_with_the_data(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    config = make_config(tmp_path, max_steps=1, batch_size=4)
    first = train_sft(tiny_model(), tokenizer, make_features(4), [], config)
    second = train_sft(tiny_model(), tokenizer, make_features(5), [], config)
    assert first.manifest.data_content_hash != second.manifest.data_content_hash


def test_a_caller_can_cite_the_corpus_hash_instead(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(2),
        [],
        make_config(tmp_path, max_steps=1, batch_size=2),
        data_content_hash="corpus-deadbeef",
    )
    assert result.manifest.data_content_hash == "corpus-deadbeef"


def test_the_model_name_defaults_to_the_checkpoint_it_came_from(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(2),
        [],
        make_config(tmp_path, max_steps=1, batch_size=2),
    )
    assert result.manifest.model_name == "Qwen2ForCausalLM"


def test_an_explicit_model_name_wins(tokenizer: TinyTokenizer, tmp_path: Path) -> None:
    result = train_sft(
        tiny_model(),
        tokenizer,
        make_features(2),
        [],
        make_config(tmp_path, max_steps=1, batch_size=2),
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )
    assert result.manifest.model_name == "Qwen/Qwen2.5-0.5B-Instruct"


def test_a_result_writes_its_log_manifest_and_summary(
    tokenizer: TinyTokenizer, tmp_path: Path
) -> None:
    features = make_features(4)
    result = train_sft(
        tiny_model(),
        tokenizer,
        features,
        features[:2],
        make_config(tmp_path, max_steps=2, batch_size=4, eval_every_steps=2),
    )
    directory = result.save(tmp_path / "artefacts")
    assert TrainLog.from_jsonl(directory / LOG_FILENAME).records == result.log.records
    assert RunManifest.from_json(directory / MANIFEST_FILENAME) == result.manifest
    summary = json.loads((directory / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["steps"] == 2
    assert summary["val_curve"][0]["step"] == 2
    assert summary["collation"]["seen"] > 0


def test_an_empty_result_reports_nan_rather_than_crashing(tmp_path: Path) -> None:
    """`final_train_loss` and `loss_reduction` are read by the report before it knows the
    run produced anything."""
    empty = SFTResult(
        log=TrainLog(),
        val_curve=(),
        best_val_loss=None,
        best_step=None,
        steps=0,
        supervised_tokens=0,
        manifest=RunManifest(
            stage="sft",
            model_name="tiny",
            data_content_hash="abc",
            trainable_parameters=1,
            total_parameters=2,
            config=make_config(tmp_path),
        ),
        adapter_path=None,
        collation=CompletionOnlyCollator(pad_token_id=0).report,
    )
    assert math.isnan(empty.final_train_loss)
    assert math.isnan(empty.loss_reduction)


def test_a_zero_final_loss_reports_an_infinite_reduction(tmp_path: Path) -> None:
    log = TrainLog()
    log.append(step=1, loss=4.0, lr=1e-3, grad_norm=1.0, tokens=8)
    log.append(step=2, loss=0.0, lr=1e-3, grad_norm=1.0, tokens=16)
    result = SFTResult(
        log=log,
        val_curve=(),
        best_val_loss=None,
        best_step=None,
        steps=2,
        supervised_tokens=16,
        manifest=RunManifest(
            stage="sft",
            model_name="tiny",
            data_content_hash="abc",
            trainable_parameters=1,
            total_parameters=2,
            config=make_config(tmp_path),
        ),
        adapter_path=None,
        collation=CompletionOnlyCollator(pad_token_id=0).report,
    )
    assert result.loss_reduction == float("inf")
