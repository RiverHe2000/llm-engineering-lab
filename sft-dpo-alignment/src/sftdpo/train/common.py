"""Machinery both trainers share: configuration, seeding, the schedule, the log, the manifest.

A training loop is easy to write and hard to trust. Everything in this module exists so that
a number a run produces can be defended afterwards. The configuration is frozen and
serialisable, so the settings cannot drift between the run and the report. The seeding is
explicit about what it does and does not buy. The learning-rate schedule is a pure function
of the step, so it can be tested at its boundaries instead of being inferred from the shape
of a loss curve. The log is append-only and persists as JSONL, so a crashed run still leaves
its history behind. The manifest ties a result to the data, the model and the library
versions that produced it.

Nothing here imports a model or a tokenizer and nothing here touches CUDA at import time, so
the whole module is exercised on CPU in milliseconds.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sftdpo.modeling.collate import DEFAULT_MAX_SEQ_LENGTH
from sftdpo.modeling.lora import trainable_parameter_report

__all__ = [
    "CONTENT_HASH_VERSION",
    "CUDA_DETERMINISM_NOTE",
    "CURVE_METRICS",
    "MAX_SEED",
    "TRACKED_LIBRARIES",
    "EvalPoint",
    "LogRecord",
    "RunManifest",
    "TrainConfig",
    "TrainLog",
    "collect_library_versions",
    "cosine_schedule_with_warmup",
    "set_determinism",
    "token_content_hash",
    "warmup_steps_for",
]

MAX_SEED = 2**32 - 1
"""Largest seed `numpy.random.seed` accepts; the ceiling is numpy's, not torch's."""

CONTENT_HASH_VERSION = "sftdpo-tokens-v1"
"""Prefix mixed into every content hash. Bumping it invalidates every recorded hash, which
is the intended blast radius of a change to how token sequences are digested."""

CURVE_METRICS: tuple[str, ...] = ("loss", "lr", "grad_norm", "tokens")

TRACKED_LIBRARIES: tuple[str, ...] = ("torch", "transformers", "peft", "numpy", "pydantic")

CUDA_DETERMINISM_NOTE = (
    "Seeding fixes the sampling, the shuffling and the weight initialisation, and on CPU "
    "that is enough for two runs of this package to agree bit for bit. On CUDA it is not. "
    "Many kernels reduce with atomics, so the order in which partial sums are added depends "
    "on how blocks happen to be scheduled, and cuBLAS may select a different GEMM algorithm "
    "for the same shapes on a different card or driver. Reduced precision compounds it: "
    "bf16 addition is not associative, so a different order is a different number. What a "
    "seed does guarantee is that the same code, on the same machine, with the same library "
    "versions and the same device count, reproduces a run closely enough that a change in "
    "the result is attributable to the change you made -- not that a loss curve is "
    "reproducible to the last decimal on someone else's GPU. Passing "
    "deterministic_algorithms=True trades throughput for stricter kernels and raises on any "
    "operation that has no deterministic implementation."
)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


class TrainConfig(BaseModel):
    """Every knob a training run reads, frozen so a run cannot edit its own settings.

    Frozen matters more than it looks. The manifest records this object, and a loop that
    could mutate its config mid-run would produce a manifest describing settings that were
    never used for most of the steps.

    The Adam moments are configuration rather than hard-coded defaults because the second
    moment in particular is worth an experiment: beta2 = 0.95 is a common choice for short
    LoRA runs, where 0.999 averages over more steps than the run even has.

    Attributes:
        learning_rate: Peak LR, reached at the end of warmup and decayed from there.
        batch_size: Examples per forward pass, before accumulation.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        epochs: Passes over the training set when `max_steps` is not set.
        max_steps: Hard cap on optimiser steps; overrides `epochs` when given.
        warmup_ratio: Fraction of the run spent warming the LR up from zero.
        weight_decay: AdamW decoupled decay.
        max_grad_norm: Global gradient-norm clip applied before every step.
        adam_beta1: First-moment decay.
        adam_beta2: Second-moment decay.
        adam_epsilon: Denominator floor.
        bf16: Run the forward and backward pass under bf16 autocast.
        seed: Seed for `set_determinism`.
        eval_every_steps: Validation interval in optimiser steps; 0 disables periodic
            validation entirely.
        log_every_steps: Logging interval in optimiser steps.
        max_seq_length: Token budget per example, enforced by the collator.
        save_best_adapter: Write the adapter of the best validation checkpoint to disk.
        output_dir: Where checkpoints, the log and the manifest are written.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    epochs: int = Field(default=1, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    adam_epsilon: float = Field(default=1e-8, gt=0.0)
    bf16: bool = False
    seed: int = Field(default=0, ge=0, le=MAX_SEED)
    eval_every_steps: int = Field(default=50, ge=0)
    log_every_steps: int = Field(default=10, ge=1)
    max_seq_length: int = Field(default=DEFAULT_MAX_SEQ_LENGTH, ge=2)
    save_best_adapter: bool = True
    output_dir: Path = Path("runs/sft")

    @property
    def effective_batch_size(self) -> int:
        """Examples contributing to one optimiser step."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def betas(self) -> tuple[float, float]:
        """The pair `torch.optim.AdamW` expects."""
        return (self.adam_beta1, self.adam_beta2)

    def steps_per_epoch(self, num_examples: int) -> int:
        """Optimiser steps in one pass over `num_examples`.

        Rounded up: a short final batch is still a step, and rounding down would silently
        drop the tail of every epoch.

        Raises:
            ValueError: If there are no examples to train on.
        """
        if num_examples < 1:
            raise ValueError(f"need at least one example, got {num_examples}")
        return math.ceil(num_examples / self.effective_batch_size)

    def total_steps(self, num_examples: int) -> int:
        """Optimiser steps the run will take.

        `max_steps` wins when it is set, and the loop cycles epochs to reach it; that is the
        setting to use for a smoke run, because it is the only one whose cost does not
        change when the dataset does.
        """
        if self.max_steps is not None:
            return self.max_steps
        return self.steps_per_epoch(num_examples) * self.epochs

    def warmup_steps(self, total_steps: int) -> int:
        """Warmup length in optimiser steps for a run of `total_steps`."""
        return warmup_steps_for(total_steps, self.warmup_ratio)

    def lr_at(self, step: int, total_steps: int) -> float:
        """The learning rate for `step`, peak times the schedule multiplier."""
        multiplier = cosine_schedule_with_warmup(step, total_steps, self.warmup_steps(total_steps))
        return self.learning_rate * multiplier


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def set_determinism(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed every generator a run draws from, and say what that is worth.

    Three generators are seeded because three are read: `random` by the batch shuffling
    here, `numpy` by anything that reaches for the legacy global (much of the scientific
    stack still does), and `torch` by weight initialisation, dropout and sampling. There is
    no fourth call for CUDA: `torch.manual_seed` delegates to `torch.cuda.manual_seed_all`
    itself, whatever the device count, and a test pins that delegation down so this stays
    true if torch changes.

    `PYTHONHASHSEED` is deliberately not touched: it is read once at interpreter start-up,
    so setting it here would be a comforting no-op. Nothing in this package depends on set
    or dict ordering across processes.

    See `CUDA_DETERMINISM_NOTE` for what a seed does and does not guarantee on a GPU.

    Args:
        seed: Value in `[0, MAX_SEED]`.
        deterministic_algorithms: Ask torch to refuse non-deterministic kernels. Off by
            default because it is slower and raises on operations that have no deterministic
            implementation, which is a poor default for a training run but exactly right for
            a reproduction attempt.

    Raises:
        ValueError: If the seed is outside the range numpy accepts. Failing here beats
            failing several minutes into a run inside a library that reseeds late.
    """
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic_algorithms:
        # cuBLAS reuses workspaces across streams unless told not to, which makes some GEMMs
        # non-deterministic; torch raises at the first such call if this is unset.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------------------


def warmup_steps_for(total_steps: int, ratio: float) -> int:
    """Warmup length in steps, floored so it can never consume the whole run.

    Flooring has a useful consequence: `warmup < total_steps` always holds, which is what
    makes the schedule reach exactly zero at the end. A run short enough that the ratio
    rounds to nothing simply has no warmup, which is the right answer for a smoke test.

    Raises:
        ValueError: On a non-positive run length or a ratio outside `[0, 1)`.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}")
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"warmup ratio must be in [0, 1), got {ratio}")
    return int(total_steps * ratio)


def cosine_schedule_with_warmup(step: int, total: int, warmup: int) -> float:
    """Learning-rate multiplier at `step`: linear warmup, then a half cosine to zero.

    A pure function of three integers rather than a `torch.optim.lr_scheduler`, because the
    schedule is the part of a training loop most likely to be wrong in a way no loss curve
    reveals -- an off-by-one that leaves the LR at its peak for the whole run still trains,
    just worse -- and a pure function can be pinned at its boundaries by a test.

    Warmup exists because Adam's second moment starts at zero, so its first few updates are
    scaled by an estimate built from almost no data; a full-size step there can move the
    adapter somewhere the rest of the run spends its budget recovering from.

    Args:
        step: Completed optimiser steps, from 0. Values beyond `total` return 0.0 rather
            than raising, so a run that takes one extra step from a short final
            accumulation group does not die at the finish line.
        total: Total optimiser steps planned.
        warmup: Steps of linear warmup.

    Returns:
        A multiplier in `[0, 1]`. It is 0 at step 0 when there is warmup, 1 at the end of
        warmup, and 0 at `total`.

    Raises:
        ValueError: On a non-positive `total`, a negative `step`, a negative `warmup`, or a
            warmup longer than the run.
    """
    if total < 1:
        raise ValueError(f"total must be at least 1, got {total}")
    if step < 0:
        raise ValueError(f"step must not be negative, got {step}")
    if warmup < 0:
        raise ValueError(f"warmup must not be negative, got {warmup}")
    if warmup > total:
        raise ValueError(f"warmup ({warmup}) cannot exceed the run length ({total})")

    if step < warmup:
        return step / warmup
    # `max(1, ...)` covers the degenerate warmup == total, where the run is all warmup and
    # the final multiplier is 1 rather than 0.
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogRecord:
    """One logged optimiser step.

    `tokens` is cumulative -- supervised tokens consumed by the run up to and including this
    step -- rather than the count for this step alone. Logging is periodic, so per-step
    counts could not be summed into a run total without silently missing every step that was
    not logged.
    """

    step: int
    loss: float
    lr: float
    grad_norm: float
    tokens: int

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"step must not be negative, got {self.step}")
        if self.tokens < 0:
            raise ValueError(f"tokens must not be negative, got {self.tokens}")
        # A NaN compares false against every bound, so these checks reject genuinely
        # negative values while still allowing a diverged run to be recorded honestly.
        if self.lr < 0:
            raise ValueError(f"lr must not be negative, got {self.lr}")
        if self.grad_norm < 0:
            raise ValueError(f"grad_norm must not be negative, got {self.grad_norm}")

    def as_dict(self) -> dict[str, float | int]:
        """JSON-serialisable view, one JSONL line."""
        return {
            "step": self.step,
            "loss": self.loss,
            "lr": self.lr,
            "grad_norm": self.grad_norm,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> LogRecord:
        """Rebuild a record from a parsed JSONL line.

        Raises:
            ValueError: If any field is missing, naming all of them at once so a malformed
                log is diagnosed in one pass rather than one key per attempt.
        """
        missing = sorted({"step", "loss", "lr", "grad_norm", "tokens"} - set(row))
        if missing:
            raise ValueError(f"log record is missing {missing}")
        return cls(
            step=int(row["step"]),
            loss=float(row["loss"]),
            lr=float(row["lr"]),
            grad_norm=float(row["grad_norm"]),
            tokens=int(row["tokens"]),
        )


@dataclass(slots=True)
class TrainLog:
    """Append-only history of a run, with JSONL persistence.

    JSONL rather than a single JSON document because the interesting failure is a run that
    dies at step 900 of 1000: a line-delimited file is complete up to the last flush, while
    a truncated JSON array is unreadable. It is also what a notebook or `jq` reads without
    ceremony.
    """

    records: list[LogRecord] = field(default_factory=list)

    def append(self, *, step: int, loss: float, lr: float, grad_norm: float, tokens: int) -> None:
        """Record one step.

        Raises:
            ValueError: If the step number goes backwards. That is the signature of a loop
                logging its micro-batch counter instead of its optimiser step, which makes
                every curve and every eval interval in the report wrong.
        """
        if self.records and step < self.records[-1].step:
            raise ValueError(
                f"step went backwards: {step} after {self.records[-1].step}; a training log "
                "must be appended in step order"
            )
        self.records.append(
            LogRecord(step=step, loss=loss, lr=lr, grad_norm=grad_norm, tokens=tokens)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[LogRecord]:
        return iter(self.records)

    @property
    def last(self) -> LogRecord | None:
        """The most recent record, or None for an empty log."""
        return self.records[-1] if self.records else None

    @property
    def total_tokens(self) -> int:
        """Supervised tokens the run consumed, read off the last cumulative count."""
        return self.records[-1].tokens if self.records else 0

    @property
    def diverged(self) -> bool:
        """Whether any recorded loss is NaN or infinite."""
        return any(not math.isfinite(record.loss) for record in self.records)

    def curve(self, metric: str = "loss") -> list[tuple[int, float]]:
        """The `(step, value)` series the report plots.

        Args:
            metric: One of `CURVE_METRICS`.

        Raises:
            ValueError: On an unknown metric, rather than returning an empty curve that a
                report would render as a blank chart.
        """
        if metric not in CURVE_METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {list(CURVE_METRICS)}")
        return [(record.step, float(getattr(record, metric))) for record in self.records]

    def to_jsonl(self, path: str | Path) -> Path:
        """Write the log as one JSON object per line.

        Newlines are forced to `\\n` on every platform: a log written on Windows and read
        back on Linux should hash to the same bytes, and a run manifest is only worth as
        much as the artefacts it can be compared against.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(record.as_dict()) + "\n" for record in self.records)
        destination.write_text(body, encoding="utf-8", newline="\n")
        return destination

    @classmethod
    def from_jsonl(cls, path: str | Path) -> TrainLog:
        """Read a log back, blank lines tolerated.

        Raises:
            ValueError: On a line that is not a JSON object, naming the line number.
        """
        log = cls()
        text = Path(path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {number} of {path} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {number} of {path} is not a JSON object")
            log.records.append(LogRecord.from_dict(row))
        return log


@dataclass(frozen=True, slots=True)
class EvalPoint:
    """One periodic validation measurement.

    Kept apart from `LogRecord` rather than folded in with empty fields, because a
    validation loss has no learning rate and no gradient norm, and a log whose columns are
    sometimes meaningless is a log people stop trusting.
    """

    step: int
    loss: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"step must not be negative, got {self.step}")

    def as_dict(self) -> dict[str, float | int]:
        """JSON-serialisable view."""
        return {"step": self.step, "loss": self.loss}


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def token_content_hash(sequences: Iterable[Sequence[int]]) -> str:
    """Digest of the token sequences a run actually consumed.

    Hashing the tokens rather than the source text is the point: it covers the tokenizer,
    the chat template and the prompt/completion boundary as well as the data, so two runs
    with the same hash really did see the same inputs. A length prefix per sequence keeps
    the digest unambiguous -- without it, `[[1, 2], [3]]` and `[[1], [2, 3]]` would hash
    alike.

    Returns:
        A hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    digest.update(CONTENT_HASH_VERSION.encode("utf-8"))
    for sequence in sequences:
        tokens = [int(token) for token in sequence]
        digest.update(f"|{len(tokens)}:".encode())
        digest.update(",".join(str(token) for token in tokens).encode("utf-8"))
    return digest.hexdigest()


def collect_library_versions(names: Sequence[str] = TRACKED_LIBRARIES) -> dict[str, str]:
    """Versions of the libraries whose behaviour a result depends on.

    A missing package is recorded as `"not installed"` rather than raising: a manifest that
    cannot be written is worse than a manifest with a gap in it, and the gap is itself
    informative when a run is reproduced somewhere the optional extras are absent.
    """
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


class RunManifest(BaseModel):
    """What produced a result: the settings, the data, the model and the software.

    Written next to every set of numbers this package reports. The question it answers is
    the one asked of any experimental claim months later -- which data, which base model,
    how much of it was trainable, and which library versions -- and none of those can be
    reconstructed from a checkpoint directory afterwards.

    No timestamp is recorded. Two runs of the same configuration on the same data should
    produce byte-identical manifests, and a clock is the one field that would guarantee they
    never do.
    """

    # `model_name` collides with pydantic's protected `model_` namespace; the field is worth
    # more than the warning, and nothing here shadows a BaseModel attribute.
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    stage: Literal["sft", "dpo"]
    model_name: str
    data_content_hash: str
    trainable_parameters: int = Field(ge=0)
    total_parameters: int = Field(ge=1)
    config: TrainConfig
    library_versions: dict[str, str] = Field(default_factory=collect_library_versions)

    @model_validator(mode="after")
    def _check_counts(self) -> RunManifest:
        if not self.data_content_hash.strip():
            raise ValueError(
                "data_content_hash must not be empty; a manifest without it "
                "cannot tie the result to the data that produced it"
            )
        if self.trainable_parameters > self.total_parameters:
            raise ValueError(
                f"trainable_parameters ({self.trainable_parameters}) exceeds "
                f"total_parameters ({self.total_parameters})"
            )
        return self

    @property
    def trainable_pct(self) -> float:
        """Trainable share of the weights, as a percentage."""
        return 100.0 * self.trainable_parameters / self.total_parameters

    @classmethod
    def build(
        cls,
        *,
        stage: Literal["sft", "dpo"],
        model: Any,
        model_name: str,
        data_content_hash: str,
        config: TrainConfig,
        library_versions: dict[str, str] | None = None,
    ) -> RunManifest:
        """Fill the parameter counts from a live model.

        Counted from the model rather than taken on trust, because a LoRA target-module name
        that matches nothing leaves a model with zero trainable parameters and a run that
        looks entirely healthy while learning nothing.
        """
        report = trainable_parameter_report(model)
        return cls(
            stage=stage,
            model_name=model_name,
            data_content_hash=data_content_hash,
            trainable_parameters=report.trainable,
            total_parameters=report.total,
            config=config,
            library_versions=library_versions or collect_library_versions(),
        )

    def to_json(self, path: str | Path) -> Path:
        """Write the manifest as indented JSON with `\\n` newlines."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        return destination

    @classmethod
    def from_json(cls, path: str | Path) -> RunManifest:
        """Read a manifest back, validating it on the way in."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
