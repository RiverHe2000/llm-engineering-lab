"""The whole experiment as nine resumable stages, each writing its own artefacts.

A real run of this project is three hours on one RTX 4070: build the corpus, evaluate the
base model, fine-tune it, evaluate again, sample thousands of completions, mine preference
pairs from them, run DPO, evaluate a third time, gate the result and write the report. Three
hours is long enough that the machine will be interrupted, so the pipeline is built around
one rule -- every stage writes files, and a stage whose files are already on disk is skipped.
Resumption is therefore the normal path rather than a recovery mode, and `--force` is what
you reach for when you want work redone.

Two consequences of that rule shape everything else here.

*Stages communicate through the run directory, never through memory.* `stage_sft` reads the
corpus back from `data/` rather than receiving a `Dataset` object, and `stage_dpo` reads
`pairs.jsonl` rather than receiving a `MiningResult`. That costs a little I/O and buys the
property the whole design exists for: any stage can be run on its own, against artefacts
produced days earlier by a process that is long gone.

*Every stage records what it did.* `StageRecord` is written for each stage, carrying its
status, the files it produced, a small summary of headline numbers and the library versions
that produced them. The two training stages additionally write the
:class:`sftdpo.train.common.RunManifest` their trainers build -- that class is deliberately
specific to a training run (it is typed to `"sft"` or `"dpo"` and counts a live model's
parameters), so it cannot describe the data or evaluation stages, and `StageRecord` is the
uniform record that can.

The model is reached through a :class:`ModelProvider` rather than loaded directly. That is
not indirection for its own sake: the Qwen tokenizer is not available offline, so the only
way to exercise this driver end to end in CI is to substitute a two-layer model and a
deterministic stand-in tokenizer. :class:`SmokeProvider` does exactly that, which is what
makes `sftdpo pipeline smoke` -- and the test that runs the entire pipeline in seconds -- a
real execution of this code rather than a mock of it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field

from sftdpo.eval.compare import ComparisonResult, Floors, compare
from sftdpo.eval.generate import (
    completion_length,
    generate_samples,
    left_pad,
    length_sorted_batches,
    resolve_pad_token_id,
)
from sftdpo.eval.metrics import EvalReport, evaluate
from sftdpo.eval.stats import DEFAULT_RESAMPLES
from sftdpo.eval.tax import AlignmentTax, ProbeReport, alignment_tax, run_probes
from sftdpo.modeling.chat import ChatFormatter, encode_text
from sftdpo.modeling.collate import DEFAULT_MAX_SEQ_LENGTH
from sftdpo.modeling.loader import LoadedModel, ModelInfo, build_tiny_model, load_model
from sftdpo.modeling.lora import attach_adapter, lora_config
from sftdpo.prefs.pairs import (
    DEFAULT_MAX_PAIRS_PER_PROMPT,
    DEFAULT_MIN_MARGIN,
    mine_pairs,
)
from sftdpo.prefs.sample import prompts_for_examples, sample_completions
from sftdpo.schemas import Example, GenerationConfig, PreferencePair, Sample, Split
from sftdpo.task.dataset import Dataset, build_dataset
from sftdpo.task.generate import render_prompt
from sftdpo.train.common import TrainConfig, collect_library_versions
from sftdpo.train.dpo import train_dpo
from sftdpo.train.dpo_loss import DPOVariant
from sftdpo.train.sft import train_sft
from sftdpo.verify.reward import Verifier, strict_verifier

__all__ = [
    "COMPARISONS",
    "DEFAULT_MODEL",
    "EVAL_VARIANTS",
    "STAGE_ORDER",
    "HubProvider",
    "ModelProvider",
    "PipelineConfig",
    "PipelineError",
    "PipelineResult",
    "RunLayout",
    "SmokeProvider",
    "Stage",
    "StageRecord",
    "StageResult",
    "TinyTokenizer",
    "build_report",
    "compare_reports",
    "config_digest",
    "generate_replies",
    "gold_fallback_pairs",
    "measure_alignment_tax",
    "probe_report",
    "publish_adapter",
    "run_pipeline",
    "smoke_config",
    "stage_compare",
    "stage_data",
    "stage_dpo",
    "stage_eval_base",
    "stage_eval_dpo",
    "stage_eval_sft",
    "stage_mine",
    "stage_report",
    "stage_sft",
]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

EVAL_VARIANTS: tuple[str, ...] = ("base", "sft", "dpo")
"""The three variants the pipeline itself produces, in the order a report reads them."""

COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("base_sft", "base", "sft"),
    ("sft_dpo", "sft", "dpo"),
    ("base_dpo", "base", "dpo"),
)
"""Each promotion gate as (name, baseline variant, candidate variant).

`base_dpo` is not redundant with the two adjacent comparisons: a stage-by-stage gate can pass
twice on differences too small to survive being compounded, and the end-to-end claim of the
project is the one from the untouched model to the final one.
"""


def config_digest(config: PipelineConfig) -> str:
    """A stable fingerprint of everything about a run except where it writes.

    `run_dir` is excluded: copying a finished run to another directory does not change what
    produced it, and including the path would make every stage look stale after a move.
    Everything else is in, because everything else can change a number.
    """
    payload = config.model_dump(mode="json", exclude={"run_dir"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


class PipelineError(RuntimeError):
    """A stage could not run on the artefacts it was given.

    Distinct from `ValueError` so that the command line can report a pipeline problem -- a
    missing upstream artefact, a mining run that produced nothing to train on -- separately
    from a bad argument, which is a different thing for a user to fix.
    """


# --------------------------------------------------------------------------------------
# Stages and the run directory
# --------------------------------------------------------------------------------------


class Stage(StrEnum):
    """The nine stages, in dependency order.

    Declaration order is the execution order: `STAGE_ORDER` is `tuple(Stage)`, and a caller
    that asks for a subset gets it back in this order rather than in the order they listed
    it, because running DPO before SFT is never what someone meant.
    """

    DATA = "data"
    EVAL_BASE = "eval_base"
    SFT = "sft"
    EVAL_SFT = "eval_sft"
    MINE = "mine"
    DPO = "dpo"
    EVAL_DPO = "eval_dpo"
    COMPARE = "compare"
    REPORT = "report"


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Where every artefact of a run lives.

    One object owns the paths so that the stage that writes a file and the stage that reads
    it cannot disagree about its name. The layout is also what makes resumption possible at
    all: `outputs` is the definition of "this stage has already run".
    """

    root: Path

    @property
    def data_dir(self) -> Path:
        """The corpus, one JSONL per split plus its manifest."""
        return self.root / "data"

    @property
    def data_stats(self) -> Path:
        """Descriptive statistics for the corpus."""
        return self.root / "data_stats.json"

    @property
    def sft_dir(self) -> Path:
        """Supervised fine-tuning logs, manifest and checkpoints."""
        return self.root / "sft"

    @property
    def sft_adapter(self) -> Path:
        """The adapter the SFT stage settled on, at a name that does not depend on why."""
        return self.sft_dir / "adapter"

    @property
    def dpo_dir(self) -> Path:
        """Preference-optimisation logs, manifest and checkpoint."""
        return self.root / "dpo"

    @property
    def dpo_adapter(self) -> Path:
        """The adapter the DPO stage produced."""
        return self.dpo_dir / "adapter"

    @property
    def samples(self) -> Path:
        """Every sampled completion, kept so mining can be re-run without the GPU."""
        return self.root / "samples.jsonl"

    @property
    def pairs(self) -> Path:
        """The mined preference pairs."""
        return self.root / "pairs.jsonl"

    @property
    def mining_stats(self) -> Path:
        """Why there are as many pairs as there are."""
        return self.root / "mining_stats.json"

    @property
    def report(self) -> Path:
        """The Markdown comparison across every variant in the run."""
        return self.root / "report.md"

    @property
    def stage_dir(self) -> Path:
        """One `StageRecord` per stage."""
        return self.root / "stages"

    @property
    def config(self) -> Path:
        """The settings that produced this run.

        Written before any stage executes, so a run directory can always answer what made
        it. Several of these settings change the numbers without changing any file name --
        `sample_batch_size` and `eval_batch_size` in particular, because batched decoding is
        not bit-identical to single-stream decoding on a GPU (see `sftdpo.prefs.sample`).
        A comparison between two variants is only sound if they were decoded the same way,
        and without this file there is no way to check that afterwards.
        """
        return self.root / "config.json"

    def eval_report(self, variant: str) -> Path:
        """The evaluation report of one variant."""
        return self.root / f"eval_{variant}.json"

    def comparison(self, name: str, suffix: str) -> Path:
        """One promotion gate, as Markdown (`md`) or as the full result (`json`)."""
        return self.root / f"compare_{name}.{suffix}"

    def stage_record(self, stage: Stage) -> Path:
        """Where a stage's record is written."""
        return self.stage_dir / f"{stage.value}.json"

    def outputs(self, stage: Stage) -> tuple[Path, ...]:
        """The artefacts whose presence means `stage` has already run.

        Checkpoint directories are listed rather than individual weight files, because the
        file names inside a PEFT adapter directory are peft's business and not this
        module's; `_complete` treats a directory as present when it is not empty.
        """
        if stage is Stage.DATA:
            return (
                *(self.data_dir / f"{name}.jsonl" for name in ("train", "val", "test")),
                self.data_dir / "manifest.json",
                self.data_stats,
            )
        if stage is Stage.EVAL_BASE:
            return (self.eval_report("base"),)
        if stage is Stage.SFT:
            return (
                self.sft_dir / "summary.json",
                self.sft_dir / "manifest.json",
                self.sft_dir / "train_log.jsonl",
                self.sft_adapter,
            )
        if stage is Stage.EVAL_SFT:
            return (self.eval_report("sft"),)
        if stage is Stage.MINE:
            return (self.samples, self.pairs, self.mining_stats)
        if stage is Stage.DPO:
            return (
                self.dpo_dir / "summary.json",
                self.dpo_dir / "manifest.json",
                self.dpo_dir / "train_log.jsonl",
                self.dpo_dir / "reward_log.jsonl",
                self.dpo_adapter,
            )
        if stage is Stage.EVAL_DPO:
            return (self.eval_report("dpo"),)
        if stage is Stage.COMPARE:
            return tuple(
                self.comparison(name, suffix)
                for name, _, _ in COMPARISONS
                for suffix in ("md", "json")
            )
        return (self.report,)


def _complete(outputs: Sequence[Path]) -> bool:
    """Whether every artefact of a stage is already on disk.

    An empty directory counts as absent: a checkpoint directory that was created and then
    never written to is the state an interrupted run leaves behind, and treating it as
    finished would resume a pipeline onto weights that do not exist.
    """
    for path in outputs:
        if path.is_dir():
            if not any(path.iterdir()):
                return False
        elif not path.is_file():
            return False
    return bool(outputs)


class StageRecord(BaseModel):
    """What one stage did, written beside its artefacts.

    Deliberately uniform across all nine stages. `RunManifest` describes a *training* run --
    it is typed to `"sft"` or `"dpo"`, carries a `TrainConfig` and counts a live model's
    parameters -- so it cannot describe corpus construction or a promotion gate. Both are
    written where they apply: the trainers write their `RunManifest` into `sft/` and `dpo/`,
    and this record sits alongside for every stage.

    No timestamp, for the same reason `RunManifest` carries none: two runs of the same
    configuration should produce identical records, and a clock guarantees they never do.
    """

    model_config = ConfigDict(frozen=True)

    stage: Stage
    status: Literal["ran", "skipped"]
    outputs: tuple[str, ...] = ()
    summary: dict[str, Any] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=collect_library_versions)
    config_digest: str = ""
    """Fingerprint of the configuration that produced these artefacts.

    The reason a stage can be skipped safely. Skipping on file presence alone lets a resumed
    run keep artefacts built under the old settings while `config.json` is rewritten with the
    new ones, so the run directory ends up describing a configuration that by construction did
    not produce what is in it: resume `--n-test 6, seed 0` with `--n-test 24, seed 99` and the
    corpus on disk is still six rows from seed 0, while every downstream number is attributed
    to seed 99. Comparing the digest turns that into a re-run instead of a silent mismatch.
    Empty on a record written before this field existed, which is treated as "cannot vouch for
    it" and re-runs the stage.
    """

    def to_json(self, path: str | Path) -> Path:
        """Write the record as indented JSON with `\\n` newlines."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        return destination

    @classmethod
    def from_json(cls, path: str | Path) -> StageRecord:
        """Read a record back, validating it on the way in."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class StageResult:
    """One stage's record together with the paths it is a record of."""

    stage: Stage
    record: StageRecord
    outputs: tuple[Path, ...]

    @property
    def skipped(self) -> bool:
        """Whether the stage found its artefacts already present and did no work."""
        return self.record.status == "skipped"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything a pipeline invocation did, stage by stage."""

    config: PipelineConfig
    layout: RunLayout
    results: tuple[StageResult, ...]

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The stages that were considered, in execution order."""
        return tuple(result.stage for result in self.results)

    @property
    def ran(self) -> tuple[Stage, ...]:
        """The stages that actually did work."""
        return tuple(result.stage for result in self.results if not result.skipped)

    @property
    def skipped(self) -> tuple[Stage, ...]:
        """The stages that found their artefacts already present."""
        return tuple(result.stage for result in self.results if result.skipped)

    @property
    def artefacts(self) -> tuple[Path, ...]:
        """Every file and directory the run is responsible for, in stage order."""
        return tuple(path for result in self.results for path in result.outputs)

    def record(self, stage: Stage) -> StageRecord:
        """One stage's record.

        Raises:
            KeyError: If the stage was not part of this invocation.
        """
        for result in self.results:
            if result.stage is stage:
                return result.record
        raise KeyError(f"stage {stage.value!r} was not part of this run")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Every setting the driver reads, frozen so a run cannot edit its own configuration.

    The fields are grouped by stage rather than alphabetically, because that is the order
    someone tuning a run reads them in. Defaults describe the real experiment on a 0.5 B
    model; `smoke_config` overrides them for the seconds-long CI run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    run_dir: Path
    model: str = DEFAULT_MODEL
    seed: int = Field(default=1, ge=0)

    n_train: int = Field(default=96, ge=1)
    n_val: int = Field(default=24, ge=0)
    n_test: int = Field(default=48, ge=1)

    sft_epochs: int = Field(default=3, ge=1)
    sft_max_steps: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_seq_length: int = Field(default=DEFAULT_MAX_SEQ_LENGTH, ge=2)
    log_every_steps: int = Field(default=10, ge=1)
    eval_every_steps: int = Field(default=50, ge=0)

    mine_split: Split = "train"
    sample_k: int = Field(default=6, ge=1)
    temperature: float = Field(default=0.9, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    sample_batch_size: int = Field(default=8, ge=1)
    sample_max_new_tokens: int = Field(default=320, ge=1)
    min_margin: float = Field(default=DEFAULT_MIN_MARGIN, gt=0.0)
    max_pairs_per_prompt: int = Field(default=DEFAULT_MAX_PAIRS_PER_PROMPT, ge=1)
    gold_pair_fallback: bool = False

    dpo_epochs: int = Field(default=2, ge=1)
    dpo_max_steps: int | None = Field(default=None, ge=1)
    dpo_batch_size: int = Field(default=1, ge=1)
    dpo_gradient_accumulation_steps: int = Field(default=4, ge=1)
    dpo_learning_rate: float = Field(default=1e-5, gt=0.0)
    """Its own learning rate, an order of magnitude below the supervised one.

    Sharing `learning_rate` between the two stages is what destroyed the first real run. At
    the supervised 1e-4 the preference stage drove the *rejected* completions' implicit
    reward from +0.13 to -7.15 while the chosen reward stayed near zero: the margin grew to
    7.5, reward accuracy reached 1.0 and the loss fell to 0.0014, so every quantity a
    training loop reports looked healthy. The policy had simply moved far enough from the
    reference to stop producing valid JSON at all --- strict validity 0.981 -> 0.000 --- and
    the promotion gate rejected it on every slice.

    DPO is not supervised learning at a different objective; it is a small correction to a
    model that already works, and its step size has to say so.
    """
    beta: float = Field(default=0.1, gt=0.0)
    variant: DPOVariant = "sigmoid"
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=0.5)
    length_normalise: bool = False

    eval_split: Split = "test"
    eval_batch_size: int = Field(default=8, ge=1)
    eval_max_new_tokens: int = Field(default=320, ge=1)

    margin: float = Field(default=0.0, ge=0.0, le=1.0)
    json_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    schema_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    max_slice_regression: float = Field(default=0.05, ge=0.0, le=1.0)
    max_field_recall_drop: float = Field(default=0.10, ge=0.0, le=1.0)
    min_field_support: int = Field(default=10, ge=0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    n_resamples: int = Field(default=DEFAULT_RESAMPLES, ge=1)

    dtype: str = "bfloat16"
    device: str | None = None
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bf16: bool = False

    @property
    def layout(self) -> RunLayout:
        """Where this run's artefacts live."""
        return RunLayout(root=Path(self.run_dir))

    def floors(self) -> Floors:
        """The absolute conditions the promotion gate applies."""
        return Floors(
            min_json_valid=self.json_floor,
            min_schema_valid=self.schema_floor,
            max_slice_regression=self.max_slice_regression,
            max_field_recall_drop=self.max_field_recall_drop,
            min_field_support=self.min_field_support,
        )

    def sampling_config(self) -> GenerationConfig:
        """Decoding settings for the preference-mining sweep.

        Sampled rather than greedy, because k identical completions cannot be ranked against
        each other and the whole stage would cost k times the GPU time for no pairs.
        """
        return GenerationConfig(
            max_new_tokens=self.sample_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )

    def eval_config(self) -> GenerationConfig:
        """Decoding settings for evaluation: greedy, so the gate pays for no decoder noise."""
        return GenerationConfig(
            max_new_tokens=self.eval_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=self.seed,
        )

    def _train_config(
        self,
        output_dir: Path,
        *,
        epochs: int,
        max_steps: int | None,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float | None = None,
    ) -> TrainConfig:
        return TrainConfig(
            learning_rate=self.learning_rate if learning_rate is None else learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            epochs=epochs,
            max_steps=max_steps,
            bf16=self.bf16,
            seed=self.seed,
            eval_every_steps=self.eval_every_steps,
            log_every_steps=self.log_every_steps,
            max_seq_length=self.max_seq_length,
            save_best_adapter=True,
            output_dir=output_dir,
        )

    def sft_train_config(self) -> TrainConfig:
        """The frozen configuration handed to `train_sft`."""
        return self._train_config(
            self.layout.sft_dir,
            epochs=self.sft_epochs,
            max_steps=self.sft_max_steps,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
        )

    def dpo_train_config(self) -> TrainConfig:
        """The frozen configuration handed to `train_dpo`.

        Its own batch size, because a DPO step is not an SFT step of the same width. Every
        pair puts two sequences through the policy and two more through the reference, so a
        batch of `n` pairs holds four times the activations and four vocabulary softmaxes of
        an SFT batch of `n`. On a 12 GiB card at 2 048 tokens that is the difference between
        fitting and not: see `sftdpo.train.dpo_loss.LOGPROB_CHUNK_ELEMENTS` for the measured
        figures. `dpo_gradient_accumulation_steps` compensates, so the effective batch --
        the thing that actually affects the optimisation -- stays under the caller's control.
        """
        return self._train_config(
            self.layout.dpo_dir,
            epochs=self.dpo_epochs,
            max_steps=self.dpo_max_steps,
            batch_size=self.dpo_batch_size,
            gradient_accumulation_steps=self.dpo_gradient_accumulation_steps,
            learning_rate=self.dpo_learning_rate,
        )


def smoke_config(run_dir: str | Path, **overrides: Any) -> PipelineConfig:
    """The seconds-long configuration: a handful of examples and a couple of steps a stage.

    Every number here is the smallest that still exercises the code path. Six examples per
    split is one per difficulty slice, so the per-slice reporting has a row for each; two
    optimiser steps is enough for an optimiser to step and a schedule to move; `k = 2` is the
    smallest sample count from which a pair could be mined at all.

    `gold_pair_fallback` is on, and it is on only here. A randomly initialised two-layer model
    emits nothing that parses, so every completion scores zero, every prompt is flat and
    mining yields no pairs -- correctly. Rather than pretend otherwise, the smoke run pairs
    each gold record against the worst sample so that the DPO stage has real input. That is a
    different training signal from the one the experiment reports, which is exactly why it is
    a flag and not the default.
    """
    settings: dict[str, Any] = {
        "run_dir": Path(run_dir),
        "model": "tiny",
        "seed": 0,
        "n_train": 6,
        "n_val": 6,
        "n_test": 6,
        "sft_max_steps": 2,
        "batch_size": 2,
        "log_every_steps": 1,
        "eval_every_steps": 1,
        "max_seq_length": 512,
        "sample_k": 2,
        "sample_batch_size": 3,
        "sample_max_new_tokens": 6,
        "gold_pair_fallback": True,
        "dpo_max_steps": 2,
        "dpo_batch_size": 2,
        "dpo_gradient_accumulation_steps": 1,
        "eval_batch_size": 3,
        "eval_max_new_tokens": 6,
        "n_resamples": 64,
        "dtype": "float32",
        "device": "cpu",
        "lora_r": 4,
        "lora_alpha": 8,
    }
    settings.update(overrides)
    return PipelineConfig(**settings)


# --------------------------------------------------------------------------------------
# Where the model comes from
# --------------------------------------------------------------------------------------


class ModelProvider(Protocol):
    """How a stage gets hold of a model and its tokenizer.

    One method rather than three, so a test stub is a dozen lines. `trainable` is a separate
    argument from `adapter` because the two combinations differ in kind: without an adapter
    it means "attach a fresh LoRA", and with one it means "load these weights so the
    optimiser can move them", which PEFT expresses as a different call.
    """

    def load(
        self, model: str, *, adapter: Path | None = None, trainable: bool = False
    ) -> LoadedModel:
        """Load `model`, optionally with `adapter` on top."""


@dataclass(frozen=True, slots=True)
class HubProvider:
    """The real path: a checkpoint from the Hugging Face cache, LoRA attached on request."""

    dtype: str = "bfloat16"
    device: str | None = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    def load(
        self, model: str, *, adapter: Path | None = None, trainable: bool = False
    ) -> LoadedModel:
        """Load a checkpoint, and make its adapter trainable when asked.

        Args:
            model: Hub id or local path.
            adapter: A saved PEFT adapter directory, or None for the bare checkpoint.
            trainable: Return a model whose adapter the optimiser may move. With no adapter
                this attaches a fresh one; with an adapter it reloads those weights as
                trainable, which `load_model` deliberately does not do because every other
                caller wants them frozen.

        Returns:
            The model, its tokenizer and the parameter counts, recounted after any wrapping
            so that a manifest cannot quote the base model's numbers for an adapted one.
        """
        if not trainable:
            return load_model(model, dtype=self.dtype, device=self.device, adapter_path=adapter)

        base = load_model(model, dtype=self.dtype, device=self.device)
        if adapter is None:
            wrapped = attach_adapter(
                base.model,
                lora_config(r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout),
            )
        else:
            from peft import PeftModel

            wrapped = PeftModel.from_pretrained(base.model, str(adapter), is_trainable=True)
        return LoadedModel(
            model=wrapped,
            tokenizer=base.tokenizer,
            info=ModelInfo.from_model(wrapped, name=model),
        )


TINY_VOCAB = 151936
"""Vocabulary of the CI model, kept at the real Qwen size so token ids stay interchangeable."""


@dataclass(slots=True)
class TinyTokenizer:
    """A deterministic stand-in tokenizer, for the smoke path and for nothing else.

    The Qwen tokenizer needs files this project deliberately does not download in CI, so the
    end-to-end test would otherwise have to stop at the first stage that tokenises anything.
    This splits text into fixed-width character chunks and maps each chunk to an id with a
    hash, which buys the three properties the pipeline actually depends on: the same text
    always produces the same ids *in any process*, the prompt's ids are a prefix of the full
    text's ids, and a completion round-trips back to its own characters.

    The hash makes it stateless where it matters. A table built in encoding order would give
    the same string different ids in different stages, so an adapter trained in one stage
    would be evaluated against a different tokenisation in the next -- silently, and with a
    plausible-looking loss curve. Collisions are possible and harmless here: two chunks
    sharing an id costs the stand-in a little fidelity and costs the pipeline nothing.

    An id the instance has never encoded decodes to `<id>` rather than to nothing, because a
    randomly initialised model emits ids from all over the vocabulary and a decoder that
    returned the empty string for them would hand the verifier nothing to score and the
    preference miner nothing to reject.

    Attributes:
        chunk: Characters per token. Chosen to keep the tiny model's sequences short: the
            prompt carries the whole JSON schema, so a byte-level stand-in would make every
            forward pass in CI ten times the size for no extra coverage.
        vocab_size: Id ceiling, matching the CI model's embedding matrix.
        pad_token_id: Filler id; also the stop token, as on many instruct checkpoints.
        eos_token_id: Stop token.
    """

    chunk: int = 64
    vocab_size: int = TINY_VOCAB
    pad_token_id: int = 0
    eos_token_id: int = 0
    # Filled in as text is encoded, so that a completion can be decoded back. Not a
    # constructor argument: the table is a consequence of what this instance has seen, and a
    # caller who supplied one would be describing a tokenisation that never happened.
    _pieces: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {self.chunk}")
        if self.vocab_size < 4:
            raise ValueError(f"vocab_size must be at least 4, got {self.vocab_size}")

    def _id_for(self, piece: str) -> int:
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
        # Ids 0 and 1 are reserved: 0 is the stop token, and leaving 1 unused keeps a spare
        # for anything that needs an id no piece of text can claim.
        return 2 + int.from_bytes(digest, "big") % (self.vocab_size - 2)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Split `text` into fixed-width chunks and map each to its id.

        Args:
            text: The text to tokenise.
            add_special_tokens: Must be False. The parameter exists because that is the
                keyword `encode_text` passes; there are no special tokens to add here, and
                silently ignoring a request for them would hide a real mis-call.

        Returns:
            One id per chunk, in order.

        Raises:
            ValueError: If `add_special_tokens` is True.
        """
        if add_special_tokens:
            raise ValueError("the stand-in tokenizer has no special tokens to add")
        ids: list[int] = []
        for start in range(0, len(text), self.chunk):
            piece = text[start : start + self.chunk]
            token = self._id_for(piece)
            self._pieces[token] = piece
            ids.append(token)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Turn ids back into text, rendering unseen ids as `<id>`."""
        parts: list[str] = []
        for token in ids:
            value = int(token)
            if value in (self.pad_token_id, self.eos_token_id):
                if not skip_special_tokens:
                    parts.append("<eos>")
                continue
            parts.append(self._pieces.get(value, f"<{value}>"))
        return "".join(parts)


@dataclass(frozen=True, slots=True)
class SmokeProvider:
    """The CI path: a two-layer Qwen2 built from config and the stand-in tokenizer.

    Nothing is downloaded and nothing touches the network, so the whole pipeline runs on a
    machine with an empty Hugging Face cache. The model name a caller passes is ignored on
    purpose -- there is only one model here -- but it is still recorded on every sample and
    every report, so the artefacts of a smoke run say what produced them.
    """

    seed: int = 0
    chunk: int = 64
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.0

    def load(
        self, model: str, *, adapter: Path | None = None, trainable: bool = False
    ) -> LoadedModel:
        """Build the tiny model, attaching or reloading a LoRA adapter as asked."""
        tokenizer = TinyTokenizer(chunk=self.chunk)
        base = build_tiny_model(self.seed)
        wrapped: Any = base
        if trainable and adapter is None:
            wrapped = attach_adapter(
                base,
                lora_config(r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout),
            )
        elif adapter is not None:
            from peft import PeftModel

            wrapped = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
        wrapped.eval()
        return LoadedModel(
            model=wrapped,
            tokenizer=tokenizer,
            info=ModelInfo.from_model(wrapped, name=model),
        )


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> Path:
    """Write indented, key-sorted JSON with `\\n` newlines, so two runs diff cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    path.write_text(body + "\n", encoding="utf-8", newline="\n")
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[BaseModel]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{row.model_dump_json()}\n" for row in rows)
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _read_pairs(path: Path) -> list[PreferencePair]:
    text = path.read_text(encoding="utf-8")
    return [PreferencePair.model_validate_json(line) for line in text.splitlines() if line.strip()]


def _require(path: Path, stage: Stage) -> Path:
    """Fail with the stage that should have produced a missing input, not with a bare path."""
    if not path.exists():
        raise PipelineError(
            f"{path} is missing; run the {stage.value!r} stage before this one, or drop "
            "--only if you meant to run the whole pipeline"
        )
    return path


def publish_adapter(source: Path | None, destination: Path, stage: Stage) -> Path:
    """Put the adapter a trainer chose at a name later stages can rely on.

    `train_sft` writes to `best/` or `final/` depending on whether validation ever improved,
    which is the right thing for a training log and the wrong thing for a downstream stage
    that has to name a directory before the run starts. Copying costs a few hundred kilobytes
    at rank 8 and removes the branch from every caller.

    Raises:
        PipelineError: If the trainer saved nothing.
    """
    if source is None:
        raise PipelineError(
            f"the {stage.value!r} stage trained but saved no adapter; there is nothing for the "
            "next stage to load"
        )
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def _evaluate_variant(
    ctx: _Context,
    *,
    variant: str,
    adapter: Path | None,
    label: str,
) -> dict[str, Any]:
    """Decode one greedy completion per evaluation example and score the lot."""
    config = ctx.config
    dataset = Dataset.load(_require(config.layout.data_dir, Stage.DATA))
    examples = dataset.examples_for(config.eval_split)
    if not examples:
        raise PipelineError(
            f"the {config.eval_split!r} split is empty; there is nothing to evaluate"
        )
    loaded = ctx.provider.load(config.model, adapter=adapter)
    samples = generate_samples(
        loaded.model,
        loaded.tokenizer,
        examples,
        model_name=label,
        formatter=ChatFormatter(tokenizer=loaded.tokenizer),
        config=config.eval_config(),
        batch_size=config.eval_batch_size,
    )
    report = evaluate(samples, examples, model=label)
    path = config.layout.eval_report(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    return {"variant": variant, "model": label, **report.overall.as_dict()}


# --------------------------------------------------------------------------------------
# The stages
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Context:
    """What every stage function is handed: the settings and the way to a model."""

    config: PipelineConfig
    provider: ModelProvider


def stage_data(ctx: _Context) -> dict[str, Any]:
    """Build the corpus and write it with its statistics.

    The statistics are an artefact rather than something printed, because the presence rates
    they carry are the only check that the absent-fields slice is still producing empty lists
    -- and that is a property no downstream test would notice losing.
    """
    config = ctx.config
    dataset = build_dataset(
        seed=config.seed,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
    )
    dataset.save(config.layout.data_dir)
    stats = dataset.stats()
    _write_json(config.layout.data_stats, stats.model_dump(mode="json"))
    return {
        "total": stats.total,
        "content_hash": stats.content_hash,
        "counts": {name: block.n for name, block in stats.splits.items()},
    }


def stage_eval_base(ctx: _Context) -> dict[str, Any]:
    """Evaluate the untouched checkpoint: the number every later claim is relative to."""
    return _evaluate_variant(ctx, variant="base", adapter=None, label=ctx.config.model)


def stage_sft(ctx: _Context) -> dict[str, Any]:
    """Fine-tune a fresh LoRA adapter on the gold records, completions only.

    The corpus hash is passed to the trainer rather than letting it hash the tokenised
    features, so the manifest cites the same identifier the data stage wrote and a reader can
    match a training run to a corpus without re-tokenising anything.
    """
    config = ctx.config
    dataset = Dataset.load(_require(config.layout.data_dir, Stage.DATA))
    loaded = ctx.provider.load(config.model, adapter=None, trainable=True)
    result = train_sft(
        loaded.model,
        loaded.tokenizer,
        dataset.train,
        dataset.val,
        config.sft_train_config(),
        model_name=config.model,
        data_content_hash=dataset.content_hash(),
    )
    result.save(config.layout.sft_dir)
    publish_adapter(result.adapter_path, config.layout.sft_adapter, Stage.SFT)
    return {
        "steps": result.steps,
        "supervised_tokens": result.supervised_tokens,
        "final_train_loss": result.final_train_loss,
        "best_val_loss": result.best_val_loss,
        "trainable_pct": result.manifest.trainable_pct,
    }


def stage_eval_sft(ctx: _Context) -> dict[str, Any]:
    """Evaluate the supervised checkpoint on the same split as the base model."""
    config = ctx.config
    return _evaluate_variant(
        ctx,
        variant="sft",
        adapter=_require(config.layout.sft_adapter, Stage.SFT),
        label=f"{config.model}+sft",
    )


def gold_fallback_pairs(
    examples: Sequence[Example],
    samples: Sequence[Sample],
    verifier: Verifier,
    *,
    min_margin: float,
    max_pairs_per_prompt: int,
) -> list[PreferencePair]:
    """Pair each gold record against the worst completion sampled for it.

    This is not the preference signal the project reports. Mining works by ranking the
    policy's *own* completions against each other, which is what makes DPO an
    on-policy-flavoured update; pairing against gold is closer to supervised training wearing
    a preference-shaped hat. It exists for one situation: a model so bad that every sample
    scores identically, where honest mining correctly yields nothing and the stages after it
    have no input at all. That is the permanent state of the randomly initialised CI model,
    so the smoke run turns this on to exercise the DPO stage, and a real run leaves it off.

    Args:
        examples: The prompts sampled from.
        samples: The completions drawn for them.
        verifier: The scorer, used both on the samples and on the gold record itself, so the
            margin is measured on one scale.
        min_margin: Smallest reward gap worth a pair.
        max_pairs_per_prompt: Cap on pairs from one prompt.

    Returns:
        The pairs, grouped by example in the order the examples were given.
    """
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.example_id, []).append(sample)

    pairs: list[PreferencePair] = []
    for example in examples:
        group = grouped.get(example.example_id)
        if not group:
            continue
        gold_text = example.gold_json
        gold_value = verifier.score(gold_text, example.gold).value
        # Sorted worst first, ties broken by sample index, so the pair carries the largest
        # margin available and two runs over the same completions emit the same pairs.
        scored = sorted(
            (
                (verifier.score(sample.text, example.gold).value, sample.sample_index, sample.text)
                for sample in group
                if sample.text.strip() and sample.text != gold_text
            ),
            key=lambda item: (item[0], item[1]),
        )
        seen: set[str] = set()
        for value, _, text in scored:
            if len(seen) >= max_pairs_per_prompt:
                break
            if text in seen or gold_value - value < min_margin:
                continue
            seen.add(text)
            pairs.append(
                PreferencePair(
                    example_id=example.example_id,
                    slice=example.slice,
                    prompt=render_prompt(example.note),
                    chosen=gold_text,
                    rejected=text,
                    chosen_reward=gold_value,
                    rejected_reward=value,
                )
            )
    return pairs


def stage_mine(ctx: _Context) -> dict[str, Any]:
    """Sample k completions per training prompt and mine preference pairs from them.

    The raw samples are written as well as the pairs. They are the expensive artefact -- this
    is the stage that costs the GPU hours -- and keeping them means the mining rules can be
    re-tuned, or the margin lowered, without decoding anything a second time.
    """
    config = ctx.config
    dataset = Dataset.load(_require(config.layout.data_dir, Stage.DATA))
    examples = dataset.examples_for(config.mine_split)
    if not examples:
        raise PipelineError(f"the {config.mine_split!r} split is empty; nothing to sample from")

    adapter = _require(config.layout.sft_adapter, Stage.SFT)
    loaded = ctx.provider.load(config.model, adapter=adapter)
    formatter = ChatFormatter(tokenizer=loaded.tokenizer)
    prompts = prompts_for_examples(examples, formatter=formatter, tokenizer=loaded.tokenizer)
    samples = sample_completions(
        loaded.model,
        loaded.tokenizer,
        prompts,
        k=config.sample_k,
        config=config.sampling_config(),
        batch_size=config.sample_batch_size,
        model_name=f"{config.model}+sft",
    )
    _write_jsonl(config.layout.samples, samples)

    verifier = strict_verifier()
    result = mine_pairs(
        samples,
        examples,
        verifier,
        min_margin=config.min_margin,
        max_pairs_per_prompt=config.max_pairs_per_prompt,
        seed=config.seed,
    )
    pairs = list(result.pairs)
    fallback = False
    if not pairs and config.gold_pair_fallback:
        pairs = gold_fallback_pairs(
            examples,
            samples,
            verifier,
            min_margin=config.min_margin,
            max_pairs_per_prompt=config.max_pairs_per_prompt,
        )
        fallback = True

    _write_jsonl(config.layout.pairs, pairs)
    summary = result.stats()
    stats = summary.as_dict()
    stats["gold_fallback_used"] = fallback
    stats["pairs_written"] = len(pairs)
    _write_json(config.layout.mining_stats, stats)
    return {
        "samples": len(samples),
        "pairs": len(pairs),
        "prompts": result.prompts_seen,
        "saturation_rate": summary.saturation_rate,
        "gold_fallback_used": fallback,
    }


def stage_dpo(ctx: _Context) -> dict[str, Any]:
    """Run DPO on the mined pairs, starting from the supervised adapter.

    The adapter is reloaded as trainable rather than a fresh one attached, so the preference
    stage continues the supervised model instead of restarting from the base -- and so the
    reference policy inside `reference_context` is the supervised model, which is the policy
    the KL term is meant to hold it near.
    """
    config = ctx.config
    pairs = _read_pairs(_require(config.layout.pairs, Stage.MINE))
    if not pairs:
        raise PipelineError(
            "no preference pairs were mined, so there is nothing to optimise; raise --k or "
            "the temperature, or accept that the policy has saturated this data"
        )
    loaded = ctx.provider.load(
        config.model,
        adapter=_require(config.layout.sft_adapter, Stage.SFT),
        trainable=True,
    )
    result = train_dpo(
        loaded.model,
        loaded.tokenizer,
        pairs,
        config.dpo_train_config(),
        beta=config.beta,
        variant=config.variant,
        label_smoothing=config.label_smoothing,
        length_normalise=config.length_normalise,
        model_name=f"{config.model}+sft",
    )
    result.save(config.layout.dpo_dir)
    publish_adapter(result.adapter_path, config.layout.dpo_adapter, Stage.DPO)
    return {
        "steps": result.steps,
        "pairs_seen": result.pairs_seen,
        "final_loss": result.final_loss,
        "final_reward_accuracy": result.final_accuracy,
        "reward_margin_slope": result.margin_slope,
    }


def stage_eval_dpo(ctx: _Context) -> dict[str, Any]:
    """Evaluate the preference-optimised checkpoint on the same split as the other two."""
    config = ctx.config
    return _evaluate_variant(
        ctx,
        variant="dpo",
        adapter=_require(config.layout.dpo_adapter, Stage.DPO),
        label=f"{config.model}+dpo",
    )


def compare_reports(
    baseline: EvalReport, candidate: EvalReport, config: PipelineConfig
) -> ComparisonResult:
    """Apply the promotion gate to two reports with this run's settings."""
    return compare(
        baseline,
        candidate,
        margin=config.margin,
        floors=config.floors(),
        alpha=config.alpha,
        n_resamples=config.n_resamples,
        seed=config.seed,
    )


def stage_compare(ctx: _Context) -> dict[str, Any]:
    """Gate every adjacent pair of variants, and the end-to-end pair as well.

    Both the Markdown and the full result are written. The Markdown is what a human reads and
    what CI attaches to a build; the JSON is what a later run reads back, and rendering the
    decision from the same object means the two can never disagree.
    """
    config = ctx.config
    layout = config.layout
    reports: dict[str, EvalReport] = {}
    for variant in EVAL_VARIANTS:
        path = layout.eval_report(variant)
        if not path.is_file():
            raise PipelineError(
                f"{path} is missing; the comparison needs every variant's evaluation report"
            )
        reports[variant] = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))

    decisions: dict[str, str] = {}
    for name, baseline, candidate in COMPARISONS:
        result = compare_reports(reports[baseline], reports[candidate], config)
        layout.comparison(name, "md").write_text(
            result.to_markdown(), encoding="utf-8", newline="\n"
        )
        layout.comparison(name, "json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8", newline="\n"
        )
        decisions[name] = result.decision.value
    return {"decisions": decisions}


def stage_report(ctx: _Context) -> dict[str, Any]:
    """Render the run directory as one Markdown document."""
    layout = ctx.config.layout
    text = build_report(layout.root)
    layout.report.parent.mkdir(parents=True, exist_ok=True)
    layout.report.write_text(text, encoding="utf-8", newline="\n")
    return {"characters": len(text)}


_STAGES: dict[Stage, Callable[[_Context], dict[str, Any]]] = {
    Stage.DATA: stage_data,
    Stage.EVAL_BASE: stage_eval_base,
    Stage.SFT: stage_sft,
    Stage.EVAL_SFT: stage_eval_sft,
    Stage.MINE: stage_mine,
    Stage.DPO: stage_dpo,
    Stage.EVAL_DPO: stage_eval_dpo,
    Stage.COMPARE: stage_compare,
    Stage.REPORT: stage_report,
}


def run_pipeline(
    config: PipelineConfig,
    *,
    provider: ModelProvider | None = None,
    stages: Sequence[Stage] | None = None,
    force: bool = False,
) -> PipelineResult:
    """Run the requested stages, skipping any whose artefacts are already on disk.

    Args:
        config: The frozen run configuration.
        provider: Where models come from; a `HubProvider` built from `config` by default.
        stages: Which stages to consider. Every stage by default, and always re-ordered into
            dependency order -- asking for DPO before SFT is never what anyone meant.
        force: Re-run stages whose artefacts already exist. The point of the default is that
            a three-hour run interrupted at hour two costs one hour to finish, not three.

    Returns:
        A `PipelineResult` holding one `StageResult` per stage considered.

    Raises:
        PipelineError: If a stage is missing an artefact an earlier stage should have written.
    """
    active = _resolve_provider(config, provider)
    selected = STAGE_ORDER if stages is None else tuple(s for s in STAGE_ORDER if s in set(stages))
    layout = config.layout
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.stage_dir.mkdir(parents=True, exist_ok=True)
    # Rewritten on every invocation, including a resumed one: if a run is resumed with a
    # different configuration the file should describe what actually produced the artefacts
    # that are there now, and a stale copy of the original settings would be worse than none.
    layout.config.write_text(
        config.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    context = _Context(config=config, provider=active)
    digest = config_digest(config)
    results: list[StageResult] = []
    for stage in selected:
        outputs = layout.outputs(stage)
        record_path = layout.stage_record(stage)
        stored = StageRecord.from_json(record_path) if record_path.is_file() else None
        reusable = stored is not None and stored.config_digest == digest
        if not force and _complete(outputs) and reusable:
            logger.info("skipping stage %s: artefacts already present", stage.value)
            # The previous run's numbers are carried forward rather than blanked, so a report
            # built after a resumed run reads the same as one built after a run in one go.
            record = StageRecord(
                stage=stage,
                status="skipped",
                outputs=tuple(str(path) for path in outputs),
                summary=stored.summary if stored is not None else {},
                config_digest=digest,
            )
        else:
            if not force and _complete(outputs) and not reusable:
                logger.info(
                    "re-running stage %s: its artefacts were built under a different configuration",
                    stage.value,
                )
            else:
                logger.info("running stage %s", stage.value)
            record = StageRecord(
                stage=stage,
                status="ran",
                outputs=tuple(str(path) for path in outputs),
                summary=_STAGES[stage](context),
                config_digest=digest,
            )
        record.to_json(record_path)
        results.append(StageResult(stage=stage, record=record, outputs=outputs))
    return PipelineResult(config=config, layout=layout, results=tuple(results))


def _resolve_provider(config: PipelineConfig, provider: ModelProvider | None) -> ModelProvider:
    if provider is not None:
        return provider
    return HubProvider(
        dtype=config.dtype,
        device=config.device,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )


# --------------------------------------------------------------------------------------
# The alignment-tax probe, which needs decoding but no gold records
# --------------------------------------------------------------------------------------


def generate_replies(
    model: Any,
    tokenizer: Any,
    instructions: Sequence[str],
    *,
    formatter: ChatFormatter,
    config: GenerationConfig,
    batch_size: int = 8,
) -> list[str]:
    """Greedily decode one reply per instruction, batched and in the caller's order.

    The alignment-tax probes have no gold record, so they cannot travel through
    `generate_samples`, which builds `Sample`s keyed by example id. The padding, bucketing and
    token-accounting helpers are imported from there rather than rewritten, so the probes are
    decoded exactly as the task is.

    Raises:
        ValueError: If `batch_size` is not positive, or the tokenizer offers no filler id.
    """
    if not instructions:
        return []
    pad_token_id = resolve_pad_token_id(tokenizer)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    encoded = [
        encode_text(tokenizer, formatter.prompt_text(text, tokenizer=tokenizer))
        for text in instructions
    ]
    device = next(model.parameters()).device
    if callable(getattr(model, "eval", None)):
        model.eval()

    replies: list[str | None] = [None] * len(instructions)
    for batch in length_sorted_batches([len(ids) for ids in encoded], batch_size):
        input_ids, attention_mask = left_pad([encoded[index] for index in batch], pad_token_id)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
            )
        width = int(input_ids.shape[1])
        for row, index in zip(generated[:, width:], batch, strict=True):
            token_ids = [int(token) for token in row.tolist()]
            produced = completion_length(
                token_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id
            )
            replies[index] = tokenizer.decode(token_ids[:produced], skip_special_tokens=True)
    return [reply for reply in replies if reply is not None]


def probe_report(
    provider: ModelProvider,
    model: str,
    *,
    adapter: Path | None,
    label: str,
    config: GenerationConfig,
    batch_size: int = 8,
) -> ProbeReport:
    """Run the held-out instruction probes through one model variant."""
    loaded = provider.load(model, adapter=adapter)
    formatter = ChatFormatter(tokenizer=loaded.tokenizer)

    def respond(instructions: Sequence[str]) -> Sequence[str]:
        return generate_replies(
            loaded.model,
            loaded.tokenizer,
            instructions,
            formatter=formatter,
            config=config,
            batch_size=batch_size,
        )

    return run_probes(respond, model=label)


def measure_alignment_tax(
    config: PipelineConfig,
    *,
    provider: ModelProvider | None = None,
    adapter: Path | None,
    baseline_adapter: Path | None = None,
    margin: float = 0.05,
) -> AlignmentTax:
    """Compare general instruction-following before and after alignment.

    Args:
        config: The run configuration, for the model name and the decoding settings.
        provider: Where models come from.
        adapter: The aligned adapter under test.
        baseline_adapter: The adapter to compare against; None means the bare checkpoint,
            which is the comparison the write-up quotes.
        margin: How much instruction-following may be given up before it counts as a
            regression.

    Returns:
        The paired probe comparison.
    """
    active = _resolve_provider(config, provider)
    settings = config.eval_config()
    before = probe_report(
        active,
        config.model,
        adapter=baseline_adapter,
        label=config.model,
        config=settings,
        batch_size=config.eval_batch_size,
    )
    after = probe_report(
        active,
        config.model,
        adapter=adapter,
        label=f"{config.model}+aligned",
        config=settings,
        batch_size=config.eval_batch_size,
    )
    return alignment_tax(
        before, after, margin=margin, n_resamples=config.n_resamples, seed=config.seed
    )


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def _rate(value: float) -> str:
    """Four decimal places, negative zero normalised, so two runs diff only where they differ."""
    return f"{0.0 if value == 0 else value:.4f}"


def _delta(value: float) -> str:
    return f"{0.0 if value == 0 else value:+.4f}"


def _number(value: Any) -> str:
    """Render a summary value without letting float formatting vary by platform."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return _rate(value)
    return str(value)


def _discovered_variants(root: Path) -> list[tuple[str, EvalReport]]:
    """Every `eval_*.json` in a run directory, pipeline variants first.

    Sorted rather than left in filesystem order, and the pipeline's own three placed ahead of
    anything else, so a run that also evaluated a larger prompted model renders it after the
    three the pipeline produced instead of wherever the directory listing put it.
    """
    found: dict[str, EvalReport] = {}
    for path in sorted(root.glob("eval_*.json")):
        variant = path.stem.removeprefix("eval_")
        try:
            found[variant] = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("ignoring %s: not a valid evaluation report", path)
    order = [name for name in EVAL_VARIANTS if name in found]
    order.extend(sorted(name for name in found if name not in EVAL_VARIANTS))
    return [(name, found[name]) for name in order]


def _summary_table(title: str, payload: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    rows = [
        f"| {key.replace('_', ' ')} | {_number(payload[key])} |" for key in keys if key in payload
    ]
    if not rows:
        return []
    return [f"### {title}", "", "| Quantity | Value |", "| --- | ---: |", *rows, ""]


def build_report(run_dir: str | Path) -> str:
    """Render every artefact in a run directory as one Markdown document.

    Written to tolerate a partial directory: each section is emitted only if its artefacts
    are there, so the same function serves a finished run and a run that stopped after
    supervised fine-tuning. The run's own path is not printed -- only its directory name --
    because a report that embeds an absolute path differs between two machines that did
    identical work.

    Args:
        run_dir: A directory written by `run_pipeline`, or by the experiment script.

    Returns:
        The report, ending in exactly one newline. Byte-identical for byte-identical inputs:
        every collection is walked in a fixed order and every float goes through one
        formatter.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")

    lines: list[str] = [f"# sft-dpo-alignment: run `{root.name}`", ""]
    lines.extend(_data_section(root))
    variants = _discovered_variants(root)
    lines.extend(_metrics_section(variants))
    lines.extend(_slice_section(variants))
    lines.extend(_parse_gap_section(variants))
    lines.extend(_training_section(root))
    lines.extend(_mining_section(root))
    lines.extend(_gate_section(root))
    if not variants:
        lines.append("No evaluation reports were found in this run directory.")
    # Exactly one trailing newline whatever the sections contributed, so a caller can print
    # the report without deciding whether to add one.
    return "\n".join(lines).rstrip("\n") + "\n"


def _data_section(root: Path) -> list[str]:
    path = root / "data_stats.json"
    if not path.is_file():
        return []
    stats = _read_json(path)
    lines = [
        "## Corpus",
        "",
        f"Seed {stats['seed']}, {stats['total']} examples, content hash "
        f"`{stats['content_hash'][:16]}`.",
        "",
        "| Split | n | Objectives present | Recommendations present | Flags present |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("train", "val", "test"):
        block = stats["splits"].get(name)
        if block is None:
            continue
        presence = block["field_presence"]
        lines.append(
            f"| {name} | {block['n']} | {_rate(presence['objectives'])} | "
            f"{_rate(presence['recommendations'])} | {_rate(presence['flags'])} |"
        )
    lines.append("")
    return lines


def _metrics_section(variants: Sequence[tuple[str, EvalReport]]) -> list[str]:
    if not variants:
        return []
    lines = [
        "## Evaluation",
        "",
        "| Variant | Model | n | JSON valid | Schema valid | Field F1 | Exact match | "
        "Mean reward |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in variants:
        block = report.overall
        lines.append(
            f"| {name} | {report.model} | {block.n} | {_rate(block.json_valid_rate)} | "
            f"{_rate(block.schema_valid_rate)} | {_rate(block.mean_field_f1)} | "
            f"{_rate(block.exact_match_rate)} | {_rate(block.mean_reward)} |"
        )
    lines.append("")
    return lines


def _slice_section(variants: Sequence[tuple[str, EvalReport]]) -> list[str]:
    if not variants:
        return []
    names = sorted({name for _, report in variants for name in report.per_slice})
    if not names:
        return []
    header = " | ".join(name for name, _ in variants)
    lines = [
        "## Schema validity per slice",
        "",
        f"| Slice | {header} |",
        "| --- |" + " ---: |" * len(variants),
    ]
    for slice_name in names:
        cells = []
        for _, report in variants:
            block = report.per_slice.get(slice_name)
            cells.append("-" if block is None else _rate(block.schema_valid_rate))
        lines.append(f"| {slice_name.value} | {' | '.join(cells)} |")
    lines.append("")
    return lines


def _parse_gap_section(variants: Sequence[tuple[str, EvalReport]]) -> list[str]:
    if not variants:
        return []
    lines = [
        "## Parse gap: strict JSON versus a repaired parse",
        "",
        "| Variant | Strict | Lenient | Gap | Repaired |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, report in variants:
        gap = report.parse_gap
        lines.append(
            f"| {name} | {_rate(gap.strict_rate)} | {_rate(gap.lenient_rate)} | "
            f"{_rate(gap.gap)} | {gap.repaired} |"
        )
    lines.extend(
        [
            "",
            "The gap is the size of the repair step a deployment would need in front of the "
            "model. It cannot be negative, because a lenient parse accepts everything a "
            "strict one does.",
            "",
        ]
    )
    return lines


def _training_section(root: Path) -> list[str]:
    sft = root / "sft" / "summary.json"
    dpo = root / "dpo" / "summary.json"
    if not sft.is_file() and not dpo.is_file():
        return []
    lines = ["## Training", ""]
    if sft.is_file():
        lines.extend(
            _summary_table(
                "Supervised fine-tuning",
                _read_json(sft),
                (
                    "steps",
                    "supervised_tokens",
                    "final_train_loss",
                    "loss_reduction",
                    "best_val_loss",
                    "trainable_parameters",
                    "trainable_pct",
                    "diverged",
                ),
            )
        )
    if dpo.is_file():
        lines.extend(
            _summary_table(
                "Direct preference optimisation",
                _read_json(dpo),
                (
                    "steps",
                    "pairs_seen",
                    "beta",
                    "variant",
                    "final_loss",
                    "initial_reward_accuracy",
                    "final_reward_accuracy",
                    "reward_margin_slope",
                    "diverged",
                ),
            )
        )
    return lines


def _mining_section(root: Path) -> list[str]:
    path = root / "mining_stats.json"
    if not path.is_file():
        return []
    stats = _read_json(path)
    table = _summary_table(
        "Preference mining",
        stats,
        (
            "prompts",
            "prompts_with_pairs",
            "pairs",
            "pairs_written",
            "mean_margin",
            "prompt_yield",
            "saturation_rate",
            "flat_prompts",
            "single_sample_prompts",
            "pairs_dropped_by_cap",
            "gold_fallback_used",
        ),
    )
    if not table:
        return []
    # `_summary_table` opens with its own `### title` heading and a blank line; the mining
    # section wants a top-level heading instead, so both are replaced rather than nested.
    body = ["## Preference mining", "", *table[2:]]
    if stats.get("gold_fallback_used"):
        body.extend(
            [
                "Mining found no rankable pair, so this run fell back to pairing each gold "
                "record against the worst sample. That is a different training signal from "
                "the one the experiment reports and is only ever enabled for the smoke run.",
                "",
            ]
        )
    if stats.get("saturation_rate", 0.0) > 0.0:
        body.extend(
            [
                "A high saturation rate with few pairs is a finished pipeline rather than a "
                "broken one: the policy has outgrown this data and the next move is harder "
                "prompts, not a lower margin.",
                "",
            ]
        )
    return body


def _gate_section(root: Path) -> list[str]:
    rows: list[str] = []
    for path in sorted(root.glob("compare_*.json")):
        try:
            result = ComparisonResult.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("ignoring %s: not a valid comparison result", path)
            continue
        rows.append(
            f"| {path.stem.removeprefix('compare_')} | {result.decision.value.upper()} | "
            f"{result.n_paired} | {_delta(result.success_diff.point)} | "
            f"[{_delta(result.success_diff.low)}, {_delta(result.success_diff.high)}] | "
            f"{result.mcnemar_p:.4f} |"
        )
    if not rows:
        return []
    return [
        "## Promotion gates",
        "",
        "| Comparison | Decision | n | Success difference | CI | McNemar p |",
        "| --- | --- | ---: | ---: | --- | ---: |",
        *rows,
        "",
        "HOLD means the candidate is not worse but has not been shown to be better. It is a "
        "first-class outcome: collapsing it into either neighbour would either ship models on "
        "noise or call an underpowered run a failure.",
        "",
    ]
