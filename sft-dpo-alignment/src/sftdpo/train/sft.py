"""The supervised fine-tuning loop, written out by hand.

`transformers.Trainer` and `trl.SFTTrainer` would both do this in a dozen lines, and both
would hide the four things this project is actually about: where the loss comes from, what
is masked out of it, how gradient accumulation is normalised, and what the learning rate was
on any given step. The DPO loop that follows is a variation on this one, and a hand-written
SFT loop is what makes the comparison between them honest -- the two differ in their
objective and in nothing else.

Three decisions here are worth reading before the code.

*The loss is the model's own.* The batch carries `labels` with every prompt position set to
`IGNORE_INDEX`, and `Qwen2ForCausalLM` computes the shifted cross-entropy over what is left.
Re-implementing that shift in the trainer would add a second place for the off-by-one to
live. The trainer's job is to mask correctly and then get out of the way, which is why
`CompletionOnlyCollator` is the only thing that decides what is supervised.

*Gradient accumulation is exact, not approximate.* Each micro-batch backpropagates its
summed token loss, and the accumulated gradient is divided by the run's token count for the
whole effective batch. The usual recipe -- divide each micro-batch's mean loss by the number
of micro-batches -- weights a short final micro-batch as heavily as a full one, so the
gradient of `k` micro-batches is not the gradient of one batch `k` times larger. Here it is,
to floating-point noise, and a test asserts exactly that.

*An all-masked batch contributes nothing rather than NaN.* Hugging Face averages the
cross-entropy over supervised positions; with none, that is 0/0. One such micro-batch would
turn every weight in the model to NaN on the next optimiser step, and the run would carry on
for hours producing nothing. The guard in `batch_loss` is also the cleanest possible
statement of the masking invariant: if the completion is entirely masked, the loss is
exactly zero and the gradients vanish, so no prompt token can be contributing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from sftdpo.modeling.chat import ChatFormatter
from sftdpo.modeling.collate import (
    IGNORE_INDEX,
    CollationReport,
    CompletionOnlyCollator,
    SFTFeature,
    encode_sft_example,
)
from sftdpo.schemas import Example
from sftdpo.train.common import (
    EvalPoint,
    RunManifest,
    TrainConfig,
    TrainLog,
    cosine_schedule_with_warmup,
    set_determinism,
    token_content_hash,
)

__all__ = [
    "LOG_FILENAME",
    "MANIFEST_FILENAME",
    "SUMMARY_FILENAME",
    "SFTResult",
    "TrainingItem",
    "accumulation_groups",
    "batch_loss",
    "encode_examples",
    "evaluate_loss",
    "resolve_pad_token_id",
    "save_adapter",
    "supervised_token_count",
    "train_sft",
]

logger = logging.getLogger(__name__)

LOG_FILENAME = "train_log.jsonl"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"

BEST_CHECKPOINT_DIR = "best"
FINAL_CHECKPOINT_DIR = "final"

TrainingItem = Example | SFTFeature
"""What the loop trains on: a dataset example, or a feature already tokenised by a caller.

Both are accepted because they answer different needs. `Example` is what the corpus holds
and what a real run passes. `SFTFeature` is what a test passes when it wants to control the
token sequence exactly -- which is the only way to construct the pathological batches the
masking invariants have to survive.
"""


def resolve_pad_token_id(tokenizer: Any) -> int:
    """The id used to pad `input_ids`, falling back to end-of-sequence.

    Reusing EOS as padding is safe here and not merely convenient: padding is masked out of
    attention and is never supervised, so the value only has to be inside the vocabulary.
    Many instruct checkpoints ship without a distinct pad token, and inventing one would
    resize the embedding matrix for no benefit.

    Raises:
        ValueError: If the tokenizer offers neither, since a collator cannot pad without a
            token and guessing zero would silently corrupt a vocabulary that uses it.
    """
    for attribute in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int) and value >= 0:
            return value
    raise ValueError(
        f"{type(tokenizer).__name__} has neither pad_token_id nor eos_token_id; "
        "pass an explicit collator instead"
    )


def encode_examples(
    items: Sequence[TrainingItem],
    *,
    formatter: ChatFormatter,
    tokenizer: Any,
) -> list[SFTFeature]:
    """Tokenise dataset examples, passing already-tokenised features straight through.

    The completion is the compact JSON of the gold record, exactly the string the verifier
    will later be asked to parse. Training on a prettified variant would teach the model
    whitespace the evaluation never rewards.

    Raises:
        TypeError: On an item that is neither an `Example` nor an `SFTFeature`.
    """
    features: list[SFTFeature] = []
    for item in items:
        if isinstance(item, SFTFeature):
            features.append(item)
        elif isinstance(item, Example):
            features.append(encode_sft_example(formatter, tokenizer, item.note, item.gold_json))
        else:
            raise TypeError(f"expected an Example or an SFTFeature, got {type(item).__name__}")
    return features


def supervised_token_count(labels: Tensor) -> int:
    """Positions in `labels` that contribute to the loss."""
    return int((labels != IGNORE_INDEX).sum().item())


def batch_loss(model: Any, batch: dict[str, Tensor]) -> tuple[Tensor, int]:
    """Mean loss per supervised token for one batch, plus how many tokens that was.

    The loss comes from the model's own `labels` handling -- one shift, one cross-entropy,
    computed where the vocabulary projection already lives -- so the trainer never has a
    second opinion about which logit predicts which token.

    Returns:
        A pair of (loss tensor still attached to the graph, supervised token count). The
        count is what makes exact gradient accumulation possible: a mean is only combinable
        with another mean if you know what each was a mean over.
    """
    tokens = supervised_token_count(batch["labels"])
    outputs = model(**batch)
    if tokens == 0:
        # Nothing to learn from. Returning `outputs.loss` here would return NaN and destroy
        # the model on the next step; returning a detached zero would break the backward
        # pass. `logits.sum() * 0` is zero with a gradient path that contributes nothing.
        return outputs.logits.sum() * 0.0, 0
    loss: Tensor = outputs.loss
    return loss, tokens


def accumulation_groups(
    order: Sequence[int],
    *,
    batch_size: int,
    accumulation: int,
) -> list[list[list[int]]]:
    """Partition an epoch's example order into optimiser steps of micro-batches.

    Returned eagerly as a nested list rather than yielded, because the loop needs the size
    of a group before it starts backpropagating through it: the final group of an epoch is
    usually short, and its normalisation depends on how short.

    Args:
        order: Example indices in the order the epoch will visit them.
        batch_size: Examples per forward pass.
        accumulation: Micro-batches per optimiser step.

    Returns:
        A list of optimiser steps, each a list of micro-batches, each a list of indices.

    Raises:
        ValueError: On a non-positive batch size or accumulation count.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if accumulation < 1:
        raise ValueError(f"accumulation must be at least 1, got {accumulation}")
    micro = [list(order[i : i + batch_size]) for i in range(0, len(order), batch_size)]
    return [micro[i : i + accumulation] for i in range(0, len(micro), accumulation)]


def _autocast(device_type: str, *, enabled: bool) -> AbstractContextManager[Any]:
    """bf16 autocast when asked for, a no-op context otherwise."""
    if not enabled:
        return contextlib.nullcontext()
    context: AbstractContextManager[Any] = torch.autocast(
        device_type=device_type, dtype=torch.bfloat16
    )
    return context


def _to_device(batch: dict[str, Tensor], device: Any) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_loss(
    model: Any,
    features: Sequence[SFTFeature],
    collator: CompletionOnlyCollator,
    *,
    batch_size: int,
    bf16: bool = False,
) -> float:
    """Validation loss, averaged over tokens rather than over batches.

    Weighting each batch by its supervised token count is what makes the number comparable:
    a mean of per-batch means depends on how the examples happened to be grouped, so the
    same model on the same data would score differently at batch size 4 and batch size 8.
    A test pins that invariance down.

    The model is put in eval mode for the duration and restored afterwards, because leaving
    a model in eval mode is a classic way to lose an afternoon to a training run whose
    dropout silently stopped.

    Raises:
        ValueError: If there is nothing to evaluate.
    """
    if not features:
        raise ValueError("evaluate_loss needs at least one validation example")

    was_training = bool(model.training)
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = _to_device(collator(features[start : start + batch_size]), device)
                with _autocast(device.type, enabled=bf16):
                    loss, tokens = batch_loss(model, batch)
                total_loss += float(loss.detach().float()) * tokens
                total_tokens += tokens
    finally:
        model.train(was_training)
    return total_loss / total_tokens if total_tokens else 0.0


def save_adapter(model: Any, directory: str | Path) -> Path:
    """Write the trainable weights to `directory`.

    For a PEFT model this is the adapter alone -- a few hundred kilobytes at rank 8 -- which
    is why a checkpoint can be written on every validation improvement without the run
    becoming disk-bound.

    Raises:
        TypeError: If the object cannot save itself.
    """
    saver = getattr(model, "save_pretrained", None)
    if not callable(saver):
        raise TypeError(f"{type(model).__name__} has no save_pretrained(); nothing to checkpoint")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    saver(str(path))
    return path


@dataclass(frozen=True, slots=True)
class SFTResult:
    """Everything a run produced, in one object the report can read.

    Deliberately not the model: the trained weights are on disk under `adapter_path`, and
    returning a live model alongside a summary invites a caller to evaluate one while
    reporting the other.
    """

    log: TrainLog
    val_curve: tuple[EvalPoint, ...]
    best_val_loss: float | None
    best_step: int | None
    steps: int
    supervised_tokens: int
    manifest: RunManifest
    adapter_path: Path | None
    collation: CollationReport

    @property
    def final_train_loss(self) -> float:
        """The last logged training loss; the final step is always logged."""
        last = self.log.last
        return last.loss if last is not None else float("nan")

    @property
    def loss_reduction(self) -> float:
        """First logged loss divided by the last -- the factor the write-up quotes.

        Infinite when the final loss is zero, which on a memorisation set is a real outcome
        rather than an error.
        """
        if not len(self.log):
            return float("nan")
        first = self.log.records[0].loss
        last = self.log.records[-1].loss
        if last <= 0.0:
            return float("inf")
        return first / last

    def summary(self) -> dict[str, Any]:
        """Flat, JSON-serialisable view for the report and for `save`."""
        return {
            "steps": self.steps,
            "supervised_tokens": self.supervised_tokens,
            "final_train_loss": self.final_train_loss,
            "loss_reduction": self.loss_reduction,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "val_curve": [point.as_dict() for point in self.val_curve],
            "diverged": self.log.diverged,
            "trainable_parameters": self.manifest.trainable_parameters,
            "trainable_pct": self.manifest.trainable_pct,
            "data_content_hash": self.manifest.data_content_hash,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "collation": self.collation.as_dict(),
        }

    def save(self, directory: str | Path) -> Path:
        """Write the log, the manifest and the summary side by side.

        Three files rather than one, because they have three lifetimes: the log grows during
        the run, the manifest is fixed before it starts, and the summary only exists once it
        has finished.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.log.to_jsonl(path / LOG_FILENAME)
        self.manifest.to_json(path / MANIFEST_FILENAME)
        (path / SUMMARY_FILENAME).write_text(
            json.dumps(self.summary(), indent=2), encoding="utf-8", newline="\n"
        )
        return path


def _trainable_parameters(model: Any) -> list[Any]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError(
            "no parameter has requires_grad=True; with an adapter attached this means the "
            "target modules matched nothing, and the run would report a healthy loss curve "
            "while learning exactly nothing"
        )
    return parameters


def _model_name(model: Any, given: str | None) -> str:
    if given is not None:
        return given
    config = getattr(model, "config", None)
    return str(getattr(config, "name_or_path", "") or type(model).__name__)


def _feature_hash(features: Sequence[SFTFeature]) -> str:
    """Hash the tokens *and* the supervision boundary of every training feature."""
    return token_content_hash([(feature.prompt_len, *feature.input_ids) for feature in features])


def train_sft(
    model: Any,
    tokenizer: Any,
    train_examples: Sequence[TrainingItem],
    val_examples: Sequence[TrainingItem],
    config: TrainConfig,
    *,
    formatter: ChatFormatter | None = None,
    collator: CompletionOnlyCollator | None = None,
    model_name: str | None = None,
    data_content_hash: str | None = None,
) -> SFTResult:
    """Fine-tune `model` on completions only, and report what happened.

    The loop, in order: seed everything; tokenise; build the optimiser over whatever is
    trainable; then, for each optimiser step, run the micro-batches of one accumulation
    group, normalise the accumulated gradient by the group's token count, clip, set the
    learning rate from the cosine schedule, and step. Validation runs on the configured
    interval and always on the final step, and each improvement writes a checkpoint.

    Args:
        model: A causal LM, usually already wrapped in a LoRA adapter. Only parameters with
            `requires_grad` are optimised, so a frozen base costs no optimiser state.
        tokenizer: Used to encode `Example`s and to choose a pad token. Ignored when every
            item is already an `SFTFeature` and an explicit collator is supplied.
        train_examples: Training items.
        val_examples: Validation items; an empty sequence disables validation, and with it
            best-checkpoint tracking.
        config: The frozen run configuration.
        formatter: Chat formatting; defaults to `ChatFormatter()` bound to `tokenizer`.
        collator: Batch construction; defaults to a `CompletionOnlyCollator` at
            `config.max_seq_length`. Injectable so a caller can round batch widths for
            tensor cores -- and so a test can supply a pathological one.
        model_name: Name recorded in the manifest; defaults to the checkpoint the model was
            loaded from.
        data_content_hash: Overrides the hash computed from the tokenised features, for a
            caller that would rather cite the corpus hash from `Dataset.content_hash`.

    Returns:
        An `SFTResult` holding the log, the validation curve, the best checkpoint and the
        manifest.

    Raises:
        ValueError: If there is no training data, or nothing in the model is trainable.
    """
    set_determinism(config.seed)

    active_formatter = formatter if formatter is not None else ChatFormatter(tokenizer=tokenizer)
    features = encode_examples(train_examples, formatter=active_formatter, tokenizer=tokenizer)
    if not features:
        raise ValueError("train_sft needs at least one training example")
    val_features = encode_examples(val_examples, formatter=active_formatter, tokenizer=tokenizer)

    active_collator = collator or CompletionOnlyCollator(
        pad_token_id=resolve_pad_token_id(tokenizer),
        max_length=config.max_seq_length,
    )

    parameters = _trainable_parameters(model)
    optimiser = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    total = config.total_steps(len(features))
    warmup = config.warmup_steps(total)
    device = next(model.parameters()).device
    shuffle = random.Random(config.seed)

    log = TrainLog()
    val_curve: list[EvalPoint] = []
    best_val_loss: float | None = None
    best_step: int | None = None
    adapter_path: Path | None = None
    tokens_seen = 0
    step = 0

    model.train()
    while step < total:
        order = list(range(len(features)))
        shuffle.shuffle(order)
        groups = accumulation_groups(
            order,
            batch_size=config.batch_size,
            accumulation=config.gradient_accumulation_steps,
        )
        for group in groups:
            if step >= total:
                break
            optimiser.zero_grad(set_to_none=True)
            group_loss = 0.0
            group_tokens = 0
            for micro in group:
                batch = _to_device(active_collator([features[i] for i in micro]), device)
                with _autocast(device.type, enabled=config.bf16):
                    loss, tokens = batch_loss(model, batch)
                # Backpropagate the summed token loss, not the mean: the division by the
                # group's total token count happens once, below, which is what makes the
                # accumulated gradient identical to that of one batch this size.
                (loss * tokens).backward()
                group_loss += float(loss.detach().float()) * tokens
                group_tokens += tokens

            if group_tokens:
                scale = 1.0 / group_tokens
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)

            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm))
            learning_rate = config.learning_rate * cosine_schedule_with_warmup(step, total, warmup)
            for param_group in optimiser.param_groups:
                param_group["lr"] = learning_rate
            optimiser.step()

            step += 1
            tokens_seen += group_tokens
            step_loss = group_loss / group_tokens if group_tokens else 0.0

            if step % config.log_every_steps == 0 or step == total:
                log.append(
                    step=step,
                    loss=step_loss,
                    lr=learning_rate,
                    grad_norm=grad_norm,
                    tokens=tokens_seen,
                )

            due = bool(config.eval_every_steps) and (
                step % config.eval_every_steps == 0 or step == total
            )
            if val_features and due:
                val_loss = evaluate_loss(
                    model,
                    val_features,
                    active_collator,
                    batch_size=config.batch_size,
                    bf16=config.bf16,
                )
                val_curve.append(EvalPoint(step=step, loss=val_loss))
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    if config.save_best_adapter:
                        adapter_path = save_adapter(model, config.output_dir / BEST_CHECKPOINT_DIR)

    if config.save_best_adapter and adapter_path is None:
        # No validation data, or no improvement ever recorded: the weights at the end of the
        # run are still the result, and a run that trains without saving anything is a run
        # that has to be repeated.
        adapter_path = save_adapter(model, config.output_dir / FINAL_CHECKPOINT_DIR)

    model.eval()
    logger.info(
        "sft finished: %d steps, %d supervised tokens, final loss %.4f",
        step,
        tokens_seen,
        log.records[-1].loss if log.records else float("nan"),
    )

    manifest = RunManifest.build(
        stage="sft",
        model=model,
        model_name=_model_name(model, model_name),
        data_content_hash=data_content_hash or _feature_hash(features),
        config=config,
    )
    return SFTResult(
        log=log,
        val_curve=tuple(val_curve),
        best_val_loss=best_val_loss,
        best_step=best_step,
        steps=step,
        supervised_tokens=tokens_seen,
        manifest=manifest,
        adapter_path=adapter_path,
        collation=active_collator.report,
    )
