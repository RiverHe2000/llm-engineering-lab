"""The DPO loop: one model, one adapter, and a reference policy that costs no memory.

This is the SFT loop with its objective replaced. The optimiser, the cosine schedule, the
exact-accumulation rule and the log are all shared with `train/sft.py`, so a comparison
between the two stages measures the objective and nothing else. What is different is where
the loss comes from: instead of the model's own cross-entropy over masked labels, every step
computes four sequence log-probabilities and hands them to :func:`sftdpo.train.dpo_loss.dpo_loss`.

*The reference policy is the policy.* A textbook DPO implementation holds two models in
memory -- the one being trained and a frozen copy that supplies ``pi_ref`` -- which for a
0.5 B model in bf16 is about a gigabyte of VRAM spent on weights that never change. With
LoRA it is unnecessary. The adapter is the only thing separating the policy from its
reference, so switching it off inside
:func:`sftdpo.modeling.lora.reference_context` reproduces the base model's
log-probabilities exactly, from the same weights. That was verified against a separately
loaded base checkpoint before this loop was written. The reference forward therefore costs
activations and time but not a second copy of the model, which on a 12 GB card is the
difference between the run fitting and not. :func:`reference_logprobs` is deliberately the
same code path as :func:`policy_logprobs` with two wrappers around it -- the context manager
and ``torch.no_grad()`` -- because a reference computed by a *different* forward-pass
structure is the classic way to introduce a constant offset between ``pi`` and ``pi_ref``
that no test would catch and that would silently rescale every implicit reward.

*Both halves travel through one forward pass.* `PreferenceCollator` pads the chosen and
rejected sides to a common width precisely so they can be concatenated along the batch
dimension. One forward of ``2 * batch`` sequences costs the same arithmetic as two forwards
of ``batch`` and half the kernel launches, and it is the shape that makes the reference pass
a single extra call rather than two.

*The step-0 identity, and exactly what it does and does not prove.* A freshly attached LoRA
adapter initialises ``B`` to zero, so the policy and the reference are the same function,
``h`` is identically zero, and the sigmoid loss is exactly ``ln 2``.

It is worth being precise about the reach of that, because an earlier version of this
docstring overstated it. The policy and the reference log-probabilities are computed by the
*same function* on the *same batch*; only the adapter differs. So any error inside that
function -- an off-by-one in the log-probability shift, a mask that supervises the prompt, a
mis-built concatenation -- is applied identically to both sides and cancels in ``h``. The
identity still holds. Replacing `sequence_logprob` with an unshifted gather, the exact
off-by-one its own docstring warns about, leaves the step-0 loss bit-for-bit ``ln 2``.

What it does test is the part no other check covers: that ``reference_context`` really
yields a different function from the policy pass, that the adapter starts as an identity,
and that the loss and the reward bookkeeping agree at the fixed point. The shift, the masking
and the concatenation are pinned instead by their own unit tests --- `test_dpo_loss.py`
perturbs `labels[:, 0]` and `logits[:, -1]` and requires the result to be unchanged, and
`test_collate.py` asserts the supervised-token count equals the completion length --- and by
`train/crosscheck.py`, which compares against a second implementation.
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
    CollationReport,
    PairFeature,
    PreferenceCollator,
    encode_preference_pair,
)
from sftdpo.modeling.lora import reference_context
from sftdpo.schemas import PreferencePair
from sftdpo.train.common import (
    RunManifest,
    TrainConfig,
    TrainLog,
    cosine_schedule_with_warmup,
    set_determinism,
    token_content_hash,
)
from sftdpo.train.dpo_loss import DPOOutput, DPOVariant, dpo_loss, sequence_logprob
from sftdpo.train.sft import (
    LOG_FILENAME,
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    accumulation_groups,
    resolve_pad_token_id,
    save_adapter,
    supervised_token_count,
)

__all__ = [
    "PAIR_KEYS",
    "REWARD_LOG_FILENAME",
    "DPOEvaluation",
    "DPOResult",
    "PreferenceItem",
    "RewardPoint",
    "concatenated_batch",
    "dpo_batch_output",
    "encode_pairs",
    "evaluate_dpo",
    "least_squares_slope",
    "pair_token_count",
    "policy_logprobs",
    "reference_logprobs",
    "train_dpo",
]

logger = logging.getLogger(__name__)

REWARD_LOG_FILENAME = "reward_log.jsonl"
"""Where the implicit-reward diagnostics are written, beside the shared training log."""

FINAL_CHECKPOINT_DIR = "final"

PAIR_KEYS: tuple[str, ...] = (
    "chosen_input_ids",
    "chosen_attention_mask",
    "chosen_labels",
    "rejected_input_ids",
    "rejected_attention_mask",
    "rejected_labels",
)
"""The six tensors `PreferenceCollator` produces, in the order they are concatenated."""

PreferenceItem = PreferencePair | PairFeature
"""What the loop trains on: a mined pair, or one a caller has already tokenised.

`PreferencePair` is what the mining stage produces and what a real run passes.
`PairFeature` is what a test passes when the token sequences themselves are the subject --
a separable synthetic pair set, or a pair whose two sides differ in length by exactly one
token -- and there is no way to construct those through a tokenizer.
"""


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------


def encode_pairs(
    items: Sequence[PreferenceItem],
    *,
    formatter: ChatFormatter,
    tokenizer: Any,
) -> list[PairFeature]:
    """Tokenise preference pairs, passing already-tokenised features straight through.

    Both sides go through the same formatter as SFT does, so a pair the model is trained to
    prefer is rendered exactly as the prompt it will later be sampled from.

    Raises:
        TypeError: On an item that is neither a `PreferencePair` nor a `PairFeature`.
    """
    features: list[PairFeature] = []
    for item in items:
        if isinstance(item, PairFeature):
            features.append(item)
        elif isinstance(item, PreferencePair):
            features.append(encode_preference_pair(formatter, tokenizer, item))
        else:
            raise TypeError(
                f"expected a PreferencePair or a PairFeature, got {type(item).__name__}"
            )
    return features


def pair_token_count(batch: dict[str, Tensor]) -> int:
    """Scored tokens in a collated pair batch, both sides together.

    Chosen and rejected are counted together because a DPO step consumes both: reporting
    only the chosen half would halve every tokens-per-second figure the write-up quotes.
    """
    return supervised_token_count(batch["chosen_labels"]) + supervised_token_count(
        batch["rejected_labels"]
    )


# --------------------------------------------------------------------------------------
# The forward pass
# --------------------------------------------------------------------------------------


def concatenated_batch(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Stack the chosen and rejected halves into the tensors of one forward pass.

    Args:
        batch: The six tensors `PreferenceCollator` returns.

    Returns:
        A triple of `(input_ids, attention_mask, labels)`, each of shape
        `(2 * pairs, width)`, with the chosen rows first. That order is a contract: every
        `chunk(2)` downstream reads the chosen half from the front, and the reference pass
        has to agree with the policy pass about which half is which.

    Raises:
        KeyError: If a key is missing, naming all of them, because a partially built batch
            is nearly always a collator that was called with SFT features by mistake.
        ValueError: If the two halves disagree in shape. The collator pads them to a common
            width for exactly this reason, and a mismatch here would otherwise surface as an
            unrelated broadcasting error inside the model.
    """
    missing = [key for key in PAIR_KEYS if key not in batch]
    if missing:
        raise KeyError(f"batch is missing {missing}; expected the keys PreferenceCollator emits")
    for suffix in ("input_ids", "attention_mask", "labels"):
        chosen, rejected = batch[f"chosen_{suffix}"], batch[f"rejected_{suffix}"]
        if chosen.shape != rejected.shape:
            raise ValueError(
                f"chosen_{suffix} has shape {tuple(chosen.shape)} but rejected_{suffix} has "
                f"{tuple(rejected.shape)}; the two halves must be padded to one width"
            )
    return (
        torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0),
        torch.cat([batch["chosen_attention_mask"], batch["rejected_attention_mask"]], dim=0),
        torch.cat([batch["chosen_labels"], batch["rejected_labels"]], dim=0),
    )


def _logprobs(
    model: Any,
    batch: dict[str, Tensor],
    *,
    length_normalise: bool,
) -> tuple[Tensor, Tensor]:
    """One concatenated forward pass, split back into (chosen, rejected) log-probabilities."""
    input_ids, attention_mask, labels = concatenated_batch(batch)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logps = sequence_logprob(logits, labels, average=length_normalise)
    chosen, rejected = logps.chunk(2, dim=0)
    return chosen, rejected


def policy_logprobs(
    model: Any,
    batch: dict[str, Tensor],
    *,
    length_normalise: bool = False,
) -> tuple[Tensor, Tensor]:
    """`log pi(y_w | x)` and `log pi(y_l | x)` under the current adapter, with gradients.

    Args:
        model: The PEFT-wrapped policy.
        batch: A collated pair batch.
        length_normalise: Return per-token averages instead of sequence sums. This changes
            the objective rather than rescaling it -- see `dpo_loss.sequence_logprob` -- and
            is the setting under which this package's IPO agrees with TRL's.

    Returns:
        A pair of `(pairs,)` tensors still attached to the graph.
    """
    return _logprobs(model, batch, length_normalise=length_normalise)


def reference_logprobs(
    model: Any,
    batch: dict[str, Tensor],
    *,
    length_normalise: bool = False,
) -> tuple[Tensor, Tensor]:
    """The same two log-probabilities under the frozen base model.

    No second model is loaded. `reference_context` disables the LoRA adapter for the
    duration of the block, which yields the base model's behaviour from the policy's own
    weights and reproduces a separately loaded base checkpoint's log-probabilities exactly.
    A DPO run therefore holds one set of weights in memory rather than two, and the
    reference costs one extra forward pass instead of a gigabyte of VRAM.

    `torch.no_grad()` is inside the context rather than around the call site so that the two
    cannot be separated: the reference is a constant of the objective, and a graph retained
    through it would both waste memory and let the optimiser move the thing it is supposed
    to be regularised towards.

    Raises:
        TypeError: Via `reference_context`, if the model has no adapter to disable. That
            failure is deliberate: a "reference" equal to the policy makes every implicit
            reward exactly zero and pins the loss at `ln 2` while looking healthy.
    """
    with reference_context(model) as reference, torch.no_grad():
        return _logprobs(reference, batch, length_normalise=length_normalise)


def dpo_batch_output(
    model: Any,
    batch: dict[str, Tensor],
    *,
    beta: float = 0.1,
    variant: DPOVariant = "sigmoid",
    label_smoothing: float = 0.0,
    length_normalise: bool = False,
) -> DPOOutput:
    """The loss and the implicit rewards for one collated pair batch.

    Two forward passes, in this order: the policy with gradients, then the reference without.
    The policy pass comes first so that a run that dies on the reference pass has already
    shown whether the policy pass fits in memory, which is the failure that actually happens
    on a card this size.
    """
    policy_chosen, policy_rejected = policy_logprobs(
        model, batch, length_normalise=length_normalise
    )
    ref_chosen, ref_rejected = reference_logprobs(model, batch, length_normalise=length_normalise)
    return dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=beta,
        variant=variant,
        label_smoothing=label_smoothing,
    )


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewardPoint:
    """The implicit-reward diagnostics of one logged step.

    Kept beside `TrainLog` rather than inside it, for the reason `EvalPoint` is kept
    separate: an SFT record has no reward and a log whose columns are sometimes meaningless
    is a log people stop reading. These five numbers are what a DPO run is actually judged
    by -- the loss alone cannot distinguish a policy that has learned to lift the chosen
    response from one that has only learned to crush the rejected one.

    Attributes:
        step: Optimiser step this was measured at.
        chosen: Mean `beta * (log pi(y_w) - log pi_ref(y_w))` over the step's pairs.
        rejected: The same for the rejected response.
        margin: `chosen - rejected`, the quantity the objective is a function of.
        accuracy: Fraction of pairs the implicit reward ranks correctly; ties count as
            misses, so a policy identical to its reference scores 0, not 0.5.
    """

    step: int
    chosen: float
    rejected: float
    margin: float
    accuracy: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"step must not be negative, got {self.step}")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"accuracy must lie in [0, 1], got {self.accuracy}")

    def as_dict(self) -> dict[str, float | int]:
        """JSON-serialisable view, one JSONL line."""
        return {
            "step": self.step,
            "reward_chosen": self.chosen,
            "reward_rejected": self.rejected,
            "reward_margin": self.margin,
            "reward_accuracy": self.accuracy,
        }


def least_squares_slope(points: Sequence[tuple[int, float]]) -> float:
    """Slope of the ordinary least-squares line through `(step, value)`.

    The reward margin of a healthy DPO run rises, but it does not rise at every step: a
    batch of hard pairs dents it, and asserting step-by-step monotonicity would produce a
    test that fails on a run that is working. The slope is the honest version of the same
    claim -- "increasing in expectation" -- and it is one number the write-up can quote.

    Returns:
        The slope, or 0.0 when fewer than two points are given or every point shares a step,
        because a single measurement has no trend and a vertical line has no slope.
    """
    if len(points) < 2:
        return 0.0
    n = float(len(points))
    mean_x = sum(float(step) for step, _ in points) / n
    mean_y = sum(value for _, value in points) / n
    covariance = sum((float(step) - mean_x) * (value - mean_y) for step, value in points)
    variance = sum((float(step) - mean_x) ** 2 for step, _ in points)
    return covariance / variance if variance > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class DPOEvaluation:
    """A held-out measurement of the objective and of how well it separates pairs.

    Accuracy is reported next to the loss because they can move apart: label smoothing and
    IPO both have finite optimal margins, so a run can push the loss up while still ranking
    every pair correctly, and the reverse is what over-optimisation looks like.
    """

    loss: float
    accuracy: float
    margin: float
    pairs: int

    def as_dict(self) -> dict[str, float | int]:
        """JSON-serialisable view."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "margin": self.margin,
            "pairs": self.pairs,
        }


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


def evaluate_dpo(
    model: Any,
    features: Sequence[PairFeature],
    collator: PreferenceCollator,
    *,
    batch_size: int,
    beta: float = 0.1,
    variant: DPOVariant = "sigmoid",
    label_smoothing: float = 0.0,
    length_normalise: bool = False,
    bf16: bool = False,
) -> DPOEvaluation:
    """The objective and the reward diagnostics over a set of pairs, without training.

    Weighted by pairs rather than by batch, so the number does not depend on how the pairs
    happened to be grouped; a test pins that invariance down. The model is put in eval mode
    and restored afterwards, because a validation helper that leaves a model in eval mode
    silently disables dropout for the rest of the run.

    Raises:
        ValueError: If there is nothing to evaluate.
    """
    if not features:
        raise ValueError("evaluate_dpo needs at least one preference pair")

    was_training = bool(model.training)
    device = next(model.parameters()).device
    model.eval()
    loss_sum = 0.0
    margin_sum = 0.0
    correct = 0.0
    seen = 0
    try:
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = _to_device(collator(features[start : start + batch_size]), device)
                with _autocast(device.type, enabled=bf16):
                    output = dpo_batch_output(
                        model,
                        batch,
                        beta=beta,
                        variant=variant,
                        label_smoothing=label_smoothing,
                        length_normalise=length_normalise,
                    )
                pairs = int(output.losses.shape[0])
                loss_sum += float(output.losses.detach().float().sum())
                margin_sum += float(output.reward_margins.float().sum())
                correct += float((output.reward_margins > 0).sum())
                seen += pairs
    finally:
        model.train(was_training)

    return DPOEvaluation(
        loss=loss_sum / seen,
        accuracy=correct / seen,
        margin=margin_sum / seen,
        pairs=seen,
    )


# --------------------------------------------------------------------------------------
# The result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DPOResult:
    """Everything a preference run produced, in one object the report can read.

    The reward curve is a first-class member rather than an afterthought: the headline claim
    of a DPO stage is that reward accuracy rose and the margin opened, and both are read off
    `rewards` rather than off the loss.
    """

    log: TrainLog
    rewards: tuple[RewardPoint, ...]
    steps: int
    pairs_seen: int
    supervised_tokens: int
    beta: float
    variant: DPOVariant
    manifest: RunManifest
    adapter_path: Path | None
    collation: CollationReport

    @property
    def final_loss(self) -> float:
        """The last logged training loss; the final step is always logged."""
        last = self.log.last
        return last.loss if last is not None else float("nan")

    @property
    def initial_accuracy(self) -> float:
        """Reward accuracy at the first logged step, NaN if nothing was logged."""
        return self.rewards[0].accuracy if self.rewards else float("nan")

    @property
    def final_accuracy(self) -> float:
        """Reward accuracy at the last logged step, NaN if nothing was logged."""
        return self.rewards[-1].accuracy if self.rewards else float("nan")

    @property
    def accuracy_gain(self) -> float:
        """How much of the preference ordering the run learned to reproduce."""
        return self.final_accuracy - self.initial_accuracy

    @property
    def margin_slope(self) -> float:
        """Least-squares trend of the reward margin, in reward units per optimiser step.

        Positive means the policy is separating chosen from rejected. Negative on a run
        whose loss is falling is the signature of a bug in the reference pass, not of a
        hard dataset.
        """
        return least_squares_slope([(point.step, point.margin) for point in self.rewards])

    def summary(self) -> dict[str, Any]:
        """Flat, JSON-serialisable view for the report and for `save`."""
        return {
            "stage": "dpo",
            "beta": self.beta,
            "variant": self.variant,
            "steps": self.steps,
            "pairs_seen": self.pairs_seen,
            "supervised_tokens": self.supervised_tokens,
            "final_loss": self.final_loss,
            "initial_reward_accuracy": self.initial_accuracy,
            "final_reward_accuracy": self.final_accuracy,
            "reward_accuracy_gain": self.accuracy_gain,
            "reward_margin_slope": self.margin_slope,
            "diverged": self.log.diverged,
            "trainable_parameters": self.manifest.trainable_parameters,
            "trainable_pct": self.manifest.trainable_pct,
            "data_content_hash": self.manifest.data_content_hash,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "collation": self.collation.as_dict(),
        }

    def save(self, directory: str | Path) -> Path:
        """Write the training log, the reward log, the manifest and the summary.

        Four files rather than one, for the reason the SFT stage writes three: they have
        different lifetimes, and the two JSONL logs are readable up to the last flush if the
        run dies mid-way.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.log.to_jsonl(path / LOG_FILENAME)
        body = "".join(json.dumps(point.as_dict()) + "\n" for point in self.rewards)
        (path / REWARD_LOG_FILENAME).write_text(body, encoding="utf-8", newline="\n")
        self.manifest.to_json(path / MANIFEST_FILENAME)
        (path / SUMMARY_FILENAME).write_text(
            json.dumps(self.summary(), indent=2), encoding="utf-8", newline="\n"
        )
        return path


def _trainable_parameters(model: Any) -> list[Any]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError(
            "no parameter has requires_grad=True; a DPO run on a frozen model would report "
            "a flat loss of ln 2 forever, because the policy would never leave the reference"
        )
    return parameters


def _model_name(model: Any, given: str | None) -> str:
    if given is not None:
        return given
    config = getattr(model, "config", None)
    return str(getattr(config, "name_or_path", "") or type(model).__name__)


def _feature_hash(features: Sequence[PairFeature]) -> str:
    """Hash both sides of every pair, prompt boundary included.

    Both sides matter: two runs that trained on the same chosen completions but different
    rejected ones did not train on the same data, and a hash over the chosen half alone
    would call them identical.
    """
    sequences: list[tuple[int, ...]] = []
    for feature in features:
        sequences.append((feature.chosen.prompt_len, *feature.chosen.input_ids))
        sequences.append((feature.rejected.prompt_len, *feature.rejected.input_ids))
    return token_content_hash(sequences)


def train_dpo(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[PreferenceItem],
    config: TrainConfig,
    *,
    beta: float = 0.1,
    variant: DPOVariant = "sigmoid",
    label_smoothing: float = 0.0,
    length_normalise: bool = False,
    formatter: ChatFormatter | None = None,
    collator: PreferenceCollator | None = None,
    model_name: str | None = None,
    data_content_hash: str | None = None,
) -> DPOResult:
    """Train `model` on preference pairs and report what the implicit rewards did.

    The loop, in order: seed everything; tokenise the pairs; build the optimiser over
    whatever is trainable; then, for each optimiser step, run the micro-batches of one
    accumulation group, normalise the accumulated gradient by the group's pair count, clip,
    set the learning rate from the shared cosine schedule, and step. Each micro-batch costs
    two forward passes -- the policy with gradients and the reference without -- and no
    second copy of the model, because the reference is this model with its adapter switched
    off inside `reference_context`.

    Accumulation is exact in the same sense as the SFT loop: each micro-batch backpropagates
    its summed pair loss and the accumulated gradient is divided once by the group's total
    pair count, so `k` micro-batches produce the gradient of one batch `k` times larger
    rather than an average of averages that over-weights a short final micro-batch.

    Args:
        model: A causal LM wrapped in a LoRA adapter. An unadapted model raises, because
            without an adapter there is nothing to disable and no reference policy.
        tokenizer: Used to encode `PreferencePair`s and to choose a pad token. Ignored when
            every item is already a `PairFeature` and an explicit collator is supplied.
        pairs: The mined preference pairs, or pre-tokenised features.
        config: The frozen run configuration, shared with the SFT stage.
        beta: KL strength of the objective. It scales the implicit reward; it is not a
            learning rate.
        variant: `"sigmoid"`, `"ipo"` or `"cdpo"`.
        label_smoothing: Assumed label-flip probability; defined only for `"cdpo"`.
        length_normalise: Score completions per token instead of per sequence. Removes the
            length bias of summed log-probabilities at the cost of changing the objective.
        formatter: Chat formatting; defaults to `ChatFormatter()` bound to `tokenizer`.
        collator: Batch construction; defaults to a `PreferenceCollator` at
            `config.max_seq_length`.
        model_name: Name recorded in the manifest.
        data_content_hash: Overrides the hash computed from the tokenised pairs.

    Returns:
        A `DPOResult` holding both logs, the manifest and the final adapter path.

    Raises:
        ValueError: If there are no pairs, or nothing in the model is trainable.
    """
    set_determinism(config.seed)

    active_formatter = formatter if formatter is not None else ChatFormatter(tokenizer=tokenizer)
    features = encode_pairs(pairs, formatter=active_formatter, tokenizer=tokenizer)
    if not features:
        raise ValueError("train_dpo needs at least one preference pair")

    active_collator = collator or PreferenceCollator(
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
    rewards: list[RewardPoint] = []
    tokens_seen = 0
    pairs_seen = 0
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
            group_pairs = 0
            group_tokens = 0
            chosen_sum = 0.0
            rejected_sum = 0.0
            correct = 0.0

            for micro in group:
                batch = _to_device(active_collator([features[i] for i in micro]), device)
                with _autocast(device.type, enabled=config.bf16):
                    output = dpo_batch_output(
                        model,
                        batch,
                        beta=beta,
                        variant=variant,
                        label_smoothing=label_smoothing,
                        length_normalise=length_normalise,
                    )
                count = int(output.losses.shape[0])
                # The summed pair loss, not the mean: the single division by the group's
                # pair count below is what makes accumulation exact.
                (output.loss * count).backward()
                group_loss += float(output.loss.detach().float()) * count
                group_pairs += count
                group_tokens += pair_token_count(batch)
                chosen_sum += float(output.chosen_rewards.float().sum())
                rejected_sum += float(output.rejected_rewards.float().sum())
                correct += float((output.reward_margins > 0).sum())

            scale = 1.0 / group_pairs
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
            pairs_seen += group_pairs

            if step % config.log_every_steps == 0 or step == total:
                chosen_mean = chosen_sum * scale
                rejected_mean = rejected_sum * scale
                log.append(
                    step=step,
                    loss=group_loss * scale,
                    lr=learning_rate,
                    grad_norm=grad_norm,
                    tokens=tokens_seen,
                )
                rewards.append(
                    RewardPoint(
                        step=step,
                        chosen=chosen_mean,
                        rejected=rejected_mean,
                        margin=chosen_mean - rejected_mean,
                        accuracy=correct * scale,
                    )
                )

    adapter_path: Path | None = None
    if config.save_best_adapter:
        # No validation split here: a DPO run is judged on reward accuracy and on the
        # downstream evaluation, so the weights at the end of the run are the result.
        adapter_path = save_adapter(model, config.output_dir / FINAL_CHECKPOINT_DIR)

    model.eval()
    last = rewards[-1] if rewards else None
    logger.info(
        "dpo finished: %d steps, %d pairs, final loss %.4f, reward accuracy %.3f",
        step,
        pairs_seen,
        log.records[-1].loss if log.records else float("nan"),
        last.accuracy if last is not None else float("nan"),
    )

    manifest = RunManifest.build(
        stage="dpo",
        model=model,
        model_name=_model_name(model, model_name),
        data_content_hash=data_content_hash or _feature_hash(features),
        config=config,
    )
    return DPOResult(
        log=log,
        rewards=tuple(rewards),
        steps=step,
        pairs_seen=pairs_seen,
        supervised_tokens=tokens_seen,
        beta=beta,
        variant=variant,
        manifest=manifest,
        adapter_path=adapter_path,
        collation=active_collator.report,
    )
