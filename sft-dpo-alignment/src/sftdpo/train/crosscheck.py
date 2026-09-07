"""The honesty check: my DPO loss against the official `trl` implementation.

A hand-written objective is only worth writing if someone can tell whether it is right. This
module answers that by running the reference implementation on inputs that are identical
down to the bit, and reporting the largest difference it can find in the loss and in the two
implicit rewards. It reports the differences whether or not they are small.

**Why the comparison is not a call to a TRL loss function.** There is not one. As of TRL
1.12.0 the objective lives inside `DPOTrainer._compute_loss`, a method roughly two hundred
lines long that runs the policy forward pass, optionally the reference forward pass, the
log-probability reduction, one of fifteen loss types, the WPO weighting and the metric
bookkeeping, and returns a single scalar. Earlier releases exposed a `DPOTrainer.dpo_loss`
method taking four log-probability tensors; that method no longer exists. Both remaining
options are worse than a function call, so this module takes the least bad one and says so:

* *Transcribing the expressions into this file* would compare my code against my copy of
  TRL's code. Every future TRL release would drift away silently, which is precisely the
  failure this module exists to prevent.
* *Standing up a real `DPOTrainer`* needs a model, a tokenizer, a `Dataset`, an
  `Accelerator` and a `TrainingArguments` with a writable output directory. It would compare
  two whole training stacks rather than two objectives, and any disagreement would be a
  research project to localise.

So `_compute_loss` is invoked as an unbound function against a hand-built stand-in `self`
carrying the dozen attributes it reads, with the policy log-probabilities supplied as logits
that realise them exactly. TRL's own arithmetic runs, from TRL's own installed source; the
day TRL changes that arithmetic, this cross-check changes with it. The cost is a dependency
on a private method, and the mitigation is that the binding is one function in one place and
fails loudly -- `TrlInterfaceError` -- rather than degrading into a comparison that quietly
passes. A cross-check that has been disabled to keep a suite green is worse than no
cross-check at all.

**What was found.** Running it produced two disagreements, both real and neither papered
over. They are recorded in :data:`KNOWN_CONVENTIONS` and reproduced by the test suite:

1. *IPO is length-normalised in TRL and not here.* TRL divides each side's log-ratio by that
   side's completion token count before the squared error, a choice its own source
   attributes to correspondence with the IPO authors and which the paper does not state.
   :func:`sftdpo.train.dpo_loss.dpo_loss` takes four log-probabilities and no lengths, so it
   cannot do this and does not pretend to. The two agree exactly when the caller passes
   per-token averages -- `sequence_logprob(..., average=True)`, which the trainer exposes as
   `length_normalise=True` -- and differ by the completion length otherwise. The check
   quantifies this rather than asserting it: the residual after accounting for the
   normalisation is reported as `normalised_loss_difference`, and it is at the level of
   float64 noise.

2. *TRL 1.12.0 has no cDPO.* Its `sigmoid` loss carries no label smoothing at all any more,
   and its `label_smoothing` argument now feeds Robust DPO (`robust`), EXO and the AOT
   losses. Robust DPO is a *different* objective from cDPO, not a rescaling:
   cDPO takes the likelihood of a noisy label, `-(1-e) log s(u) - e log s(-u)`, while rDPO
   takes the unbiased estimator `[-(1-e) log s(u) + e log s(-u)] / (1 - 2e)`. Comparing them
   would be comparing two different formulas and calling the difference a bug. The smoothed
   likelihood does survive inside TRL's `aot` loss, which computes exactly cDPO's expression
   on a batch-sorted `h`; sorting is the identity for a single pair, so cDPO is cross-checked
   against `aot` one pair at a time, and it agrees to float64 noise.

Neither difference is a defect in this package, and both are worth knowing before quoting a
number against a paper. `trl` is an optional extra (`pip install -e ".[crosscheck]"`); when
it is absent the result says so in `reason` and `status` is `"unavailable"`, which is a
different thing from agreement and must never be reported as one.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import torch
from torch import Tensor

from sftdpo.train.dpo_loss import IGNORE_INDEX, DPOVariant, dpo_loss, sequence_logprob

__all__ = [
    "DEFAULT_TOLERANCE",
    "KNOWN_CONVENTIONS",
    "TRL_LOSS_TYPES",
    "CrossCheckBatch",
    "CrossCheckResult",
    "CrossCheckStatus",
    "TrlInterfaceError",
    "crosscheck_against_trl",
    "crosscheck_all",
    "trl_available",
    "trl_version",
]

DEFAULT_TOLERANCE = 1e-9
"""Largest difference two implementations of the same formula may show, in float64.

Nine decimal places is far tighter than any tolerance a training run would need, and that is
the point: the comparison is run in float64 so that a genuine difference in the arithmetic
stands several orders of magnitude clear of rounding. In float32 the same batch agrees only
to about 1e-7, which is not enough resolution to tell a formula difference from noise.
"""

TRL_LOSS_TYPES: dict[DPOVariant, str] = {
    "sigmoid": "sigmoid",
    "ipo": "ipo",
    "cdpo": "aot",
}
"""Which of TRL's fifteen loss types implements each of this package's three variants.

`cdpo` maps to `aot` rather than to `sigmoid` because TRL 1.12.0's sigmoid loss ignores
`label_smoothing` entirely; `aot` still computes the label-smoothed log-sigmoid likelihood,
over a batch-sorted `h` that is the identity for one pair. See the module docstring.
"""

KNOWN_CONVENTIONS: dict[DPOVariant, str] = {
    "ipo": (
        "TRL divides each side's log-ratio by that side's completion token count before the "
        "squared error; dpo_loss takes no lengths and cannot. The two agree exactly on "
        "per-token-averaged log-probabilities (length_normalise=True)."
    ),
    "cdpo": (
        "TRL 1.12.0 dropped label smoothing from its sigmoid loss and reuses the argument "
        "for Robust DPO, a different objective. cDPO's expression survives inside TRL's "
        "'aot' loss, which is compared one pair at a time so its batch sort is the identity."
    ),
}
"""Documented differences of convention, quoted verbatim into any result that hits one."""

CrossCheckStatus = Literal["agree", "explained", "disagree", "unavailable"]
"""Outcome of a comparison.

`"agree"` means every difference is within tolerance. `"explained"` means a difference
exceeded it but a documented convention accounts for it exactly, leaving a residual within
tolerance. `"disagree"` means neither -- the case this module exists to make impossible to
miss. `"unavailable"` means `trl` is not installed and nothing was compared.
"""


class TrlInterfaceError(RuntimeError):
    """Raised when `trl` is installed but its loss cannot be driven any more.

    Deliberately not caught and turned into a skip. This module reaches into
    `DPOTrainer._compute_loss`, and the day that method's shape changes the honest outcome is
    a loud failure that gets the binding fixed -- not a green suite whose cross-check has
    quietly stopped comparing anything.
    """


def trl_available() -> bool:
    """Whether the optional `trl` extra is installed.

    Checked by import machinery rather than by a try/except import, so nothing is executed
    and the answer is the same whether or not the module has already been imported.
    """
    return importlib.util.find_spec("trl") is not None


def trl_version() -> str:
    """The installed `trl` version, or `"not installed"`.

    Recorded in every result: a cross-check is a statement about two specific
    implementations, and the other one has a version number.
    """
    try:
        return importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


# --------------------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossCheckBatch:
    """Four sequence log-probabilities and the completion lengths that produced them.

    The lengths are carried because TRL's IPO needs them and mine does not, which is exactly
    the difference the check has to be able to see. Everything is float64: see
    :data:`DEFAULT_TOLERANCE`.

    Attributes:
        policy_chosen: `log pi(y_w | x)`, one entry per pair, all strictly negative.
        policy_rejected: `log pi(y_l | x)`.
        ref_chosen: `log pi_ref(y_w | x)`.
        ref_rejected: `log pi_ref(y_l | x)`.
        chosen_tokens: Scored tokens in each chosen completion.
        rejected_tokens: Scored tokens in each rejected completion.
    """

    policy_chosen: Tensor
    policy_rejected: Tensor
    ref_chosen: Tensor
    ref_rejected: Tensor
    chosen_tokens: tuple[int, ...]
    rejected_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        tensors = {
            "policy_chosen": self.policy_chosen,
            "policy_rejected": self.policy_rejected,
            "ref_chosen": self.ref_chosen,
            "ref_rejected": self.ref_rejected,
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional, got shape {tuple(tensor.shape)}")
            if tensor.shape != self.policy_chosen.shape:
                raise ValueError(
                    f"{name} has {tensor.shape[0]} entries but policy_chosen has "
                    f"{self.policy_chosen.shape[0]}; the four are aligned per pair"
                )
            if bool((tensor >= 0.0).any()):
                raise ValueError(
                    f"{name} must hold log-probabilities, which are negative; a value of zero "
                    "or above cannot be realised as logits and would make the comparison a "
                    "test of the reconstruction rather than of the loss"
                )
        for name, counts in (
            ("chosen_tokens", self.chosen_tokens),
            ("rejected_tokens", self.rejected_tokens),
        ):
            if len(counts) != self.size:
                raise ValueError(f"{name} has {len(counts)} entries, expected {self.size}")
            if any(count < 1 for count in counts):
                raise ValueError(f"{name} must be positive; got {counts}")

    @property
    def size(self) -> int:
        """Number of preference pairs."""
        return int(self.policy_chosen.shape[0])

    @property
    def width(self) -> int:
        """Sequence length the synthesised batch needs: the longest completion plus a prompt."""
        return 1 + max((*self.chosen_tokens, *self.rejected_tokens))

    @classmethod
    def synthetic(
        cls,
        size: int = 8,
        *,
        seed: int = 0,
        tokens: int = 1,
        spread: float = 4.0,
    ) -> CrossCheckBatch:
        """A pseudo-random batch spanning both signs of the log-ratio.

        Both signs matter. A batch where the policy already prefers every chosen response
        exercises only the saturating tail of the sigmoid, where two implementations can
        differ by a lot and still round to the same float.

        Args:
            size: Pairs in the batch.
            seed: Seed for the draw; the generator is local, so this never disturbs global
                torch state.
            tokens: Scored tokens per completion. One makes TRL's IPO length normalisation a
                no-op; more than one makes it bite.
            spread: Scale of the log-probabilities drawn.
        """
        if size < 1:
            raise ValueError(f"size must be at least 1, got {size}")
        if tokens < 1:
            raise ValueError(f"tokens must be at least 1, got {tokens}")
        generator = torch.Generator().manual_seed(seed)
        draws = [
            -spread * torch.rand(size, generator=generator, dtype=torch.float64) - 0.05
            for _ in range(4)
        ]
        return cls(
            policy_chosen=draws[0],
            policy_rejected=draws[1],
            ref_chosen=draws[2],
            ref_rejected=draws[3],
            chosen_tokens=(tokens,) * size,
            rejected_tokens=(tokens,) * size,
        )


# --------------------------------------------------------------------------------------
# The result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossCheckResult:
    """What the comparison found, including when it found a difference.

    Attributes:
        status: See :data:`CrossCheckStatus`.
        variant: The objective compared.
        trl_loss_type: The TRL loss type it was compared against.
        beta: KL strength used on both sides.
        label_smoothing: Smoothing used on both sides.
        pairs: Preference pairs compared.
        tolerance: Largest difference treated as agreement.
        loss_difference: Largest absolute per-pair loss difference.
        chosen_reward_difference: Largest absolute difference in `beta * (pi - pi_ref)` for
            the chosen response.
        rejected_reward_difference: The same for the rejected response.
        normalised_loss_difference: For a variant with a documented length convention, the
            residual once that convention is applied; None when there is none to apply.
        note: The documented convention, when one was needed to explain the numbers.
        reason: Why nothing was compared, when `status` is `"unavailable"`.
        trl_version: The version of `trl` that produced the other side.
    """

    status: CrossCheckStatus
    variant: DPOVariant
    trl_loss_type: str
    beta: float
    label_smoothing: float
    pairs: int
    tolerance: float
    loss_difference: float
    chosen_reward_difference: float
    rejected_reward_difference: float
    normalised_loss_difference: float | None
    note: str
    reason: str
    trl_version: str

    @property
    def available(self) -> bool:
        """Whether the comparison ran at all."""
        return self.status != "unavailable"

    @property
    def agrees(self) -> bool:
        """Whether the two implementations computed the same numbers.

        `"explained"` is not agreement. It means the difference is understood and documented,
        which is a claim about the write-up rather than about the arithmetic.
        """
        return self.status == "agree"

    @property
    def reward_difference(self) -> float:
        """The larger of the two implicit-reward differences."""
        return max(self.chosen_reward_difference, self.rejected_reward_difference)

    @property
    def max_difference(self) -> float:
        """The single number to quote: the largest difference found anywhere."""
        return max(self.loss_difference, self.reward_difference)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for the report."""
        return {
            "status": self.status,
            "variant": self.variant,
            "trl_loss_type": self.trl_loss_type,
            "trl_version": self.trl_version,
            "beta": self.beta,
            "label_smoothing": self.label_smoothing,
            "pairs": self.pairs,
            "tolerance": self.tolerance,
            "loss_difference": self.loss_difference,
            "chosen_reward_difference": self.chosen_reward_difference,
            "rejected_reward_difference": self.rejected_reward_difference,
            "normalised_loss_difference": self.normalised_loss_difference,
            "note": self.note,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        if self.status == "unavailable":
            return f"cross-check unavailable: {self.reason}"
        return (
            f"{self.variant} vs trl {self.trl_version} '{self.trl_loss_type}' on "
            f"{self.pairs} pairs: {self.status}, max difference {self.max_difference:.3e}"
        )


def _unavailable(variant: DPOVariant, beta: float, label_smoothing: float) -> CrossCheckResult:
    return CrossCheckResult(
        status="unavailable",
        variant=variant,
        trl_loss_type=TRL_LOSS_TYPES[variant],
        beta=beta,
        label_smoothing=label_smoothing,
        pairs=0,
        tolerance=DEFAULT_TOLERANCE,
        loss_difference=float("nan"),
        chosen_reward_difference=float("nan"),
        rejected_reward_difference=float("nan"),
        normalised_loss_difference=None,
        note="",
        reason=(
            "trl is not installed; it is an optional extra, so install it with "
            'pip install -e ".[crosscheck]" to run the comparison'
        ),
        trl_version="not installed",
    )


# --------------------------------------------------------------------------------------
# Realising log-probabilities as logits
# --------------------------------------------------------------------------------------


def _logit_for(target: float, vocab: int) -> float:
    """The logit that makes one token's log-probability exactly `target`.

    With every other logit at zero, `log_softmax([x, 0, ..., 0])[0] = x - log(e^x + V - 1)`,
    and setting that equal to `target` gives the closed form below. Solving analytically
    rather than optimising matters: the two implementations must read the *same* logits, so
    any residual in the reconstruction would show up as a difference between them.
    """
    return math.log(vocab - 1) - math.log(math.expm1(-target))


def _synthesise(batch: CrossCheckBatch, *, vocab: int) -> dict[str, Tensor]:
    """Build logits, ids and masks whose log-probabilities are the batch's, exactly.

    TRL's loss reduces logits, not log-probabilities: `_compute_loss` insists on running the
    policy forward pass itself. Rather than weaken the comparison by handing the two sides
    different numbers, the batch's log-probabilities are turned back into logits that
    realise them, and *both* implementations then reduce those same logits with their own
    code. That widens the check from the loss to the log-probability reduction: the shift
    convention, the masking of unscored positions and the ragged-length handling are all
    compared as well.

    Each position of a completion gets a different logit, so a disagreement about which
    logit predicts which token would change the totals rather than cancelling out.
    """
    rows = list(batch.chosen_tokens) + list(batch.rejected_tokens)
    totals = torch.cat([batch.policy_chosen, batch.policy_rejected]).tolist()
    width = batch.width

    input_ids = torch.zeros((2 * batch.size, width), dtype=torch.long)
    completion_mask = torch.zeros((2 * batch.size, width), dtype=torch.long)
    logits = torch.zeros((2 * batch.size, width, vocab), dtype=torch.float64)

    for row, (count, total) in enumerate(zip(rows, totals, strict=True)):
        # Split the sequence total over its tokens with distinct, non-uniform shares, so no
        # two positions carry the same number and a mis-shift cannot cancel.
        weights = [1.0 + index for index in range(count)]
        scale = total / sum(weights)
        for index in range(count):
            token = (row + index + 1) % vocab
            position = index + 1
            input_ids[row, position] = token
            completion_mask[row, position] = 1
            logits[row, position - 1, token] = _logit_for(weights[index] * scale, vocab)

    labels = input_ids.masked_fill(completion_mask == 0, IGNORE_INDEX)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "completion_mask": completion_mask,
        "labels": labels,
        "logits": logits,
    }


class _IdentityAccelerator:
    """The two gather methods `_compute_loss` calls, on a single process.

    `Accelerator` itself is not constructed: it initialises distributed state, reads the
    environment and prints, none of which a loss comparison needs. On one process both
    gathers are the identity, which is what makes the stand-in faithful rather than merely
    convenient.
    """

    device = torch.device("cpu")

    def gather_for_metrics(self, tensor: Tensor) -> Tensor:
        return tensor

    def gather(self, tensor: Tensor) -> Tensor:
        return tensor


def _trl_stub(*, loss_type: str, beta: float, label_smoothing: float) -> Any:
    """The attributes TRL's `_compute_loss` reads off `self`, and nothing else.

    Listing them explicitly is the point: if a future TRL reads an attribute this does not
    carry, the call raises `AttributeError` and :func:`_run_trl` turns it into a
    `TrlInterfaceError` naming the attribute, rather than the check silently comparing the
    wrong thing.
    """
    return SimpleNamespace(
        model=SimpleNamespace(training=False),
        accelerator=_IdentityAccelerator(),
        aux_loss_enabled=False,
        ld_alpha=None,
        precompute_ref_logps=True,
        ref_model=None,
        f_divergence_type="reverse_kl",
        loss_types=[loss_type],
        loss_weights=[1.0],
        beta=beta,
        label_smoothing=label_smoothing,
        use_weighting=False,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens=0,
    )


def _run_trl(
    tensors: dict[str, Tensor],
    *,
    row: int,
    pairs: int,
    ref_chosen: Tensor,
    ref_rejected: Tensor,
    loss_type: str,
    beta: float,
    label_smoothing: float,
) -> tuple[float, float, float]:
    """Run TRL's own loss on one pair, returning (loss, chosen reward, rejected reward).

    One pair per call, for two reasons. TRL returns only the batch mean of its per-sequence
    losses, so a per-pair comparison is the only one available; and its `aot` losses sort the
    batch, which is the identity on a single pair and would otherwise make the comparison
    depend on the draw.

    Raises:
        TrlInterfaceError: If TRL's private loss method has moved or changed shape.
    """
    try:
        from trl.trainer.dpo_trainer import DPOTrainer
    except ImportError as exc:  # pragma: no cover - guarded by trl_available()
        raise TrlInterfaceError(f"trl is installed but cannot be imported: {exc}") from exc

    compute = getattr(DPOTrainer, "_compute_loss", None)
    if not callable(compute):
        raise TrlInterfaceError(
            "trl.trainer.dpo_trainer.DPOTrainer has no _compute_loss; the DPO objective has "
            "moved, so this cross-check is comparing nothing until the binding is updated"
        )

    select = [row, row + pairs]
    inputs = {
        "input_ids": tensors["input_ids"][select],
        "attention_mask": tensors["attention_mask"][select],
        "completion_mask": tensors["completion_mask"][select],
        "ref_chosen_logps": ref_chosen[row : row + 1],
        "ref_rejected_logps": ref_rejected[row : row + 1],
    }
    logits = tensors["logits"][select]

    def forward(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=logits)

    stub = _trl_stub(loss_type=loss_type, beta=beta, label_smoothing=label_smoothing)
    try:
        loss = compute(stub, forward, inputs, False)
    except (AttributeError, KeyError, TypeError) as exc:
        raise TrlInterfaceError(
            f"trl's DPOTrainer._compute_loss could not be driven directly ({exc!r}); its "
            "interface has changed and the cross-check must be rebound before it means "
            "anything"
        ) from exc
    metrics = stub._metrics["eval"]
    return (
        float(loss),
        float(metrics["rewards/chosen"][-1]),
        float(metrics["rewards/rejected"][-1]),
    )


def _mine(
    tensors: dict[str, Tensor],
    batch: CrossCheckBatch,
    *,
    beta: float,
    variant: DPOVariant,
    label_smoothing: float,
    length_normalise: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """My loss and rewards, reduced from the same logits TRL is given."""
    logps = sequence_logprob(tensors["logits"], tensors["labels"], average=length_normalise)
    policy_chosen, policy_rejected = logps.chunk(2, dim=0)
    ref_chosen, ref_rejected = batch.ref_chosen, batch.ref_rejected
    if length_normalise:
        ref_chosen = ref_chosen / torch.tensor(batch.chosen_tokens, dtype=ref_chosen.dtype)
        ref_rejected = ref_rejected / torch.tensor(batch.rejected_tokens, dtype=ref_rejected.dtype)
    output = dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=beta,
        variant=variant,
        label_smoothing=label_smoothing,
    )
    return output.losses, output.chosen_rewards, output.rejected_rewards


def crosscheck_against_trl(
    batch: CrossCheckBatch,
    *,
    beta: float = 0.1,
    variant: DPOVariant = "sigmoid",
    label_smoothing: float = 0.0,
    tolerance: float = DEFAULT_TOLERANCE,
    vocab: int = 8,
) -> CrossCheckResult:
    """Compare `dpo_loss` against `trl` on inputs both implementations read identically.

    The batch's log-probabilities are realised as logits; TRL reduces them with
    `selective_log_softmax` inside its own `_compute_loss`, this package reduces them with
    `sequence_logprob`, and the two objectives are then evaluated on those reductions. The
    reference log-probabilities are handed to both sides as numbers, so nothing about the
    reference forward pass enters the comparison.

    Args:
        batch: The pairs to compare on.
        beta: KL strength, passed to both implementations.
        variant: Which of this package's three objectives to check.
        label_smoothing: Passed to both; only `"cdpo"` uses it.
        tolerance: Largest absolute difference counted as agreement.
        vocab: Vocabulary of the synthesised logits. Small is fine and fast; the value only
            has to admit distinct token ids per position.

    Returns:
        A `CrossCheckResult`. When `trl` is absent, `status` is `"unavailable"` and no
        difference is reported -- which is not the same as agreement and is never presented
        as one.

    Raises:
        TrlInterfaceError: If `trl` is installed but its loss cannot be driven.
        ValueError: On a tolerance that is not positive, or a vocabulary below two.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if vocab < 2:
        raise ValueError(f"vocab must be at least 2, got {vocab}")
    if variant not in TRL_LOSS_TYPES:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(TRL_LOSS_TYPES)}")
    if not trl_available():
        return _unavailable(variant, beta, label_smoothing)

    loss_type = TRL_LOSS_TYPES[variant]
    tensors = _synthesise(batch, vocab=vocab)
    mine_losses, mine_chosen, mine_rejected = _mine(
        tensors,
        batch,
        beta=beta,
        variant=variant,
        label_smoothing=label_smoothing,
        length_normalise=False,
    )

    trl_losses: list[float] = []
    trl_chosen: list[float] = []
    trl_rejected: list[float] = []
    for row in range(batch.size):
        loss, chosen, rejected = _run_trl(
            tensors,
            row=row,
            pairs=batch.size,
            ref_chosen=batch.ref_chosen,
            ref_rejected=batch.ref_rejected,
            loss_type=loss_type,
            beta=beta,
            label_smoothing=label_smoothing,
        )
        trl_losses.append(loss)
        trl_chosen.append(chosen)
        trl_rejected.append(rejected)

    reference = torch.tensor(trl_losses, dtype=torch.float64)
    loss_difference = float((mine_losses - reference).abs().max())
    chosen_difference = float(
        (mine_chosen - torch.tensor(trl_chosen, dtype=torch.float64)).abs().max()
    )
    rejected_difference = float(
        (mine_rejected - torch.tensor(trl_rejected, dtype=torch.float64)).abs().max()
    )

    normalised: float | None = None
    note = ""
    if variant == "ipo":
        # The one documented convention that can be applied and measured rather than merely
        # asserted: recompute my loss on per-token averages and see what is left over.
        normalised_losses, _, _ = _mine(
            tensors,
            batch,
            beta=beta,
            variant=variant,
            label_smoothing=label_smoothing,
            length_normalise=True,
        )
        normalised = float((normalised_losses - reference).abs().max())

    worst = max(loss_difference, chosen_difference, rejected_difference)
    if worst <= tolerance:
        status: CrossCheckStatus = "agree"
    # Both implicit rewards, not just the chosen one. The length-normalisation convention
    # explains a difference in the *loss*; it explains nothing about a reward, so a
    # discrepancy in either of them has to fall through to "disagree". Omitting the rejected
    # reward here meant an injected 1.0 error in TRL's rejected reward was reported as
    # "explained" with a zero exit code -- the one outcome this module exists to prevent.
    elif normalised is not None and max(normalised, chosen_difference, rejected_difference) <= (
        tolerance
    ):
        status = "explained"
        note = KNOWN_CONVENTIONS["ipo"]
    else:
        status = "disagree"
        note = KNOWN_CONVENTIONS.get(variant, "")

    return CrossCheckResult(
        status=status,
        variant=variant,
        trl_loss_type=loss_type,
        beta=beta,
        label_smoothing=label_smoothing,
        pairs=batch.size,
        tolerance=tolerance,
        loss_difference=loss_difference,
        chosen_reward_difference=chosen_difference,
        rejected_reward_difference=rejected_difference,
        normalised_loss_difference=normalised,
        note=note,
        reason="",
        trl_version=trl_version(),
    )


def crosscheck_all(
    batch: CrossCheckBatch | None = None,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.1,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[CrossCheckResult, ...]:
    """Run the comparison for every variant, in the order the write-up reports them.

    `cdpo` is checked with a non-zero smoothing by default, because at `eps = 0` it reduces
    to the sigmoid objective exactly and would test nothing that the first result has not
    already tested.
    """
    pairs = batch if batch is not None else CrossCheckBatch.synthetic()
    variants: Sequence[DPOVariant] = ("sigmoid", "ipo", "cdpo")
    return tuple(
        crosscheck_against_trl(
            pairs,
            beta=beta,
            variant=variant,
            label_smoothing=label_smoothing if variant == "cdpo" else 0.0,
            tolerance=tolerance,
        )
        for variant in variants
    )
