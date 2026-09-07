"""LoRA configuration, attachment and the reference-policy trick DPO depends on.

Adapters are targeted at the attention projections *and* the MLP projections. The common
"q_proj and v_proj only" recipe comes from the original paper's GPT-3 experiments, where the
budget was chosen to make a 175 B model tractable. This task is different in kind: the model
has to learn an output *format* -- always emit one JSON object, always these keys, always
these enum spellings -- and that behaviour lives in the feed-forward blocks as much as in
attention. On a 0.5 B model the whole seven-projection adapter is still well under a percent
of the weights (0.167 % on the tiny CI model), so the cheaper recipe buys nothing worth the
lost capacity.

`reference_context` is the memory result that makes DPO fit on one 12 GB card: PEFT's
`disable_adapter()` yields the frozen base behaviour from the same weights, so the reference
policy costs nothing beyond the adapter it switches off.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import LoraConfig

__all__ = [
    "DEFAULT_TARGET_MODULES",
    "QWEN2_ATTENTION_PROJECTIONS",
    "QWEN2_MLP_PROJECTIONS",
    "TrainableReport",
    "attach_adapter",
    "default_target_modules",
    "lora_config",
    "reference_context",
    "trainable_parameter_report",
]

QWEN2_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
QWEN2_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_TARGET_MODULES = QWEN2_ATTENTION_PROJECTIONS + QWEN2_MLP_PROJECTIONS


def default_target_modules() -> tuple[str, ...]:
    """The seven Qwen2 projections an adapter is attached to.

    Returns:
        The four attention projections followed by the three MLP projections, as a tuple so
        that a caller cannot mutate the shared default.
    """
    return tuple(DEFAULT_TARGET_MODULES)


def lora_config(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    targets: Sequence[str] | None = None,
) -> LoraConfig:
    """Build the PEFT config for this project.

    `alpha = 2r` keeps the effective update scale (`alpha / r`) at 2 when the rank is tuned,
    so a rank sweep measures capacity rather than learning rate. Dropout defaults to zero
    because the datasets here are small and generated, and DPO in particular wants the
    policy and the reference to differ only by the adapter -- dropout inside the adapter
    would add noise to that comparison.

    Args:
        r: LoRA rank.
        alpha: LoRA scaling numerator.
        dropout: Dropout applied inside the adapter.
        targets: Module names to adapt; None uses `default_target_modules()`.

    Returns:
        A `peft.LoraConfig` with `task_type="CAUSAL_LM"`.

    Raises:
        ValueError: On a non-positive rank or alpha, a dropout outside [0, 1), an empty
            target list, or duplicate target names.
    """
    if r < 1:
        raise ValueError(f"LoRA rank must be at least 1, got {r}")
    if alpha < 1:
        raise ValueError(f"LoRA alpha must be at least 1, got {alpha}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")
    chosen = default_target_modules() if targets is None else tuple(targets)
    if not chosen:
        raise ValueError("target_modules must not be empty")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"duplicate target modules: {chosen}")

    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(chosen),
        bias="none",
        task_type="CAUSAL_LM",
    )


def attach_adapter(model: Any, cfg: LoraConfig) -> Any:
    """Wrap a base model in its LoRA adapter, freezing everything else.

    Args:
        model: A causal-LM, e.g. from `build_tiny_model` or `load_model`.
        cfg: The config from `lora_config`.

    Returns:
        The `PeftModel`. The base object is modified in place by PEFT, so the return value
        and the argument share their weights; the return value is the one to train.
    """
    from peft import get_peft_model

    return get_peft_model(model, cfg)


@dataclass(frozen=True, slots=True)
class TrainableReport:
    """How much of the model the optimiser will actually touch."""

    trainable: int
    total: int

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ValueError(f"total parameter count must be positive, got {self.total}")
        if not 0 <= self.trainable <= self.total:
            raise ValueError(
                f"trainable ({self.trainable}) must be between 0 and total ({self.total})"
            )

    @property
    def percentage(self) -> float:
        """Trainable parameters as a percentage of the total."""
        return 100.0 * self.trainable / self.total

    def as_dict(self) -> dict[str, float | int]:
        """Manifest-friendly view."""
        return {
            "trainable": self.trainable,
            "total": self.total,
            "percentage": self.percentage,
        }

    def __str__(self) -> str:
        return f"{self.trainable:,} trainable of {self.total:,} ({self.percentage:.3f} %)"


def trainable_parameter_report(model: Any) -> TrainableReport:
    """Count trainable and total parameters of a model.

    Counted from `model.parameters()` rather than from the config, so the number reflects
    what PEFT actually attached -- a target-module name that matches nothing is a silent
    no-op otherwise, and the first sign of it is a report that says 0 trainable.

    Raises:
        ValueError: If the model has no parameters.
    """
    parameters = list(model.parameters())
    total = sum(int(p.numel()) for p in parameters)
    trainable = sum(int(p.numel()) for p in parameters if p.requires_grad)
    return TrainableReport(trainable=trainable, total=total)


@contextlib.contextmanager
def reference_context(model: Any) -> Iterator[Any]:
    """Run the base model's behaviour on the policy's own weights.

    DPO needs log-probabilities from two policies: the one being trained and a frozen
    reference. With LoRA the second is not a second model -- switching the adapter off
    inside `peft`'s `disable_adapter()` reproduces the base model's log-probabilities
    exactly, which was verified against a separately loaded base checkpoint. A 0.5 B
    reference copy would cost roughly a gigabyte of VRAM in bf16; this costs nothing, and
    on a 12 GB card that is the difference between the run fitting and not.

    Args:
        model: A `PeftModel`.

    Yields:
        The same model, with its adapter disabled for the duration of the block.

    Raises:
        TypeError: If the model is not adapter-wrapped. Yielding the model unchanged would
            be worse than failing: the "reference" would be the policy itself, the implicit
            reward difference would be identically zero, and the loss would sit at ln 2
            while looking perfectly healthy.
    """
    disable = getattr(model, "disable_adapter", None)
    if not callable(disable):
        raise TypeError(
            f"{type(model).__name__} has no disable_adapter(); reference_context needs a "
            "PEFT model, so attach an adapter before asking for a reference policy"
        )
    with disable():
        yield model
