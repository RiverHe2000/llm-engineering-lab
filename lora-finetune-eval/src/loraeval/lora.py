"""Low-Rank Adaptation (Hu et al., 2021) implemented from scratch on top of ``nn.Linear``.

The idea: a fine-tuned weight is ``W + ΔW`` and ``ΔW`` has low intrinsic rank, so
parameterise it as ``ΔW = B @ A`` with ``A: [r, in]``, ``B: [out, r]`` and ``r << min(in, out)``.
Only ``A`` and ``B`` are trained (``0.1-1%`` of the parameters), the base weight stays
frozen, and at inference ``ΔW`` can be *merged* into ``W`` so there is zero latency cost.

Conventions match the reference implementation (microsoft/LoRA) and PEFT:
* ``A`` is Kaiming-uniform initialised, ``B`` is zero, so the adapted model equals the
  base model at step 0 (a property the tests verify).
* The update is scaled by ``alpha / r`` so that changing ``r`` does not require re-tuning
  the learning rate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: str = r"\.(q_lin|v_lin)$"
    """Regex matched against ``model.named_modules()`` names; every ``nn.Linear`` that
    matches is wrapped. The default targets DistilBERT's query/value projections, the
    choice made in the original paper's GPT-3 experiments."""

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        try:
            re.compile(self.target_modules)
        except re.error as exc:
            raise ValueError(f"target_modules is not a valid regex: {exc}") from exc

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class LoRALinear(nn.Module):
    """``y = W x + b + (alpha / r) * B A x`` with ``W``/``b`` frozen."""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = cfg.r
        self.scaling = cfg.scaling
        self.lora_A = nn.Parameter(torch.empty(cfg.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, cfg.r))
        self.lora_dropout: nn.Module = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.merged = False
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # nn.Linear's default init

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> Tensor:
        """The dense update ``ΔW = (alpha / r) * B @ A`` of shape ``[out, in]``."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.base(x)
        if self.merged:
            return result
        # Two skinny matmuls (in->r, r->out) instead of materialising ΔW every step.
        update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return result + update.to(result.dtype) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ΔW into the base weight: inference then costs exactly one matmul."""
        if not self.merged:
            self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_(self.delta_weight().to(self.base.weight.dtype))
            self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:g}, merged={self.merged}"


# --------------------------------------------------------------------------------------
# Model surgery
# --------------------------------------------------------------------------------------


def _parent_and_attr(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    *path, attr = qualified_name.split(".")
    parent = model
    for part in path:
        parent = getattr(parent, part)
    return parent, attr


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Replace every ``nn.Linear`` whose name matches ``cfg.target_modules`` with a
    :class:`LoRALinear`. Returns the qualified names that were wrapped."""
    pattern = re.compile(cfg.target_modules)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and pattern.search(name)
    ]
    if not targets:
        raise ValueError(f"no nn.Linear module matched target_modules={cfg.target_modules!r}")
    for name, module in targets:
        parent, attr = _parent_and_attr(model, name)
        setattr(parent, attr, LoRALinear(module, cfg))
    return [name for name, _ in targets]


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def mark_only_lora_as_trainable(model: nn.Module, extra_trainable: str | None = None) -> None:
    """Freeze everything except LoRA matrices and parameters matching ``extra_trainable``
    (typically the freshly initialised classification head)."""
    extra = re.compile(extra_trainable) if extra_trainable else None
    for name, param in model.named_parameters():
        is_lora = ".lora_A" in name or ".lora_B" in name
        is_extra = extra is not None and extra.search(name) is not None
        param.requires_grad_(is_lora or is_extra)


def merge_all(model: nn.Module) -> int:
    n = 0
    for module in lora_modules(model).values():
        module.merge()
        n += 1
    return n


def unmerge_all(model: nn.Module) -> int:
    n = 0
    for module in lora_modules(model).values():
        module.unmerge()
        n += 1
    return n


def lora_state_dict(model: nn.Module, extra: str | None = None) -> dict[str, Tensor]:
    """Only the adapter weights (plus ``extra`` matches), i.e. what needs to be shipped:
    for DistilBERT with r=8 on q/v that is ~0.3 MB instead of 268 MB."""
    pattern = re.compile(extra) if extra else None
    out: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".lora_A" in name or ".lora_B" in name or (pattern and pattern.search(name)):
            out[name] = tensor.detach().cpu().clone()
    return out


def load_lora_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Strict partial load: every key in ``state`` must exist in the model."""
    model_keys = set(model.state_dict().keys())
    missing = sorted(set(state) - model_keys)
    if missing:
        raise KeyError(f"adapter keys not present in model: {missing[:5]}")
    model.load_state_dict(state, strict=False)


@dataclass(frozen=True)
class ParamCount:
    total: int
    trainable: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (
            f"{self.trainable:,} trainable / {self.total:,} total "
            f"({100 * self.trainable_fraction:.3f}%)"
        )


def count_parameters(model: nn.Module) -> ParamCount:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParamCount(total=total, trainable=trainable)
