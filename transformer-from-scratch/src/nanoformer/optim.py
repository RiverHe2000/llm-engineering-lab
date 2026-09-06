"""Optimiser construction and the learning-rate schedule."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Two parameter groups: decayed (matrices, embeddings) and non-decayed (norm gains,
    biases).

    Decaying 1-D parameters pulls RMSNorm gains toward zero, which shrinks activations
    and hurts training; the ``p.dim() >= 2`` rule is the standard GPT-2 / nanoGPT fix.
    Tied weights appear once because ``named_parameters`` de-duplicates shared tensors.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """AdamW with decoupled weight decay; uses the fused CUDA kernel when available."""
    groups = build_param_groups(model, weight_decay)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))


def lr_at(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
) -> float:
    """Linear warm-up followed by cosine decay to ``min_lr`` (Loshchilov & Hutter, 2017).

    * Warm-up (``step < warmup_steps``) ramps from ``base_lr / warmup_steps`` to ``base_lr``
      so that Adam's second-moment estimates settle before large updates are taken.
    * After ``total_steps`` the schedule holds at ``min_lr``.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_span = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_span
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)
