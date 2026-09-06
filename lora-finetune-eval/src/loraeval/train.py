"""Fine-tuning loop: AdamW + linear warm-up/decay, bf16 autocast, gradient clipping,
per-epoch validation on macro-F1, early stopping and best-checkpoint restoration."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from loraeval.data import Example
from loraeval.evaluate import predict
from loraeval.metrics import macro_f1, negative_log_likelihood
from loraeval.models import load_trainable_state_dict, trainable_state_dict

log = logging.getLogger(__name__)

Precision = Literal["auto", "fp32", "bf16"]


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    precision: Precision = "auto"
    seed: int = 42
    patience: int = 2
    """Stop after this many epochs without a val macro-F1 improvement (0 = never)."""
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0 or self.max_grad_norm < 0 or self.patience < 0:
            raise ValueError("weight_decay, max_grad_norm and patience must be non-negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.precision not in ("auto", "fp32", "bf16"):
            raise ValueError(f"unknown precision {self.precision!r}")


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches. Together with a seeded DataLoader generator
    this makes CPU runs bit-reproducible (``tests/test_train.py``)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def autocast_dtype(precision: Precision, device: torch.device) -> torch.dtype | None:
    """bf16 has fp32's exponent range, so no loss scaling is needed (unlike fp16)."""
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def warmup_linear_decay(step: int, total_steps: int, warmup_steps: int) -> float:
    """Multiplier for the base LR: linear ramp over ``warmup_steps`` then linear decay
    to zero at ``total_steps`` (the BERT fine-tuning recipe)."""
    if step < 0 or total_steps <= 0 or warmup_steps < 0:
        raise ValueError("invalid schedule arguments")
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    remaining = total_steps - step
    return max(0.0, remaining / max(1, total_steps - warmup_steps))


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW over *trainable* parameters only, no decay on biases/LayerNorm gains."""
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    lr: float
    seconds: float


@dataclass
class TrainResult:
    history: list[EpochStats] = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_f1: float = -math.inf
    stopped_early: bool = False
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Example],
    val_loader: DataLoader[Example],
    cfg: TrainConfig,
    *,
    n_classes: int,
) -> TrainResult:
    """Train, select the best epoch by validation macro-F1, and leave ``model`` holding
    the best weights."""
    device = resolve_device(cfg.device)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_linear_decay(s, total_steps, warmup_steps)
    )
    amp_dtype = autocast_dtype(cfg.precision, device)

    result = TrainResult()
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    t_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        loss_sum, n_seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(**batch)
            loss: Tensor = out.loss
            loss.backward()  # type: ignore[no-untyped-call]
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            bs = int(batch["labels"].shape[0])
            loss_sum += float(loss.detach()) * bs
            n_seen += bs

        val_pred = predict(model, val_loader, device, amp_dtype)
        val_f1 = macro_f1(val_pred.labels, val_pred.preds, n_classes)
        val_loss = negative_log_likelihood(val_pred.probs, val_pred.labels)
        stats = EpochStats(
            epoch=epoch,
            train_loss=loss_sum / max(n_seen, 1),
            val_loss=val_loss,
            val_macro_f1=val_f1,
            lr=float(scheduler.get_last_lr()[0]),
            seconds=time.perf_counter() - t_epoch,
        )
        result.history.append(stats)
        log.info(
            "epoch %d: train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f (%.1fs)",
            epoch,
            stats.train_loss,
            stats.val_loss,
            stats.val_macro_f1,
            stats.seconds,
        )

        if val_f1 > result.best_val_macro_f1:
            result.best_val_macro_f1 = val_f1
            result.best_epoch = epoch
            best_state = trainable_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg.patience and epochs_without_improvement >= cfg.patience:
                result.stopped_early = True
                log.info("early stopping at epoch %d (best epoch %d)", epoch, result.best_epoch)
                break

    if best_state is not None:
        load_trainable_state_dict(model, best_state)
    result.total_seconds = time.perf_counter() - t_start
    return result
