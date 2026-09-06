"""Training loop with mixed precision, gradient accumulation, clipping, evaluation,
JSONL metrics and *exactly resumable* checkpoints.

"Exactly resumable" means: train N steps straight, or train k steps, checkpoint, reload
and train N-k more — both yield bit-identical weights on CPU (``tests/test_trainer.py``).
That requires checkpointing the optimiser, the AMP scaler and every RNG the loop uses.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from nanoformer.config import ModelConfig
from nanoformer.data import TokenDataset
from nanoformer.model import Transformer
from nanoformer.optim import configure_optimizer, lr_at

log = logging.getLogger(__name__)

Precision = Literal["fp32", "bf16", "fp16"]


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    batch_size: int = 32
    block_size: int = 128
    grad_accum_steps: int = 1
    max_steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    checkpoint_interval: int = 500
    log_interval: int = 10
    precision: Precision = "fp32"
    device: str = "auto"
    seed: int = 1337
    compile: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive = ["batch_size", "block_size", "grad_accum_steps", "max_steps", "eval_iters"]
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ["eval_interval", "checkpoint_interval", "log_interval", "warmup_steps"]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.lr <= 0 or self.min_lr < 0 or self.min_lr > self.lr:
            raise ValueError("need 0 <= min_lr <= lr and lr > 0")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative (0 disables clipping)")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"unknown precision {self.precision!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.block_size * self.grad_accum_steps


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _autocast_dtype(precision: Precision) -> torch.dtype | None:
    return {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


class Trainer:
    """Owns the model, optimiser, RNGs and metrics for one training run."""

    def __init__(
        self,
        model: Transformer,
        train_data: TokenDataset,
        val_data: TokenDataset,
        cfg: TrainConfig,
    ) -> None:
        if train_data.block_size != cfg.block_size or val_data.block_size != cfg.block_size:
            raise ValueError("dataset block_size must match TrainConfig.block_size")
        if cfg.block_size > model.cfg.max_seq_len:
            raise ValueError("block_size cannot exceed the model's max_seq_len")
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        if cfg.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 autocast requires CUDA; use bf16 or fp32 on CPU")

        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = configure_optimizer(
            self.model,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            device_type=self.device.type,
        )
        self.scaler = GradScaler(self.device.type, enabled=(cfg.precision == "fp16"))
        self._amp_dtype = _autocast_dtype(cfg.precision)
        self.data_rng = torch.Generator().manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.step = 0
        self.history: list[dict[str, float]] = []
        self.best_val_loss = math.inf
        self.out_dir = Path(cfg.out_dir)
        self._compiled = torch.compile(self.model) if cfg.compile else self.model

    # ----- helpers -------------------------------------------------------------------
    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_dtype is not None,
        )

    def _set_lr(self) -> float:
        lr = lr_at(
            self.step,
            base_lr=self.cfg.lr,
            warmup_steps=self.cfg.warmup_steps,
            total_steps=self.cfg.max_steps,
            min_lr=self.cfg.min_lr,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ----- one optimisation step -----------------------------------------------------
    def train_step(self) -> dict[str, float]:
        """Accumulate ``grad_accum_steps`` micro-batches, clip, and take one optimiser step."""
        self.model.train()
        lr = self._set_lr()
        total_loss = 0.0
        for _ in range(self.cfg.grad_accum_steps):
            x, y = self.train_data.get_batch(self.cfg.batch_size, self.data_rng, self.device)
            with self._autocast():
                out = self._compiled(x, y)
            assert out.loss is not None
            loss = out.loss / self.cfg.grad_accum_steps
            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach())

        self.scaler.unscale_(self.optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.grad_clip if self.cfg.grad_clip > 0 else math.inf,
            )
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return {"loss": total_loss, "lr": lr, "grad_norm": grad_norm}

    # ----- evaluation ----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Mean loss over ``eval_iters`` batches of train and val data.

        Uses a *fresh, fixed-seed* generator so that every evaluation sees the same
        batches (comparable across steps) without disturbing the training data stream.
        """
        self.model.eval()
        result: dict[str, float] = {}
        for name, ds in (("train", self.train_data), ("val", self.val_data)):
            rng = torch.Generator().manual_seed(self.cfg.seed + 1)
            losses: list[float] = []
            for _ in range(self.cfg.eval_iters):
                x, y = ds.get_batch(self.cfg.batch_size, rng, self.device)
                with self._autocast():
                    out = self._compiled(x, y)
                assert out.loss is not None
                losses.append(float(out.loss))
            result[f"{name}_loss"] = sum(losses) / len(losses)
        self.model.train()
        return result

    # ----- full loop -----------------------------------------------------------------
    def train(self) -> list[dict[str, float]]:
        """Run until ``max_steps``; returns the metrics history (also written as JSONL)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.out_dir / "metrics.jsonl"
        log.info(
            "training %s params on %s, %d tokens/step",
            f"{self.model.num_params():,}",
            self.device,
            self.cfg.tokens_per_step,
        )
        t_start = time.perf_counter()
        with metrics_path.open("a", encoding="utf-8") as fh:
            while self.step < self.cfg.max_steps:
                if self.cfg.eval_interval and self.step % self.cfg.eval_interval == 0:
                    self._evaluate_and_track_best(fh)
                t0 = time.perf_counter()
                stats = self.train_step()
                dt = time.perf_counter() - t0
                if self.cfg.log_interval and self.step % self.cfg.log_interval == 0:
                    tok_s = self.cfg.tokens_per_step / max(dt, 1e-9)
                    self._record(fh, {"step": self.step, **stats, "tokens_per_sec": tok_s})
                if (
                    self.cfg.checkpoint_interval
                    and self.step % self.cfg.checkpoint_interval == 0
                    and self.step < self.cfg.max_steps
                ):
                    self.save_checkpoint(self.out_dir / f"ckpt_{self.step:06d}.pt")
            self._evaluate_and_track_best(fh, extra={"elapsed_sec": time.perf_counter() - t_start})
        self.save_checkpoint(self.out_dir / "ckpt_final.pt")
        return self.history

    def _evaluate_and_track_best(
        self, fh: Any, extra: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Evaluate, log, and keep ``ckpt_best.pt`` at the lowest validation loss seen.

        Early stopping by checkpoint selection: in the data-limited regime the model
        keeps improving on train long after val turns around, so "final" is rarely "best".
        """
        row = {"step": self.step, **self.evaluate(), **(extra or {})}
        if row["val_loss"] < self.best_val_loss:
            self.best_val_loss = row["val_loss"]
            self.save_checkpoint(self.out_dir / "ckpt_best.pt")
            row["is_best"] = 1.0
        self._record(fh, row)
        return row

    def _record(self, fh: Any, row: dict[str, float]) -> None:
        self.history.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        log.info(
            " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        )

    # ----- checkpointing -------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "model_config": self.model.cfg.to_dict(),
            "train_config": self.cfg.to_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "step": self.step,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "data_rng_state": self.data_rng.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if self.device.type == "cuda" else None,
        }
        torch.save(state, path)

    @classmethod
    def resume(
        cls,
        path: str | Path,
        train_data: TokenDataset,
        val_data: TokenDataset,
        *,
        device: str | None = None,
    ) -> Trainer:
        """Rebuild a trainer from a checkpoint so that training continues *exactly*."""
        state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        model = Transformer(ModelConfig.from_dict(state["model_config"]))
        model.load_state_dict(state["model_state"])
        cfg = TrainConfig.from_dict(state["train_config"])
        if device is not None:
            cfg.device = device
        trainer = cls(model, train_data, val_data, cfg)
        trainer.optimizer.load_state_dict(state["optimizer_state"])
        trainer.scaler.load_state_dict(state["scaler_state"])
        trainer.step = int(state["step"])
        trainer.history = list(state["history"])
        trainer.best_val_loss = float(state.get("best_val_loss", math.inf))
        trainer.data_rng.set_state(state["data_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if trainer.device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"])
        return trainer


def load_model(path: str | Path, device: str = "cpu") -> Transformer:
    """Load just the model weights from a checkpoint (inference use)."""
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig.from_dict(state["model_config"]))
    model.load_state_dict(state["model_state"])
    return model.to(device).eval()


def loss_to_perplexity(loss: float) -> float:
    return math.exp(loss)


def to_tensor_stats(t: Tensor) -> dict[str, float]:
    """Small helper used by the CLI to print weight statistics."""
    return {"mean": float(t.mean()), "std": float(t.std()), "absmax": float(t.abs().max())}
