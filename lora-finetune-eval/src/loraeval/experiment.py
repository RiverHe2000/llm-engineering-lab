"""One experiment = one config -> one run directory with everything needed to audit it:

runs/<name>/
    config.json        the exact configuration (after overrides)
    metrics.json       parameter budget, split sizes, per-epoch history, test report
    predictions.npz    test-set probabilities + labels (for paired tests later)
    weights.safetensors only the trained weights (LoRA + head: ~0.3 MB)
"""

from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import nn

from loraeval.data import (
    LABEL_NAMES,
    Agreement,
    DataSplit,
    Example,
    TokenizerLike,
    label_distribution,
    load_financial_phrasebank,
    make_loader,
    stratified_split,
)
from loraeval.evaluate import EvalReport, Predictions, evaluate_predictions, predict
from loraeval.lora import LoRAConfig, ParamCount
from loraeval.models import (
    DEFAULT_HEAD_PATTERN,
    Strategy,
    apply_strategy,
    load_pretrained,
    load_trainable_state_dict,
    trainable_state_dict,
)
from loraeval.train import (
    TrainConfig,
    TrainResult,
    autocast_dtype,
    resolve_device,
    set_seed,
    train_model,
)

log = logging.getLogger(__name__)


@dataclass
class DataConfig:
    agreement: Agreement = "all"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    max_length: int = 128
    seed: int = 42


@dataclass
class ExperimentConfig:
    name: str
    model_name: str = "distilbert/distilbert-base-uncased"
    strategy: Strategy = "lora"
    lora: LoRAConfig | None = field(default_factory=LoRAConfig)
    head_pattern: str = DEFAULT_HEAD_PATTERN
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_dir: str = "runs"
    n_boot: int = 1000

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError("name must be a non-empty single path component")
        if self.strategy == "lora" and self.lora is None:
            raise ValueError("strategy='lora' requires a lora section")
        if self.n_boot <= 0:
            raise ValueError("n_boot must be positive")

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        d = dict(d)
        lora_raw = d.pop("lora", "unset")
        lora: LoRAConfig | None
        if lora_raw == "unset":
            lora = LoRAConfig() if d.get("strategy", "lora") == "lora" else None
        else:
            lora = LoRAConfig(**lora_raw) if lora_raw is not None else None
        data = DataConfig(**d.pop("data", {}))
        train = TrainConfig(**d.pop("train", {}))
        return cls(lora=lora, data=data, train=train, **d)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(apply_overrides(raw, overrides))


def apply_overrides(config: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """``section.key=value`` overrides, values parsed as YAML (``train.lr=1e-4``)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return config


@dataclass
class RunResult:
    run_dir: Path
    params: ParamCount
    split: DataSplit
    train: TrainResult
    report: EvalReport
    predictions: Predictions


# ----- injection points (monkeypatched in tests to stay offline) ------------------------


def _load_examples(cfg: ExperimentConfig) -> list[Example]:
    return load_financial_phrasebank(cfg.data.agreement)


def _load_tokenizer(cfg: ExperimentConfig) -> TokenizerLike:
    from transformers import AutoTokenizer

    return cast(TokenizerLike, AutoTokenizer.from_pretrained(cfg.model_name))


def _load_model(cfg: ExperimentConfig) -> nn.Module:
    return load_pretrained(cfg.model_name, len(LABEL_NAMES))


# ----- the experiment -------------------------------------------------------------------


def run_experiment(
    cfg: ExperimentConfig,
    *,
    model: nn.Module | None = None,
    tokenizer: TokenizerLike | None = None,
    examples: Sequence[Example] | None = None,
) -> RunResult:
    set_seed(cfg.train.seed)
    t0 = time.perf_counter()

    all_examples = list(examples) if examples is not None else _load_examples(cfg)
    split = stratified_split(
        all_examples,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.data.seed,
    )
    tok = tokenizer if tokenizer is not None else _load_tokenizer(cfg)
    loader_kw = {"batch_size": cfg.train.batch_size, "max_length": cfg.data.max_length}
    train_loader = make_loader(split.train, tok, shuffle=True, seed=cfg.train.seed, **loader_kw)
    val_loader = make_loader(split.val, tok, shuffle=False, **loader_kw)
    test_loader = make_loader(split.test, tok, shuffle=False, **loader_kw)

    net = model if model is not None else _load_model(cfg)
    params = apply_strategy(net, cfg.strategy, cfg.lora, cfg.head_pattern)
    log.info("[%s] %s | split=%s", cfg.name, params, split.sizes())

    train_result = train_model(net, train_loader, val_loader, cfg.train, n_classes=len(LABEL_NAMES))

    device = resolve_device(cfg.train.device)
    preds = predict(net, test_loader, device, autocast_dtype(cfg.train.precision, device))
    report = evaluate_predictions(preds, LABEL_NAMES, n_boot=cfg.n_boot, seed=cfg.data.seed)
    log.info(
        "[%s] test accuracy %s | macro-F1 %s | ECE %.3f",
        cfg.name,
        report.accuracy,
        report.macro_f1,
        report.ece,
    )

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    metrics: dict[str, Any] = {
        "params": vars(params),
        "split": {
            "sizes": split.sizes(),
            "label_distribution": {
                "train": label_distribution(split.train),
                "val": label_distribution(split.val),
                "test": label_distribution(split.test),
            },
        },
        "train": train_result.to_dict(),
        "test": report.to_dict(),
        "wall_seconds": time.perf_counter() - t0,
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python": platform.python_version(),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preds.save(run_dir / "predictions.npz")
    save_weights(net, run_dir / "weights.safetensors")
    return RunResult(run_dir, params, split, train_result, report, preds)


def save_weights(model: nn.Module, path: Path) -> None:
    from safetensors.torch import save_file

    state = {k: v.contiguous() for k, v in trainable_state_dict(model).items()}
    save_file(state, str(path))


def load_run_model(run_dir: str | Path) -> tuple[nn.Module, TokenizerLike, ExperimentConfig]:
    """Rebuild a run's model (base weights + trained deltas) for inference."""
    from safetensors.torch import load_file

    run_dir = Path(run_dir)
    cfg = ExperimentConfig.from_dict(
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    )
    net = _load_model(cfg)
    apply_strategy(net, cfg.strategy, cfg.lora, cfg.head_pattern)
    load_trainable_state_dict(net, load_file(str(run_dir / "weights.safetensors")))
    net.eval()
    return net, _load_tokenizer(cfg), cfg
