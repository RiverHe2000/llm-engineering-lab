"""Model and tokenizer loading, plus the parameter accounting a run manifest records.

Two loading paths exist on purpose. `build_tiny_model` instantiates a two-layer Qwen2 from
config with no download at all, so every test in this package -- masking, LoRA wiring, the
reference-policy context, the DPO loss identity -- runs on a real `Qwen2ForCausalLM` on CPU
in under a second. `load_model` is the real path and imports `transformers` lazily, so that
importing this module (which the collator tests do transitively) does not pay for it.

`HF_HUB_OFFLINE` is honoured explicitly rather than left to the hub library's own handling,
because the failure it prevents is expensive: a training run that reaches for the network
mid-epoch on a machine that has the weights cached already.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "DTYPE_ALIASES",
    "TINY_MODEL_CONFIG",
    "LoadedModel",
    "ModelInfo",
    "build_tiny_model",
    "hub_offline",
    "load_model",
    "resolve_device",
    "resolve_dtype",
]

TINY_MODEL_CONFIG: dict[str, Any] = {
    # The vocabulary is full size so that token ids from a real Qwen tokenizer are valid
    # here; everything else is shrunk. Tied embeddings keep the parameter count at 9.8 M
    # instead of 19 M, which is what makes a CPU test loop tolerable.
    "vocab_size": 151936,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
}

DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config string onto a torch dtype.

    Args:
        name: One of the keys of `DTYPE_ALIASES`, case-insensitive.

    Returns:
        The corresponding `torch.dtype`.

    Raises:
        ValueError: If the name is not a known alias. Failing here is much cheaper than
            failing after a multi-gigabyte download.
    """
    key = name.strip().lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(DTYPE_ALIASES)}")
    return DTYPE_ALIASES[key]


def resolve_device(device: str | None = None) -> str:
    """Pick a device, defaulting to CUDA when it is actually available.

    Args:
        device: `"cpu"`, `"cuda"`, `"auto"`, or None. None and `"auto"` are the same.

    Returns:
        The concrete device string.

    Raises:
        ValueError: If CUDA is requested but not available, or the string is unknown. An
            explicit `"cuda"` is a statement about the machine, so silently falling back to
            CPU would turn a configuration error into a run that is fifty times slower.
    """
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device 'cuda' requested but torch reports no CUDA device")
        return "cuda"
    raise ValueError(f"unknown device {device!r}; expected 'cpu', 'cuda' or 'auto'")


def hub_offline() -> bool:
    """Whether `HF_HUB_OFFLINE` asks us to stay off the network."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUTHY


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Parameter accounting for the run manifest.

    Recorded per run because "the adapter was 0.17 % of the weights" is the claim a reader
    of the results will want to check, and it cannot be reconstructed after the fact from a
    checkpoint directory alone.
    """

    name: str
    total_parameters: int
    trainable_parameters: int
    dtype: str
    device: str

    @property
    def trainable_pct(self) -> float:
        """Trainable share of all parameters, as a percentage."""
        return 100.0 * self.trainable_parameters / self.total_parameters

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, for writing straight into a manifest."""
        return {
            "name": self.name,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "trainable_pct": self.trainable_pct,
            "dtype": self.dtype,
            "device": self.device,
        }

    @classmethod
    def from_model(cls, model: Any, *, name: str) -> ModelInfo:
        """Count the parameters of a live model.

        Raises:
            ValueError: If the model has no parameters, which would otherwise surface much
                later as a division by zero in `trainable_pct`.
        """
        parameters = list(model.parameters())
        if not parameters:
            raise ValueError(f"model {name!r} has no parameters to count")
        total = sum(int(p.numel()) for p in parameters)
        trainable = sum(int(p.numel()) for p in parameters if p.requires_grad)
        return cls(
            name=name,
            total_parameters=total,
            trainable_parameters=trainable,
            dtype=str(parameters[0].dtype).removeprefix("torch."),
            device=parameters[0].device.type,
        )


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A model, its tokenizer and their accounting, kept together.

    Returned as one object rather than a tuple because every caller needs all three and a
    tuple invites them to be separated and then mismatched -- a tokenizer from one
    checkpoint with weights from another is a silent accuracy loss, not a crash.
    """

    model: Any
    tokenizer: Any
    info: ModelInfo


def build_tiny_model(seed: int = 0, **overrides: Any) -> Any:
    """Instantiate the two-layer Qwen2 used by the tests.

    No weights are downloaded: the architecture is built from a config, so this works on a
    machine with no Hugging Face cache and no network at all.

    The RNG is forked rather than seeded globally. Seeding `torch` from a helper would make
    a test's model depend on whether an earlier test happened to call it, which is exactly
    the kind of order dependence that makes a suite flaky.

    Args:
        seed: Seed for the weight initialisation.
        **overrides: Config fields to change, e.g. `num_hidden_layers=1`.

    Returns:
        A randomly initialised `Qwen2ForCausalLM` in float32 on CPU.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**{**TINY_MODEL_CONFIG, **overrides})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Qwen2ForCausalLM(config)


def _validated_adapter_path(adapter_path: str | Path | None) -> Path | None:
    if adapter_path is None:
        return None
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"adapter directory does not exist: {path}")
    return path


def load_model(
    name: str,
    *,
    dtype: str = "bfloat16",
    device: str | None = None,
    adapter_path: str | Path | None = None,
) -> LoadedModel:
    """Load a real checkpoint, optionally with a trained LoRA adapter on top.

    Every cheap check -- dtype alias, device availability, adapter directory -- happens
    before the first expensive call, so a typo costs a millisecond rather than a download.

    Args:
        name: Hub id or local path, e.g. `"Qwen/Qwen2.5-0.5B-Instruct"`.
        dtype: Alias understood by `resolve_dtype`. bf16 is the default because it trains
            stably without a loss scaler on any Ampere-or-later card.
        device: `"cpu"`, `"cuda"`, `"auto"` or None.
        adapter_path: Directory holding a saved PEFT adapter, or None for the base model.

    Returns:
        The model, its tokenizer and a `ModelInfo` describing what was loaded.

    Raises:
        ValueError: On an unknown dtype or an unavailable device.
        FileNotFoundError: If `adapter_path` does not exist.
    """
    torch_dtype = resolve_dtype(dtype)
    resolved_device = resolve_device(device)
    adapter = _validated_adapter_path(adapter_path)
    offline = hub_offline()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=offline)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch_dtype,
        local_files_only=offline,
    )
    model = model.to(resolved_device)

    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)

    # Loaded in eval mode: sampling and evaluation are the common case, and a trainer calls
    # .train() itself. The reverse default would leave dropout on during evaluation.
    model.eval()
    return LoadedModel(
        model=model, tokenizer=tokenizer, info=ModelInfo.from_model(model, name=name)
    )
