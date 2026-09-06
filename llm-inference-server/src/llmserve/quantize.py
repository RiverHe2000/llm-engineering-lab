"""Inference-time optimisations that need no retraining.

* **Dynamic INT8 quantisation** (CPU): weights of every ``nn.Linear`` are stored as int8
  and activations are quantised on the fly per batch. 4x smaller weights, 1.5-3x faster
  matmuls on x86 (fbgemm), typically < 0.5 perplexity loss. GPU paths use different tools
  (bitsandbytes / AWQ / GPTQ), which is why this is gated to CPU.
* **torch.compile**: optional; a clear win for steady-state decode on Linux+CUDA, brittle
  on Windows, so it is off by default and explicit.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, cast

import torch
from torch import nn

log = logging.getLogger(__name__)


def quantization_backend_available() -> bool:
    engines = torch.backends.quantized.supported_engines
    return any(e in engines for e in ("fbgemm", "x86", "qnnpack", "onednn"))


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    if not quantization_backend_available():
        raise RuntimeError("no quantized backend available on this platform")
    with warnings.catch_warnings():
        # torch.ao eager-mode quantization is deprecated in favour of torchao; it still ships
        # in torch 2.11 and is the only dependency-free INT8 path, so use it deliberately.
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized = torch.ao.quantization.quantize_dynamic(  # type: ignore[no-untyped-call]
            model, {nn.Linear}, dtype=torch.qint8
        )
    n = count_quantized_linears(quantized)
    log.info("dynamic INT8 quantization applied to %d Linear layers", n)
    return cast(nn.Module, quantized)


def count_quantized_linears(model: nn.Module) -> int:
    from torch.ao.nn.quantized.dynamic import Linear as QLinear

    return sum(1 for m in model.modules() if isinstance(m, QLinear))


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    compiled: Any = torch.compile(model)
    log.info("torch.compile enabled")
    return cast(nn.Module, compiled)


def parameter_bytes(model: nn.Module) -> int:
    """Bytes held by floating-point parameters (quantised weights are packed and excluded)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())
