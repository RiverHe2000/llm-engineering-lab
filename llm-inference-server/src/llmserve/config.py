"""Service configuration: typed, validated, overridable through ``LLMSERVE_*`` env vars.

Twelve-factor style — the same image runs with a different model or batch size by
changing the environment, never the code.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLMSERVE_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    model_name: str = "distilbert/distilgpt2"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"

    # Batching
    max_batch_size: int = Field(8, ge=1, le=256)
    batch_window_ms: float = Field(10.0, ge=0.0, le=1000.0)
    """How long the scheduler waits for more requests after the first one arrives."""
    queue_maxsize: int = Field(256, ge=1)
    request_timeout_s: float = Field(60.0, gt=0.0)

    # Limits enforced at the API boundary
    max_context: int = Field(1024, ge=16)
    max_new_tokens_limit: int = Field(256, ge=1)
    max_prompt_chars: int = Field(8000, ge=1)

    # Optimisations
    quantize_int8: bool = False
    torch_compile: bool = False
    seed: int | None = None

    log_level: str = "INFO"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("LLMSERVE_DEVICE=cuda but no CUDA device is available")
        return torch.device(self.device)

    def resolve_dtype(self, device: torch.device) -> torch.dtype:
        if self.dtype == "auto":
            return torch.bfloat16 if device.type == "cuda" else torch.float32
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
