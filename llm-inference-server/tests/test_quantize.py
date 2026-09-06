from __future__ import annotations

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from conftest import build_model
from llmserve.engine import GenerationEngine, GenerationRequest
from llmserve.quantize import (
    count_quantized_linears,
    maybe_compile,
    parameter_bytes,
    quantization_backend_available,
    quantize_dynamic_int8,
)
from llmserve.sampling import SamplingParams

needs_backend = pytest.mark.skipif(
    not quantization_backend_available(), reason="no quantized backend on this platform"
)


@needs_backend
def test_dynamic_int8_replaces_linears_and_still_generates(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    model = build_model("qwen2", tokenizer)
    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    fp32_bytes = parameter_bytes(model)
    quantized = quantize_dynamic_int8(model)
    assert count_quantized_linears(quantized) == n_linear
    assert parameter_bytes(quantized) < fp32_bytes  # packed int8 weights are not fp params

    engine = GenerationEngine(quantized, tokenizer, device="cpu", max_context=64)
    out = engine.generate_batch(
        [
            GenerationRequest(
                "a", "profit rose", 4, SamplingParams(temperature=0.0), stop_on_eos=False
            ),
            GenerationRequest(
                "b", "the bank said today", 4, SamplingParams(temperature=0.0), stop_on_eos=False
            ),
        ]
    )
    assert [r.completion_tokens for r in out] == [4, 4]
    assert all(0 <= t < len(tokenizer) for r in out for t in r.token_ids)


def test_maybe_compile_off_returns_same_object(tokenizer: PreTrainedTokenizerFast) -> None:
    model = build_model("qwen2", tokenizer)
    assert maybe_compile(model, enabled=False) is model
