from __future__ import annotations

import json
import logging
import sys

import pytest
import torch
from pydantic import ValidationError

from llmserve.config import Settings
from llmserve.logging_utils import JsonFormatter, configure_logging, log_event, request_id_var


def test_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings(device="cpu").max_batch_size == 8
    monkeypatch.setenv("LLMSERVE_MAX_BATCH_SIZE", "3")
    monkeypatch.setenv("LLMSERVE_MODEL_NAME", "some/model")
    monkeypatch.setenv("LLMSERVE_QUANTIZE_INT8", "true")
    s = Settings(device="cpu")
    assert s.max_batch_size == 3 and s.model_name == "some/model" and s.quantize_int8 is True


def test_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        Settings(max_batch_size=0)
    with pytest.raises(ValidationError):
        Settings(batch_window_ms=-1)
    with pytest.raises(ValidationError):
        Settings(device="tpu")
    monkeypatch.setenv("LLMSERVE_REQUEST_TIMEOUT_S", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_device_and_dtype_resolution() -> None:
    s = Settings(device="cpu")
    assert s.resolve_device().type == "cpu"
    assert s.resolve_dtype(torch.device("cpu")) == torch.float32
    assert Settings(dtype="bfloat16").resolve_dtype(torch.device("cpu")) == torch.bfloat16
    assert Settings(dtype="float16").resolve_dtype(torch.device("cpu")) == torch.float16
    assert Settings(dtype="auto").resolve_dtype(torch.device("cuda")) == torch.bfloat16
    assert Settings(device="auto").resolve_device().type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            Settings(device="cuda").resolve_device()


def test_json_logging_carries_request_id(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_lines=True)
    logger = logging.getLogger("llmserve.test")
    token = request_id_var.set("abc123")
    try:
        log_event(logger, "hello", tokens=5)
    finally:
        request_id_var.reset(token)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["msg"] == "hello" and record["request_id"] == "abc123" and record["tokens"] == 5
    assert record["level"] == "INFO" and record["ts"].endswith("Z")

    configure_logging("DEBUG", json_lines=False)
    assert logging.getLogger().level == logging.DEBUG


def test_json_formatter_includes_exception() -> None:
    fmt = JsonFormatter()
    try:
        raise ValueError("bad")
    except ValueError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    out = json.loads(fmt.format(record))
    assert "ValueError: bad" in out["exc"]
