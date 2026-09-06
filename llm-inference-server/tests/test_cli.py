from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from llmserve import cli
from llmserve.config import Settings
from llmserve.engine import GenerationEngine, GenerationRequest
from llmserve.quantize import count_quantized_linears
from llmserve.sampling import SamplingParams


@pytest.fixture
def offline_engine(
    monkeypatch: pytest.MonkeyPatch, qwen_engine: GenerationEngine
) -> GenerationEngine:
    monkeypatch.setattr(
        GenerationEngine, "from_settings", classmethod(lambda _cls, _settings: qwen_engine)
    )
    return qwen_engine


@pytest.mark.usefixtures("offline_engine")
def test_generate_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "generate",
            "--device",
            "cpu",
            "--prompt",
            "profit rose",
            "--prompt",
            "the bank",
            "--max-tokens",
            "3",
            "--temperature",
            "0",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "[cli-0]" in out and "[cli-1]" in out and "finish=length tokens=3" in out


@pytest.mark.usefixtures("offline_engine")
def test_bench_command_writes_markdown_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    md_path, json_path = tmp_path / "b.md", tmp_path / "b.json"
    args = [
        "bench",
        "--device",
        "cpu",
        "--batch-sizes",
        "1,2",
        "--max-tokens",
        "2",
        "--repeats",
        "1",
        "--out",
        str(md_path),
        "--json-out",
        str(json_path),
        "--title",
        "bench-title",
    ]
    assert cli.main(args) == 0
    md = md_path.read_text(encoding="utf-8")
    assert md.startswith("## bench-title") and "| 2 |" in md
    assert "## bench-title" in capsys.readouterr().out
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [r["batch_size"] for r in payload["rows"]] == [1, 2] and payload["env"][
        "device"
    ] == "cpu"

    assert cli.main([*args, "--append"]) == 0
    assert md_path.read_text(encoding="utf-8").count("## bench-title") == 2


@pytest.mark.usefixtures("offline_engine")
def test_serve_command_hands_app_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    code = cli.main(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--device",
            "cpu",
            "--plain-logs",
            "--max-batch-size",
            "2",
            "--batch-window-ms",
            "5",
        ]
    )
    assert code == 0
    assert captured["host"] == "0.0.0.0" and captured["port"] == 9000
    assert captured["app"].title == "llmserve"
    assert captured["app"].state.settings.max_batch_size == 2


def test_settings_from_args_flags() -> None:
    args = cli.build_parser().parse_args(
        [
            "bench",
            "--quantize-int8",
            "--torch-compile",
            "--model-name",
            "m",
            "--max-context",
            "128",
            "--device",
            "cpu",
        ]
    )
    s = cli._settings_from_args(args)
    assert s.quantize_int8 and s.torch_compile and s.model_name == "m" and s.max_context == 128


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("HF_HUB_OFFLINE") == "1", reason="offline environment")
def test_from_settings_loads_a_real_model() -> None:
    settings = Settings(model_name="distilbert/distilgpt2", device="cpu", max_context=128)
    engine = GenerationEngine.from_settings(settings)
    out = engine.generate_batch(
        [
            GenerationRequest(
                "a", "The bank said", 5, SamplingParams(temperature=0.0), stop_on_eos=False
            )
        ]
    )[0]
    assert out.completion_tokens == 5 and out.text.strip()

    quantized = GenerationEngine.from_settings(
        Settings(
            model_name="distilbert/distilgpt2", device="cpu", max_context=128, quantize_int8=True
        )
    )
    assert count_quantized_linears(quantized.model) >= 1  # GPT-2 uses Conv1D except lm_head
    q_out = quantized.generate_batch(
        [
            GenerationRequest(
                "a", "The bank said", 5, SamplingParams(temperature=0.0), stop_on_eos=False
            )
        ]
    )[0]
    assert q_out.completion_tokens == 5
