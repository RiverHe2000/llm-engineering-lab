from __future__ import annotations

import pytest

from llmserve.benchmark import (
    environment_info,
    median,
    render_markdown,
    run_benchmark,
    summarize_latencies,
)
from llmserve.engine import GenerationEngine


def test_summarize_latencies() -> None:
    stats = summarize_latencies([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats.mean_ms == 30.0 and stats.p50_ms == 30.0
    assert stats.p95_ms == pytest.approx(48.0)
    assert stats.min_ms == 10.0 and stats.max_ms == 50.0
    assert median([3.0, 1.0, 2.0]) == 2.0
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_run_benchmark_and_render(qwen_engine: GenerationEngine) -> None:
    rows = run_benchmark(
        qwen_engine, prompt="the bank said", batch_sizes=[1, 2], max_new_tokens=3, repeats=2
    )
    assert [r.batch_size for r in rows] == [1, 2]
    for r in rows:
        assert r.new_tokens_per_request == 3 and r.repeats == 2 and r.prompt_tokens == 3
        assert r.tokens_per_second > 0 and r.latency.p50_ms > 0
        assert r.ms_per_token_per_request == pytest.approx(r.latency.p50_ms / 3)
        assert "tokens_per_second" in r.to_dict()
    env = environment_info(qwen_engine)
    assert env["device"] == "cpu" and env["model"] == "tiny-qwen2" and "torch" in env
    md = render_markdown(rows, "tiny", env)
    assert md.startswith("## tiny") and "| 2 | 3 | 3 |" in md and "1.00x" in md


@pytest.mark.parametrize(
    "kwargs",
    [{"batch_sizes": [0]}, {"max_new_tokens": 0}, {"repeats": 0}],
)
def test_invalid_benchmark_args(qwen_engine: GenerationEngine, kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {"prompt": "x", "batch_sizes": [1], "max_new_tokens": 1, "repeats": 1}
    with pytest.raises(ValueError):
        run_benchmark(qwen_engine, **{**base, **kwargs})  # type: ignore[arg-type]
