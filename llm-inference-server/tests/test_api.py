from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llmserve.api import create_app
from llmserve.config import Settings
from llmserve.engine import GenerationEngine


@pytest.fixture
def client(qwen_engine: GenerationEngine, settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings, engine=qwen_engine)
    with TestClient(app) as c:
        yield c


def test_health_ready_models_metrics(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"ready": True, "model": "tiny-qwen2", "device": "cpu"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny-qwen2"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "llmserve_requests_total" in metrics.text
    assert "llmserve_batch_size" in metrics.text


def test_completion_round_trip(client: TestClient) -> None:
    resp = client.post(
        "/v1/completions",
        json={"prompt": "profit rose", "max_tokens": 5, "temperature": 0, "stop_on_eos": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"].startswith("cmpl-") and body["model"] == "tiny-qwen2"
    assert body["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}
    assert isinstance(body["text"], str)
    assert body["engine_latency_ms"] > 0 and body["total_latency_ms"] >= body["engine_latency_ms"]
    assert resp.headers["X-Request-ID"]
    assert 'llmserve_requests_total{status="ok"} 1.0' in client.get("/metrics").text


def test_request_id_is_propagated(client: TestClient) -> None:
    resp = client.post(
        "/v1/completions",
        json={"prompt": "the bank", "max_tokens": 2, "temperature": 0},
        headers={"X-Request-ID": "trace-123"},
    )
    assert resp.headers["X-Request-ID"] == "trace-123"
    assert resp.json()["id"] == "cmpl-trace-123"


def test_seeded_requests_are_reproducible_over_http(client: TestClient) -> None:
    payload = {
        "prompt": "the market",
        "max_tokens": 8,
        "temperature": 1.2,
        "seed": 5,
        "stop_on_eos": False,
    }
    a = client.post("/v1/completions", json=payload).json()["text"]
    b = client.post("/v1/completions", json=payload).json()["text"]
    assert a == b


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": "x", "max_tokens": 0},
        {"prompt": "x", "temperature": 3.0},
        {"prompt": "x", "top_p": 0.0},
        {"prompt": "x", "repetition_penalty": 0.5},
        {"prompt": "x", "unknown_field": 1},
        {"prompt": 123},
    ],
)
def test_validation_errors_are_422(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/v1/completions", json=payload).status_code == 422


def test_server_limits_are_400(client: TestClient) -> None:
    too_many = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 17})
    assert too_many.status_code == 400 and "max_tokens" in too_many.json()["detail"]
    too_long = client.post("/v1/completions", json={"prompt": "y" * 401, "max_tokens": 1})
    assert too_long.status_code == 400 and "prompt" in too_long.json()["detail"]
    # 63 prompt tokens + 16 new tokens > max_context 64 -> engine-level rejection
    ctx = client.post("/v1/completions", json={"prompt": " ".join(["a"] * 70), "max_tokens": 16})
    assert ctx.status_code == 400 and "exceeds max_context" in ctx.json()["detail"]
    assert 'llmserve_requests_total{status="bad_request"} 1.0' in client.get("/metrics").text


def test_not_ready_before_startup(qwen_engine: GenerationEngine, settings: Settings) -> None:
    app = create_app(settings, engine=qwen_engine)
    client = TestClient(app)  # no context manager: lifespan never runs
    assert client.get("/ready").status_code == 503
    assert client.get("/ready").json()["ready"] is False
    assert client.post("/v1/completions", json={"prompt": "x"}).status_code == 503
    assert client.get("/v1/models").json()["data"] == []


def test_concurrent_requests_share_engine_batches(client: TestClient) -> None:
    payload = {"prompt": "shares rose", "max_tokens": 4, "temperature": 0, "stop_on_eos": False}

    def call(_: int) -> int:
        return int(client.post("/v1/completions", json=payload).status_code)

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(call, range(6)))
    assert statuses == [200] * 6
    stats = client.app.state.batcher.stats  # type: ignore[attr-defined]
    assert stats.requests >= 6
    assert max(stats.batch_sizes) > 1, "requests arriving together should be batched"
