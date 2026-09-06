"""Prometheus metrics. One private registry so several app instances (tests) coexist."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "llmserve_requests_total",
    "Completion requests by outcome",
    ["status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "llmserve_request_latency_seconds",
    "End-to-end request latency (queue wait + generation)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)
GENERATED_TOKENS = Counter(
    "llmserve_generated_tokens_total", "Completion tokens produced", registry=REGISTRY
)
BATCH_SIZE = Histogram(
    "llmserve_batch_size",
    "Requests per engine batch",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    registry=REGISTRY,
)
QUEUE_DEPTH = Gauge("llmserve_queue_depth", "Requests waiting for a batch", registry=REGISTRY)
ENGINE_BUSY = Gauge("llmserve_engine_busy", "1 while the engine is generating", registry=REGISTRY)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
