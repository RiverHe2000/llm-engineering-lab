from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

import pytest

from llmserve.engine import GenerationRequest, GenerationResult
from llmserve.scheduler import DynamicBatcher, QueueFullError


class FakeEngine:
    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.delay = delay
        self.fail = fail
        self.batches: list[list[str]] = []

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        self.batches.append([r.request_id for r in requests])
        if self.fail:
            raise RuntimeError("boom")
        time.sleep(self.delay)
        return [
            GenerationResult(r.request_id, f"out:{r.prompt}", [1], 1, 1, "length", 1.0)
            for r in requests
        ]


def req(i: int) -> GenerationRequest:
    return GenerationRequest(f"r{i}", f"p{i}", max_new_tokens=1)


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def test_concurrent_requests_are_batched() -> None:
    async def main() -> tuple[FakeEngine, DynamicBatcher, list[GenerationResult]]:
        eng = FakeEngine(delay=0.05)
        b = DynamicBatcher(eng, max_batch_size=4, batch_window_s=0.05, queue_maxsize=16)
        await b.start()
        assert b.running
        results = await asyncio.gather(*[b.submit(req(i)) for i in range(6)])
        await b.stop()
        return eng, b, list(results)

    eng, b, results = run(main())
    assert [r.text for r in results] == [f"out:p{i}" for i in range(6)]
    sizes = [len(x) for x in eng.batches]
    assert sum(sizes) == 6 and max(sizes) <= 4 and max(sizes) > 1
    assert b.stats.requests == 6 and b.stats.batches == len(sizes)
    assert b.stats.mean_batch_size == pytest.approx(6 / len(sizes))
    assert not b.running


def test_engine_error_propagates_to_every_waiter_and_batcher_survives() -> None:
    async def main() -> None:
        eng = FakeEngine(fail=True)
        b = DynamicBatcher(eng, max_batch_size=4, batch_window_s=0.02, queue_maxsize=16)
        await b.start()
        outcomes = await asyncio.gather(
            *[b.submit(req(i)) for i in range(3)], return_exceptions=True
        )
        assert all(isinstance(o, RuntimeError) and str(o) == "boom" for o in outcomes)
        eng.fail = False
        ok = await b.submit(req(9))
        assert ok.text == "out:p9"
        await b.stop()

    run(main())


def test_queue_full_gives_back_pressure() -> None:
    async def main() -> None:
        eng = FakeEngine(delay=0.2)
        b = DynamicBatcher(eng, max_batch_size=1, batch_window_s=0.0, queue_maxsize=2)
        await b.start()
        tasks = [asyncio.create_task(b.submit(req(i))) for i in range(3)]
        await asyncio.sleep(0)  # let every submit reach the queue before the worker runs
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert sum(isinstance(o, QueueFullError) for o in outcomes) == 1
        assert sum(isinstance(o, GenerationResult) for o in outcomes) == 2
        await b.stop()

    run(main())


def test_zero_window_still_batches_requests_that_queued_meanwhile() -> None:
    async def main() -> list[list[str]]:
        eng = FakeEngine(delay=0.1)
        b = DynamicBatcher(eng, max_batch_size=4, batch_window_s=0.0, queue_maxsize=16)
        await b.start()
        first = asyncio.create_task(b.submit(req(0)))
        await asyncio.sleep(0.02)  # r0 is now in flight; r1 and r2 queue up behind it
        rest = [asyncio.create_task(b.submit(req(i))) for i in (1, 2)]
        await asyncio.gather(first, *rest)
        await b.stop()
        return eng.batches

    batches = run(main())
    assert batches == [["r0"], ["r1", "r2"]]


def test_everything_already_queued_forms_one_batch() -> None:
    async def main() -> list[list[str]]:
        eng = FakeEngine()
        b = DynamicBatcher(eng, max_batch_size=4, batch_window_s=0.0, queue_maxsize=16)
        await b.start()
        await asyncio.gather(*[b.submit(req(i)) for i in range(3)])
        await b.stop()
        return eng.batches

    assert run(main()) == [["r0", "r1", "r2"]]


def test_stop_fails_pending_and_inflight_requests() -> None:
    async def main() -> list[object]:
        eng = FakeEngine(delay=0.3)
        b = DynamicBatcher(eng, max_batch_size=1, batch_window_s=0.0, queue_maxsize=8)
        await b.start()
        tasks = [asyncio.create_task(b.submit(req(i))) for i in range(2)]
        await asyncio.sleep(0.05)  # r0 in flight, r1 queued
        await b.stop()
        return list(await asyncio.gather(*tasks, return_exceptions=True))

    outcomes = run(main())
    assert all(isinstance(o, RuntimeError) for o in outcomes)


def test_lifecycle_errors() -> None:
    async def main() -> None:
        b = DynamicBatcher(FakeEngine(), max_batch_size=2, batch_window_s=0.0, queue_maxsize=2)
        with pytest.raises(RuntimeError, match="not started"):
            await b.submit(req(0))
        await b.start()
        with pytest.raises(RuntimeError, match="already started"):
            await b.start()
        await b.stop()
        await b.stop()  # idempotent

    run(main())


@pytest.mark.parametrize(
    "kwargs", [{"max_batch_size": 0}, {"batch_window_s": -1.0}, {"queue_maxsize": 0}]
)
def test_invalid_configuration(kwargs: dict[str, float]) -> None:
    base: dict[str, float] = {"max_batch_size": 1, "batch_window_s": 0.0, "queue_maxsize": 1}
    with pytest.raises(ValueError):
        DynamicBatcher(FakeEngine(), **{**base, **kwargs})  # type: ignore[arg-type]
