"""Dynamic batching: coalesce concurrent requests into one engine call.

Policy: when a request arrives and the engine is free, wait at most ``batch_window`` for
more requests (up to ``max_batch_size``), then run them together. Throughput scales with
batch size on GPUs (the weights are read once per step regardless of batch), so a few
milliseconds of added latency buy a large throughput gain under load — the same trade-off
TensorFlow Serving's and Triton's dynamic batchers make. The engine runs in a worker thread
so the event loop keeps serving health checks and metrics during generation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Protocol

from llmserve.engine import GenerationRequest, GenerationResult
from llmserve.metrics import BATCH_SIZE, ENGINE_BUSY, QUEUE_DEPTH

log = logging.getLogger(__name__)


class EngineLike(Protocol):
    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]: ...


class QueueFullError(RuntimeError):
    """Back-pressure signal: the API turns this into HTTP 503."""


@dataclass
class _Pending:
    request: GenerationRequest
    future: asyncio.Future[GenerationResult]


@dataclass
class BatcherStats:
    batches: int = 0
    requests: int = 0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def mean_batch_size(self) -> float:
        return self.requests / self.batches if self.batches else 0.0


class DynamicBatcher:
    def __init__(
        self,
        engine: EngineLike,
        *,
        max_batch_size: int = 8,
        batch_window_s: float = 0.01,
        queue_maxsize: int = 256,
    ) -> None:
        if max_batch_size <= 0 or batch_window_s < 0 or queue_maxsize <= 0:
            raise ValueError("invalid batcher configuration")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_window_s = batch_window_s
        self.queue_maxsize = queue_maxsize
        self.stats = BatcherStats()
        self._queue: asyncio.Queue[_Pending] | None = None
        self._task: asyncio.Task[None] | None = None
        self._inflight: list[_Pending] = []

    # ----- lifecycle ------------------------------------------------------------------
    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("batcher already started")
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)
        self._task = asyncio.create_task(self._run(), name="llmserve-batcher")

    async def stop(self) -> None:
        if self._task is None or self._queue is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        abandoned = list(self._inflight)
        while not self._queue.empty():
            abandoned.append(self._queue.get_nowait())
        for pending in abandoned:
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("server shutting down"))
        self._inflight = []
        self._task = None
        self._queue = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ----- submission -----------------------------------------------------------------
    async def submit(self, request: GenerationRequest) -> GenerationResult:
        if self._queue is None:
            raise RuntimeError("batcher not started")
        if self._queue.full():
            raise QueueFullError(f"queue is full ({self.queue_maxsize} waiting)")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerationResult] = loop.create_future()
        self._queue.put_nowait(_Pending(request, future))
        QUEUE_DEPTH.set(self._queue.qsize())
        return await future

    # ----- worker ---------------------------------------------------------------------
    async def _collect_batch(self) -> list[_Pending]:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        batch = [await self._queue.get()]
        # Take everything already waiting, then wait up to the window for stragglers.
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        deadline = loop.time() + self.batch_window_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        QUEUE_DEPTH.set(self._queue.qsize())
        return batch

    async def _run(self) -> None:
        while True:
            batch = await self._collect_batch()
            BATCH_SIZE.observe(len(batch))
            self.stats.batches += 1
            self.stats.requests += len(batch)
            self.stats.batch_sizes.append(len(batch))
            ENGINE_BUSY.set(1)
            self._inflight = batch
            try:
                results = await asyncio.to_thread(
                    self.engine.generate_batch, [p.request for p in batch]
                )
            except Exception as exc:
                log.exception("engine batch failed")
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(exc)
                self._inflight = []
                continue
            finally:
                # On cancellation (shutdown) _inflight stays populated so stop() can fail
                # the futures of the batch the worker thread is still computing.
                ENGINE_BUSY.set(0)
            for p, r in zip(batch, results, strict=True):
                if not p.future.done():  # the client may have timed out meanwhile
                    p.future.set_result(r)
            self._inflight = []
