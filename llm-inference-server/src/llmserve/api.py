"""HTTP surface: an OpenAI-style ``/v1/completions`` plus liveness, readiness and metrics.

Validation happens at the boundary (Pydantic) so the engine only ever sees well-formed
requests; overload and timeouts map to 503/504 instead of stack traces.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from llmserve import __version__
from llmserve.config import Settings
from llmserve.engine import ContextLengthError, GenerationEngine, GenerationRequest
from llmserve.logging_utils import log_event, request_id_var
from llmserve.metrics import GENERATED_TOKENS, REQUEST_LATENCY, REQUESTS, render
from llmserve.sampling import SamplingParams
from llmserve.scheduler import DynamicBatcher, QueueFullError

log = logging.getLogger(__name__)


# ----- schemas ----------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(64, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_k: int = Field(0, ge=0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    seed: int | None = None
    stop_on_eos: bool = True


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    model: str
    text: str
    finish_reason: str
    usage: Usage
    engine_latency_ms: float
    total_latency_ms: float


class ReadyResponse(BaseModel):
    ready: bool
    model: str | None = None
    device: str | None = None


# ----- app --------------------------------------------------------------------------------


def create_app(settings: Settings | None = None, engine: GenerationEngine | None = None) -> FastAPI:
    """Build the application. ``engine`` can be injected (tests, custom models)."""
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = False
        eng = engine if engine is not None else GenerationEngine.from_settings(cfg)
        batcher = DynamicBatcher(
            eng,
            max_batch_size=cfg.max_batch_size,
            batch_window_s=cfg.batch_window_ms / 1000.0,
            queue_maxsize=cfg.queue_maxsize,
        )
        await batcher.start()
        app.state.engine = eng
        app.state.batcher = batcher
        app.state.ready = True
        log.info("ready: model=%s device=%s", eng.model_name, eng.device)
        try:
            yield
        finally:
            app.state.ready = False
            await batcher.stop()

    app = FastAPI(title="llmserve", version=__version__, lifespan=lifespan)
    app.state.settings = cfg
    app.state.ready = False

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        log_event(
            log,
            "http",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            request_id=rid,
        )
        return response

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(body: CompletionRequest, request: Request) -> CompletionResponse:
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="model not ready")
        if body.max_tokens > cfg.max_new_tokens_limit:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens exceeds server limit of {cfg.max_new_tokens_limit}",
            )
        if len(body.prompt) > cfg.max_prompt_chars:
            raise HTTPException(
                status_code=400, detail=f"prompt exceeds {cfg.max_prompt_chars} characters"
            )
        gen_req = GenerationRequest(
            request_id=request_id_var.get(),
            prompt=body.prompt,
            max_new_tokens=body.max_tokens,
            sampling=SamplingParams(
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                repetition_penalty=body.repetition_penalty,
                seed=body.seed,
            ),
            stop_on_eos=body.stop_on_eos,
        )
        batcher: DynamicBatcher = request.app.state.batcher
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(batcher.submit(gen_req), timeout=cfg.request_timeout_s)
        except QueueFullError as exc:
            REQUESTS.labels(status="overloaded").inc()
            raise HTTPException(status_code=503, detail="server busy, retry later") from exc
        except TimeoutError as exc:
            REQUESTS.labels(status="timeout").inc()
            raise HTTPException(status_code=504, detail="generation timed out") from exc
        except ContextLengthError as exc:
            REQUESTS.labels(status="bad_request").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        total_ms = (time.perf_counter() - t0) * 1000.0
        REQUESTS.labels(status="ok").inc()
        GENERATED_TOKENS.inc(result.completion_tokens)
        REQUEST_LATENCY.observe(total_ms / 1000.0)
        return CompletionResponse(
            id=f"cmpl-{result.request_id}",
            model=request.app.state.engine.model_name,
            text=result.text,
            finish_reason=result.finish_reason,
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
            engine_latency_ms=round(result.latency_ms, 2),
            total_latency_ms=round(total_ms, 2),
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness: the process is up (used by container orchestrators to restart)."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready", response_model=ReadyResponse)
    async def ready(request: Request, response: Response) -> ReadyResponse:
        """Readiness: the model is loaded and the batcher is running (gate for traffic)."""
        is_ready = bool(request.app.state.ready)
        if not is_ready:
            response.status_code = 503
            return ReadyResponse(ready=False)
        eng: GenerationEngine = request.app.state.engine
        return ReadyResponse(ready=True, model=eng.model_name, device=str(eng.device))

    @app.get("/metrics")
    async def metrics() -> Response:
        payload, content_type = render()
        return Response(content=payload, media_type=content_type)

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, object]:
        name = request.app.state.engine.model_name if request.app.state.ready else None
        return {"object": "list", "data": [{"id": name, "object": "model"}] if name else []}

    return app
