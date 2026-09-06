# llmserve · a production-style LLM inference server

A production-style serving stack for any Hugging Face causal language model: KV-cached
batched generation that is *provably* equivalent to single-sequence decoding, an async
dynamic batcher with back-pressure, INT8 quantisation for CPU, Prometheus metrics,
structured logs with correlation ids, an OpenAI-style HTTP API, and a Docker image.

| | |
|---|---|
| Quality gates | `ruff`, `mypy --strict`, **68 tests** (offline, CPU, ≈ 10 s), **96 % branch coverage** |
| Models exercised | random tiny Qwen2 + GPT-2 in tests; `distilgpt2` and `Qwen2.5-0.5B-Instruct` in the benchmarks |
| Headline | Dynamic batching: **29× throughput at batch 32 for +7 % latency** (Qwen2.5-0.5B, RTX 4070); 16 concurrent HTTP requests answered in 2.4 s instead of ~23 s sequentially |

---

## 1. Architecture

```
 HTTP client ──► FastAPI (validation, limits, request-id, metrics)
                    │  await batcher.submit(request)
                    ▼
             DynamicBatcher (asyncio)          ── bounded queue → 503 when full
               collect ≤ max_batch within batch_window
                    │  asyncio.to_thread(engine.generate_batch, batch)
                    ▼
             GenerationEngine (torch, one worker thread)
               tokenize (left pad) → prefill → decode loop with KV cache
               evict finished rows (DynamicCache.batch_select_indices)
               per-request sampling params & seeds
                    │
                    ▼
             results → futures → HTTP responses (+ Prometheus, JSON logs)
```

| Component | File | Notes |
|---|---|---|
| Engine | `engine.py` | Left padding + explicit `position_ids`; prefill once, then one token/step; finished rows are dropped from the batch and cache; EOS vs length finish reasons; context-length enforcement; vectorised sampling when parameters are uniform, per-row otherwise (seeded requests are always per-row so a request's output never depends on its batch-mates). |
| Scheduler | `scheduler.py` | Take everything already queued, wait up to `batch_window_ms` for stragglers, cap at `max_batch_size`; engine runs in a thread so the event loop keeps serving `/health`; queue bound → `QueueFullError` → HTTP 503; shutdown fails pending **and in-flight** futures. |
| API | `api.py` | `POST /v1/completions` (Pydantic: ranges, `extra="forbid"`), `/health` (liveness), `/ready` (readiness, 503 until the model is loaded), `/metrics`, `/v1/models`. Server limits → 400; overload → 503; timeout → 504. `X-Request-ID` in, out, and in every log line. |
| Sampling | `sampling.py` | Temperature, top-k, top-p, repetition penalty, seeds — pure tensor functions, unit-tested on hand-built logits. |
| Optimisation | `quantize.py` | Dynamic INT8 for `nn.Linear` on CPU (fbgemm/oneDNN); optional `torch.compile`. |
| Config | `config.py` | `pydantic-settings`, every knob via `LLMSERVE_*` env vars (twelve-factor). |
| Observability | `metrics.py`, `logging_utils.py` | Counters/histograms for requests by status, latency, tokens, batch size, queue depth; JSON-lines logs. |
| Benchmark | `benchmark.py` | Tokens/s and p50/p95 batch latency per batch size; Markdown + JSON output. |
| Packaging | `Dockerfile`, `docker-compose.yml` | Multi-stage CPU image, non-root user, health check, model cache volume. |

---

## 2. Results (RTX 4070, bf16; CPU = 20-core desktop) — full tables in [docs/BENCHMARK.md](docs/BENCHMARK.md)

**Batching trade-off, Qwen2.5-0.5B-Instruct, 64 new tokens per request**

| Batch | Tokens/s | Batch p50 latency | ms / token / request | Speed-up |
|---:|---:|---:|---:|---:|
| 1 | 48.5 | 1 342 ms | 21.0 | 1.0× |
| 4 | 187.8 | 1 341 ms | 21.0 | 3.9× |
| 8 | 378.8 | 1 315 ms | 20.5 | 7.8× |
| 16 | 755.1 | 1 335 ms | 20.9 | 15.6× |
| 32 | 1 408.1 | 1 434 ms | 22.4 | 29.0× |

Decode at batch 1 is bound by kernel-launch overhead (a 0.5 B model in eager PyTorch does
~21 ms/token here, far from the memory-bandwidth limit), so adding rows is almost free
until batch 32 — exactly the regime where batching is the single most valuable serving
optimisation. `distilgpt2` (82 M) shows the same shape: 363 → 8 922 tokens/s (24.6×).

**CPU, `distilgpt2`, 32 new tokens**

| Precision | Batch 1 tokens/s | Batch 8 tokens/s | Note |
|---|---:|---:|---|
| fp32 | 114 | 582 | |
| dynamic INT8 | 145 (+27 %) | 793 (+36 %) | `nn.Linear` only; GPT-2's `Conv1D` layers stay fp32, so the gain is modest — a Linear-based model (Qwen, LLaMA) benefits more |

**Live server smoke test** (`scripts/run_benchmarks.sh`; artefacts in `docs/smoke_*`):
Qwen2.5-0.5B-Instruct served on CUDA; 16 concurrent `curl`s of 60 tokens each — all 200,
mean 2.45 s, max 2.68 s; Prometheus shows they were executed as **one batch of 15 plus
one of 1** rather than 16 sequential calls of ~1.45 s each. Sample completion:

> *Name three risks a bank faces when lending to small businesses.* →
> "1. **Credit Risk**: This is the risk that the bank could lose the rights to the loaned
> funds if the borrower defaults … 2. **Market Risk**: …"

---

## 3. Tests: what each one proves

| Test | Property |
|---|---|
| `test_engine::test_batched_left_padded_greedy_matches_single_sequence` | Batched + padded + cached decoding equals unbatched, uncached decoding token-for-token (Qwen2 **and** GPT-2). |
| `test_engine::test_rows_finishing_early_are_dropped_without_changing_others` | Evicting finished rows from the KV cache does not perturb the survivors. |
| `test_engine::test_per_request_seed_is_reproducible_and_batch_independent` | A seeded request gives the same tokens alone or batched with strangers. |
| `test_engine::test_eos_stops_generation` / `test_context_length_is_enforced` | Finish reasons and context limits. |
| `test_scheduler::test_concurrent_requests_are_batched` | Concurrent submits coalesce into batches ≤ `max_batch_size`; results map back to the right futures. |
| `test_scheduler::test_queue_full_gives_back_pressure` | A bounded queue rejects instead of degrading. |
| `test_scheduler::test_stop_fails_pending_and_inflight_requests` | Shutdown never leaves a client hanging (this test found a real bug). |
| `test_api::*` | 422 on malformed input, 400 on server limits (including engine-level context errors), 503 before readiness, request-id propagation, metrics exposure, and that **concurrent HTTP requests really share engine batches**. |
| `test_quantize`, `test_benchmark`, `test_cli`, `test_config` | INT8 path generates valid tokens; benchmark maths; CLI wiring with an injected engine; env-var configuration. |

---

## 4. Run it

```bash
pip install -e ".[dev]" && pytest                       # offline tests

llmserve serve --model-name Qwen/Qwen2.5-0.5B-Instruct --device cuda --port 8000
curl -s localhost:8000/ready
curl -s -X POST localhost:8000/v1/completions -H "Content-Type: application/json" \
     -d '{"prompt": "The RBA said on Tuesday that", "max_tokens": 40, "temperature": 0.7, "seed": 1}'
curl -s localhost:8000/metrics | grep llmserve_batch_size

llmserve bench --model-name distilbert/distilgpt2 --device cpu --quantize-int8 --batch-sizes 1,4,8
llmserve generate --prompt "Explain a credit default swap in two sentences." --max-tokens 60

docker compose up --build                               # CPU image, INT8, model cache volume
```

Every setting is an environment variable (`LLMSERVE_MAX_BATCH_SIZE`, `LLMSERVE_BATCH_WINDOW_MS`,
`LLMSERVE_QUANTIZE_INT8`, …) — see `config.py`.

---

## 5. Design decisions and honest limits

* **Dynamic, not continuous, batching.** New requests do not join a running batch; finished
  rows leave it. This captures most of the throughput gain at moderate concurrency with a
  small, testable scheduler. vLLM-style iteration-level scheduling with paged KV blocks is
  the next step and is described in the interview notes.
* **Correctness before speed.** The oracle test (batched vs single-sequence decoding) is the
  contract every optimisation must keep. It caught the need for explicit `position_ids`
  under left padding on the first run.
* **No streaming yet.** Responses are returned whole; SSE streaming with time-to-first-token
  metrics is the obvious next feature and fits the existing per-step loop.
* **Eager PyTorch.** No CUDA graphs / `torch.compile` in the reported numbers (Windows
  Triton support is fragile). Compile is a flag; on Linux it would cut the per-step overhead
  that dominates small-batch decode.
* **`torch.ao` quantisation is deprecated** upstream in favour of `torchao`; it is isolated
  behind one function so the migration is local.

See [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md).

---

## Related projects

This repository is one of three standalone projects that together cover a Transformer's
life-cycle — build it, adapt it, serve it — all held to the same engineering standard
(ruff, `mypy --strict`, offline CPU test suites with coverage gates, matrix CI):

* **transformer-from-scratch** (`nanoformer`) — a LLaMA-style decoder, byte-level BPE and an
  AMP trainer with bit-exact resume, written from first principles.
* **lora-finetune-eval** (`loraeval`) — LoRA implemented from scratch and a statistically
  rigorous evaluation harness, applied to financial sentiment classification.
* **llm-inference-server** (`llmserve`) — KV-cached batched generation, dynamic batching,
  INT8, Prometheus metrics, FastAPI and Docker for any Hugging Face causal LM.
