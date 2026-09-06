# Interview notes — LLM inference serving

## Generation mechanics

**Walk me through one request.** API validates the body (Pydantic: ranges, unknown fields
rejected), checks server limits (max tokens, prompt size), builds a `GenerationRequest` and
awaits the batcher. The batcher coalesces it with other waiting requests and hands the
batch to the engine in a worker thread. The engine tokenises with **left padding**,
computes `position_ids` from the attention mask, runs one prefill forward pass, then one
forward pass per generated token with the KV cache; rows that hit EOS or their token
budget are evicted from the batch. Results resolve each request's future; the API records
metrics and returns usage + latency.

**Why left padding?** With a KV cache the next token is always read from the *last*
position of every row. Right padding would put pad tokens there for shorter prompts. Left
padding aligns all "last real tokens" at the end; the attention mask hides the pads and
explicit `position_ids` (cumsum of the mask − 1) make each row see positions 0..n−1 as if
it were alone. `test_batched_left_padded_greedy_matches_single_sequence` checks this token
for token against unbatched, uncached decoding for both Qwen2 and GPT-2.

**Why is the KV cache the bottleneck rather than compute?** Decode does one token per
step: the FLOPs are tiny but every step re-reads all model weights *and* the whole cache.
Per token the cache costs `2 · layers · kv_heads · head_dim · bytes`; for a 7 B model in
fp16 that is ~0.5 MB per token, so a 4 k-token context is 2 GB per sequence. That is why
GQA, quantised caches and paged allocation (vLLM) matter more than FLOPs at inference.

**Dynamic batching vs continuous batching.** This project does *dynamic* batching: a
batch is formed at the start and rows are evicted as they finish
(`DynamicCache.batch_select_indices`), but no new request joins mid-flight. Continuous
(iteration-level) batching — vLLM, TGI — admits new sequences at every decode step, which
needs per-sequence cache blocks and a scheduler that mixes prefill and decode. I chose
dynamic + eviction because it captures most of the throughput gain for a moderate
concurrency level with a fraction of the complexity, and I documented the gap honestly.

**Why does throughput rise with batch size?** Each decode step reads all the weights once
regardless of batch size; a batch of 16 amortises that read over 16 tokens. Throughput
grows almost linearly until the GPU becomes compute-bound or the cache fills memory.
The benchmark table in the README shows the curve and the latency price per request.

## Sampling

**Temperature, top-k, top-p, repetition penalty — and their order.** Divide logits by
temperature, keep top-k, keep the nucleus of mass p, softmax, sample. Repetition penalty
is applied first, on raw logits, to tokens already in prompt + generation (positive logits
divided, negative multiplied — the HF/CTRL rule, so it always lowers the probability).
Greedy is temperature 0. Each rule is a pure tensor function with its own test.

**Per-request seeds inside a batch.** A seeded request must produce the same tokens
whether it is alone or batched with strangers, so seeded rows are sampled individually
with their own generator; only unseeded, identical-parameter rows take the vectorised
path. `test_per_request_seed_is_reproducible_and_batch_independent` pins this down.

## Serving concerns

**Liveness vs readiness.** `/health` says the process is alive (restart me if not);
`/ready` says the model is loaded and the batcher is running (send me traffic). Kubernetes
uses them differently: a failing liveness probe restarts the pod, a failing readiness
probe only removes it from the load balancer. Loading a 7 B model can take a minute; a
single "health" endpoint would either restart-loop or send traffic to an empty server.

**Back-pressure.** The queue is bounded; when full the API returns 503 immediately instead
of accepting work it cannot finish (which would only grow latency for everyone). Clients
retry with back-off. `request_timeout_s` bounds the wait; a timed-out request's future is
simply not resolved when its batch completes.

**Why run the engine in a thread?** Generation is CPU/GPU-bound and would block the event
loop, so health checks and metrics would stall. `asyncio.to_thread` keeps the loop
responsive; PyTorch releases the GIL inside kernels, so the thread is cheap.

**Observability.** JSON logs with a correlation id (`X-Request-ID`, propagated through a
`ContextVar` so the engine's log lines carry it); Prometheus counters/histograms for
requests by status, latency, generated tokens, batch size and queue depth. From those you
derive the SLOs that matter: p95 latency, tokens/s, and the batch-size distribution
that tells you whether batching is actually happening under real traffic.

**Graceful shutdown.** On lifespan exit the batcher is cancelled and every pending or
in-flight future is failed with a clear error so clients get a response instead of a hung
connection (`test_stop_fails_pending_and_inflight_requests` — a bug I found by writing the
test: a `finally` block was clearing the in-flight list on cancellation).

## Optimisation

**Dynamic INT8 quantisation.** Weights stored as int8 with a per-tensor scale, activations
quantised on the fly per batch, matmuls in int8 (fbgemm / oneDNN). No calibration data
needed, minimal accuracy loss, 4x smaller Linear weights, and a CPU speed-up that is
largest when the model is memory-bound. GPU quantisation uses different kernels (AWQ,
GPTQ, bitsandbytes NF4) which is why this path is CPU-only. `torch.ao.quantization` is
deprecated in favour of `torchao`; I kept it because it is the only dependency-free path
and isolated it behind one function so the swap is local.

**What would I do next for a real deployment?** Continuous batching with paged KV,
speculative decoding for latency, streaming responses (SSE) with time-to-first-token as
the headline metric, prefix caching for shared system prompts, tensor parallelism for
> 13 B models, and a load test that drives the batch-size histogram to the operating point.
For most teams the honest answer is "run vLLM behind this API"; the value of this project
is knowing *why* each of those features exists.

## Correctness testing strategy

* Oracle tests: batched/cached/padded decoding vs. plain single-sequence decoding.
* Fake engine for the scheduler so batching policy is tested without a model.
* Random tiny models from two architecture families (Linear-based Qwen2, Conv1D-based
  GPT-2) so the engine is not accidentally coupled to one.
* API tested through the ASGI app with an injected engine — validation codes, limits,
  correlation ids, readiness gating, and that concurrent HTTP requests really share
  engine batches.
