# Interview notes — Transformer from scratch

Questions this project prepares me to answer, with the answer I would give and the code
that backs it. Numbers refer to `configs/small.yaml` unless stated.

## Attention

**Write down attention and its cost.**
`softmax(QKᵀ/√d_head + M) V`, computed per head. `layers.py::attention_reference` is the
literal formula; the model uses `F.scaled_dot_product_attention` which fuses it. Time and
memory are O(T²·d) per layer for the score matrix; FlashAttention avoids materialising the
T×T matrix by tiling the softmax, giving O(T) memory at the same FLOPs.

**Why divide by √d?** Dot products of two random d-dimensional vectors have variance ∝ d, so
without scaling the softmax saturates for large heads and gradients vanish.

**How is causality enforced, and how did you test it?** A boolean mask where query i sees
keys ≤ i (`causal_mask`); with a KV cache the query block sits at the *end* of the key
sequence, so the mask is offset by `t_key − t_query`. The test perturbs future tokens and
asserts earlier logits are unchanged to 1e-6.

**Multi-head vs grouped-query vs multi-query.** Query heads are always `n_heads`; K/V heads
are shared in groups. With 6 query heads and 2 KV heads the cache is 3× smaller: per token
`2 · n_layers · kv_heads · head_dim · bytes = 2·6·2·64·2 = 3 KB` in bf16 instead of 9 KB.
That is the memory that bounds batch size at inference, which is why LLaMA-2 70B and
Mistral use GQA. Implementation: `k.repeat_interleave(n_rep, dim=1)` after the cache
update, so the cache itself stays small.

## Positions

**Why RoPE rather than learned absolute embeddings?** RoPE rotates each q/k pair by an angle
proportional to position, so the dot product depends only on the *relative* offset
(tested in `test_relative_position_property`). It adds no parameters, extrapolates better,
and is compatible with KV caching because keys are rotated once at their own position.

**Why do low dimensions rotate fast and high dimensions slowly?** Frequency `i` is
`θ^(−2i/d)`; fast dimensions resolve local order, slow ones long-range distance. Raising θ
(10k → 500k in LLaMA-3) stretches the slow band for longer contexts.

## Normalisation and residuals

**RMSNorm vs LayerNorm.** RMSNorm drops the mean subtraction and bias: `x / rms(x) · g`.
Same stabilising effect, ~15 % cheaper, standard since LLaMA. I compute the statistic in
fp32 even under bf16 autocast; `test_half_precision_input_does_not_overflow` shows why
(300² overflows fp16).

**Pre-norm vs post-norm.** Pre-norm keeps the residual stream un-normalised, so the identity
path has unit gain and gradients flow straight through 6 or 60 layers. Post-norm
(original Transformer) needs careful warm-up and is unstable at depth.

**Why scale the output projections at init?** Each block *adds* to the residual stream;
with n_layers blocks the variance grows ∝ 2·n_layers. Dividing the init std of `wo` and
`w_down` by √(2·n_layers) keeps the stream's variance O(1) (GPT-2 trick, checked in
`test_residual_projections_use_scaled_init`).

## Training

**Why AdamW with decay/no-decay groups?** Weight decay on RMSNorm gains and biases pulls
them to zero and shrinks activations; the standard rule is "decay only tensors with
dim ≥ 2" (`optim.py::build_param_groups`). Tied embeddings appear once.

**Why warm-up + cosine?** Adam's second-moment estimate is noisy in the first steps; a
linear warm-up prevents huge early updates. Cosine decay to 10 % of the peak gives most of
the training at a high rate and a clean anneal at the end (`optim.py::lr_at`, unit-tested
for its shape).

**bf16 vs fp16 autocast.** bf16 has fp32's 8-bit exponent, so no loss scaling is needed and
overflow is a non-issue; fp16 needs `GradScaler`. Both are supported; the loss is always
reduced in fp32.

**Gradient accumulation.** `grad_accum_steps` micro-batches are summed before one optimiser
step; the loss is divided by the accumulation count so the gradient equals that of the large
batch. Tokens/step = batch × block × accum is what the LR was tuned for.

**What does "exactly resumable" require?** Model, optimiser state (Adam moments), scaler
state, step counter, the data-sampling generator, and the global torch RNG. Miss any one
and the resumed run silently diverges. `test_resume_is_bit_exact` compares every weight
with `torch.equal`.

**Throughput and utilisation.** 225 k tokens/s × 6 × 9.4 M params ≈ 12.7 TFLOP/s. That is
on the order of 10 % of an RTX 4070's dense bf16 peak — expected for a 9 M-param model,
which is kernel-launch bound; MFU rises with model width, not with more steps.

**Your model overfit. Why, and what did you do?** 415 k training tokens against 9.4 M
parameters and 16 k tokens/step: one epoch is 25 steps. At 3 000 steps train loss hit 0.09
and val rose to 6.3. Fixes, in order of value: stop early on val (best-val checkpoint),
dropout 0.2, fewer steps; the real fix is more data. This is the "tokens per parameter"
lesson of Chinchilla in miniature.

## Inference

**What does the KV cache buy?** Without it, generating token t recomputes attention over
all t positions: O(T²) per sequence overall. With it, each step attends 1 query to T
cached keys: O(T). Memory becomes the constraint, hence GQA, paged attention (vLLM) and
quantised caches.

**Pre-allocate or `torch.cat`?** `cat` copies the whole cache every step (quadratic
traffic, allocator churn). `KVCache` writes into a fixed buffer and returns a view.

**Sampling controls.** Temperature scales logits; top-k keeps the k largest; top-p keeps
the smallest set with cumulative mass ≥ p (always at least the top-1). Greedy is
temperature 0. Finished rows are frozen on EOS so batched output stays rectangular.

## Tokenizer

**Why byte-level BPE?** No unknown tokens: every string is a byte sequence. Merges are
learned by repeatedly joining the most frequent adjacent pair inside pre-tokenised words;
the GPT-2 regex keeps merges from crossing word/punctuation boundaries so vocabulary
entries are linguistically meaningful (" the" is one token after training).

**Why not shuffle before the train/val split?** With random-window sampling a shuffled
split would put validation windows inside training text: leakage. The split is contiguous
(last 10 % of the file).

## Engineering

**How do you know the fused attention kernel is correct?** Every layer can be switched to
the reference implementation (`model.set_attention_impl(False)`); tests compare the two on
MHA/GQA/MQA and with a cache. The same pattern — keep a slow oracle next to the fast path —
is how I would validate a custom Triton kernel.

**What would change to train a 7 B model?** FlashAttention-2, FSDP/ZeRO-3 sharding of
parameters and optimiser state, sequence packing with document masks, checkpoint sharding,
a real data pipeline (memory-mapped shards, deterministic per-rank sampling), and
logging of MFU, not just loss.
