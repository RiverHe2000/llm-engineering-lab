# Interview notes — LoRA and evaluation

## LoRA

**Explain LoRA in two sentences.** Fine-tuning changes a weight matrix by `ΔW`, and
empirically `ΔW` has low intrinsic rank; so parameterise `ΔW = B·A` with rank `r` (say 8),
train only `A` and `B`, and keep `W` frozen. Parameters drop from `d_in·d_out` to
`r·(d_in + d_out)` per matrix — 1 % of the model here — and at inference `ΔW` is merged
into `W`, so there is no extra latency.

**Why is `B` initialised to zero?** So the adapted model is *exactly* the base model at
step 0: training starts from the pretrained function, not from a randomly perturbed one.
`A` gets a normal Kaiming init so the product has a non-degenerate gradient
(`test_equals_base_at_init`, `test_gradients_flow_only_into_adapter`).

**What does `alpha` do?** The update is scaled by `α/r`. Keeping `α/r` constant while
changing `r` keeps the effective step size similar, so you can sweep `r` without re-tuning
the learning rate. In this repo `α = 2r` throughout.

**Which matrices should get adapters?** The original paper found q and v sufficient for
GPT-3; QLoRA found "all linear layers" better for large models. On this 67 M-param model
and 1.6 k examples, q/v (1.1 % params) beat all-linear (1.9 %) by 1.5 points — not
significant, but no evidence that more adapters help at this scale. It is a hyper-parameter;
measure it.

**How do merge/unmerge work and why test them?** `merge` does `W += (α/r)·B·A` in place;
`unmerge` subtracts it. The tests check merged and unmerged forward passes agree to 1e-6
and that unmerge restores `W`. The failure mode this catches: merging twice, or merging in
a lower dtype than the update was computed in.

**LoRA vs full fine-tuning: when would you still do full FT?** When the task needs
knowledge the base model lacks (new language, new modality), when data is plentiful
(> 100 k examples) and the compute exists, or when you need to change the tokenizer or
embeddings. For domain adaptation of a classifier on a few thousand examples, LoRA is the
default: cheaper, one base model shared across tasks, and the low-rank constraint acts as
a regulariser (which is a plausible reason it edged out full FT here).

**Memory: why is LoRA cheaper to *train*, not just to store?** Adam keeps two moments per
trainable parameter (8 bytes in fp32) plus the gradient. For 67 M params that is ~800 MB
of optimiser state; for 740 k LoRA params it is ~9 MB. Activations still have to be stored
for the backward pass through the frozen layers, which is why QLoRA adds 4-bit base
weights and paged optimisers to fit 65 B models on one GPU.

## Evaluation

**Why macro-F1 rather than accuracy?** The data is 61 % neutral. A model that predicts
"neutral" for everything scores 61 % accuracy and 25 % macro-F1. Macro-F1 weights the
13 % negative class — the one a bank actually cares about — equally.

**Why bootstrap confidence intervals?** A single accuracy on 340 sentences is a point
estimate with ±3–4 points of sampling noise. The percentile bootstrap resamples the test
set with replacement 1 000 times and reports the 2.5/97.5 percentiles; it needs no
distributional assumption and works for any metric, including macro-F1 which has no closed
form. `test_interval_shrinks_with_more_data` checks the width falls with n.

**Why McNemar rather than comparing two accuracies?** Two models scored on the *same*
examples are paired. Only the discordant pairs (A right/B wrong, A wrong/B right) carry
information about which is better; McNemar tests whether those counts are symmetric. An
unpaired comparison of two accuracies ignores that most examples are easy for both models
and has far less power. Exact binomial for small counts, χ² with continuity correction
otherwise.

**Rank 16 got p = 0.03 — is LoRA-16 better than full FT?** I would not claim it. Six
models were compared against the same baseline; at α = 0.05 one false positive in six is
expected. The effect (6 vs 0 discordant pairs, 1.8 points) is within the CI overlap. The
defensible statement is "LoRA r ∈ {4, 8, 16} is indistinguishable from full fine-tuning."

**What is expected calibration error and why report it?** Bin predictions by confidence;
within each bin compare average confidence with empirical accuracy; ECE is the
size-weighted mean gap. A classifier feeding a decision threshold, an expected-loss
calculation or a human-review trigger must be calibrated, not just accurate. All runs are
at 0.03–0.05 here; fine-tuned Transformers are often much worse (overconfident), in which
case temperature scaling on the validation set is the fix.

**What is in a run directory and why?** `config.json` (exact config after overrides),
`metrics.json` (parameter budget, split sizes and label distribution, per-epoch history,
test report, environment), `predictions.npz` (per-example probabilities and labels — needed
for any later paired test), `weights.safetensors` (only trained tensors). Anyone can
re-derive the table from these files without the model or GPU.

**How do you keep the test set clean?** Stratified split with a recorded seed, performed
once before any training; validation drives early stopping and model selection, the test
set is touched once per run to produce `predictions.npz`. The split sizes and class
distribution are written to `metrics.json` so a reviewer can check them.

## Model risk / governance angle

**How would you validate an LLM-based classifier under a model-risk framework?**
The same evidence this repo produces: a documented data lineage (source, license, subset,
split), a fixed and auditable test set, performance with uncertainty, comparison against a
simpler challenger model (the linear probe here), calibration, per-class behaviour on the
minority class, reproducibility (seeds, environment recorded), and monitoring hooks
(saved predictions for drift comparison). The things that differ from a scorecard are the
challenger set (a small fine-tuned model vs a prompted LLM) and the need for robustness
checks (paraphrase, negation, entity swaps), which would be the next test file to add.
