# llm-engineering-lab

[![transformer-from-scratch](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/transformer-from-scratch-ci.yml/badge.svg)](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/transformer-from-scratch-ci.yml)
[![lora-finetune-eval](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/lora-finetune-eval-ci.yml/badge.svg)](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/lora-finetune-eval-ci.yml)
[![llm-inference-server](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/llm-inference-server-ci.yml/badge.svg)](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/llm-inference-server-ci.yml)
[![sft-dpo-alignment](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/sft-dpo-alignment-ci.yml/badge.svg)](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/sft-dpo-alignment-ci.yml)

Four self-contained projects covering the life-cycle of a Transformer language model —
**build it, adapt it, serve it, align it** — each written to the standard I would hold
production code to: typed, linted, tested for mathematical *properties*, reproducible, and
documented with the trade-offs made.

| # | Project | What it demonstrates | Headline result |
|---|---|---|---|
| 01 | [transformer-from-scratch](transformer-from-scratch/) | Transformer, tokenizer and resumable trainer from first principles | 9.4 M non-embedding parameters; validation perplexity **27.3** on Tiny Shakespeare, one RTX 4070. [Evidence](transformer-from-scratch/README.md#2-results) |
| 02 | [lora-finetune-eval](lora-finetune-eval/) | Parameter-efficient adaptation with paired evaluation | **1.1% trainable parameters**; 95.9% vs 94.4% full-FT accuracy on 340 Financial PhraseBank examples. No significant difference detected; equivalence is unproven. [Evidence](lora-finetune-eval/docs/RESULTS.md) |
| 03 | [llm-inference-server](llm-inference-server/) | Correct batched generation, back-pressure and serving observability | **29× throughput** comparing batch 32 with batch 1 on Qwen2.5-0.5B and one RTX 4070. This is an eager-PyTorch batching comparison, not a vLLM comparison. [Evidence](llm-inference-server/docs/BENCHMARK.md) |
| 04 | [sft-dpo-alignment](sft-dpo-alignment/) | Detecting reward hacking with field-level release checks | DPO reached **98.8% schema validity** while deleting all optional flags; exact match fell **43.8% → 5.0%**. The corrected gate **rejects** the model. [Failure analysis](sft-dpo-alignment/docs/RESULTS.md) |

Companion repositories: [`genai-platform-lab`](https://github.com/RiverHe2000/genai-platform-lab)
(RAG, agents with guardrails, an LLM gateway) and [`mlops-lab`](https://github.com/RiverHe2000/mlops-lab)
(MLflow lifecycle, SageMaker deployment, drift monitoring).

---

## Start here

For an applied LLM role, start with the SFT/DPO failure analysis and its CPU smoke command.
For a serving role, start with the inference benchmark and its documented limits. The
committed GPU reports are historical experiments; offline CI checks implementation and
reproducibility contracts, and does not rerun those model-quality claims.

## Why these four

* **01 answers "do you understand the model?"** — every block is written, not imported, and
  every block has a test that checks a *property* (causality, RoPE relative-position
  invariance, cache/no-cache equivalence, closed-form parameter counts, bit-exact resume),
  not just a shape.
* **02 answers "can you adapt it and prove the result?"** — the LoRA maths is implemented
  and verified (merge/unmerge, zero-init equivalence), and the evaluation is the kind a
  model-validation function would accept: intervals, paired significance, calibration, saved
  predictions, recorded splits.
* **03 answers "can you run it for other people?"** — the concerns that appear only in
  production: padding correctness in batches, cache eviction, back-pressure, timeouts,
  readiness vs liveness, correlation ids, metrics, and a benchmark that quantifies the
  batching trade-off.
* **04 answers "can you change what it does, and show the change survived a test?"** — the
  objective is derived and implemented rather than imported, cross-checked against the
  official TRL implementation to 1e-14, and the result is reported with a paired interval,
  an exact McNemar test, per-slice non-regression and an explicit deployability floor. The
  two memory findings that came out of running it for real are written down as findings.

---

## Engineering standard (identical across the four)

| Gate | Tooling |
|---|---|
| Lint + format | `ruff` (E, F, W, I, N, UP, B, SIM, C4, PT, RUF, PIE, RET, ARG) |
| Types | `mypy --strict` on `src/` **and** `tests/` |
| Tests | `pytest` with branch-coverage gates; every test runs offline on CPU in seconds using randomly initialised tiny models and in-memory tokenizers |
| Reproducibility | explicit seeds and RNG ownership; project 01 checkpoints *every* RNG so resume is bit-exact; project 02 records split sizes and label distributions per run |
| CI | one workflow per project, path-filtered, on Python 3.12 and 3.13 with CPU-only torch and `HF_HUB_OFFLINE=1` so a network dependency in a test is a failure |
| Docs | each project: README with results, `docs/INTERVIEW_NOTES.md`, generated result files |

```bash
python -m venv .venv && source .venv/bin/activate    # .venv\Scripts\activate on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
make install          # editable install of all four, dev extras
make all              # ruff + mypy + pytest for all four (what CI runs)
make PROJECT=lora-finetune-eval test
```

Hardware for the reported numbers: one NVIDIA RTX 4070 (12 GB), Windows 11, PyTorch 2.11
(cu128), transformers 5.16.

---

## Layout

```
llm-engineering-lab/
├── transformer-from-scratch/   nanoformer: model, tokenizer, trainer, generation, CLI
├── lora-finetune-eval/         loraeval: LoRA, strategies, metrics, experiment runner, CLI
├── llm-inference-server/       llmserve: engine, scheduler, API, quantization, benchmark, Docker
├── sft-dpo-alignment/          sftdpo: task, verifier-reward, SFT, preference mining, DPO, gates
├── .github/workflows/          one path-filtered CI workflow per project
└── Makefile                    install / lint / type / test / all
```

Each project is independently installable and has its own README — start with the one
closest to the role you are hiring for.
