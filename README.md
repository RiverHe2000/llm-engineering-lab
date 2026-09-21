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
| 01 | [transformer-from-scratch](transformer-from-scratch/) — `nanoformer` | Decoder-only Transformer (RMSNorm, RoPE, GQA, SwiGLU, KV cache) + byte-level BPE + AMP trainer with bit-exact resume | 9.4 M params → **val ppl 27.3** on Tiny Shakespeare in 65 s on one RTX 4070; **131 tests, 98.8 % coverage** |
| 02 | [lora-finetune-eval](lora-finetune-eval/) — `loraeval` | LoRA implemented from first principles, three fine-tuning strategies, bootstrap CIs, McNemar paired tests, calibration, auditable run artefacts | LoRA r = 8 (1.1 % trainable) **matches full fine-tuning** on Financial PhraseBank: 95.9 % vs 94.4 %, p = 0.125; **83 tests, 98 % coverage** |
| 03 | [llm-inference-server](llm-inference-server/) — `llmserve` | KV-cached batched generation with left padding and row eviction, async dynamic batching, INT8, Prometheus metrics, FastAPI, Docker | Dynamic batching: **29× throughput at batch 32 for +7 % latency** (Qwen2.5-0.5B); 16 concurrent requests in 2.4 s vs 23 s sequential; **68 tests, 96 % coverage** |
| 04 | [sft-dpo-alignment](sft-dpo-alignment/) — `sftdpo` | LoRA supervised fine-tuning then DPO written from the paper (sigmoid/IPO/cDPO), with the preference label supplied by a deterministic verifier instead of a human or a judge model; paired statistics and a promotion gate with floors | Qwen2.5-0.5B on a schema-constrained extraction task: **schema-valid output 23.1 % → 83.1 %** after supervised fine-tuning and **98.8 %** after alignment, on 4.4 M trainable parameters (0.88 %), and the JSON repair step in front of the model becomes unnecessary (23 → **0** completions repaired). The aligned 0.5 B beats a prompted **Qwen3-4B** on every metric (**0.988 vs 0.763**, +0.225 paired [+0.163, +0.294]) — though the 4 B is perfect on four of six slices and collapses only where the input is long or the record large. Two failures are the write-up's real subject: a preference run that **destroyed the model** while every training metric looked healthy, and a later run that passed every gate rule while **deleting an optional field from all 160 records** because the reward made silence cheaper than being right 83 % of the time. The gate now compares recall per field and rejects it. My DPO loss agrees with TRL to **1.07e-14**; **1 709 tests, 99.6 % coverage** |

Companion repositories: [`genai-platform-lab`](https://github.com/RiverHe2000/genai-platform-lab)
(RAG, agents with guardrails, an LLM gateway) and [`mlops-lab`](https://github.com/RiverHe2000/mlops-lab)
(MLflow lifecycle, SageMaker deployment, drift monitoring).

---

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
