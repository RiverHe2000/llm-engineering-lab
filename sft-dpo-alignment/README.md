# sft-dpo-alignment — `sftdpo`

[![CI](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/sft-dpo-alignment-ci.yml/badge.svg)](https://github.com/RiverHe2000/llm-engineering-lab/actions/workflows/sft-dpo-alignment-ci.yml)

Supervised fine-tuning and then **Direct Preference Optimisation** of a 0.5 B instruct model
on a schema-constrained extraction task, where the preference label comes from a
**deterministic verifier** instead of a human or a judge model.

The project starts from a measured failure. A companion project of mine,
[`llm-app-ops-loop`](https://github.com/RiverHe2000/advice-ai-lab), has a prompt-regression
gate that refused to promote either of two candidate prompts, because both scored 0 out of 2
on the cases that required JSON output. Prompt engineering did not move it. This project
asks whether training does, and holds the answer to the same paired statistics.

The baseline says the problem is real. Qwen2.5-0.5B-Instruct, prompted with the full JSON
Schema, produces parseable JSON on 74.4 % of a 160-example test set — and a **schema-valid**
record on **23.1 %**, with an exact match rate of **0 %**.

```bash
pip install -e ".[dev,crosscheck]"

sftdpo pipeline smoke --out runs/smoke     # the whole experiment on a 9.8 M model, CPU, seconds
sftdpo crosscheck --tolerance 1e-5         # my DPO loss against the official TRL implementation
bash scripts/run_experiments.sh            # the real thing: one GPU, a few hours
```

---

## The idea

DPO needs pairs: for one prompt, a completion a human preferred and one they did not. Human
labels are the expensive part, and a judge model replaces them with a second model's
opinion — which is exactly the thing being measured, and which cannot be right about JSON
validity in a way that a parser is not more right about.

Here the task has a *checkable* answer. A completion either parses or it does not, satisfies
the schema or does not, and matches the gold record field by field or does not. So the
verifier scores completions, and the pairs fall out: sample k completions from the
fine-tuned model, score them, pair the best against the worst where the margin is large
enough. The preference signal is as trustworthy as the schema, which is to say exact.

That also gives the pipeline a natural end state worth reporting rather than hiding: **a
model good enough that almost every sample is perfect yields almost no pairs.** The mining
stage counts those saturated prompts, because the count says how much signal was left.

---

## What is built

**A task with a known answer.** A deterministic generator turns a seed into (adviser note,
gold `AdviceRecord`) pairs — eight top-level fields, three of them compound, thirteen
distinct names in all and three levels deep at `recommendations[i].product`, `extra="forbid"`,
in Australian wealth-management vocabulary. Six difficulty slices, each reported separately so
an average cannot hide a regression: clean, distractor (a superseded figure corrected later
in the note), mixed formats (dates and money written three ways), absent fields (the gold
really is empty — a test that the model invents nothing), long context, many items.

**A verifier that is the reward.** Robust JSON recovery with every repair named; schema
validation translated into a typed violation taxonomy with JSON paths; normalised field
comparison with the choices justified — money to the cent, `objectives` and `flags` as sets
because order carries no meaning, `recommendations` as an ordered list matched on product
name so a correct set in a different order is not scored as six errors. Two configurations
are kept: a **strict** verifier, which demands the completion be exactly one JSON object and
nothing else, and a **lenient** one, which permits repairs. The gap between the two scores is
reported, because "the model emits JSON if you allow a repair" and "the model emits JSON" are
different claims and only one of them lets you delete the repair step.

**Training written out rather than delegated.** A hand-written SFT loop (AdamW, cosine
schedule with warmup, gradient accumulation and clipping, prompt-masked loss, best-checkpoint
tracking) and a hand-written DPO loop, so the masking and the objective are visible rather
than inherited. The DPO loss is implemented from the paper with three variants — sigmoid,
IPO and cDPO — each derived in its docstring.

**A reference policy that costs no memory.** `peft_model.disable_adapter()` reproduces the
base model's log-probabilities exactly, so the reference policy is the same weights with the
adapter switched off. DPO here carries no second copy of the model.

**The whole experiment as nine resumable stages.** Data, evaluate base, SFT, evaluate,
sample and mine, DPO, evaluate, gate, report. Every stage writes files and a stage whose
files exist is skipped, because a multi-hour run on one GPU will be interrupted. The run
directory records the settings that produced it, including the decoding batch sizes.

---

## Two things the cross-check found

`sftdpo crosscheck` compares this package's DPO loss against `trl` 1.12.0 on identical
inputs. They agree to **1.07e-14** across all three variants — but getting there surfaced two
differences worth knowing before quoting a number against a paper, and both are recorded
rather than smoothed over:

1. **TRL's IPO is length-normalised and the paper does not say so.** TRL divides each side's
   log-ratio by its completion token count; its own source attributes the choice to
   correspondence with the IPO authors. This implementation takes four log-probabilities and
   no lengths, so it cannot do that and does not pretend to. The two agree exactly when the
   caller passes per-token averages, and the residual is reported rather than asserted away.
2. **TRL 1.12.0 has no cDPO.** Its `label_smoothing` now feeds Robust DPO, which is a
   different formula, not a rescaling: cDPO takes the likelihood of a noisy label,
   rDPO takes an unbiased estimator. The smoothed likelihood survives inside TRL's `aot`
   loss, where sorting is the identity for a single pair, so cDPO is checked against that.

---

## Results

Supervised fine-tuning took schema-valid output from **23.1 % to 83.1 %** on 0.88 % of the
weights, and made the JSON repair step in front of the model unnecessary — 23 completions
needed one before, none after.

The first preference run then **destroyed the model**, and that is the more useful result.
Reward accuracy reached 1.000, the loss fell to 0.0014 and the margin grew from 0.8 to 7.5,
while strict JSON validity went 0.981 → **0.000**. The implicit rewards say why in one line:
the rejected completions' reward fell to −7.15 while the chosen stayed near zero, so the
whole margin was bought by crushing the rejected side rather than lifting the chosen one —
the exact failure `dpo_loss.py` names as the reason it reports two rewards instead of one
margin. The promotion gate rejected it on every slice. The cause was a learning rate shared
with the supervised stage; the fix is a separate one, and the failed run's artefacts are
committed rather than deleted.

The corrected run took schema validity to **98.8 %**, and the aligned 0.5 B beat a prompted
**Qwen3-4B** — eight times its size — on every metric (schema validity 0.988 against 0.763,
paired difference +0.225 [+0.163, +0.294], McNemar p = 2.8e-10). Every gate said promote. One
number disagreed: exact match fell 0.438 → 0.050. Following it up found that the preference stage
had taught the model to **stop emitting the optional `flags` array entirely** — 0 of 160
records, flag recall 0.610 → 0.000 — and that this *raised* the reward, because omitting an
optional field is always schema-valid while getting it wrong is not. Nothing malfunctioned;
the verifier measured what it was told to measure. No aggregate over records could catch a
field that vanished from every record, and no slice regressed. `EvalReport` now carries
per-field coverage and the gate carries `max_field_recall_drop`, which turns that promotion
into a **REJECT** naming the field and the number.

See [`docs/RESULTS.md`](docs/RESULTS.md) for all of it: base against SFT against DPO, per
slice, with paired intervals and the gates; the mining yield and what the rejected
completions were being taught to stop doing; the field-deletion finding in full; the
comparison against much larger models that were only prompted; and the alignment tax. Design
decisions and their trade-offs are in [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md).

---

## Engineering standard

| Gate | Result |
|---|---|
| Lint and format | `ruff` (pinned 0.16.6), broad rule set, line length 100 |
| Types | `mypy --strict` over `src/` **and** `tests/` |
| Tests | **1 709 tests, 99.6 % branch coverage**, CPU, offline, no downloads |
| CI | Python 3.12 and 3.13; the TRL cross-check; the whole pipeline end to end on a 9.8 M-parameter model |

The CI model is a two-layer Qwen2 built from a config — 9.8 M parameters, no download, and a
real execution of every stage rather than a mock of it. Real weights are needed only for the
numbers in `docs/RESULTS.md`.

```bash
make all                     # ruff + mypy + pytest
make smoke                   # the whole pipeline in seconds on CPU
make crosscheck              # against TRL
```

---

## Relation to the other projects here

[`lora-finetune-eval`](../lora-finetune-eval/) implements LoRA from first principles and
compares fine-tuning strategies on an **encoder** doing classification, with bootstrap
intervals and McNemar tests. This project is the generative counterpart: a **decoder**, an
alignment stage after the supervised one, and a preference signal produced by a verifier
rather than by labels. [`transformer-from-scratch`](../transformer-from-scratch/) builds the
model, [`llm-inference-server`](../llm-inference-server/) serves it; this one changes its
behaviour and measures whether the change survived a paired test.
