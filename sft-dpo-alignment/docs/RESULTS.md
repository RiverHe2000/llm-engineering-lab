# Results

Every number here is produced by `bash scripts/run_experiments.sh` and traces to a file under
[`docs/experiments/`](experiments/). Hardware: one NVIDIA RTX 4070 (12 GB), Windows 11,
PyTorch 2.11 (cu128), transformers 5.16, peft 0.20, trl 1.12.

Configuration of the run reported below: Qwen2.5-0.5B-Instruct, LoRA r = 8 on the seven
standard Qwen projections (**4.40 M trainable, 0.883 % of the model**), 400 training / 80
validation / 160 test examples stratified across the six slices, 3 supervised epochs,
k = 5 sampled completions per prompt at temperature 0.9, 1 preference epoch.

---

## 1. The defect this project starts from

A 0.5 B instruct model, prompted with the task and the full JSON Schema, on 160 held-out
examples:

| | Base model, prompted |
|---|---:|
| Parseable JSON (strict) | 0.7438 |
| **Schema-valid record** | **0.2313** |
| Field F1 against gold | 0.5265 |
| Exact match | 0.0000 |

Three in four completions are JSON; fewer than one in four is a record the schema accepts.
That gap is the whole problem: the output *looks* structured and is not usable. Prompt
engineering did not close it — this is the same floor a promotion gate in my
[`llm-app-ops-loop`](https://github.com/ChuanHe-PhD/advice-ai-lab) project refused to promote
either candidate prompt past.

Where the base model fails, counted by the verifier's taxonomy over 160 completions
(one headline violation each):

| Violation | Count |
|---|---:|
| `missing_field` | 44 |
| `unparseable` | 41 |
| `bad_enum` | 30 |
| `out_of_range` | 3 |
| `wrong_type` | 3 |
| `extra_field` | 2 |

---

## 2. Supervised fine-tuning

300 optimiser steps, 227 844 supervised tokens, final training loss 0.00056, best validation
loss 0.00083.

| | Base | **After SFT** |
|---|---:|---:|
| Parseable JSON (strict) | 0.7438 | **0.9750** |
| **Schema-valid record** | 0.2313 | **0.8313** |
| Field F1 | 0.5265 | **0.8914** |
| Exact match | 0.0000 | **0.4375** |
| Mean verifier reward | 0.5109 | **0.8961** |

**Schema validity 0.231 → 0.831 on 4.4 M trainable parameters**, and exact match — every one
of the thirty-odd fields right, in one object — from 0 to 0.438. The violation profile says
what was learned rather than only how much:

| Violation | Base | After SFT |
|---|---:|---:|
| `bad_enum` | 30 | **0** |
| `unparseable` | 41 | 4 |
| `missing_field` | 44 | 8 |
| `out_of_range` | 3 | 5 |
| `wrong_type` | 3 | 10 |
| `extra_field` | 2 | **0** |

The enum errors are gone completely: the base model invented risk-profile and action labels,
and the fine-tuned model uses the five and four the schema allows. What remains is a
different, smaller problem — a handful of missing fields and out-of-range numbers — which is
the kind of residue a preference stage might reach.

### The repair step can be deleted

| | Strict parse | Lenient parse | Completions repaired |
|---|---:|---:|---:|
| Base | 0.7438 | 0.8875 | **23** |
| After SFT | 0.9750 | 0.9750 | **0** |

The base model needs a repair step in front of it — fenced blocks, preambles, trailing
prose — and 23 of 160 completions depend on one. After fine-tuning the strict and lenient
rates are identical and nothing is repaired, so the deployment can drop that stage and the
strict rate becomes the honest number to quote. This is the difference between "the model
emits JSON if you allow a repair" and "the model emits JSON."

### Per slice, all six

| Slice | n | JSON valid | Schema valid | Field F1 | (base schema) |
|---|---:|---:|---:|---:|---:|
| clean | 27 | 1.0000 | 0.9630 | 0.9939 | 0.2593 |
| distractor | 27 | 0.9259 | 0.7407 | 0.8715 | 0.2963 |
| mixed_formats | 27 | 1.0000 | 0.7407 | 0.9093 | 0.2222 |
| absent_fields | 27 | 1.0000 | 0.9630 | 0.8896 | 0.5185 |
| long_context | 26 | 0.9231 | 0.7692 | **0.7137** | 0.0000 |
| many_items | 26 | 1.0000 | 0.8077 | 0.9664 | 0.0769 |

The spread is the reason the slices are reported rather than averaged, and the two extremes
are the two that a four-row table would have left out.

`long_context` has the **worst field F1 of any slice, 0.714**, and the lowest JSON validity
too. The model has to produce a well-formed record from 600 words of meeting narrative, and
where it manages that it still gets some of the contents wrong — the failure is moving from
*format* to *reading*, and no amount of further format training will touch the second half.
That is the single most useful thing this table says, and the mean of 0.891 hides it.

`many_items` goes from the **worst** slice under prompting (schema validity 0.077, field F1
0.071 — the base model essentially cannot produce a record with six recommendations in it) to
0.808 and 0.966. That is the largest movement on the page, and it says the base model's
failure there was structural rather than a reading problem.

`mixed_formats` and `distractor` share the lowest schema validity at 0.741 for opposite
reasons: the first gets the content right and the date or number format wrong (field F1
0.909), the second reads the superseded figure the note corrects later (field F1 0.872).
Those two are the split the two metrics exist to show.

---

## 3. Preference mining

The fine-tuned model sampled five completions per prompt at temperature 0.9 over the 400
training prompts, and the verifier scored all 2 000 and paired them.

| | |
|---|---:|
| Completions sampled | 2 000 |
| Prompts with at least one usable pair | 377 of 400 (**94.3 %**) |
| Pairs mined | **671** |
| Mean reward margin | 0.384 |
| Prompts where every sample scored the same | 16 |
| Prompts saturated at a perfect score | 1 (**0.25 %**) |

The saturation count is the number worth reading first, because it is the one that predicts
whether preference optimisation has anything left to work with. One prompt in four hundred
was solved perfectly five times out of five; the model is nowhere near the ceiling, which is
consistent with the 83.1 % schema validity the supervised stage reached.

**What the rejected side is teaching the model to stop doing**, counted over the losing
completion of each pair:

| Violation on the rejected completion | Count |
|---|---:|
| `unparseable` | 145 |
| `extra_field` | 144 |
| `missing_field` | 142 |
| `wrong_type` | 81 |
| `out_of_range` | 28 |
| `bad_enum` | 4 |
| no schema violation, lower field accuracy | 127 |

That last row is the interesting one. A fifth of the pairs are between two schema-valid
records, so the preference signal is not only about format: it also carries which of two
well-formed answers read the note better.

### The margin was calibrated against a record shape the corpus never produces

The default minimum margin was 0.05, chosen so that "one wrong field is enough to make a
pair". A field is worth `0.6 / n` where `n` is the gold record's leaf-path count, and the
figure came from a ten-path record — but the generator's narrowest record has **13** paths
and its widest **35**, so a single wrong field is worth between 0.0171 and 0.0462 and cleared
the old margin on **none** of the 640 examples. The filter was not conservative; it was above
the entire distribution, and the mining stage could not produce a single-field pair at all.

Re-mining the same 2 000 completions at the corrected 0.015:

| | margin 0.05 | margin 0.015 |
|---|---:|---:|
| Pairs | 567 | **671** |
| Prompt yield | 0.868 | **0.943** |
| Mean margin | 0.449 | 0.384 |

104 more pairs, and the mean margin falls because the finer distinctions the old value
excluded are exactly the ones now included. A default calibrated against a fixture rather
than against the data is worse than an arbitrary one, because it looks principled.

---

## 4. Direct preference optimisation, and the run that destroyed the model

The first preference run reached a **reward accuracy of 1.000**, a final loss of **0.0014**
and a reward margin that grew steadily from 0.80 to 7.46. Every quantity a training loop
prints looked healthy.

It produced a model that emits **no valid JSON at all**: strict validity 0.981 → **0.000**,
all 160 test completions unparseable, field F1 0.847 → 0.000. It is far worse than the
untrained base model it started from.

*(Those two SFT figures, and the gate table below, were measured by the evaluator as it stood
at the time; the adversarial-review fixes described in section 8 changed it, and the rest of
this page was regenerated in one pass afterwards. They are quoted here because both sides of
this particular comparison came from that evaluator, so it is internally consistent — and
because the collapsed adapter no longer exists to re-score. Nothing in the collapse argument
turns on the third decimal place of a number that went to zero.)*

### The diagnostic said exactly what happened

The implicit rewards are logged separately for a reason, and this is it:

| Step | Chosen reward | Rejected reward | Margin | Accuracy |
|---:|---:|---:|---:|---:|
| 10 | +0.93 | +0.13 | 0.80 | 0.50 |
| 50 | +1.61 | −2.61 | 4.22 | 1.00 |
| 90 | −3.23 | −4.58 | 1.35 | 0.50 |
| 130 | −1.42 | −9.69 | 8.28 | 1.00 |
| 168 | +0.31 | **−7.15** | 7.46 | 1.00 |

The margin was bought entirely by **crushing the rejected completions**, not by lifting the
chosen ones: the rejected reward fell by more than seven nats while the chosen reward
wandered around zero. The policy moved that far from the reference by degrading, and what it
degraded was the ability to produce the format at all.

This failure mode is named in `dpo_loss.py`'s own docstring, written before the run:

> Their individual values say whether a run is lifting the chosen response or merely
> crushing the rejected one; `h` alone cannot tell those apart, and the second is the failure
> mode that quietly degrades a model while the loss curve looks healthy.

The reason the module takes four log-probabilities instead of one margin is to make that
distinguishable. It was, and the two-column table above is the whole argument for the design.

### The gate caught it

| Comparison | Decision | Why |
|---|---|---|
| SFT → DPO | **REJECT** | every one of the six slices regressed, worst 0.9615; CI lower bound −0.8812 |
| base → DPO | **REJECT** | still worse than the untrained model: five slices regressed, CI lower bound −0.2000 |

A gate that only compared means would have had to argue about −0.88; the per-slice
non-regression rule rejected on the first slice it looked at. The exit code is 1, so a CI
step gating on this stops the release without anyone reading the table.

### The cause was a shared learning rate

The preference stage inherited the supervised one, 1e-4. That is an ordinary rate for
supervised fine-tuning and a very large one for DPO, which is a small correction to a model
that already works rather than a fresh objective to be minimised. `PipelineConfig` now
carries `dpo_learning_rate` separately, defaulting to 1e-5, the same way it carries a separate
`dpo_batch_size` — a preference step puts four sequences through a model where a supervised
step puts one, and the two stages had no business sharing either setting.

The collapsed run's evidence is kept under
[`collapsed-dpo/`](experiments/collapsed-dpo/) — its reward log, training log, evaluation
report and both rejected gates — because a negative result that is deleted is not a result.

### The corrected run

Same data, same 168 steps, same 671 pairs, `dpo_learning_rate = 1e-5`:

| Step | Chosen reward | Rejected reward | Margin | Accuracy |
|---:|---:|---:|---:|---:|
| 10 | +4.67 | +4.61 | 0.06 | 0.50 |
| 50 | +4.56 | +3.56 | 1.01 | 1.00 |
| 90 | +3.04 | +3.37 | −0.33 | 0.50 |
| 130 | +3.46 | +2.35 | 1.11 | 0.75 |
| 168 | **+3.92** | **+1.46** | 2.46 | 1.00 |

Both implicit rewards stay positive for all 168 steps, and the margin is bought by holding
the chosen completions up rather than by pushing the rejected ones down. The reference is the
*base* checkpoint — `reference_context` switches the adapter off, and the adapter being
trained is the supervised one — so the +4.67 at step 10 is mostly what supervised fine-tuning
had already earned before preference training moved anything. That is what makes the collapsed
run's step 10 legible in hindsight: at the same point it read **+0.93**, nearly four nats
below where SFT had left it. It had spent the supervised stage's entire head start inside
forty pairs.

The margin ends at 2.46 rather than 7.46, the slope is 0.0089 rather than 0.0378, and the
final loss is 0.1607 rather than 0.0014. A DPO loss near zero means every pair is already
ranked with certainty, and nothing in the objective stops the policy moving further once it
is — so the healthy run is the one that did *not* saturate.

**The two runs' summary files are identical except for those two numbers.** Both record 168
steps, 671 pairs, 219 852 supervised tokens, reward accuracy 0.5 → 1.0, `diverged: false`,
and the same data hash. Every field a training dashboard would show is the same for the run
that worked and the run that destroyed the model. Only `reward_log.jsonl`, which keeps the
two implicit rewards apart instead of reporting their difference, tells them apart.

### What it produced

| Metric | base | +SFT | +DPO |
|---|---:|---:|---:|
| JSON valid (strict) | 0.7438 | 0.9750 | **1.0000** |
| Schema valid | 0.2313 | 0.8313 | **0.9875** |
| Mean field F1 | 0.5265 | 0.8914 | **0.9133** |
| Mean reward | 0.5109 | 0.8961 | **0.9455** |
| Exact match | 0.0000 | **0.4375** | 0.0500 |
| Completions needing a repair | 23 | 0 | **0** |

Schema validity per slice, which is what the gate pairs on:

| Slice | n | base | +SFT | +DPO |
|---|---:|---:|---:|---:|
| clean | 27 | 0.259 | 0.963 | 0.963 |
| distractor | 27 | 0.296 | 0.741 | **1.000** |
| mixed_formats | 27 | 0.222 | 0.741 | **1.000** |
| absent_fields | 27 | 0.519 | 0.963 | **1.000** |
| long_context | 26 | 0.000 | 0.769 | **1.000** |
| many_items | 26 | 0.077 | 0.808 | 0.962 |

Preference optimisation finished what the supervised stage started, and it finished it where
the supervised stage was weakest: `mixed_formats` and `distractor` were the two lowest slices
after SFT at 0.741 and are both 1.000 after DPO. `long_context` goes from a model that never
once produced a schema-valid record, through 0.769, to 1.000.

| Comparison | Decision | Evidence |
|---|---|---|
| base → SFT | **PROMOTE** | +0.6000 [+0.5186, +0.6813], McNemar b/c = 4/100, p = 4.72e−25 |
| base → DPO | **PROMOTE** | +0.7562 [+0.6875, +0.8187], McNemar b/c = 0/121, p = 7.52e−37 |
| SFT → DPO | **REJECT** | +0.1562 [+0.1000, +0.2125], p = 4.17e−07 — and rejected anyway |

The third row is not a typo, and it is the most useful line on this page.

Artefacts: [`eval_base.json`](experiments/eval_base.json),
[`eval_sft.json`](experiments/eval_sft.json), [`eval_dpo.json`](experiments/eval_dpo.json),
[`compare_base_sft.md`](experiments/compare_base_sft.md),
[`compare_base_dpo.md`](experiments/compare_base_dpo.md),
[`compare_sft_dpo.md`](experiments/compare_sft_dpo.md),
[`dpo_reward_log.jsonl`](experiments/dpo_reward_log.jsonl),
[`dpo_summary.json`](experiments/dpo_summary.json).

---

## 5. The promotion that should not have been one

At the time that run finished, the gate had four rules: two absolute floors, a per-slice
non-regression rule, and two significance tests that must agree. SFT → DPO cleared every one
of them — schema validity +0.1562 with a confidence interval excluding zero, McNemar
p = 4.17e−07, no floor breached, all six slices up or level — and the gate said **PROMOTE**.

One number disagreed. Exact match fell **0.4375 → 0.0500** — 70 perfect records down to 8.
**62 examples lost it and not one gained it**, and on those 62 the mean field F1 went from
1.000 to 0.967: each was now wrong in about one field out of thirty. Exact match is not what
the gate pairs on, so nothing was blocking. Pulling that thread produced the most useful
result in this project.

The evaluation report stored grades, not completions, so finding out *what* changed meant
regenerating those examples under both adapters and diffing against gold. Every difference was
in one field, in one direction:

```
test-clean-0003       flags[0]  gold 'centrelink_means_test_impact'   sft same  dpo absent
test-distractor-0007  flags[0]  gold 'capital_gains_on_switch'        sft same  dpo absent
test-many_items-0012  flags[1]  gold 'transfer_balance_cap_check...'  sft same  dpo absent
```

So the whole test split was regenerated under both adapters and the `flags` array counted
directly — `python scripts/field_coverage_probe.py --field flags`, whose output is
[`flags_hedge.json`](experiments/flags_hedge.json):

| | +SFT | +DPO |
|---|---:|---:|
| Records emitting at least one flag | 111 / 160 | **0 / 160** |
| Records with no `flags` key at all | 38 / 156 parsed | **160 / 160** |
| Flags emitted in total | 147 | **0** |
| Flag recall (of 200 gold flags) | 0.610 | **0.000** |
| Flag precision | 0.830 | — |

**The preference stage taught the model to delete a field.** Not to get it wrong — to stop
emitting it, on every record in the split.

And the reward went *up*, which is the part that matters. `flags` is optional, so omitting it
is always schema-valid, while emitting it risks a bad enum member, a wrong type, or a flag
gold did not ask for. The SFT model was right about `flags` 83 % of the times it tried, which
is worse than it did on the record as a whole — so under a reward of 0.2 parse + 0.2 schema +
0.6 field F1, the arithmetic favours silence. The preference pairs were labelled by exactly
that reward, and DPO found exactly what they rewarded. Nothing malfunctioned: **the verifier
reported what it measured, and it did not measure whether a field still existed.**

Nothing in the gate could catch it either. Mean F1 rose, mean reward rose, schema validity
rose, and all six slices rose. An average over records cannot see a field that vanished from
every record, because the loss is spread evenly across all of them and is more than paid for
by the errors that vanished with it.

### The rule that was added, and what it changed

`EvalReport` now carries a `per_field` block — expected, emitted and correct path counts per
top-level field, aggregated from the `FieldScore`s the reward already produced, so it costs no
extra model call — and `Floors` carries `max_field_recall_drop` (0.10) alongside
`min_field_support` (10 gold paths), so a rarely-asked-for field cannot block a release on
noise. The block is now in every committed evaluation report, and it says the whole story in
eight rows:

| Field | Gold paths | +SFT recall | +DPO recall | Δ |
|---|---:|---:|---:|---:|
| client_name | 160 | 0.9750 | 1.0000 | +0.0250 |
| record_date | 160 | 0.9313 | 0.9875 | +0.0563 |
| risk_profile | 160 | 0.9688 | 1.0000 | +0.0312 |
| objectives | 411 | 0.9562 | 1.0000 | +0.0438 |
| recommendations | 1513 | 0.8387 | 0.8691 | +0.0304 |
| fees | 320 | 0.8938 | 1.0000 | +0.1062 |
| review_months | 160 | 0.9375 | 1.0000 | +0.0625 |
| **flags** | 200 | **0.6100** | **0.0000** | **−0.6100** |

Seven fields improved. One was deleted. With that block present the same two checkpoints over
the same test split give the opposite decision:

```
Decision: REJECT
- field 'flags' lost 0.6100 of its recall (0.6100 to 0.0000 over 200 gold paths),
  tolerance 0.1000
```

Same models, same split, same statistics — a different answer, because the report now contains
the fact the decision needed. **The gate was never wrong about what it was looking at. It was
looking at the wrong thing, and it took a metric it did not gate on to notice.**

### The gate still promotes base → DPO, and it is right to

Worth sitting with, because it looks like an inconsistency and is not. The base model's flag
recall is **0.000** — it emitted two flags across 160 records and got neither right. Against
that baseline the DPO model loses nothing, so no field rule fires and the promotion stands on
+0.7562 schema validity.

The gate's answer therefore depends on what is being replaced, and both answers are correct.
Replacing the prompted base model with the aligned one is a large improvement in every
respect. Replacing the *supervised* model with the aligned one buys +0.156 schema validity and
pays for it by dropping a field the supervised model could do something with. Whether that
trade is worth making is a product decision — but it is now a decision someone is asked to
make, rather than one made silently by an average.

### Three failures, and only one of them is a bug

* The **collapse** was a misconfiguration, caught by an instrument built before the run.
* The **field deletion** was not a bug anywhere. Every component did what it was specified to
  do. It is a specification defect, and it propagated cleanly: reward → preference data →
  policy → gate, each step faithfully carrying the same wrong assumption, that a missing
  optional field is cheap.
* The **gate** was incomplete in a way only the second could reveal, because its per-slice
  rule was built for losses that concentrate and this loss was perfectly uniform.

Buying preference labels from a verifier removes the annotator, not the specification problem.
It moves it into code — where it can be tested, versioned and gated. That is the argument for
doing it this way, and this section is the evidence for both halves of it.

---

## 6. Fine-tune a small model, or prompt a large one?

The decision the project exists to inform, on one test set with one verifier. Every model
below answered the same 160 examples; the two prompted models got the same prompt, schema
included, that the base model got.

| | 0.5 B prompted | 1.5 B prompted | **4 B prompted** | **0.5 B fine-tuned** |
|---|---:|---:|---:|---:|
| Parameters | 0.5 B | 1.5 B | 4 B | 0.5 B + 4.4 M |
| JSON valid (strict) | 0.7438 | 0.8125 | 0.7625 | **1.0000** |
| Schema valid | 0.2313 | 0.3688 | 0.7625 | **0.9875** |
| Mean field F1 | 0.5265 | 0.5544 | 0.7122 | **0.9133** |
| Mean reward | 0.5109 | 0.5689 | 0.7323 | **0.9455** |
| Completions needing a repair | 23 | 15 | 12 | **0** |

**A fine-tuned 0.5 B beats a prompted model eight times its size on every metric**, and the
gate agrees: +0.2250 schema validity [+0.1625, +0.2938], McNemar b/c = 1/37, p = 2.84e−10,
decision **PROMOTE**. Against the 1.5 B the difference is +0.6188 [+0.5437, +0.6937].

Scale does help — 0.231 → 0.369 → 0.763 schema validity is a real curve, and Qwen3-4B is a
much better instruction follower than either Qwen2.5 model. But the per-slice breakdown says
where its remaining failure lives, and it is not spread evenly:

| Slice | n | 4 B prompted | 0.5 B fine-tuned |
|---|---:|---:|---:|
| clean | 27 | **1.0000** | 0.9630 |
| distractor | 27 | 1.0000 | 1.0000 |
| mixed_formats | 27 | 1.0000 | 1.0000 |
| absent_fields | 27 | 1.0000 | 1.0000 |
| long_context | 26 | 0.4615 | **1.0000** |
| many_items | 26 | 0.0769 | **0.9615** |

The 4 B is **perfect on four slices out of six** — better than the fine-tuned model on
`clean`, in fact — and then falls off a cliff on the two that make the output long: 0.462 when
the note runs to 600 words, 0.077 when the record needs six recommendations in it. Its whole
deficit is there. **38 of its 160 completions do not parse strictly at all**, and a repair step
recovers 12 of those — fenced blocks and preambles — which is the failure mode a schema in the
prompt is supposed to prevent and does not, at any of the three sizes tested.

That is a more useful answer than the aggregate. If the inputs are short and the records
small, prompting a 4 B is competitive and needs no training run. If either grows — and in this
domain both do — the small fine-tuned model is not slightly better, it is the difference
between 0.08 and 0.96.

### What alignment cost

Training a small model hard on one output format is expected to damage its general
instruction-following, so a held-out set of generic instructions with deterministic checkable
properties is run before and after:

| | Before | After |
|---|---:|---:|
| Probes passed | 7 / 12 (0.5833) | 5 / 12 (0.4167) |

Difference −0.1667, 95 % CI [−0.4167, +0.0000], probes lost/gained 2/0, McNemar p = 0.5000,
**outside** the 0.05 non-inferiority margin.

The honest reading is that this measurement is underpowered rather than reassuring. Twelve
probes cannot separate a real 17-point regression from two unlucky flips: the interval reaches
from a catastrophic loss to no loss at all, and the exact test cannot reject chance. What can
be said is that the point estimate is negative, that no probe improved, and that the number is
reported next to the task gains rather than omitted. Sizing this probe set to the point where
it could actually decide is the clearest piece of unfinished work on this page.

Artefacts: [`eval_mid.json`](experiments/eval_mid.json),
[`eval_big.json`](experiments/eval_big.json),
[`compare_big_dpo.md`](experiments/compare_big_dpo.md),
[`compare_mid_dpo.md`](experiments/compare_mid_dpo.md),
[`alignment_tax.md`](experiments/alignment_tax.md), [`tax.json`](experiments/tax.json), and
the whole run in [`report.md`](experiments/report.md).

---

## 7. The DPO loss against the official TRL implementation

`sftdpo crosscheck --tolerance 1e-5`, in float64 on identical inputs, 8 pairs:

| Variant | TRL loss type | Maximum absolute difference |
|---|---|---:|
| sigmoid | `sigmoid` | 1.110e-16 |
| ipo | `ipo` | 1.066e-14 |
| cdpo | `aot` | 1.110e-16 |

They agree. Getting there surfaced two differences in convention that are recorded rather
than smoothed over, because either would make a number quoted against a paper wrong:

1. **TRL's IPO is length-normalised and the paper does not say so.** TRL divides each side's
   log-ratio by its completion token count before the squared error; its own source
   attributes the choice to correspondence with the IPO authors. This implementation takes
   four log-probabilities and no lengths, so it cannot do that. The two agree exactly when
   the caller passes per-token averages (`length_normalise=True`), and the residual after
   accounting for the normalisation is reported rather than asserted away.
2. **TRL 1.12.0 has no cDPO.** Its `label_smoothing` argument now feeds Robust DPO, which is
   a different objective, not a rescaling: cDPO takes the likelihood of a noisy label,
   `-(1-e)·log σ(u) - e·log σ(-u)`, while rDPO takes the unbiased estimator
   `[-(1-e)·log σ(u) + e·log σ(-u)] / (1-2e)`. The smoothed likelihood does survive inside
   TRL's `aot` loss, which computes exactly that expression on a batch-sorted `h`; sorting is
   the identity for a single pair, so cDPO is cross-checked against `aot` one pair at a time.

CI runs this comparison on every push and fails the build if the difference exceeds 1e-5.

---

## 8. Engineering findings from the run itself

Before the real-model stage, seven reviewers and a three-lens refutation panel were run over
this project and its companion; 35 of 52 candidate findings survived. Four of them changed
what this page can honestly claim, and all four were invisible in a suite that was passing
1 652 tests:

* **The preference margin was unreachable** — see the calibration table in section 3. The
  claim in the docstring was true of a fixture and false of the corpus.
* **A step-zero identity that could not fail for the reason claimed.** The docstring said the
  `ln 2` identity exercised the log-probability shift, the masking and the concatenation. It
  does not: both sides are computed by the same function on the same batch, so an error there
  cancels. Substituting the exact off-by-one the code warns about leaves the loss bit-for-bit
  `ln 2`. The identity pins the reference context and the adapter initialisation, and the
  shift and masking are pinned by their own tests; the docstring now says so.
* **The cross-check could report a real disagreement as "explained".** Its `explained` branch
  compared the loss residual and the chosen implicit reward and forgot the rejected one, so an
  injected error in TRL's rejected reward came back `explained` with a zero exit code — the
  one outcome the module exists to prevent. A regression test now injects exactly that.
* **A documented histogram that nothing produces.** The parser names every repair it applies,
  and the module said the evaluator reports a histogram over them. It reports a count. The
  taxonomy is on `ParseOutcome` for a caller that wants it; the aggregate is one integer, and
  the docstring now says which.

One more, found by exhaustive search rather than by review: `field_f1` is **not** monotone in
correctness when the gold record holds two recommendations with the same product name, because
the name-based alignment re-derives on every call. Repairing one name towards gold can then
lose more matched paths than it gains, and the miner would put the more-correct completion on
the rejected side. The generator draws products without replacement, so this never occurs in
the 640 examples here, and a test asserts both the break and its unreachability — so the day
the corpus changes, the test fails rather than the reward silently inverting.


**A 1 024-token budget truncated every example.** The prompt carries the whole JSON Schema,
so measured prompt lengths run from a median of 1 072 tokens on the shortest slice to a
maximum of 1 726 on `long_context`, and prompt plus gold answer reaches 1 964. The first real
run was configured at the unremarkable default of 1 024 and dropped **every preference pair
it had built**, both sides truncated away. It failed loudly only because the collators count
what they drop; a pipeline that padded silently would have completed, reported a falling
loss, and trained on prompts with their answers cut off. The budget is now a measured
constant with a `network`-marked test that checks the corpus against it.

**A vocabulary-sized softmax that did not fit.** Scoring label tokens means a softmax over
Qwen's 151 936 entries. One DPO pair at 2 048 tokens is two rows of 2 047 × 151 936 scores:
1.16 GiB in bfloat16 and 2.32 GiB again for each float32 copy. Measured peak above its
inputs:

| Implementation | One pair | Two pairs | Difference from float32 reference |
|---|---:|---:|---|
| Single-shot `log_softmax(x.float())` | 6.95 GiB | 13.90 GiB | — (the reference) |
| `-F.cross_entropy(...)` on bf16 logits | 2.32 GiB | 4.63 GiB | **18.5 nats** — wrong |
| Chunked along the position axis | 3.71 GiB | 7.19 GiB | **0.0** — bit-identical |

The first killed the run in the LoRA layer of the reference forward pass with 25 GiB
allocated. The second is the idiomatic memory fix and it is wrong: on bfloat16 logits it does
its arithmetic in bfloat16, and a DPO margin lives at the scale of a fraction of a nat, so
that optimisation would have been invisible in a loss curve and fatal to the result. The
third is what is implemented, and a test asserts the equivalence with `torch.equal` rather
than a tolerance, on both the values and the gradients.

**Batched decoding is not bit-identical to single-stream decoding.** Padding changes the
kernel shapes and bfloat16 addition is not associative, so a near-tied argmax can tip: on the
0.5 B, five of six greedy completions matched between a batch of six and a batch of one, and
the sixth differed in the scored answer (`8500` against `"$8,500"`). The invariant that does
hold, and the one the experiments rely on, is *same batch composition and same seed, same
output* — which is why the decoding batch sizes are written into the run configuration
alongside the seed.

That invariant was then tested by accident, and held. Every report on this page was
regenerated in one pass at the end, because the base and SFT reports predated the review fixes
above and disagreed with the current evaluator — the base model's strict validity read 0.7875
under the old one and 0.7438 under the new. Two of the five reports had already been produced
by the *current* code in earlier, separate processes: the DPO report and the 1.5 B report both
came back **bit-identical**, to every one of the fifteen figures each carries. So the numbers
here are reproducible across processes, and the earlier disagreement was a code difference
rather than decoder noise — which is exactly the distinction that would have been impossible
to draw if the regeneration had not been done wholesale.
