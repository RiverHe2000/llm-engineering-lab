# Interview notes — SFT then DPO with a verifier as the preference signal

## The premise

**Why this task?** Because it has a checkable answer and a measured failure. A 0.5 B instruct
model asked for one JSON object against a fixed schema produces parseable JSON 74.4 % of the
time and a schema-valid record 23.1 % of the time. That is not a prompt problem — I hit the
same floor in an ops-loop project whose regression gate refused to promote either candidate
prompt because both scored 0/2 on the JSON cases. The interesting question is whether
training moves a number that prompting cannot.

**Why a verifier instead of a judge model or human labels?** DPO needs a preference between
two completions. On a task with a schema, a parser is strictly more reliable than a model at
saying which of two completions is valid, and it is free. So the reward is
`w1·parses + w2·schema_valid + w3·field_f1`, provably in [0, 1], and the pairs are mined by
sampling k completions and pairing the best against the worst where the margin is wide
enough. The signal is exactly as trustworthy as the schema.

**What is the honest limitation of that?** It only works where correctness is checkable. This
is verifier-shaped alignment, not a general preference method, and the write-up says so. The
transferable part is the pattern: wherever a deterministic check exists, it beats a judge
model at labelling, and it tells you *what* the model is being taught to stop doing — the
mining stage reports the distribution of the rejected completions' headline violation.

## Data

**Why synthetic?** Because the gold answer has to be known exactly for the reward to be
exact, and because the difficulty slices have to be constructed rather than found. The cost
is that nothing here proves the method works on real adviser notes, which the results say
plainly.

**What the slices are for.** Reported separately so an average cannot hide a regression. The
two that earn their place: `absent_fields`, where the gold really is empty and the test is
that the model invents nothing; and `distractor`, where a figure is stated and then corrected
later in the note ("actually, make that $12,000") so the gold takes the correction. A model
that pattern-matches the first number it sees fails that slice and nothing else.

**Splits are disjoint by construction** and asserted to be — no note text may appear in two
splits — and the corpus carries a content hash so a result can prove which data produced it.

## The verifier

**Strict versus lenient parsing, and why both.** Strict mode requires the completion to be
exactly one JSON object and nothing else, which is the behaviour training is trying to
produce. Lenient mode recovers from what small models actually emit: a fenced block, a
"Here is the JSON:" preamble, single quotes, trailing commas, Python `None`. Every repair is
named on the result. The gap between the two rates is a headline number, because it is the
size of the repair step a deployment would still need in front of the model.

**Why `extra="forbid"`?** A model that invents a plausible extra key has not followed the
schema. A permissive validator scores that as success, and then the deployment discovers it
downstream.

**Why is `2026-02-30` a violation rather than a string?** Because the field is a date. Type
checking that stops at "it is a string of the right shape" is the kind of validation that
passes everything and catches nothing.

**Normalisation choices, each defended.** Money to the cent; percentages to two decimals;
`objectives` and `flags` compared as sets because their order carries no meaning;
`recommendations` compared as an ordered list matched on product name, so a correct set in a
different order is not scored as six separate errors. Field F1 is computed over the union of
present paths, so a missing field and an invented one both cost.

## Training

**Prompt masking.** Labels are aligned with `input_ids`, not shifted, because every causal-LM
head in `transformers` shifts internally; masking position `i` means "do not ask the model to
predict token `i`". Get it wrong and the loss still falls, the text still looks fluent, and
the model has been taught to reproduce the prompt. The tests assert three invariants:
nothing before the completion is supervised, the count of supervised positions equals the
completion length, and padding is never supervised.

**Why write the loops by hand?** The masking, the schedule and the accumulation are the parts
an interviewer will ask about, and a `Trainer` call hides all three. The DPO loop also has to
do something a stock trainer does not — use the same weights as both policy and reference —
so it would have needed rewriting anyway.

**The reference policy costs no memory.** `peft_model.disable_adapter()` is a context manager
that yields the base model's behaviour, and it reproduces the base log-probabilities exactly
(verified). So DPO here carries one set of weights, not two: the reference forward pass costs
activations and time, but not a second copy of the model in VRAM.

**The step-zero identity, and what it actually proves.** At step zero with a freshly attached
adapter, policy and reference coincide, so the sigmoid loss must be exactly
`ln 2 = 0.693147…` for any beta. It is asserted in the unit tests and again in the training
test.

The interesting part of this answer is the limit of it, and I had it wrong in the first
draft. The two sides are computed by the same function on the same batch, so an error inside
that function — an off-by-one in the shift, a mask that supervises the prompt — is applied to
both and cancels. I checked: swapping in the exact off-by-one the docstring warns about
leaves the step-zero loss bit-for-bit `ln 2`. What the identity does pin down is that the
reference context genuinely yields a different function from the policy pass and that the
adapter starts as an identity. The shift and the masking are pinned by their own tests, which
perturb `labels[:, 0]` and `logits[:, -1]` and require the result to be unchanged, and by the
cross-check against TRL. A test that cannot fail for the reason you claim is worse than no
test, because it buys confidence it has not earned.

## The run that destroyed the model, which is the answer to "tell me about a failure"

**What happened.** The first preference run reached a reward accuracy of 1.000, a final loss
of 0.0014 and a reward margin that grew from 0.8 to 7.5. Every number a training loop prints
said it had worked. The resulting model produced **no valid JSON at all** — strict validity
0.981 → 0.000, all 160 test completions unparseable, worse than the untrained base model.

**How I knew why within a minute.** The implicit rewards are logged separately, and the
rejected reward had fallen from +0.13 to −7.15 while the chosen reward wandered around zero.
The entire margin was bought by crushing the rejected completions rather than by lifting the
chosen ones, so the policy walked far enough from the reference that it stopped producing the
format at all. That is the exact failure the module's docstring names as the reason the loss
function takes four log-probabilities instead of one margin — the margin cannot tell the two
apart, and the two have opposite meanings.

**What caused it.** The preference stage had inherited the supervised learning rate, 1e-4.
Ordinary for supervised fine-tuning, very large for DPO, which is a small correction to a
model that already works. The configuration now carries `dpo_learning_rate` separately, the
same way it already carries a separate `dpo_batch_size` — a preference step puts four
sequences through a model where a supervised step puts one, and the two stages had no
business sharing either setting.

**What caught it.** The promotion gate, on the first rule it evaluated: every one of the six
slices regressed past the tolerance, so the decision was REJECT with exit code 1 before any
argument about means was needed. The base-to-DPO gate rejected as well, which is the check
that says the pipeline had gone backwards rather than merely sideways.

**Why I would tell this story rather than hide it.** Three parts of the design earned their
keep in one run: reporting the two implicit rewards separately, gating per slice rather than
on an average, and keeping the failed run's artefacts. If I had reported only the loss curve
and the margin I would have shipped a model that cannot produce its own output format, and I
would have had a plausible-looking table to justify it with.

## The DPO objective

**The derivation, briefly.** Bradley-Terry over a preference, plus the closed form of the
KL-regularised RLHF optimum, gives `r(x,y) = β log(π/π_ref) + β log Z(x)`. Both completions
share a prompt, so `Z(x)` cancels and maximum likelihood on the observed preferences is the
DPO loss. No reward model is ever fitted: the policy is the reward model, read through that
log-ratio.

**Why does the function take four log-probabilities rather than one margin?** Because the two
implicit rewards have to be reported separately. Their individual values say whether a run is
lifting the chosen response or merely crushing the rejected one — the second degrades the
model while the loss curve looks healthy, and the margin alone cannot tell them apart.

**Length bias.** The summed log-probability is what the derivation assumes and it is
length-biased: every extra token adds another negative number, so preference training on sums
learns to prefer short responses. Length-averaging removes the bias but no longer corresponds
to any sequence probability, so it changes the objective rather than rescaling it. Both are
exposed and there is a named test showing the ranking flip.

**Numerics.** `logsigmoid`, never `log(sigmoid(x))`, with a test that the loss stays finite at
log-ratio differences of ±200.

## Two engineering findings worth the airtime

**The memory bug that killed the first real run.** Scoring label tokens means a softmax over
Qwen's 151 936-entry vocabulary. One DPO pair at 2 048 tokens is two rows of 2 047 × 151 936
scores — 1.16 GiB in bfloat16 and 2.32 GiB for each float32 copy the promotion makes. Done in
one shot that peaks at **6.95 GiB above its inputs for a single pair** and 13.9 GiB for two,
which is how the first run died with 25 GiB allocated. Chunking the promotion and the gather
along the position axis brings it to **3.71 GiB** and **7.19 GiB**, and it is not an
approximation: each position's softmax reads only that position's row, so the values and the
gradients are **bit-identical**, which a test asserts with `torch.equal` rather than a
tolerance.

The obvious alternative is worth knowing because it is wrong. Replacing the whole thing with
`-F.cross_entropy(flat_logits, targets, reduction="none")` is the memory-efficient idiom,
needs 2.32 GiB, and on bfloat16 logits does its arithmetic in bfloat16: the sequence
log-probabilities came out **18.5 nats** from the float32 answer. A DPO margin lives at the
scale of a fraction of a nat. That optimisation would have been invisible in a loss curve and
fatal to the result.

**The default that truncated every example.** The prompt carries the whole JSON Schema, so
nothing here is short: measured prompt lengths run from a median of 1 072 tokens to a maximum
of 1 726, and prompt plus gold reaches 1 964. A 1 024-token budget — the value that looks
unremarkable in a fine-tuning script — therefore truncates every single example, and on a
preference pair it truncates both sides away and leaves nothing to prefer. It was loud only
because the collators count what they drop; a pipeline that padded silently would have
completed, reported a falling loss, and trained on prompts with their answers cut off. The
budget is now a measured constant, and a `network`-marked test checks the corpus against it.

**Batched decoding is not bit-identical to single-stream decoding.** Padding changes the
kernel shapes and bfloat16 addition is not associative, so a near-tied argmax can tip the
other way: measured on the 0.5 B, five of six greedy completions matched between batch-6 and
batch-1, and the sixth differed in the scored answer (`8500` versus `"$8,500"`). So the
invariant an experiment can rely on is *same batch composition, same seed, same output* —
which is why the decoding batch sizes are recorded in the run configuration, and why
comparing two variants means holding them fixed rather than hoping they do not matter.

## Evaluation

**Paired everything.** The variants answer the same test examples, so comparisons are paired:
a paired bootstrap for the difference with a percentile interval, and an exact McNemar test
on the per-example success indicators. Implemented from the definitions, no SciPy, tested
against hand-computed values including b = c = 0 giving p = 1.0.

**Floors, not just averages.** The gate carries an absolute minimum on JSON validity that a
candidate must clear regardless of its mean, and a per-slice non-regression rule. A model
that lifts the average while losing a slice has not earned promotion, and the exit code says
so.

**Per-field recall, because the slice rule was not enough.** The corrected DPO run passed
every rule above and had, it turned out, stopped emitting the optional `flags` array on all
160 test records — recall 0.610 → 0.000. The reward went up as a result: `flags` is optional,
so omitting it is always schema-valid, while emitting it risks a bad enum member or a flag
gold did not ask for, and the model was right about it only 83 % of the times it tried. The
per-slice rule could not see it because the loss was perfectly uniform: every slice lost the
same small amount and gained more back. The gate now compares recall per top-level field and
rejects a candidate that stopped producing one, with a support threshold so a rare field
cannot block a release on noise.

The interview version of this is short. *The verifier reported what it measured. It did not
measure whether a field still existed, so the preference data did not either, so the model
learned that silence was cheap, and the gate agreed. Buying labels from a verifier moves the
specification problem into code; it does not remove it. What it buys you is that the
specification is now testable* — which is why the fix is thirty lines and a floor rather than
a re-labelling exercise.

**The alignment tax is measured, not assumed away.** Fine-tuning hard on one output format
can damage general instruction-following, so a held-out set of generic instructions with
deterministic checkable properties is run before and after and reported as a paired
comparison. A drop there is a cost of the method, and reporting it is the difference between
an experiment and an advertisement.

**Why compare against a prompted larger model?** Because that is the decision a team actually
faces: fine-tune a small model, or prompt a big one. The comparison holds the test set and
the verifier fixed, so the answer is a trade-off rather than a single number.

The answer here is more interesting than the headline. The fine-tuned 0.5 B beats a prompted
Qwen3-4B on every aggregate — schema validity 0.988 against 0.763, +0.225 paired [+0.163,
+0.294] — but the 4 B is *perfect* on four of the six slices and better than the fine-tuned
model on `clean`. Its entire deficit is on the two slices that make the output long:
`long_context` 0.462 and `many_items` 0.077. So the honest recommendation is conditional. For
short notes and small records, prompt the 4 B and skip the training run. For long notes or
records with six recommendations in them, the gap is 0.077 against 0.962, and no amount of
prompting closes that. An aggregate would have supported the same decision for the wrong
reason, which is the sort of thing that only stays right by accident.
