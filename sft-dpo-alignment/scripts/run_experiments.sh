#!/usr/bin/env bash
# Regenerates everything docs/RESULTS.md cites, with the configuration that produced it.
#
# The question this experiment answers: a 0.5 B instruct model cannot reliably emit one JSON
# object that satisfies a fixed schema. Does training fix what prompting could not, and how
# does the trained small model compare with a much larger model that was only prompted?
#
# The whole experiment is one `sftdpo pipeline run` -- data, baseline, SFT, mining, DPO, the
# gates and the report -- followed by the two prompted larger models, the gates against them,
# and the alignment tax. Every stage writes to the run directory and is skipped if its
# artefacts already exist, so an interrupted run resumes.
#
# The flags below ARE the documented run: they are checked against
# docs/experiments/run_config.json by tests/test_scripts.py, so this file cannot drift from
# the numbers it claims to reproduce. An earlier draft carried different split sizes and
# left the preference-stage learning rate at the supervised default -- which is exactly the
# configuration that destroyed the model in section 4 of the results. A reproduction script
# that reproduces the failure rather than the result is worse than none.
#
# Needs a CUDA device. Budget roughly two hours on one RTX 4070 for the pipeline and another
# forty minutes for the comparisons.
#
# Usage:  bash scripts/run_experiments.sh [--quick]
set -euo pipefail

cd "$(dirname "$0")/.."
RUN=${SFTDPO_RUN_DIR:-runs/main}
OUT=docs/experiments
mkdir -p "$RUN" "$OUT"
export TOKENIZERS_PARALLELISM=false
# Deliberately no HF_HUB_OFFLINE here: on a machine that has not seen these checkpoints the
# first run has to be allowed to download them. Set it yourself once they are cached.

SMALL=${SFTDPO_SMALL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PY=${PYTHON:-python}

QUICK=""
if [ "${1:-}" = "--quick" ]; then
  # A smoke-sized run on real weights. Placed after the documented flags so it overrides them.
  QUICK="--n-train 120 --n-val 40 --n-test 60 --sft-epochs 1 --k 4"
fi

echo "== 1. The pipeline: data, baseline, SFT, mining, DPO, gates, report"
# shellcheck disable=SC2086
"$PY" -m sftdpo pipeline run --out "$RUN" --model "$SMALL" --seed 1 \
  --n-train 400 --n-val 80 --n-test 160 \
  --sft-epochs 3 --k 5 --temperature 0.9 \
  --dpo-epochs 1 --dpo-lr 1e-5 --dpo-batch-size 1 --dpo-grad-accum 4 \
  --beta 0.1 --variant sigmoid $QUICK

echo
echo "== 2. The three gates, at the margin and floor the results report"
# The pipeline's own compare stage runs at margin 0 and no floor; the committed gates use a
# 2-point non-inferiority margin and a 0.95 JSON-validity floor, so they are re-run here.
"$PY" -m sftdpo eval compare "$RUN/eval_base.json" "$RUN/eval_sft.json" \
  --margin 0.02 --json-floor 0.95 | tee "$OUT/compare_base_sft.md"
"$PY" -m sftdpo eval compare "$RUN/eval_sft.json"  "$RUN/eval_dpo.json" \
  --margin 0.02 --json-floor 0.95 | tee "$OUT/compare_sft_dpo.md"
"$PY" -m sftdpo eval compare "$RUN/eval_base.json" "$RUN/eval_dpo.json" \
  --margin 0.02 --json-floor 0.95 | tee "$OUT/compare_base_dpo.md"

echo
echo "== 3. Prompted larger models, their gates, and the alignment tax"
bash scripts/run_comparisons.sh "$RUN"

echo
echo "== 4. My DPO loss against the official TRL implementation"
"$PY" -m sftdpo crosscheck --tolerance 1e-5 | tee "$OUT/crosscheck.txt"
"$PY" -m sftdpo crosscheck --tolerance 1e-5 --json > "$OUT/crosscheck.json"

echo
echo "== 5. Which field the preference stage stopped producing"
"$PY" scripts/field_coverage_probe.py --run "$RUN" --field flags --out "$OUT/flags_hedge.json"

echo
echo "== 6. Copy the evidence out of the run directory"
# `runs/` is git-ignored: it holds adapters and sampled completions, which are hundreds of
# megabytes and regenerable. The evaluation reports and manifests are neither, and every
# number in docs/RESULTS.md is read off them, so they are copied where they can be committed.
"$PY" -m sftdpo report "$RUN" > "$OUT/report.md"
[ -f "$RUN/config.json" ] && cp "$RUN/config.json" "$OUT/run_config.json"
[ -f "$RUN/data/manifest.json" ] && cp "$RUN/data/manifest.json" "$OUT/data_manifest.json"
for name in data_stats mining_stats; do
  [ -f "$RUN/$name.json" ] && cp "$RUN/$name.json" "$OUT/$name.json"
done
for stage in sft dpo; do
  for artefact in manifest summary; do
    [ -f "$RUN/$stage/$artefact.json" ] && cp "$RUN/$stage/$artefact.json" "$OUT/${stage}_$artefact.json"
  done
  [ -f "$RUN/$stage/train_log.jsonl" ] && cp "$RUN/$stage/train_log.jsonl" "$OUT/${stage}_train_log.jsonl"
done
[ -f "$RUN/dpo/reward_log.jsonl" ] && cp "$RUN/dpo/reward_log.jsonl" "$OUT/dpo_reward_log.jsonl"
ls -1 "$OUT"

echo "Done. Every number in docs/RESULTS.md should trace to a file under $OUT."
