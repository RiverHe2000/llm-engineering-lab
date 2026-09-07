#!/usr/bin/env bash
# Regenerates everything docs/RESULTS.md cites.
#
# The question this experiment answers: a 0.5 B instruct model cannot reliably emit one JSON
# object that satisfies a fixed schema. Does training fix what prompting could not, and how
# does the trained small model compare with a much larger model that was only prompted?
#
# Needs a CUDA device. Budget roughly three hours on one RTX 4070. Every stage writes to the
# run directory and is skipped if its output already exists, so an interrupted run resumes.
#
# Usage:  bash scripts/run_experiments.sh [--quick]
set -euo pipefail

cd "$(dirname "$0")/.."
RUN=${SFTDPO_RUN_DIR:-runs/main}
OUT=docs/experiments
mkdir -p "$RUN" "$OUT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SMALL=${SFTDPO_SMALL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
MID=${SFTDPO_MID_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
# A hub id, not a local path: this script has to run on a machine that is not mine.
# Point $SFTDPO_BIG_MODEL at a local directory to use one that is already downloaded.
BIG=${SFTDPO_BIG_MODEL:-Qwen/Qwen3-4B-Instruct-2507}

N_TRAIN=600; N_VAL=120; N_TEST=200; SFT_EPOCHS=3; SAMPLE_K=6; DPO_EPOCHS=2
if [ "${1:-}" = "--quick" ]; then
  N_TRAIN=120; N_VAL=40; N_TEST=60; SFT_EPOCHS=1; SAMPLE_K=4; DPO_EPOCHS=1
fi

echo "== 1. Data"
sftdpo data build --seed 1 --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" \
  --out "$RUN/data"
sftdpo data stats "$RUN/data" | tee "$OUT/data_stats.txt"

echo
echo "== 2. Baseline: the untrained 0.5B, prompted"
sftdpo eval run --model "$SMALL" --data "$RUN/data" --split test --out "$RUN/eval_base.json"

echo
echo "== 3. Supervised fine-tuning (LoRA)"
sftdpo sft train --model "$SMALL" --data "$RUN/data" --epochs "$SFT_EPOCHS" \
  --out "$RUN/sft"
sftdpo eval run --model "$SMALL" --adapter "$RUN/sft/adapter" --data "$RUN/data" \
  --split test --out "$RUN/eval_sft.json"

echo
echo "== 4. Preference mining: the SFT model samples, the verifier labels"
sftdpo prefs mine --model "$SMALL" --adapter "$RUN/sft/adapter" --data "$RUN/data" \
  --split train --k "$SAMPLE_K" --out "$RUN/pairs.jsonl" | tee "$OUT/mining_stats.txt"

echo
echo "== 5. Direct Preference Optimisation"
sftdpo dpo train --model "$SMALL" --adapter "$RUN/sft/adapter" --pairs "$RUN/pairs.jsonl" \
  --epochs "$DPO_EPOCHS" --beta 0.1 --variant sigmoid --out "$RUN/dpo"
sftdpo eval run --model "$SMALL" --adapter "$RUN/dpo/adapter" --data "$RUN/data" \
  --split test --out "$RUN/eval_dpo.json"

echo
echo "== 6. The comparison that matters: a prompted model four to eight times the size"
sftdpo eval run --model "$MID" --data "$RUN/data" --split test --out "$RUN/eval_mid.json"
sftdpo eval run --model "$BIG" --data "$RUN/data" --split test --out "$RUN/eval_big.json"

echo
echo "== 7. Gates and comparisons"
sftdpo eval compare "$RUN/eval_base.json" "$RUN/eval_sft.json" \
  --margin 0.02 --json-floor 0.95 | tee "$OUT/compare_base_sft.md"
sftdpo eval compare "$RUN/eval_sft.json"  "$RUN/eval_dpo.json" \
  --margin 0.02 --json-floor 0.95 | tee "$OUT/compare_sft_dpo.md"
sftdpo eval compare "$RUN/eval_big.json"  "$RUN/eval_dpo.json" \
  --margin 0.02 --json-floor 0.95 | tee "$OUT/compare_big_dpo.md"

echo
echo "== 8. What alignment cost: general instruction-following before and after"
sftdpo eval tax --model "$SMALL" --adapter "$RUN/dpo/adapter" \
  --baseline-adapter none --out "$RUN/tax.json" | tee "$OUT/alignment_tax.md"

echo
echo "== 9. My DPO loss against the official TRL implementation"
sftdpo crosscheck --tolerance 1e-5 | tee "$OUT/crosscheck.txt"

echo
echo "== 10. Full report"
sftdpo report "$RUN" | tee "$OUT/report.md"

echo
echo "== 11. Copy the evidence out of the run directory"
# `runs/` is git-ignored: it holds adapters and sampled completions, which are hundreds of
# megabytes and regenerable. The evaluation reports and manifests are neither, and every
# number in docs/RESULTS.md is read off them, so they are copied where they can be committed.
for name in base sft dpo mid big; do
  [ -f "$RUN/eval_$name.json" ] && cp "$RUN/eval_$name.json" "$OUT/eval_$name.json"
done
[ -f "$RUN/config.json" ] && cp "$RUN/config.json" "$OUT/run_config.json"
[ -f "$RUN/data/manifest.json" ] && cp "$RUN/data/manifest.json" "$OUT/data_manifest.json"
for stage in sft dpo; do
  [ -f "$RUN/$stage/manifest.json" ] && cp "$RUN/$stage/manifest.json" "$OUT/${stage}_manifest.json"
  [ -f "$RUN/$stage/train_log.jsonl" ] && cp "$RUN/$stage/train_log.jsonl" "$OUT/${stage}_train_log.jsonl"
done
# Into their own directory: a stage record and an evaluation report share a file name and
# the stage record is the smaller of the two, so a flat copy would silently lose the report.
mkdir -p "$OUT/stages"
[ -d "$RUN/stages" ] && cp "$RUN"/stages/*.json "$OUT/stages/" 2>/dev/null || true
ls -1 "$OUT"

echo "Done. Every number in docs/RESULTS.md should trace to a file under $OUT."
