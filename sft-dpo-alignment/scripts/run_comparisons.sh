#!/usr/bin/env bash
# The comparisons the pipeline itself does not run: prompted models several times the size of
# the one that was fine-tuned, and the alignment tax.
#
# Separate from run_experiments.sh because these need no training and can be re-run on their
# own after the pipeline has finished, which is what happens when a comparison model changes.
#
# Usage:  bash scripts/run_comparisons.sh [run-directory]
set -euo pipefail

cd "$(dirname "$0")/.."
RUN=${1:-runs/main}
OUT=docs/experiments
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false
# No HF_HUB_OFFLINE here: the 4 B checkpoint is a hub id, and forcing offline mode on a
# machine that has not downloaded it yet turns the first run into an immediate failure.

SMALL=${SFTDPO_SMALL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
MID=${SFTDPO_MID_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
# A hub id, not a local path: this script has to run on a machine that is not mine.
# Point $SFTDPO_BIG_MODEL at a local directory to use one that is already downloaded.
BIG=${SFTDPO_BIG_MODEL:-Qwen/Qwen3-4B-Instruct-2507}

PY=${PYTHON:-python}

echo "== A prompted model three times the size"
"$PY" -m sftdpo eval run --model "$MID" --data "$RUN/data" --split test \
  --label "mid-prompted" --out "$RUN/eval_mid.json"

echo
echo "== A prompted model eight times the size"
"$PY" -m sftdpo eval run --model "$BIG" --data "$RUN/data" --split test \
  --label "big-prompted" --out "$RUN/eval_big.json"

echo
echo "== The comparison the decision actually turns on"
# A fine-tuned 0.5B against a prompted 4B on the same test set, scored by the same verifier.
"$PY" -m sftdpo eval compare "$RUN/eval_big.json" "$RUN/eval_dpo.json" \
  --margin 0.02 --json-floor 0.95 | tee "$OUT/compare_big_dpo.md"
"$PY" -m sftdpo eval compare "$RUN/eval_mid.json" "$RUN/eval_dpo.json" \
  --margin 0.02 --json-floor 0.95 | tee "$OUT/compare_mid_dpo.md"

echo
echo "== What the alignment cost: general instruction-following before and after"
"$PY" -m sftdpo eval tax --model "$SMALL" --adapter "$RUN/dpo/adapter" \
  --baseline-adapter none --out "$RUN/tax.json" | tee "$OUT/alignment_tax.md"

echo
echo "== Copy the evidence where it can be committed"
for name in base sft dpo mid big; do
  [ -f "$RUN/eval_$name.json" ] && cp "$RUN/eval_$name.json" "$OUT/eval_$name.json"
done
[ -f "$RUN/tax.json" ] && cp "$RUN/tax.json" "$OUT/tax.json"
ls -1 "$OUT"
