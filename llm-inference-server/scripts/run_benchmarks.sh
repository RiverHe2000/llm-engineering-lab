#!/usr/bin/env bash
# Reproduces docs/BENCHMARK.md, docs/sample_generation.txt and the live-server smoke test.
# Usage (from the project directory, with the venv active): bash scripts/run_benchmarks.sh
set -euo pipefail
mkdir -p docs
rm -f docs/BENCHMARK.md

echo "== GPU benchmarks"
llmserve bench --model-name distilbert/distilgpt2 --device cuda --batch-sizes 1,2,4,8,16,32 \
  --max-tokens 64 --repeats 3 --title "distilgpt2 (82M) on RTX 4070, bf16" \
  --out docs/BENCHMARK.md --json-out docs/bench_distilgpt2_cuda.json
llmserve bench --model-name Qwen/Qwen2.5-0.5B-Instruct --device cuda --batch-sizes 1,2,4,8,16,32 \
  --max-tokens 64 --repeats 3 --title "Qwen2.5-0.5B-Instruct (494M) on RTX 4070, bf16" \
  --out docs/BENCHMARK.md --append --json-out docs/bench_qwen05b_cuda.json

echo "== CPU benchmarks (fp32 vs dynamic INT8)"
llmserve bench --model-name distilbert/distilgpt2 --device cpu --batch-sizes 1,4,8 \
  --max-tokens 32 --repeats 2 --title "distilgpt2 on CPU, fp32" --out docs/BENCHMARK.md --append
llmserve bench --model-name distilbert/distilgpt2 --device cpu --quantize-int8 --batch-sizes 1,4,8 \
  --max-tokens 32 --repeats 2 --title "distilgpt2 on CPU, dynamic INT8" --out docs/BENCHMARK.md --append

echo "== sample generation"
PROMPT=$'<|im_start|>user\nExplain in two sentences what a credit default swap is.<|im_end|>\n<|im_start|>assistant\n'
llmserve generate --model-name Qwen/Qwen2.5-0.5B-Instruct --device cuda --prompt "$PROMPT" \
  --max-tokens 80 --temperature 0.7 --seed 0 > docs/sample_generation.txt

echo "== live server smoke test"
LLMSERVE_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct LLMSERVE_DEVICE=cuda LLMSERVE_MAX_BATCH_SIZE=16 \
  llmserve serve --port 8765 > docs/server_log.jsonl 2>&1 &
for _ in $(seq 1 90); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/ready | grep -q 200; then break; fi
  sleep 2
done
curl -s http://127.0.0.1:8765/ready > docs/smoke_ready.json
cat > docs/smoke_request.json <<'JSON'
{"prompt": "<|im_start|>user\nName three risks a bank faces when lending to small businesses.<|im_end|>\n<|im_start|>assistant\n", "max_tokens": 60, "temperature": 0.7, "seed": 1}
JSON
curl -s -X POST http://127.0.0.1:8765/v1/completions -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-1" -d @docs/smoke_request.json > docs/smoke_completion.json
echo "-- 16 concurrent requests (status, seconds)"
seq 1 16 | xargs -P 16 -I{} curl -s -o /dev/null -w "%{http_code} %{time_total}\n" \
  -X POST http://127.0.0.1:8765/v1/completions -H "Content-Type: application/json" \
  -d @docs/smoke_request.json > docs/smoke_concurrent.txt
cat docs/smoke_concurrent.txt
curl -s http://127.0.0.1:8765/metrics | grep -E "^llmserve_(requests_total|batch_size_bucket|batch_size_count|generated_tokens_total|request_latency_seconds_count)" > docs/smoke_metrics.txt
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*llmserve*serve*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" || true
echo "BENCH_DONE"
