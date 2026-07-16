#!/usr/bin/env bash
set -euo pipefail

: "${TASK8_FORMAL_ROOT:?TASK8_FORMAL_ROOT must name an immutable formal run directory}"
PYTHON=/root/miniconda3/envs/eviagri/bin/python
CODE_ROOT=/root/EviAgri-VL/server
CONFIG="$CODE_ROOT/configs/static_qlora_v1.json"
ADAPTER=/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/formal/adapter

cd "$CODE_ROOT"
"$PYTHON" -c 'import json,sys; report=json.load(open(sys.argv[1])); assert report.get("passed") is True' \
  "$TASK8_FORMAL_ROOT/leakage_report.json"

"$PYTHON" run_task8_inference.py \
  --jobs "$TASK8_FORMAL_ROOT/data/inference_jobs.jsonl" \
  --leakage-report "$TASK8_FORMAL_ROOT/leakage_report.json" \
  --config "$CONFIG" \
  --adapter-dir "$ADAPTER" \
  --output-root "$TASK8_FORMAL_ROOT/run"

"$PYTHON" evaluate_task8.py \
  --prediction "$TASK8_FORMAL_ROOT/run/B0/predictions.jsonl" \
  --prediction "$TASK8_FORMAL_ROOT/run/B1/predictions.jsonl" \
  --prediction "$TASK8_FORMAL_ROOT/run/B2/predictions.jsonl" \
  --prediction "$TASK8_FORMAL_ROOT/run/B3/predictions.jsonl" \
  --output "$TASK8_FORMAL_ROOT/run/metrics.json"

cd "$TASK8_FORMAL_ROOT"
sha256sum \
  data/manifest.sha256 \
  leakage_report.json \
  run/B0/predictions.jsonl \
  run/B1/predictions.jsonl \
  run/B2/predictions.jsonl \
  run/B3/predictions.jsonl \
  run/run_summary.json \
  run/metrics.json \
  > completion.sha256
sha256sum -c completion.sha256

