#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/envs/eviagri/bin/python
CODE_ROOT=/root/EviAgri-VL/server
CONFIG="$CODE_ROOT/configs/static_qlora_v1.json"
ADAPTER=/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/formal/adapter
SMOKE_ROOT=${TASK8_SMOKE_ROOT:-/root/autodl-tmp/EviAgriDiag/experiments/task8_causal_audit/2026-07-15/smoke}

cd "$CODE_ROOT"
"$PYTHON" -c 'import json,sys; report=json.load(open(sys.argv[1])); assert report.get("passed") is True' \
  "$SMOKE_ROOT/leakage_report.json"

"$PYTHON" run_task8_inference.py \
  --jobs "$SMOKE_ROOT/data/inference_jobs.jsonl" \
  --leakage-report "$SMOKE_ROOT/leakage_report.json" \
  --config "$CONFIG" \
  --adapter-dir "$ADAPTER" \
  --output-root "$SMOKE_ROOT/run"

"$PYTHON" evaluate_task8.py \
  --prediction "$SMOKE_ROOT/run/B0/predictions.jsonl" \
  --prediction "$SMOKE_ROOT/run/B1/predictions.jsonl" \
  --prediction "$SMOKE_ROOT/run/B2/predictions.jsonl" \
  --prediction "$SMOKE_ROOT/run/B3/predictions.jsonl" \
  --output "$SMOKE_ROOT/run/metrics.json"

"$PYTHON" validate_task8_smoke.py \
  --audit-manifest "$SMOKE_ROOT/data/audit_manifest.jsonl" \
  --inference-jobs "$SMOKE_ROOT/data/inference_jobs.jsonl" \
  --leakage-report "$SMOKE_ROOT/leakage_report.json" \
  --run-root "$SMOKE_ROOT/run" \
  --output "$SMOKE_ROOT/smoke_gate.json"
