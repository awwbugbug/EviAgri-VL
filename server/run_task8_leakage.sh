#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/envs/eviagri/bin/python
CODE_ROOT=/root/EviAgri-VL/server
DATA_ROOT=/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v1
SMOKE_ROOT=${TASK8_SMOKE_ROOT:-/root/autodl-tmp/EviAgriDiag/experiments/task8_causal_audit/2026-07-15/smoke}

cd "$CODE_ROOT"
"$PYTHON" audit_task8_leakage.py \
  --train-jsonl "$DATA_ROOT/train.jsonl" \
  --val-jsonl "$DATA_ROOT/val.jsonl" \
  --test-jsonl "$DATA_ROOT/test.jsonl" \
  --audit-manifest "$SMOKE_ROOT/data/audit_manifest.jsonl" \
  --inference-jobs "$SMOKE_ROOT/data/inference_jobs.jsonl" \
  --output "$SMOKE_ROOT/leakage_report.json"
