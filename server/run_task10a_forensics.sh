#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/eviagri/bin/python}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-/root/autodl-tmp/EviAgriDiag/experiments/task9d_v22_micro/2026-07-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10a}"

exec "${PYTHON_BIN}" /root/EviAgri-VL/server/run_task10a_forensics.py \
  --historical-root "${HISTORICAL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-predictions 352 \
  --expected-families 32
