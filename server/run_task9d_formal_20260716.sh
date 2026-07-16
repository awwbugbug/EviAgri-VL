#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16
OUT="$ROOT/evaluation/formal_metrics_v1"
LOG="$ROOT/evaluation/formal_metrics_v1.log"

if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "refusing to overwrite existing formal evaluation output" >&2
  exit 17
fi

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ $rc -ne 0 ]]; then
    mkdir -p "$OUT"
    printf '{"stage":"formal_evaluation","exit_code":%d}\n' "$rc" > "$OUT/failure.json"
  fi
  exit "$rc"
}
trap on_exit EXIT

/root/miniconda3/envs/eviagri/bin/python \
  /root/EviAgri-VL/server/run_task9d_evaluation.py \
  --manifest "$ROOT/preparation/protocol/evaluation_protocol/manifest.jsonl" \
  --inference-root "$ROOT/evaluation/inference" \
  --class-bands "$ROOT/preparation/protocol/class_bands.json" \
  --pretraining-gate "$ROOT/preparation/pretraining_gate.json" \
  --output-root "$OUT" \
  --repetitions 1000 \
  --seed 20260716 \
  > "$LOG" 2>&1
