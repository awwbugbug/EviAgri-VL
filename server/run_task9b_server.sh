#!/usr/bin/env bash
set -uo pipefail

STATUS_ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task9b_protocol_build/2026-07-15
OUTPUT_ROOT=/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v2_protocol/2026-07-15
SOURCE_DIR=/root/autodl-tmp/EviAgriDiag/datasets/derived/eviagridiag_detection_v1/vlm_sft
LOCKED_MANIFEST=/root/autodl-tmp/EviAgriDiag/experiments/task8_causal_audit/2026-07-15/formal_clean_v2/data/family_manifest.jsonl
REVIEWED_PAIRS=/root/autodl-tmp/EviAgriDiag/experiments/task8_causal_audit/2026-07-15/near_duplicate_review_v2.json
PYTHON=/root/miniconda3/envs/eviagri/bin/python

mkdir -p "$STATUS_ROOT"
if [[ -e "$OUTPUT_ROOT" ]]; then
  printf '{"state":"failed","reason":"output_exists","at":"%s"}\n' "$(date -Iseconds)" > "$STATUS_ROOT/status.json"
  exit 20
fi
printf '{"state":"running","started_at":"%s"}\n' "$(date -Iseconds)" > "$STATUS_ROOT/status.json"

"$PYTHON" /root/EviAgri-VL/server/run_task9b_freeze.py \
  --source-dir "$SOURCE_DIR" \
  --locked-manifest "$LOCKED_MANIFEST" \
  --reviewed-pairs "$REVIEWED_PAIRS" \
  --output-root "$OUTPUT_ROOT" \
  --seed 20260715 > "$STATUS_ROOT/run.log" 2>&1
code=$?

if [[ $code -eq 0 ]]; then
  printf '{"state":"completed","exit_code":0,"completed_at":"%s"}\n' "$(date -Iseconds)" > "$STATUS_ROOT/status.json"
else
  printf '{"state":"failed","exit_code":%d,"failed_at":"%s"}\n' "$code" "$(date -Iseconds)" > "$STATUS_ROOT/status.json"
fi
exit "$code"
