#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/eviagri/bin/python
PROJECT=/root/EviAgri-VL/task11b0_code_20260727_protocol_v1
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
BASE_ROWS=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/formal_features/feature_rows.jsonl
PLANTSEG=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/dataset/manifest.jsonl
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task11b_repprobe/2026-07-27/protocol_v1/attempt_01

test ! -e "$ROOT"
test -x "$PY"
test -d "$MODEL"
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)

"$PY" "$PROJECT/task11b0_query_rep_smoke.py" \
  --config "$PROJECT/task11b0_query_rep_smoke.json" \
  --base-rows "$BASE_ROWS" \
  --plantseg-manifest "$PLANTSEG" \
  --model-path "$MODEL" \
  --output-root "$ROOT"
