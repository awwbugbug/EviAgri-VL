#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/eviagri/bin/python
EVAL_PY=/root/miniconda3/bin/python
PROJECT=/root/EviAgri-VL/task11b1_code_20260727_protocol_v1
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
BASE_MANIFEST=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/protocol/manifest.jsonl
PLANTDOC_MANIFEST=/root/autodl-tmp/EviAgriDiag/experiments/task11a2_plantdoc_real_null/2026-07-23/features/feature_rows.jsonl
PLANTSEG_MANIFEST=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/dataset/manifest.jsonl
VISION_BASE=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/formal_features
VISION_STRESS=/root/autodl-tmp/EviAgriDiag/experiments/task11_confidence_router/2026-07-23/task11a_r4/stress_features
VISION_PLANTDOC=/root/autodl-tmp/EviAgriDiag/experiments/task11a2_plantdoc_real_null/2026-07-23/features
VISION_PLANTSEG=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/features
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task11b1_repprobe/2026-07-27/protocol_v1/attempt_01
QUERY='Determine whether the image contains a visible agricultural pest. Base the decision only on image pixels.'

test ! -e "$ROOT"
test -x "$PY" && test -x "$EVAL_PY" && test -d "$MODEL"
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)
mkdir -p "$ROOT"

extract () {
  local name="$1" manifest="$2" mode="$3"
  "$PY" "$PROJECT/extract_task11b1_query_features.py" \
    --manifest "$manifest" --model-path "$MODEL" --output-root "$ROOT/query_$name" \
    --query "$QUERY" --expected-layer 27 --expected-query-tokens 19 --mode "$mode"
  (cd "$ROOT/query_$name" && sha256sum -c completion.sha256)
}

extract base "$BASE_MANIFEST" plain
extract stress "$BASE_MANIFEST" stress
extract plantdoc "$PLANTDOC_MANIFEST" plain
extract plantseg "$PLANTSEG_MANIFEST" plain

"$EVAL_PY" "$PROJECT/evaluate_task11b1_repprobe.py" \
  --config "$PROJECT/task11b1_repprobe.json" --output-root "$ROOT/evaluation" \
  --vision-base "$VISION_BASE" --vision-stress "$VISION_STRESS" \
  --vision-plantdoc "$VISION_PLANTDOC" --vision-plantseg "$VISION_PLANTSEG" \
  --query-base "$ROOT/query_base" --query-stress "$ROOT/query_stress" \
  --query-plantdoc "$ROOT/query_plantdoc" --query-plantseg "$ROOT/query_plantseg"
(cd "$ROOT/evaluation" && sha256sum -c completion.sha256)
