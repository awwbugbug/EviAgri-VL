#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/bin/python
PROJECT=/root/EviAgri-VL/task11b1_eval_code_20260727_protocol_v1_attempt02
A1=/root/autodl-tmp/EviAgriDiag/experiments/task11b1_repprobe/2026-07-27/protocol_v1/attempt_01
A2=/root/autodl-tmp/EviAgriDiag/experiments/task11b1_repprobe/2026-07-27/protocol_v1/attempt_02
VBASE=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/formal_features
VSTRESS=/root/autodl-tmp/EviAgriDiag/experiments/task11_confidence_router/2026-07-23/task11a_r4/stress_features
VPD=/root/autodl-tmp/EviAgriDiag/experiments/task11a2_plantdoc_real_null/2026-07-23/features
VPS=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/features

test ! -e "$A2"
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)
mkdir -p "$A2"
"$PY" "$PROJECT/evaluate_task11b1_repprobe.py" \
  --config "$PROJECT/task11b1_repprobe.json" --output-root "$A2/evaluation" \
  --vision-base "$VBASE" --vision-stress "$VSTRESS" --vision-plantdoc "$VPD" --vision-plantseg "$VPS" \
  --query-base "$A1/query_base" --query-stress "$A1/query_stress" \
  --query-plantdoc "$A1/query_plantdoc" --query-plantseg "$A1/query_plantseg"
(cd "$A2/evaluation" && sha256sum -c completion.sha256)
