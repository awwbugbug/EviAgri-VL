#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/bin/python
PROJECT=/root/EviAgri-VL/task11c0_code_20260727_protocol_v1
BASE=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/protocol/manifest.jsonl
PLANTSEG=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/dataset/manifest.jsonl
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task11c0_local_crop/2026-07-27/protocol_v1/attempt_01

test ! -e "$ROOT"
test -x "$PY"
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)
"$PY" "$PROJECT/build_task11c0_local_crop_smoke.py" \
  --base-manifest "$BASE" --plantseg-manifest "$PLANTSEG" --output-root "$ROOT"
(cd "$ROOT" && sha256sum -c completion.sha256)
