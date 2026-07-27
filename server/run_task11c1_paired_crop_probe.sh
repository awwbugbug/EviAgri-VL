#!/usr/bin/env bash
set -euo pipefail

PY_BASE=/root/miniconda3/bin/python
PY_VISION=/root/miniconda3/envs/eviagri/bin/python
PROJECT=/root/EviAgri-VL/task11c1_code_20260727_protocol_v1
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
BASE_MANIFEST=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/protocol/manifest.jsonl
PLANTSEG_MANIFEST=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/dataset/manifest.jsonl
CROP_ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task11c0_local_crop/2026-07-27/protocol_v2/attempt_01
BASE_FEATURES=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/formal_features
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task11c1_paired_crop_probe/2026-07-27/protocol_v1/attempt_02

test ! -e "$ROOT"
test -x "$PY_BASE" && test -x "$PY_VISION" && test -d "$MODEL"
test "$(sha256sum "$BASE_MANIFEST" | cut -d' ' -f1)" = 84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90
test "$(sha256sum "$PLANTSEG_MANIFEST" | cut -d' ' -f1)" = 7196eb45259b851908c362dfbbb08e1b5b81f65f36dfe1a25d36692e63efc025
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)
(cd "$CROP_ROOT" && sha256sum -c completion.sha256)
(cd "$BASE_FEATURES" && sha256sum -c completion.sha256)
mkdir -p "$ROOT"

"$PY_BASE" "$PROJECT/build_task11c1_pair_manifests.py" \
  --crop-root "$CROP_ROOT" --base-manifest "$BASE_MANIFEST" \
  --plantseg-manifest "$PLANTSEG_MANIFEST" --output-root "$ROOT/pairs"
(cd "$ROOT/pairs" && sha256sum -c completion.sha256)

for condition in full local; do
  "$PY_VISION" "$PROJECT/extract_task10b_features.py" \
    --manifest "$ROOT/pairs/${condition}_manifest.jsonl" --model-path "$MODEL" \
    --output-root "$ROOT/features_${condition}" \
    --config-version task11c1-${condition}-feature-config-1 \
    --summary-version task11c1-${condition}-feature-summary-1
  (cd "$ROOT/features_${condition}" && sha256sum -c completion.sha256)
done

"$PY_BASE" "$PROJECT/evaluate_task11c1_paired_crop.py" \
  --base-root "$BASE_FEATURES" --full-root "$ROOT/features_full" \
  --local-root "$ROOT/features_local" --pair-root "$ROOT/pairs" \
  --output-root "$ROOT/evaluation" --repetitions 1000
(cd "$ROOT/evaluation" && sha256sum -c completion.sha256)
