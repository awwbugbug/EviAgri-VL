#!/usr/bin/env bash
set -euo pipefail

PY_BASE=/root/miniconda3/bin/python
PY_VISION=/root/miniconda3/envs/eviagri/bin/python
PROJECT=/root/EviAgri-VL/task12b_code_20260727_protocol_v1
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
BASE_MANIFEST=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/protocol/manifest.jsonl
PLANTSEG_MANIFEST=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/dataset/manifest.jsonl
PRIOR_CROP=/root/autodl-tmp/EviAgriDiag/experiments/task11c0_local_crop/2026-07-27/protocol_v2/attempt_01/manifest.jsonl
TASK12A_DATASET=/root/autodl-tmp/EviAgriDiag/experiments/task12a_conditional_complementarity/2026-07-27/protocol_v1/attempt_01/dataset
BASE_FEATURES=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/formal_features
PLANTSEG_FEATURES=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/features
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task12b_independent_replication/2026-07-27/protocol_v1/attempt_01

test ! -e "$ROOT"
test -x "$PY_BASE" && test -x "$PY_VISION" && test -d "$MODEL"
test "$(sha256sum "$BASE_MANIFEST" | cut -d' ' -f1)" = 84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90
test "$(sha256sum "$PLANTSEG_MANIFEST" | cut -d' ' -f1)" = 7196eb45259b851908c362dfbbb08e1b5b81f65f36dfe1a25d36692e63efc025
test "$(sha256sum "$PRIOR_CROP" | cut -d' ' -f1)" = 87ee2135d3eedf19e4b6c2426edcb6733abbf327ffbd835489531fb0b8df5e8b
test "$(sha256sum "$TASK12A_DATASET/manifest.jsonl" | cut -d' ' -f1)" = 0f0c56f423f2d174a2e9a6cf846142041eea1b20fc7100f39f87519a95c30279
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)
(cd "$BASE_FEATURES" && sha256sum -c completion.sha256)
(cd "$PLANTSEG_FEATURES" && sha256sum -c completion.sha256)
(cd "$(dirname "$PRIOR_CROP")" && sha256sum -c completion.sha256)
(cd "$TASK12A_DATASET" && sha256sum -c completion.sha256)
mkdir -p "$ROOT"

"$PY_BASE" "$PROJECT/build_task12a_local_dataset.py" \
  --base-manifest "$BASE_MANIFEST" --plantseg-manifest "$PLANTSEG_MANIFEST" \
  --prior-crop-manifest "$PRIOR_CROP" \
  --additional-used-manifest "$TASK12A_DATASET/manifest.jsonl" \
  --output-root "$ROOT/dataset"
(cd "$ROOT/dataset" && sha256sum -c completion.sha256)

"$PY_VISION" "$PROJECT/extract_task10b_features.py" \
  --manifest "$ROOT/dataset/manifest.jsonl" --model-path "$MODEL" \
  --output-root "$ROOT/local_features" \
  --config-version task12b-local-feature-config-1 \
  --summary-version task12b-local-feature-summary-1
(cd "$ROOT/local_features" && sha256sum -c completion.sha256)

"$PY_BASE" "$PROJECT/evaluate_task12a_complementarity.py" \
  --base-root "$BASE_FEATURES" --plantseg-root "$PLANTSEG_FEATURES" \
  --local-root "$ROOT/local_features" --dataset-root "$ROOT/dataset" \
  --output-root "$ROOT/evaluation" --repetitions 1000
(cd "$ROOT/evaluation" && sha256sum -c completion.sha256)

