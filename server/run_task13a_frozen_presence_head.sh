#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/bin/python
PROJECT=/root/EviAgri-VL/task13a_code_20260727_protocol_v1
BASE=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/formal_features
PLANTDOC=/root/autodl-tmp/EviAgriDiag/experiments/task11a2_plantdoc_real_null/2026-07-23/features
PLANTSEG=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/features
TASK11C=/root/autodl-tmp/EviAgriDiag/experiments/task11c1_paired_crop_probe/2026-07-27/protocol_v1/attempt_02/features_full/feature_rows.jsonl
TASK12A=/root/autodl-tmp/EviAgriDiag/experiments/task12a_conditional_complementarity/2026-07-27/protocol_v1/attempt_01/dataset/manifest.jsonl
TASK12B=/root/autodl-tmp/EviAgriDiag/experiments/task12b_independent_replication/2026-07-27/protocol_v1/attempt_01/dataset/manifest.jsonl
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task13a_frozen_presence_head/2026-07-27/protocol_v1/attempt_01

test ! -e "$ROOT"
test -x "$PY"
test -f "$PROJECT/evaluate_task13a_presence_head.py"
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)
"$PY" -c 'import numpy, sklearn'

test "$(sha256sum "$BASE/features.npy" | cut -d' ' -f1)" = 5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc
test "$(sha256sum "$BASE/feature_rows.jsonl" | cut -d' ' -f1)" = 2ad5192520a2fdbf1b1f058cfd987d6ad121985f62239e41234ae0d2d2a25ffd
test "$(sha256sum "$PLANTDOC/features.npy" | cut -d' ' -f1)" = 412815de2d6addd61b2863b9ec5227879888ae04250aabd5d736cce70159907a
test "$(sha256sum "$PLANTDOC/feature_rows.jsonl" | cut -d' ' -f1)" = 11f9a72a735b9f4b90634cd3dc3d8fe49ce4584f10f2bc89c04db7b33cfaf8f2
test "$(sha256sum "$PLANTSEG/features.npy" | cut -d' ' -f1)" = e05f01467c70ec334656f1702e1e0ec8fd4c5d8a14a7dad1616b9d98fc62b618
test "$(sha256sum "$PLANTSEG/feature_rows.jsonl" | cut -d' ' -f1)" = 7623b5fccd48650a15f6112e3c6564c0541f488c8bdb144dfb21552c8a48ae1c
test "$(sha256sum "$TASK11C" | cut -d' ' -f1)" = 7ff2eb8f5085ae9fce7534808429f6bb7e5f2769fa40d6af82d82971ec353686
test "$(sha256sum "$TASK12A" | cut -d' ' -f1)" = 0f0c56f423f2d174a2e9a6cf846142041eea1b20fc7100f39f87519a95c30279
test "$(sha256sum "$TASK12B" | cut -d' ' -f1)" = 3e1e3b2cbb4f48da01a4042244cf460a80357659bed06e2d50650c185b216bc1
(cd "$BASE" && sha256sum -c completion.sha256)
(cd "$PLANTDOC" && sha256sum -c completion.sha256)
(cd "$PLANTSEG" && sha256sum -c completion.sha256)

"$PY" "$PROJECT/evaluate_task13a_presence_head.py" \
  --base-root "$BASE" --plantdoc-root "$PLANTDOC" --plantseg-root "$PLANTSEG" \
  --prior-rows "$TASK11C" --prior-rows "$TASK12A" --prior-rows "$TASK12B" \
  --output-root "$ROOT" --repetitions 1000
(cd "$ROOT" && sha256sum -c completion.sha256)
