#!/usr/bin/env bash
set -euo pipefail

PY_BASE=/root/miniconda3/bin/python
PY_VISION=/root/miniconda3/envs/eviagri/bin/python
PROJECT=/root/EviAgri-VL/task14a_code_20260727_protocol_v1
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
POS_TRAIN=/root/autodl-tmp/EviAgriDiag/datasets/derived/eviagridiag_detection_v1/vlm_sft/train_evidence_positive.jsonl
POS_VAL=/root/autodl-tmp/EviAgriDiag/datasets/derived/eviagridiag_detection_v1/vlm_sft/val_evidence_positive.jsonl
PROVENANCE=/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v2_1_protocol/2026-07-15/private/provenance.jsonl
LOCKED=/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v2_1_protocol/2026-07-15/private/locked_exclusion.json
TASK10=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/protocol
PLANTSEG=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3/dataset/manifest.jsonl
TASK11C=/root/autodl-tmp/EviAgriDiag/experiments/task11c1_paired_crop_probe/2026-07-27/protocol_v1/attempt_02/features_full/feature_rows.jsonl
TASK12A=/root/autodl-tmp/EviAgriDiag/experiments/task12a_conditional_complementarity/2026-07-27/protocol_v1/attempt_01/dataset/manifest.jsonl
TASK12B=/root/autodl-tmp/EviAgriDiag/experiments/task12b_independent_replication/2026-07-27/protocol_v1/attempt_01/dataset/manifest.jsonl
TASK13A=/root/autodl-tmp/EviAgriDiag/experiments/task13a_frozen_presence_head/2026-07-27/protocol_v1/attempt_01/manifest.jsonl
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task14a_full_frame_token_oracle/2026-07-27/protocol_v1/attempt_01

test ! -e "$ROOT"
test -x "$PY_BASE" && test -x "$PY_VISION" && test -d "$MODEL"
test -f "$PROJECT/build_task14a_oracle_protocol.py"
test -f "$PROJECT/extract_task14a_oracle_tokens.py"
test -f "$PROJECT/evaluate_task14a_token_oracle.py"
(cd "$PROJECT" && sha256sum -c code_manifest.sha256)
"$PY_BASE" -c 'import numpy, sklearn'
"$PY_VISION" -c 'import numpy, PIL, torch, transformers; assert torch.cuda.is_available()'

test "$(sha256sum "$POS_TRAIN" | cut -d' ' -f1)" = 62e65ede77b23c451d60a30074c3ed8d7772c962f711d1c377eeda61b2c82829
test "$(sha256sum "$POS_VAL" | cut -d' ' -f1)" = 9a392d51e55ecb17ee08529c3545b1ad78964f9d5c234bd17c47abbeb2cc865d
test "$(sha256sum "$PROVENANCE" | cut -d' ' -f1)" = 24e6051330509f68059ea55054e759ff0eaf20cb42c10a375859c66d870e544a
test "$(sha256sum "$LOCKED" | cut -d' ' -f1)" = e93d9b906c8e0bbbbecfdcf5cf63626d3a60c8cbff8eabb0f05704577ea6eacd
test "$(sha256sum "$TASK10/manifest.jsonl" | cut -d' ' -f1)" = 84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90
test "$(sha256sum "$TASK10/selected_classes.json" | cut -d' ' -f1)" = 9b10428a946e0d19ae2af47b23d19e9d6704e87665782974909974d696b4efbc
test "$(sha256sum "$PLANTSEG" | cut -d' ' -f1)" = 7196eb45259b851908c362dfbbb08e1b5b81f65f36dfe1a25d36692e63efc025
test "$(sha256sum "$TASK11C" | cut -d' ' -f1)" = 7ff2eb8f5085ae9fce7534808429f6bb7e5f2769fa40d6af82d82971ec353686
test "$(sha256sum "$TASK12A" | cut -d' ' -f1)" = 0f0c56f423f2d174a2e9a6cf846142041eea1b20fc7100f39f87519a95c30279
test "$(sha256sum "$TASK12B" | cut -d' ' -f1)" = 3e1e3b2cbb4f48da01a4042244cf460a80357659bed06e2d50650c185b216bc1
test "$(sha256sum "$TASK13A" | cut -d' ' -f1)" = 17cf0585bf19e30e1b943ac595f52e2ad639f9d58d00df20b7d6762f65e1f9da

mkdir -p "$ROOT"
"$PY_BASE" "$PROJECT/build_task14a_oracle_protocol.py" \
  --positive-path "$POS_TRAIN" --positive-path "$POS_VAL" \
  --provenance "$PROVENANCE" --selected-classes "$TASK10/selected_classes.json" \
  --prior-positive "$TASK10/manifest.jsonl" --plantseg "$PLANTSEG" \
  --prior-used "$TASK11C" --prior-used "$TASK12A" --prior-used "$TASK12B" --prior-used "$TASK13A" \
  --locked-exclusion "$LOCKED" --output-root "$ROOT/protocol"
(cd "$ROOT/protocol" && sha256sum -c completion.sha256)

"$PY_BASE" - "$ROOT/protocol/protocol_report.json" <<'PY'
import json, sys
report=json.load(open(sys.argv[1],encoding="utf-8"))
if report.get("status")!="PASSED_PROTOCOL" or report.get("row_count")!=144:
    raise SystemExit("Task14A metadata gate did not pass")
if report.get("cross_split_component_overlap")!=0 or report.get("task8_locked_content_read") is not False:
    raise SystemExit("Task14A isolation gate did not pass")
PY

"$PY_VISION" "$PROJECT/extract_task14a_oracle_tokens.py" \
  --manifest "$ROOT/protocol/manifest.jsonl" --model-path "$MODEL" \
  --output-root "$ROOT/features"
(cd "$ROOT/features" && sha256sum -c completion.sha256)

"$PY_BASE" "$PROJECT/evaluate_task14a_token_oracle.py" \
  --protocol-root "$ROOT/protocol" --feature-root "$ROOT/features" \
  --output-root "$ROOT/evaluation" --repetitions 1000
(cd "$ROOT/evaluation" && sha256sum -c completion.sha256)
