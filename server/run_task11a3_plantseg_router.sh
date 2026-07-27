#!/usr/bin/env bash
set -euo pipefail

PY_BASE=/root/miniconda3/bin/python
PY_VISION=/root/miniconda3/envs/eviagri/bin/python
PROJECT=/root/EviAgri-VL/task11a3_router_code_20260727_v4
RAW=/root/autodl-tmp/EviAgriDiag/datasets/raw/plantseg_official_2026-07-23/plantseg.zip
BLIND=/root/autodl-tmp/EviAgriDiag/datasets/derived/plantseg_blind_source_audit/2026-07-27
HUMAN=/root/autodl-tmp/EviAgriDiag/datasets/derived/plantseg_final_human_audit/2026-07-27
TASK10B=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/formal_features
TASK11A2=/root/autodl-tmp/EviAgriDiag/experiments/task11a2_plantdoc_real_null/2026-07-23/features
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task11a3_plantseg_real_null/2026-07-27/formal_v3
DATASET="$ROOT/dataset"
FEATURES="$ROOT/features"
EVALUATION="$ROOT/evaluation"

EXPECTED_ARCHIVE_SIZE=1057281724
EXPECTED_ARCHIVE_MD5=9358a66dff88cdd15c4fe009763c40a3
EXPECTED_FINAL_AUDIT_SHA=56d8256949315a420cc57fd71bfeb72105020eb5900de500f2c520cb6ae9efb2
EXPECTED_FINAL_REPORT_SHA=1ace86dcabff6c3ceecdaac16dc8cce72252568eb3d2552b4e3449bff2e06f3d

failure() {
  code=$?
  if test -d "$ROOT"; then
    printf '{"state":"failed","exit_code":%s}\n' "$code" > "$ROOT/failure.json"
  else
    printf '{"state":"failed","exit_code":%s}\n' "$code" > "${ROOT}.failure.json"
  fi
  exit "$code"
}
trap failure ERR

test ! -e "$ROOT"
test ! -e "${ROOT}.failure.json"
test -x "$PY_BASE"
test -x "$PY_VISION"
test -d "$MODEL"
test "$(stat -c %s "$RAW")" = "$EXPECTED_ARCHIVE_SIZE"
test "$(md5sum "$RAW" | cut -d' ' -f1)" = "$EXPECTED_ARCHIVE_MD5"
test "$(sha256sum "$HUMAN/final_human_audit.jsonl" | cut -d' ' -f1)" = "$EXPECTED_FINAL_AUDIT_SHA"
test "$(sha256sum "$HUMAN/final_audit_report.json" | cut -d' ' -f1)" = "$EXPECTED_FINAL_REPORT_SHA"
(cd "$HUMAN" && sha256sum -c completion.sha256)
(cd "$BLIND" && sha256sum -c completion.sha256)
(cd "$TASK10B" && sha256sum -c completion.sha256)
(cd "$TASK11A2" && sha256sum -c completion.sha256)

declare -A CODE_SHA=(
  [build_task11a3_router_dataset.py]=b7b914c65f89adcda7b307ec4b0ff7f2a520aef647de100a79bd2bb63ebffe8f
  [extract_task10b_features.py]=0c012f2fbfac21d95f03bcf8c9725dc7920cf1b4768e78b751b739b54d5db3f7
  [evaluate_task11a3_plantseg_null.py]=bad8a0871e170b184a6be826906ca4b251e845bfac7347ede7635e7749a06875
  [task10_audit_common.py]=6d5ca689de59290ed8bda4ea6137075135c11c1e95830f07dd41b1b6a0ba60c3
  [evaluate_task10b_probe.py]=bb60ef488f1a871feed1065a6602ae73897c0d9adf5b8e428b2fae37948d719a
  [task11a_confidence_router.py]=d50f261dda28478c172e9ccdbd84b391d34eb3222fafa1725859f5e25a39c070
)
for name in "${!CODE_SHA[@]}"; do
  test "$(sha256sum "$PROJECT/$name" | cut -d' ' -f1)" = "${CODE_SHA[$name]}"
done

echo stage=materialize_290_original_images
"$PY_BASE" "$PROJECT/build_task11a3_router_dataset.py" \
  --archive "$RAW" \
  --audit-manifest "$BLIND/audit_manifest.jsonl" \
  --private-index "$BLIND/candidate_index_private.jsonl" \
  --final-audit "$HUMAN/final_human_audit.jsonl" \
  --final-report "$HUMAN/final_audit_report.json" \
  --task10b-feature-rows "$TASK10B/feature_rows.jsonl" \
  --task11a2-feature-rows "$TASK11A2/feature_rows.jsonl" \
  --output-root "$DATASET"
(cd "$DATASET" && sha256sum -c completion.sha256)

cat > "$ROOT/contract.json" <<EOF
{
  "version": "task11a3-router-run-contract-1",
  "expected_images": 290,
  "seeds": [17, 29, 43],
  "temperature": 0.18887372662036642,
  "threshold": 0.63,
  "bootstrap_repetitions": 1000,
  "training_on_plantseg": false,
  "threshold_selection_on_plantseg": false,
  "task8_locked_set_read": false,
  "task11b_started": false
}
EOF

echo stage=extract_frozen_visual_features
"$PY_VISION" "$PROJECT/extract_task10b_features.py" \
  --manifest "$DATASET/manifest.jsonl" \
  --model-path "$MODEL" \
  --output-root "$FEATURES" \
  --config-version task11a3-plantseg-feature-config-1 \
  --summary-version task11a3-plantseg-feature-summary-1
(cd "$FEATURES" && sha256sum -c completion.sha256)

echo stage=evaluate_frozen_router_once
"$PY_BASE" "$PROJECT/evaluate_task11a3_plantseg_null.py" \
  --base-feature-root "$TASK10B" \
  --null-feature-root "$FEATURES" \
  --dataset-root "$DATASET" \
  --task11a2-feature-rows "$TASK11A2/feature_rows.jsonl" \
  --output-root "$EVALUATION" \
  --repetitions 1000
(cd "$EVALUATION" && sha256sum -c completion.sha256)

"$PY_BASE" - "$ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
evaluation = json.loads((root / "evaluation" / "run_summary.json").read_text())
summary = {
    "version": "task11a3-router-run-summary-1",
    "state": "completed",
    "image_count": 290,
    "dataset_manifest_sha256": hashlib.sha256((root / "dataset" / "manifest.jsonl").read_bytes()).hexdigest(),
    "features_sha256": hashlib.sha256((root / "features" / "features.npy").read_bytes()).hexdigest(),
    "evaluation_decision": evaluation["decision"],
}
(root / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (root / "completion.sha256").open("x") as handle:
    for name in ("contract.json", "run_summary.json"):
        handle.write(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n")
PY
(cd "$ROOT" && sha256sum -c completion.sha256)
echo stage=completed
