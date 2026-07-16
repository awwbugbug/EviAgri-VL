#!/usr/bin/env bash
set -euo pipefail

FEATURE_PY=/root/miniconda3/envs/eviagri/bin/python
EVAL_PY=/root/miniconda3/bin/python
PROJECT=/root/EviAgri-VL/server
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2
PROTOCOL="$ROOT/protocol"
SMOKE="$ROOT/smoke_8_r3"
FEATURE_OUT="$ROOT/formal_features"
EVAL_OUT="$ROOT/evaluation"
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct

(
  cd "$PROTOCOL"
  sha256sum -c completion.sha256
)
(
  cd "$SMOKE"
  sha256sum -c completion.sha256
)

"$FEATURE_PY" - "$PROTOCOL/protocol_report.json" "$SMOKE/run_summary.json" <<'PY'
import json
import sys

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
smoke = json.load(open(sys.argv[2], encoding="utf-8"))
if protocol.get("status") != "PASSED_PROTOCOL" or protocol.get("row_count") != 320:
    raise SystemExit("Task 10B protocol gate is not passed")
if smoke.get("state") != "completed" or smoke.get("feature_count") != 8:
    raise SystemExit("Task 10B 8-image smoke gate is not passed")
if smoke.get("all_parameters_frozen") is not True:
    raise SystemExit("Task 10B smoke did not prove frozen parameters")
PY

test ! -e "$FEATURE_OUT"
test ! -e "$EVAL_OUT"

"$FEATURE_PY" "$PROJECT/extract_task10b_features.py" \
  --manifest "$PROTOCOL/manifest.jsonl" \
  --model-path "$MODEL" \
  --output-root "$FEATURE_OUT"
(
  cd "$FEATURE_OUT"
  sha256sum -c completion.sha256
)

"$EVAL_PY" -c 'import numpy, scipy, sklearn'
"$EVAL_PY" "$PROJECT/evaluate_task10b_probe.py" \
  --feature-root "$FEATURE_OUT" \
  --output-root "$EVAL_OUT" \
  --repetitions 1000
(
  cd "$EVAL_OUT"
  sha256sum -c completion.sha256
)
