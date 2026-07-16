#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/envs/eviagri/bin/python
EVALUATOR=/root/EviAgri-VL/server/evaluate_static_qlora.py
CONFIG=/root/EviAgri-VL/server/configs/static_qlora_v1.json
FORMAL=/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/formal
RUN_SUMMARY="$FORMAL/run_summary.json"
ADAPTER="$FORMAL/adapter/adapter_model.safetensors"
EVALUATION="$FORMAL/evaluation"
LOG="$EVALUATION/evaluation.log"
SESSION=static_qlora_eval

if screen -ls 2>/dev/null | grep -E "[.]$SESSION[[:space:]].*[(](Detached|Attached)[)]" >/dev/null; then
  echo "refusing to launch: screen session $SESSION already exists" >&2
  exit 1
fi
if [ ! -f "$RUN_SUMMARY" ] || [ ! -f "$ADAPTER" ]; then
  echo "refusing to launch: completed formal run_summary.json or adapter_model.safetensors is missing" >&2
  exit 1
fi
"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("completed") is True else 1)' "$RUN_SUMMARY" || {
  echo 'refusing to launch: formal "completed" is not true' >&2
  exit 1
}
if [ -f "$FORMAL/failure.json" ]; then
  echo "refusing to launch: formal failure.json exists" >&2
  exit 1
fi
if [ -f "$EVALUATION/evaluation_summary.json" ]; then
  echo "evaluation is already complete" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "refusing to launch: GPU already has a compute process" >&2
  exit 1
fi

mkdir -p "$EVALUATION"
screen -L -Logfile "$LOG" -dmS static_qlora_eval \
  "$PYTHON" "$EVALUATOR" \
  --config "$CONFIG" \
  --splits val test \
  --output-root "$EVALUATION"

sleep 3
if ! screen -ls 2>/dev/null | grep -E "[.]$SESSION[[:space:]].*[(](Detached|Attached)[)]" >/dev/null; then
  echo "evaluation exited during launch; inspect $LOG and $EVALUATION/status.json" >&2
  exit 1
fi
echo "evaluation started in detached screen session: $SESSION"
echo "status: bash /root/EviAgri-VL/server/check_static_qlora_evaluation_status.sh"
