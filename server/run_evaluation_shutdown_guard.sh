#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/envs/eviagri/bin/python
GUARD=/root/EviAgri-VL/server/evaluation_shutdown_guard.py
EVALUATION=/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/formal/evaluation
LOG="$EVALUATION/shutdown_guard.log"
SESSION=static_qlora_shutdown_guard

if [ "${1:-}" != "--worker" ]; then
  if screen -ls 2>/dev/null | grep -E "[.]$SESSION[[:space:]].*[(](Detached|Attached)[)]" >/dev/null; then
    echo "shutdown guard is already running"
    exit 0
  fi
  mkdir -p "$EVALUATION"
  screen -L -Logfile "$LOG" -dmS "$SESSION" bash "$0" --worker
  sleep 2
  if ! screen -ls 2>/dev/null | grep -E "[.]$SESSION[[:space:]].*[(](Detached|Attached)[)]" >/dev/null; then
    echo "shutdown guard failed to start; inspect $LOG" >&2
    exit 1
  fi
  echo "shutdown guard started in detached screen session: $SESSION"
  exit 0
fi

while true; do
  set +e
  "$PYTHON" "$GUARD" \
    --evaluation-root "$EVALUATION" \
    --expected-val 3052 \
    --expected-test 7596 \
    --screen-name static_qlora_eval \
    --write-manifest
  result=$?
  set -e

  case "$result" in
    0)
      echo "completion gate passed; syncing artifacts before platform shutdown"
      sync
      sleep 10
      /usr/bin/shutdown
      exit 0
      ;;
    10)
      sleep 60
      ;;
    *)
      echo "guard_blocked: completion gate failed; automatic shutdown suppressed" >&2
      exit "$result"
      ;;
  esac
done
