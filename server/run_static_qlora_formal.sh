#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/envs/eviagri/bin/python
TRAINER=/root/EviAgri-VL/server/train_static_qlora.py
CONFIG=/root/EviAgri-VL/server/configs/static_qlora_v1.json
EXPERIMENT=/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1
SMOKE_GATE="$EXPERIMENT/smoke/smoke_gate.json"
FORMAL_DIR="$EXPERIMENT/formal"
LAUNCH_LOG="$EXPERIMENT/formal-launch.log"
LAUNCH_RECORD="$EXPERIMENT/preparation/formal_launch.txt"
SESSION=static_qlora_v1

if screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
  echo "refusing to launch: screen session $SESSION already exists" >&2
  exit 1
fi
if [ ! -f "$SMOKE_GATE" ]; then
  echo "refusing to launch: missing $SMOKE_GATE" >&2
  exit 1
fi
grep -q '"passed": true' "$SMOKE_GATE" || {
  echo "refusing to launch: smoke gate is not passed" >&2
  exit 1
}
"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("passed") is True and len(r.get("gates", {})) == 6 and all(r["gates"].values()) else 1)' "$SMOKE_GATE" || {
  echo "refusing to launch: not all six smoke gates are true" >&2
  exit 1
}

required_bytes=$((20 * 1024 * 1024 * 1024))
available_bytes=$(df --output=avail -B1 /root/autodl-tmp | tail -n 1 | tr -d ' ')
if [ "$available_bytes" -lt "$required_bytes" ]; then
  echo "refusing to launch: less than 20 GiB free on /root/autodl-tmp" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "refusing to launch: GPU already has a compute process" >&2
  exit 1
fi
if [ -d "$FORMAL_DIR" ] && find "$FORMAL_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty formal output: $FORMAL_DIR" >&2
  exit 1
fi
if [ -e "$LAUNCH_LOG" ]; then
  echo "refusing to overwrite existing launch log: $LAUNCH_LOG" >&2
  exit 1
fi

screen -L -Logfile "$LAUNCH_LOG" -dmS static_qlora_v1 /bin/bash -lc \
  "'$PYTHON' '$TRAINER' --config '$CONFIG' --mode formal; exit_code=\$?; if [ -d '$FORMAL_DIR' ]; then mv '$LAUNCH_LOG' '$FORMAL_DIR/train.log'; fi; exit \$exit_code"

sleep 3
if ! screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
  echo "formal training exited during launch; inspect $LAUNCH_LOG or $FORMAL_DIR/failure.json" >&2
  exit 1
fi
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "session=$SESSION"
  echo "config=$CONFIG"
  echo "smoke_gate=$SMOKE_GATE"
} > "$LAUNCH_RECORD"
echo "formal training started in detached screen session: $SESSION"
echo "status: bash /root/EviAgri-VL/server/check_static_qlora_status.sh"
