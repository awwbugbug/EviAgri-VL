#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?usage: run_task9d_matrix.sh CONFIG}
GATE=${2:?usage: run_task9d_matrix.sh CONFIG PRETRAINING_GATE}
PYTHON=${PYTHON:-/root/miniconda3/envs/eviagri/bin/python}
"$PYTHON" - "$GATE" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
if gate.get("passed") is not True:
    raise SystemExit("Task 9D pretraining gate is not passed")
PY
for variant in A B C; do
  for seed in 17 29 43; do
    "$PYTHON" /root/EviAgri-VL/server/train_task9d.py --config "$CONFIG" --variant "$variant" --seed "$seed" --mode formal
  done
done
