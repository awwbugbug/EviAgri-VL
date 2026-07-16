#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?usage: run_task9d_smoke.sh CONFIG}
PYTHON=${PYTHON:-/root/miniconda3/envs/eviagri/bin/python}
for variant in A B C; do
  "$PYTHON" /root/EviAgri-VL/server/train_task9d.py --config "$CONFIG" --variant "$variant" --seed 17 --mode smoke
done
