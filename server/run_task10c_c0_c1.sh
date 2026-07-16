#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/eviagri/bin/python
CODE=/root/EviAgri-VL/server
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10c_c0_c1
SOURCE=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/protocol/manifest.jsonl
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
PROTOCOL="$ROOT/protocol"

test ! -e "$ROOT"
mkdir -p "$ROOT"

"$PY" "$CODE/task10c_contract.py" \
  --manifest "$SOURCE" \
  --model-path "$MODEL" \
  --output-root "$PROTOCOL"
(
  cd "$PROTOCOL"
  sha256sum -c completion.sha256
)

for seed in 17 29 43; do
  "$PY" "$CODE/train_task10c_smoke.py" \
    --protocol-root "$PROTOCOL" \
    --model-path "$MODEL" \
    --experiment-root "$ROOT" \
    --seed "$seed"
  (
    cd "$ROOT/training/seed_$seed"
    sha256sum -c completion.sha256
  )

  "$PY" "$CODE/run_task10c_smoke_inference.py" \
    --protocol-root "$PROTOCOL" \
    --model-path "$MODEL" \
    --adapter-root "$ROOT/training/seed_$seed" \
    --output-root "$ROOT/inference/seed_$seed" \
    --seed "$seed"
  (
    cd "$ROOT/inference/seed_$seed"
    sha256sum -c completion.sha256
  )
done

"$PY" "$CODE/evaluate_task10c_smoke.py" \
  --protocol-root "$PROTOCOL" \
  --experiment-root "$ROOT" \
  --output-root "$ROOT/evaluation"
(
  cd "$ROOT/evaluation"
  sha256sum -c completion.sha256
)

echo TASK10C_C0_C1_COMPLETED
