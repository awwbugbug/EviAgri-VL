#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/eviagri/bin/python
CODE=/root/EviAgri-VL/server
C1=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10c_c0_c1
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10c_c2_learning_curve
PROTOCOL="$C1/protocol"
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
D2=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/evaluation

verify_completion() {
  local directory="$1"
  (
    cd "$directory"
    sha256sum -c completion.sha256
  )
}

test ! -e "$ROOT"
verify_completion "$PROTOCOL"
verify_completion "$C1/evaluation"
verify_completion "$D2"
test "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$C1/evaluation/task10c_c1_decision_report.json")" = PASS_C1_ENGINEERING
mkdir -p "$ROOT"

for seed in 17 29 43; do
  "$PY" "$CODE/train_task10c_c2.py" \
    --protocol-root "$PROTOCOL" \
    --model-path "$MODEL" \
    --experiment-root "$ROOT" \
    --seed "$seed"
  verify_completion "$ROOT/training/seed_$seed"

  for step in 8 16 32 64; do
    step_padded=$(printf '%03d' "$step")
    verify_completion "$ROOT/training/seed_$seed/checkpoints/step_$step_padded"
    "$PY" "$CODE/run_task10c_c2_inference.py" \
      --protocol-root "$PROTOCOL" \
      --model-path "$MODEL" \
      --split smoke_dev \
      --model-kind adapter \
      --seed "$seed" \
      --checkpoint-step "$step" \
      --adapter-root "$ROOT/training/seed_$seed/checkpoints/step_$step_padded" \
      --output-root "$ROOT/inference/smoke/seed_$seed/step_$step_padded"
    verify_completion "$ROOT/inference/smoke/seed_$seed/step_$step_padded"
  done

  "$PY" "$CODE/run_task10c_c2_inference.py" \
    --protocol-root "$PROTOCOL" \
    --model-path "$MODEL" \
    --split dev \
    --model-kind adapter \
    --seed "$seed" \
    --checkpoint-step 64 \
    --adapter-root "$ROOT/training/seed_$seed/checkpoints/step_064" \
    --output-root "$ROOT/inference/dev/seed_$seed"
  verify_completion "$ROOT/inference/dev/seed_$seed"
done

"$PY" "$CODE/run_task10c_c2_inference.py" \
  --protocol-root "$PROTOCOL" \
  --model-path "$MODEL" \
  --split dev \
  --model-kind base \
  --output-root "$ROOT/inference/dev/base"
verify_completion "$ROOT/inference/dev/base"

"$PY" "$CODE/score_task10c_c2_candidates.py" \
  --protocol-root "$PROTOCOL" \
  --model-path "$MODEL" \
  --model-kind base \
  --output-root "$ROOT/candidates/base"
verify_completion "$ROOT/candidates/base"

for seed in 17 29 43; do
  "$PY" "$CODE/score_task10c_c2_candidates.py" \
    --protocol-root "$PROTOCOL" \
    --model-path "$MODEL" \
    --model-kind adapter \
    --seed "$seed" \
    --checkpoint-step 64 \
    --adapter-root "$ROOT/training/seed_$seed/checkpoints/step_064" \
    --output-root "$ROOT/candidates/seed_$seed"
  verify_completion "$ROOT/candidates/seed_$seed"
done

"$PY" "$CODE/evaluate_task10c_c2.py" \
  --protocol-root "$PROTOCOL" \
  --experiment-root "$ROOT" \
  --task10b-evaluation-root "$D2" \
  --output-root "$ROOT/evaluation" \
  --repetitions 1000 \
  --bootstrap-seed 20260717
verify_completion "$ROOT/evaluation"

echo TASK10C_C2_COMPLETED_STOP
