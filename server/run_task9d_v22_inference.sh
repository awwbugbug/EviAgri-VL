#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/eviagri/bin/python
CODE=/root/EviAgri-VL/server
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task9d_v22_micro/2026-07-16
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
MANIFEST="$ROOT/protocol/evaluation_manifest.jsonl"
INFERENCE="$ROOT/inference"
EVALUATION="$ROOT/evaluation"

test -f "$ROOT/matrix_status.json"
"$PY" -c 'import json,sys; from pathlib import Path; d=json.loads(Path(sys.argv[1]).read_text()); assert d == {"state":"completed","runs":6,"steps_per_run":64}' "$ROOT/matrix_status.json"
test ! -e "$INFERENCE"
test ! -e "$EVALUATION"

for arm in Control TaxMask; do
  for seed in 17 29 43; do
    run="$ROOT/runs/$arm/seed_$seed"
    cd "$run"
    sha256sum -c completion.sha256
  done
done

for arm in Control TaxMask; do
  for seed in 17 29 43; do
    group="${arm}${seed}"
    output="$INFERENCE/$group"
    test ! -e "$output"
    echo "INFERENCE_START group=$group"
    "$PY" "$CODE/run_task9d_inference.py" \
      --model-path "$MODEL" \
      --manifest "$MANIFEST" \
      --output "$output" \
      --group "$group" \
      --adapter "$ROOT/runs/$arm/seed_$seed/adapter"
    cd "$output"
    sha256sum -c completion.sha256
    echo "INFERENCE_DONE group=$group"
  done
done

"$PY" "$CODE/run_task9d_v22_evaluation.py" \
  --manifest "$MANIFEST" \
  --inference-root "$INFERENCE" \
  --output-root "$EVALUATION"
cd "$EVALUATION"
sha256sum -c completion.sha256
echo "V22_EVALUATION_COMPLETED"
