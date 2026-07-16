#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/eviagri/bin/python
CODE=/root/EviAgri-VL/server
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task9d_v22_micro/2026-07-16
PROTOCOL="$ROOT/protocol"
AUDIT="$ROOT/loss_audit/loss_reduction_audit.json"
IMAGE_ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16/preparation/protocol
BASE_CONFIG=/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16/runs/B/seed_29/config.snapshot.json

cd "$ROOT/protocol"
sha256sum -c completion.sha256
cd "$ROOT/loss_audit"
sha256sum -c completion.sha256
"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text()); assert d["passed"] is True; assert d["reduction"] == "per_example_active_token_mean_then_batch_mean"' "$AUDIT"

for arm in Control TaxMask; do
  for seed in 17 29 43; do
    output="$ROOT/runs/$arm/seed_$seed"
    test ! -e "$output"
    echo "START arm=$arm seed=$seed"
    "$PY" "$CODE/train_task9d_v22.py" \
      --base-config "$BASE_CONFIG" \
      --protocol-root "$PROTOCOL" \
      --image-root "$IMAGE_ROOT" \
      --experiment-root "$ROOT" \
      --loss-audit "$AUDIT" \
      --arm "$arm" \
      --seed "$seed"
    cd "$output"
    sha256sum -c completion.sha256
    echo "DONE arm=$arm seed=$seed"
  done
done

"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"state":"completed","runs":6,"steps_per_run":64},indent=2)+"\n")' "$ROOT/matrix_status.json"
echo "MATRIX_COMPLETED"
