#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?usage: run_task9d_inference_matrix.sh EXPERIMENT_ROOT}
MODEL=${2:?usage: run_task9d_inference_matrix.sh EXPERIMENT_ROOT MODEL_PATH}
PYTHON=${PYTHON:-/root/miniconda3/envs/eviagri/bin/python}
SERVER=/root/EviAgri-VL/server
MANIFEST="$ROOT/preparation/protocol/evaluation_protocol/manifest.jsonl"

"$PYTHON" - "$ROOT" "$MANIFEST" <<'PY'
import hashlib, json, math, pathlib, sys
root, manifest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if sum(1 for _ in manifest.open()) != 2560:
    raise SystemExit("Task 9D inference manifest must contain exactly 2560 rows")
for variant in "ABC":
    for seed in (17, 29, 43):
        run = root / f"runs/{variant}/seed_{seed}"
        summary = json.loads((run / "run_summary.json").read_text())
        adapter = pathlib.Path(summary["adapter"]["path"])
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
        finite = all(math.isfinite(float(row[key])) for row in summary["log_history"]
                     for key in ("loss", "eval_loss", "grad_norm") if key in row)
        if not (summary.get("completed") is True and summary.get("optimizer_steps") == 192
                and summary.get("actual_training_rows") == 1536 and finite
                and digest == summary["adapter"]["sha256"]):
            raise SystemExit(f"Task 9D training artifact gate failed: {variant}{seed}")
PY

"$PYTHON" "$SERVER/run_task9d_inference.py" \
  --model-path "$MODEL" --manifest "$MANIFEST" \
  --output "$ROOT/evaluation/inference/Base" --group Base

for variant in A B C; do
  for seed in 17 29 43; do
    "$PYTHON" "$SERVER/run_task9d_inference.py" \
      --model-path "$MODEL" --manifest "$MANIFEST" \
      --output "$ROOT/evaluation/inference/${variant}${seed}" --group "${variant}${seed}" \
      --adapter "$ROOT/runs/${variant}/seed_${seed}/adapter"
  done
done
