#!/usr/bin/env bash
set -euo pipefail

/root/miniconda3/envs/eviagri/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path

evaluation = Path("/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/formal/evaluation")
status_path = evaluation / "status.json"  # evaluation/status.json
screen = subprocess.run(["screen", "-ls"], text=True, capture_output=True)
screen_output = screen.stdout + screen.stderr
screen_running = any(
    ".static_qlora_eval" in line and ("(Detached)" in line or "(Attached)" in line)
    for line in screen_output.splitlines()
)

def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())

progress = {
    split: {
        "predictions": line_count(evaluation / split / "predictions.jsonl"),
        "metrics_ready": (evaluation / split / "metrics.json").is_file(),
    }
    for split in ("val", "test")
}
written_status = None
if status_path.is_file():
    written_status = json.loads(status_path.read_text(encoding="utf-8"))
failures = sorted(path.name for path in evaluation.glob("failure_*.json"))
completed = (evaluation / "evaluation_summary.json").is_file()
if completed:
    state = "completed"
elif failures and not screen_running:
    state = "failed"
elif screen_running:
    state = "running"
elif evaluation.exists():
    state = "stopped"
else:
    state = "not_started"
gpu_query = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
    capture_output=True,
)
report = {
    "state": state,
    "screen_running": screen_running,
    "progress": progress,
    "writer_status": written_status,
    "failures": failures,
    "gpu_used_total_utilization_mib_percent": gpu_query.stdout.strip(),
}
print(json.dumps(report, indent=2))
PY
