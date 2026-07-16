#!/usr/bin/env bash
set -euo pipefail

/root/miniconda3/envs/eviagri/bin/python - <<'PY'
import json
import re
import subprocess
from pathlib import Path

experiment = Path("/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1")
formal = experiment / "formal"
launch_log = experiment / "formal-launch.log"
final_log = formal / "train.log"
screen = subprocess.run(["screen", "-ls"], text=True, capture_output=True)
screen_running = ".static_qlora_v1" in (screen.stdout + screen.stderr)

checkpoints = sorted(
    (path.name for path in formal.glob("checkpoint-*") if path.is_dir()),
    key=lambda name: int(name.rsplit("-", 1)[-1]),
)
latest_loss = None
latest_step = None
total_steps = None
state_files = [formal / name / "trainer_state.json" for name in checkpoints]
if (formal / "trainer_state.json").is_file():
    state_files.append(formal / "trainer_state.json")
for state_path in reversed(state_files):
    if not state_path.is_file():
        continue
    state = json.loads(state_path.read_text(encoding="utf-8"))
    latest_step = state.get("global_step")
    for row in reversed(state.get("log_history", [])):
        if "loss" in row:
            latest_loss = row["loss"]
            latest_step = row.get("step", latest_step)
            break
    if latest_loss is not None:
        break

log_path = final_log if final_log.is_file() else launch_log
if log_path.is_file():
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if latest_loss is None:
        matches = list(re.finditer(r"['\"]loss['\"]\s*:\s*([0-9.eE+-]+)", text))
        if matches:
            latest_loss = float(matches[-1].group(1))
    progress_matches = [
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(r"(\d+)/(\d+)\s+\[", text)
    ]
    total_steps = next((total for _done, total in progress_matches if total > 100), None)
    training_progress = [
        done for done, total in progress_matches if total_steps is not None and total == total_steps
    ]
    if training_progress:
        latest_step = max(latest_step or 0, training_progress[-1])

gpu_query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
    capture_output=True,
)
gpu = gpu_query.stdout.strip() if gpu_query.returncode == 0 else gpu_query.stderr.strip()
completed = (formal / "run_summary.json").is_file()
failed = (formal / "failure.json").is_file()
if completed:
    state_name = "completed"
elif failed:
    state_name = "failed"
elif screen_running:
    state_name = "running"
elif formal.exists():
    state_name = "stopped"
else:
    state_name = "not_started"

report = {
    "state": state_name,
    "screen_running": screen_running,
    "latest_step": latest_step,
    "total_steps": total_steps,
    "latest_loss": latest_loss,
    "gpu_used_total_utilization_mib_percent": gpu,
    "checkpoints": checkpoints,
    "completed": completed,
    "failed": failed,
    "log": str(log_path),
}
if formal.exists():
    status_path = formal / "status.json"
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)
print(json.dumps(report, indent=2))
PY
