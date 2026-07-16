from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


READY = "ready"
WAITING = "waiting"
BLOCKED = "blocked"


@dataclass(frozen=True)
class GateReport:
    state: str
    evaluator_active: bool
    counts: dict[str, int]
    reasons: list[str]


def screen_session_active(output: str, session_name: str) -> bool:
    marker = f".{session_name}"
    return any(
        marker in line and ("(Detached)" in line or "(Attached)" in line)
        for line in output.splitlines()
    )


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_summary(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, "evaluation_summary.json is missing"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"evaluation_summary.json is invalid: {error}"
    if summary.get("completed") is not True:
        return summary, 'evaluation_summary.json does not contain "completed": true'
    return summary, None


def evaluate_shutdown_gate(
    evaluation_root: Path,
    expected_counts: dict[str, int],
    evaluator_active: bool,
) -> GateReport:
    evaluation_root = Path(evaluation_root)
    counts = {
        split: _line_count(evaluation_root / split / "predictions.jsonl")
        for split in expected_counts
    }
    failures = sorted(evaluation_root.glob("failure_*.json"))
    if failures:
        return GateReport(
            BLOCKED,
            evaluator_active,
            counts,
            ["failure reports exist: " + ", ".join(path.name for path in failures)],
        )

    _, summary_error = _load_summary(evaluation_root / "evaluation_summary.json")
    reasons = [summary_error] if summary_error else []
    for split, expected in expected_counts.items():
        actual = counts[split]
        if actual != expected:
            reasons.append(f"{split} predictions: expected {expected}, found {actual}")
        for filename in ("metrics.json", "failures.jsonl"):
            if not (evaluation_root / split / filename).is_file():
                reasons.append(f"{split}/{filename} is missing")

    if reasons:
        state = WAITING if evaluator_active and summary_error == "evaluation_summary.json is missing" else BLOCKED
        return GateReport(state, evaluator_active, counts, reasons)
    if evaluator_active:
        return GateReport(WAITING, True, counts, ["evaluator screen is still active"])
    return GateReport(READY, False, counts, [])


def write_checksum_manifest(evaluation_root: Path) -> Path:
    evaluation_root = Path(evaluation_root)
    relative_paths = [Path("evaluation_summary.json"), Path("status.json")]
    for split in ("val", "test"):
        relative_paths.extend(
            Path(split) / name
            for name in ("predictions.jsonl", "metrics.json", "failures.jsonl")
        )
    manifest = evaluation_root / "completion_sha256.txt"
    lines = []
    for relative_path in relative_paths:
        path = evaluation_root / relative_path
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative_path.as_posix()}\n")
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8", newline="\n")
    temporary.replace(manifest)
    return manifest


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-safe evaluation completion gate")
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--expected-val", required=True, type=int)
    parser.add_argument("--expected-test", required=True, type=int)
    parser.add_argument("--screen-name", default="static_qlora_eval")
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screen = subprocess.run(["screen", "-ls"], text=True, capture_output=True)
    active = screen_session_active(screen.stdout + screen.stderr, args.screen_name)
    report = evaluate_shutdown_gate(
        args.evaluation_root,
        {"val": args.expected_val, "test": args.expected_test},
        evaluator_active=active,
    )
    payload = asdict(report)
    if report.state == READY and args.write_manifest:
        payload["checksum_manifest"] = str(write_checksum_manifest(args.evaluation_root))
    _write_json_atomic(args.evaluation_root / "shutdown_guard_status.json", payload)
    print(json.dumps(payload, indent=2))
    raise SystemExit({READY: 0, WAITING: 10, BLOCKED: 20}[report.state])


if __name__ == "__main__":
    main()
