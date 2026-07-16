"""Comprehensive pre-training gate for Task 9D formal matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from task9d_data import validate_v2_record


QV = re.compile(r"^model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$")


def decide_pretraining(checks: dict[str, bool]) -> dict[str, Any]:
    informational = {"three_step_generation_schema_valid"}
    failures = [key for key, value in checks.items() if key not in informational and value is not True]
    risks = [f"{key}=false" for key in sorted(informational) if checks.get(key) is False]
    return {
        "passed": not failures,
        "blocking_failures": failures,
        "informational_risks": risks,
        "rule": "all engineering/leakage gates block; three-step free-generation quality is reported but final schema rules remain blocking at evaluation",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pretraining(experiment_root: str | Path, output: str | Path) -> dict[str, Any]:
    experiment_root, output = Path(experiment_root), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing existing Task 9D pretraining gate: {output}")
    prepared = experiment_root / "preparation/protocol"
    protocol = json.loads((prepared / "freeze_report.json").read_text(encoding="utf-8"))
    eval_report = json.loads((prepared / "evaluation_protocol/report.json").read_text(encoding="utf-8"))
    details: dict[str, Any] = {"protocol": protocol, "eval_protocol": eval_report}
    checks = {
        "protocol": protocol.get("passed") is True and protocol.get("task8_locked_set_read") is False,
        "eval_protocol": eval_report.get("passed") is True and eval_report.get("rows") == 2560
                         and eval_report.get("task8_locked_set_read") is False,
    }

    gate_details = {}
    for variant in "ABC":
        report = json.loads((experiment_root / f"preparation/shortcut_gate/{variant}/metrics.json").read_text(encoding="utf-8"))
        gate_details[variant] = report["decision"]
    checks["shortcut_gates"] = all(value.get("decision") == "PASS" for value in gate_details.values())
    details["shortcut_gates"] = gate_details

    expected_roles = {
        "A": {"positive": 768, "semantic_negative": 768},
        "B": {"positive": 512, "semantic_negative": 512, "visual_counterfactual": 512},
        "C": {"positive": 512, "semantic_negative": 512, "visual_counterfactual": 512},
    }
    schedule_details = {}
    schedule_ok = True
    total_records = 0
    for variant in "ABC":
        train = _read_jsonl(prepared / f"variants/{variant}/train_schedule.jsonl")
        val = _read_jsonl(prepared / f"variants/{variant}/val.jsonl")
        total_records += len(train) + len(val)
        counts = dict(Counter(str(row["role"]) for row in train))
        variant_ok = len(train) == 1536 and len(val) == 576 and counts == expected_roles[variant]
        template_role: dict[str, Counter] = defaultdict(Counter)
        for row_index, row in enumerate(train + val):
            try:
                validate_v2_record(row["model"])
                reference = Path(str(row["model"]["messages"][1]["content"][0]["image"]))
                if reference.is_absolute() or ".." in reference.parts or not (prepared / reference).is_file():
                    variant_ok = False
            except Exception:
                variant_ok = False
            if row_index < len(train):
                template_role[str(row["role"])][str(row["template_id"])] += 1
        if variant == "C" and len({tuple(sorted(value.items())) for value in template_role.values()}) != 1:
            variant_ok = False
        schedule_ok &= variant_ok
        schedule_details[variant] = {"train_rows": len(train), "val_rows": len(val),
                                     "role_counts": counts, "valid": variant_ok}
    checks["schedule_and_schema"] = schedule_ok and total_records == 6336
    details["schedule_and_schema"] = {"records_checked": total_records, "variants": schedule_details}

    smoke_details, reload_details = {}, {}
    smoke_ok = reload_ok = True
    schema_valid = True
    for variant in "ABC":
        run = json.loads((experiment_root / f"smoke/{variant}/seed_17/run_summary.json").read_text(encoding="utf-8"))
        adapter = Path(run["adapter"]["path"])
        finite = all(math.isfinite(float(row[key])) for row in run["log_history"]
                     for key in ("loss", "eval_loss", "grad_norm") if key in row)
        targets_ok = len(run["lora_targets"]) == 72 and all(QV.fullmatch(name) for name in run["lora_targets"])
        hash_ok = adapter.is_file() and _sha256(adapter) == run["adapter"]["sha256"]
        one_ok = run.get("completed") is True and run.get("optimizer_steps") == 3 and finite and targets_ok \
                 and hash_ok and int(run["peak_vram_reserved_bytes"]) < 40 * 1024**3
        smoke_ok &= one_ok
        smoke_details[variant] = {"passed": one_ok, "losses": [row["loss"] for row in run["log_history"] if "loss" in row],
                                  "peak_vram_reserved_bytes": run["peak_vram_reserved_bytes"],
                                  "adapter_sha256": run["adapter"]["sha256"]}
        reload_summary = json.loads((experiment_root / f"smoke_reload/{variant}/run_summary.json").read_text(encoding="utf-8"))
        prediction = json.loads((experiment_root / f"smoke_reload/{variant}/predictions.jsonl").read_text(encoding="utf-8"))
        one_reload = reload_summary.get("state") == "completed" and reload_summary.get("prediction_count") == 1 \
                     and bool(str(prediction.get("raw_text", "")).strip())
        reload_ok &= one_reload
        try:
            value = json.loads(prediction["raw_text"])
            validate_v2_record({"id": "smoke", "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "image", "image": "opaque"},
                                              {"type": "text", "text": "query"}]},
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(value)}]},
            ]})
            one_schema = True
        except Exception:
            one_schema = False
        schema_valid &= one_schema
        reload_details[variant] = {"passed": one_reload, "three_step_schema_valid": one_schema,
                                   "prediction_sha256": reload_summary["predictions_sha256"]}
    checks["smoke_training"] = smoke_ok
    checks["adapter_reload_generation"] = reload_ok
    checks["three_step_generation_schema_valid"] = schema_valid
    details["smoke_training"] = smoke_details
    details["adapter_reload"] = reload_details

    decision = decide_pretraining(checks)
    report = {"version": "task9d-pretraining-gate-v1", **decision, "checks": checks, "details": details}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_pretraining(args.experiment_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
