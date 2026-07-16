"""One-to-one Task 9B v2 -> v2.1 prompt-distribution repair."""

from __future__ import annotations

import copy
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from task9b_protocol import build_prompt
from task9b_v21_exact_match import ExactMatchInfeasible, match_all_strata
from validate_task9b_freeze import validate_freeze


QUERY_PATTERN = re.compile(r"queried pest '(.+?)'")


def _user_text(model_row: dict[str, Any]) -> str:
    try:
        return model_row["messages"][1]["content"][1]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("model row does not follow frozen system/user/assistant layout") from exc


def _set_user_text(model_row: dict[str, Any], value: str) -> None:
    model_row["messages"][1]["content"][1]["text"] = value


def _query_name(model_row: dict[str, Any]) -> str:
    match = QUERY_PATTERN.search(_user_text(model_row))
    if match is None:
        raise ValueError("cannot extract queried pest name from frozen prompt")
    return match.group(1)


def repair_records(
    model_by_id: Mapping[str, dict[str, Any]],
    provenance: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    original_provenance = [dict(row) for row in provenance]
    if set(model_by_id) != {str(row["id"]) for row in original_provenance}:
        raise ValueError("model/provenance ID sets are not aligned")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original_provenance:
        by_family[str(row["family_id"])].append(row)

    class_names: dict[int, str] = {}
    families = []
    expected_roles = Counter({"positive": 1, "semantic_negative": 1, "visual_counterfactual": 1})
    for family_id, rows in sorted(by_family.items()):
        if Counter(str(row["role"]) for row in rows) != expected_roles:
            raise ValueError(f"family {family_id} does not have exact 1:1:1 roles")
        positive = next(row for row in rows if row["role"] == "positive")
        class_id = int(positive["query_class_id"])
        name = _query_name(model_by_id[str(positive["id"])])
        if class_id in class_names and class_names[class_id] != name:
            raise ValueError(f"class {class_id} has inconsistent visible names")
        class_names[class_id] = name
        families.append(
            {
                "family_id": family_id,
                "split": str(positive["split"]),
                "template_id": str(positive["template_id"]),
                "positive_query_class_id": class_id,
                "present_class_ids": list(positive["present_class_ids"]),
            }
        )

    matching = match_all_strata(families)
    missing_names = set(matching["assignment"].values()) - set(class_names)
    if missing_names:
        raise ValueError(f"matched query classes lack visible names: {sorted(missing_names)}")

    repaired_models = {str(key): copy.deepcopy(value) for key, value in model_by_id.items()}
    repaired_provenance = [copy.deepcopy(row) for row in original_provenance]
    for row in repaired_provenance:
        if row["role"] != "semantic_negative":
            continue
        class_id = int(matching["assignment"][str(row["family_id"])])
        if class_id in {int(value) for value in row["present_class_ids"]}:
            raise AssertionError("exact matcher emitted a present semantic query")
        row["query_class_id"] = class_id
        model = repaired_models[str(row["id"])]
        _set_user_text(model, build_prompt(str(row["template_id"]), class_names[class_id]))

    original_family_sequence = [str(row["family_id"]) for row in original_provenance]
    repaired_family_sequence = [str(row["family_id"]) for row in repaired_provenance]
    if original_family_sequence != repaired_family_sequence:
        raise AssertionError("repair deleted, duplicated, reordered, or moved a family row")
    for old, new in zip(original_provenance, repaired_provenance):
        if old["split"] != new["split"] or old["template_id"] != new["template_id"]:
            raise AssertionError("repair moved a family across split/template")
        if old["role"] != "semantic_negative" and old != new:
            raise AssertionError("repair modified a non-semantic row")

    max_tv = max(float(item["total_variation"]) for item in matching["strata"])
    report = {
        **{key: value for key, value in matching.items() if key != "assignment"},
        "version": "task9b-v21-repair-report-1",
        "changed_fields": ["semantic_negative.query_class_id", "semantic_negative.user_prompt"],
        "max_total_variation": max_tv,
        "rows_before": len(original_provenance),
        "rows_after": len(repaired_provenance),
        "no_family_deleted": True,
        "no_family_duplicated": True,
        "no_family_moved_across_split": True,
    }
    return repaired_models, repaired_provenance, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def repair_frozen_dataset(source_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    source_root, output_root = Path(source_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing existing v2.1 output: {output_root}")
    source_freeze = json.loads((source_root / "freeze_report.json").read_text(encoding="utf-8"))
    if source_freeze.get("passed") is not True:
        raise ValueError("source v2 freeze report is not passed")
    model_paths = {
        "train": source_root / "model" / "train.jsonl",
        "val": source_root / "model" / "val.jsonl",
        "dev": source_root / "dev_audit" / "model.jsonl",
    }
    ordered_models = {split: _read_jsonl(path) for split, path in model_paths.items()}
    model_by_id = {str(row["id"]): row for rows in ordered_models.values() for row in rows}
    provenance = _read_jsonl(source_root / "private" / "provenance.jsonl")

    # Feasibility is solved completely before the output directory is created.
    repaired_models, repaired_provenance, matching_report = repair_records(model_by_id, provenance)

    output_root.mkdir(parents=True)
    image_output = output_root / "images"
    image_output.mkdir()
    hardlinked = 0
    for source in sorted((source_root / "images").iterdir()):
        if not source.is_file():
            continue
        destination = image_output / source.name
        os.link(source, destination)
        if os.stat(source).st_ino != os.stat(destination).st_ino:
            raise AssertionError("image reuse is not a hardlink")
        hardlinked += 1

    for split, rows in ordered_models.items():
        destination = output_root / ("dev_audit/model.jsonl" if split == "dev" else f"model/{split}.jsonl")
        _write_jsonl(destination, (repaired_models[str(row["id"])] for row in rows))
    private = output_root / "private"
    _write_jsonl(private / "provenance.jsonl", repaired_provenance)
    for name in ("locked_exclusion.json", "split_manifest.json", "source_inputs.json"):
        value = json.loads((source_root / "private" / name).read_text(encoding="utf-8"))
        (private / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    matching_report["hardlinked_images"] = hardlinked
    matching_report["source_protocol_manifest_sha256"] = __import__("hashlib").sha256(
        (source_root / "protocol_manifest.json").read_bytes()
    ).hexdigest()
    (private / "exact_matching_report.json").write_text(
        json.dumps(matching_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    split_counts = Counter(row["split"] for row in repaired_provenance if row["role"] == "positive")
    summary = {
        "version": "task9b-v2.1-exact-query-match-1",
        "families_by_split": dict(sorted(split_counts.items())),
        "rows_by_split": {key: value * 3 for key, value in sorted(split_counts.items())},
        "family_count": matching_report["family_count_after"],
        "row_count": matching_report["rows_after"],
        "family_bijection": True,
        "max_query_distribution_total_variation": matching_report["max_total_variation"],
    }
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    locked = json.loads((private / "locked_exclusion.json").read_text(encoding="utf-8"))
    freeze = validate_freeze(output_root, locked_exclusion=locked)
    return {"matching": matching_report, "summary": summary, "freeze": freeze}


def assess_frozen_feasibility(source_root: str | Path) -> dict[str, Any]:
    source_root = Path(source_root)
    paths = [
        source_root / "model" / "train.jsonl",
        source_root / "model" / "val.jsonl",
        source_root / "dev_audit" / "model.jsonl",
    ]
    rows = [row for path in paths for row in _read_jsonl(path)]
    model_by_id = {str(row["id"]): row for row in rows}
    provenance = _read_jsonl(source_root / "private" / "provenance.jsonl")
    _, _, report = repair_records(model_by_id, provenance)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--feasibility-only", action="store_true")
    parser.add_argument("--status-report", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.feasibility_only and arguments.output_root is None:
        parser.error("--output-root is required unless --feasibility-only is used")
    if arguments.status_report.exists():
        raise FileExistsError(f"refusing existing status report: {arguments.status_report}")
    try:
        if arguments.feasibility_only:
            result = {"state": "feasible", "matching": assess_frozen_feasibility(arguments.source_root)}
        else:
            result = {"state": "completed", **repair_frozen_dataset(arguments.source_root, arguments.output_root)}
    except ExactMatchInfeasible as exc:
        result = {"state": "blocked", "matching": exc.report}
        arguments.status_report.parent.mkdir(parents=True, exist_ok=True)
        arguments.status_report.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(2)
    arguments.status_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.status_report.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
