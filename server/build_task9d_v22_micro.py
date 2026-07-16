"""Build the frozen, paired Task 9D v2.2 micro protocol from Task 9D only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROLES = ("positive", "semantic_negative", "visual_counterfactual")
DEFAULT_BAND_QUOTAS = {"head": 11, "medium": 11, "tail": 10}
EXPECTED_DEV_VIEWS = {"canonical", "native_0", "native_1", "native_2", "unseen_alpha", "unseen_beta"}
EXPECTED_DEV_NULLS = {"semantic_null", "source_visual_null", "blank", "blur", "shuffle"}


def _digest(value: str) -> str:
    return hashlib.sha256(f"task9d-v22-micro|{value}".encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _target_class(row: dict[str, Any]) -> int:
    target = json.loads(row["model"]["messages"][2]["content"][0]["text"])
    value = target["diagnosis"]["pest_id"]
    if not isinstance(value, int):
        raise ValueError("positive schedule target lacks integer pest_id")
    return value


def _group_schedule(rows: list[dict[str, Any]], split: str):
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[str(row["family_id"])].append(row)
    by_class: dict[int, list[str]] = defaultdict(list)
    for family, values in families.items():
        counts = Counter(str(row["role"]) for row in values)
        if counts != Counter({role: 1 for role in ROLES}):
            raise ValueError(f"{split} family {family} does not contain exact three roles")
        positive = next(row for row in values if row["role"] == "positive")
        by_class[_target_class(positive)].append(family)
    return families, by_class


def _group_full_dev(rows: list[dict[str, Any]]):
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[str(row["family_id"])].append(row)
    full: dict[str, list[dict[str, Any]]] = {}
    by_class: dict[int, list[str]] = defaultdict(list)
    for family, values in families.items():
        if len(values) != 11:
            continue
        positive = [
            row for row in values
            if row["role"] == "positive" and row["condition"] == "original"
        ]
        views = {str(row["prompt_view"]) for row in positive}
        nulls = {str(row["condition"]) for row in values if row["condition"] != "original"}
        if len(positive) != 6 or views != EXPECTED_DEV_VIEWS or nulls != EXPECTED_DEV_NULLS:
            raise ValueError(f"dev family {family} has an incomplete audit condition set")
        canonical = next(row for row in positive if row["prompt_view"] == "canonical")
        class_id = int(canonical["query_class_id"])
        if any(int(row["query_class_id"]) != class_id for row in positive):
            raise ValueError(f"dev family {family} positive prompt views disagree on class")
        full[family] = values
        by_class[class_id].append(family)
    return full, by_class


def _pick(values: list[Any], count: int, namespace: str) -> list[Any]:
    ranked = sorted(values, key=lambda value: (_digest(f"{namespace}|{value}"), str(value)))
    if len(ranked) < count:
        raise ValueError(f"insufficient candidates for {namespace}: {len(ranked)} < {count}")
    return ranked[:count]


def build_micro_protocol(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    class_bands: dict[str, str],
    *,
    band_quotas: dict[str, int] | None = None,
    total_exposures: int = 512,
) -> dict[str, Any]:
    quotas = dict(DEFAULT_BAND_QUOTAS if band_quotas is None else band_quotas)
    train_family_ids = {str(row["family_id"]) for row in train_rows}
    val_family_ids = {str(row["family_id"]) for row in val_rows}
    dev_family_ids = {str(row["family_id"]) for row in evaluation_rows}
    overlaps = {
        "train_val": len(train_family_ids & val_family_ids),
        "train_dev": len(train_family_ids & dev_family_ids),
        "val_dev": len(val_family_ids & dev_family_ids),
    }
    if any(overlaps.values()):
        raise ValueError(f"family overlap across split: {overlaps}")
    train_families, train_classes = _group_schedule(train_rows, "train")
    val_families, val_classes = _group_schedule(val_rows, "val")
    dev_families, dev_classes = _group_full_dev(evaluation_rows)
    eligible = [
        class_id for class_id, families in train_classes.items()
        if len(families) >= 2 and len(val_classes.get(class_id, [])) >= 1
        and len(dev_classes.get(class_id, [])) >= 1 and str(class_id) in class_bands
    ]
    selected_ids: list[int] = []
    availability: dict[str, int] = {}
    for band, quota in sorted(quotas.items()):
        candidates = [class_id for class_id in eligible if class_bands[str(class_id)] == band]
        availability[band] = len(candidates)
        if len(candidates) < quota:
            raise ValueError(
                f"infeasible strict class quota for {band}: {len(candidates)} < {quota}"
            )
        selected_ids.extend(_pick(candidates, quota, f"class-{band}"))
    selected = [
        {"class_id": class_id, "band": class_bands[str(class_id)]}
        for class_id in sorted(selected_ids)
    ]
    selected_train: list[str] = []
    selected_val: list[str] = []
    selected_dev: list[str] = []
    for class_id in selected_ids:
        selected_train.extend(_pick(train_classes[class_id], 2, f"train-family-{class_id}"))
        selected_val.extend(_pick(val_classes[class_id], 1, f"val-family-{class_id}"))
        selected_dev.extend(_pick(dev_classes[class_id], 1, f"dev-family-{class_id}"))
    unique_train = [
        copy.deepcopy(row) for family in selected_train for row in train_families[family]
    ]
    unique_train.sort(key=lambda row: (_digest(f"unique-train|{row['id']}"), str(row["id"])))
    pools = {
        role: sorted(
            [row for row in unique_train if row["role"] == role],
            key=lambda row: (_digest(f"exposure|{role}|{row['id']}"), str(row["id"])),
        ) for role in ROLES
    }
    exposure_counts = Counter()
    train_schedule: list[dict[str, Any]] = []
    role_offsets = Counter()
    for index in range(total_exposures):
        role = ROLES[index % len(ROLES)]
        source = pools[role][role_offsets[role] % len(pools[role])]
        role_offsets[role] += 1
        row = copy.deepcopy(source)
        source_id = str(row["id"])
        identifier = _digest(f"exposure|{index}|{source_id}")[:32]
        row["source_schedule_id"] = source_id
        row["id"] = identifier
        row["model"]["id"] = identifier
        row["schedule_index"] = index
        train_schedule.append(row)
        exposure_counts[role] += 1
    chosen_val = [copy.deepcopy(row) for family in selected_val for row in val_families[family]]
    chosen_eval = [copy.deepcopy(row) for family in selected_dev for row in dev_families[family]]
    chosen_val.sort(key=lambda row: (_digest(f"val|{row['id']}"), str(row["id"])))
    chosen_eval.sort(key=lambda row: (_digest(f"dev|{row['id']}"), str(row["id"])))
    report = {
        "version": "task9d-v22-micro-protocol-v1",
        "passed": True,
        "task8_locked_set_read": False,
        "selected_class_count": len(selected),
        "band_quotas": quotas,
        "eligible_by_band": availability,
        "unique_train_families": len(selected_train),
        "val_families": len(selected_val),
        "dev_families": len(selected_dev),
        "unique_train_rows": len(unique_train),
        "train_exposures": len(train_schedule),
        "val_rows": len(chosen_val),
        "evaluation_rows": len(chosen_eval),
        "role_exposures": dict(sorted(exposure_counts.items())),
        "split_family_overlap": overlaps,
        "arms_share_exact_schedule": True,
    }
    return {
        "selected_classes": selected,
        "unique_train_rows": unique_train,
        "train_schedule": train_schedule,
        "val_rows": chosen_val,
        "evaluation_rows": chosen_eval,
        "report": report,
    }


def write_micro_protocol(result: dict[str, Any], output_root: Path, inputs: list[Path]) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite v2.2 protocol: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    files = {
        "selected_classes.json": result["selected_classes"],
        "protocol_report.json": result["report"],
    }
    for name, value in files.items():
        _write_json(output_root / name, value)
    _write_jsonl(output_root / "unique_train.jsonl", result["unique_train_rows"])
    _write_jsonl(output_root / "train_schedule.jsonl", result["train_schedule"])
    _write_jsonl(output_root / "val.jsonl", result["val_rows"])
    _write_jsonl(output_root / "evaluation_manifest.jsonl", result["evaluation_rows"])
    input_hashes = {str(path): _sha256(path) for path in inputs}
    _write_json(output_root / "input_sha256.json", input_hashes)
    names = [
        "selected_classes.json", "protocol_report.json", "unique_train.jsonl",
        "train_schedule.jsonl", "val.jsonl", "evaluation_manifest.jsonl", "input_sha256.json",
    ]
    (output_root / "completion.sha256").write_text(
        "".join(f"{_sha256(output_root / name)}  {name}\n" for name in names), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        args.protocol_root / "variants/B/train_schedule.jsonl",
        args.protocol_root / "variants/B/val.jsonl",
        args.protocol_root / "evaluation_protocol/manifest.jsonl",
        args.protocol_root / "class_bands.json",
    ]
    result = build_micro_protocol(
        _read_jsonl(inputs[0]), _read_jsonl(inputs[1]), _read_jsonl(inputs[2]),
        json.loads(inputs[3].read_text(encoding="utf-8")),
    )
    write_micro_protocol(result, args.output_root, inputs)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
