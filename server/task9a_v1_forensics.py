from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_INPUT_MARKERS = (
    "task8",
    "formal_clean_v2",
    "locked_confirmatory",
    "task9_dev_audit",
)
VIEWS = ("user_prompt", "system_user_prompt", "prompt_metadata")
EXPECTED_OUTPUT_KEYS = (
    "evidence_present",
    "evidence_bbox",
    "visible_attributes",
    "diagnosis",
    "reliability",
)
TOKEN = re.compile(r"[a-z0-9_]+")


def assert_allowed_input(path: Path) -> None:
    lowered = str(Path(path)).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_INPUT_MARKERS):
        raise ValueError(f"forbidden audit input: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path: Path, split: str) -> list[dict[str, Any]]:
    path = Path(path)
    assert_allowed_input(path)
    if not path.is_file():
        raise ValueError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"record at {path}:{line_number} must be an object")
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"missing record id at {path}:{line_number}")
            if record_id in seen:
                raise ValueError(f"duplicate record id: {record_id}")
            seen.add(record_id)
            if row.get("split") != split:
                raise ValueError(
                    f"split mismatch for {record_id}: expected {split}, found {row.get('split')}"
                )
            present = (row.get("target") or {}).get("evidence_present")
            if not isinstance(present, bool):
                raise ValueError(f"target.evidence_present must be boolean: {record_id}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _label(row: dict[str, Any]) -> str:
    return "positive" if row["target"]["evidence_present"] else "null"


def _canonical_target(row: dict[str, Any]) -> str:
    return json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))


def _assistant_text(row: dict[str, Any]) -> str | None:
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        texts = [
            item.get("text")
            for item in message.get("content") or []
            if item.get("type") == "text" and isinstance(item.get("text"), str)
        ]
        return "\n".join(texts) if texts else None
    return None


def _json_quality(row: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    text = _assistant_text(row)
    try:
        parsed = json.loads(text) if isinstance(text, str) else None
    except json.JSONDecodeError:
        parsed = None
    syntax_valid = isinstance(parsed, dict)
    schema_valid = bool(
        syntax_valid
        and tuple(parsed.keys()) == EXPECTED_OUTPUT_KEYS
        and isinstance(parsed.get("evidence_present"), bool)
        and (
            parsed.get("evidence_bbox") is None
            or (
                isinstance(parsed.get("evidence_bbox"), list)
                and len(parsed["evidence_bbox"]) == 4
                and all(isinstance(value, (int, float)) for value in parsed["evidence_bbox"])
            )
        )
        and isinstance(parsed.get("visible_attributes"), list)
        and isinstance(parsed.get("reliability"), str)
    )
    semantic_consistency = False
    if schema_valid and parsed["evidence_present"]:
        diagnosis = parsed.get("diagnosis")
        semantic_consistency = bool(
            parsed.get("evidence_bbox") is not None
            and isinstance(diagnosis, dict)
            and diagnosis.get("pest_id") is not None
            and isinstance(diagnosis.get("pest_name"), str)
            and diagnosis.get("pest_name")
        )
    elif schema_valid:
        diagnosis = parsed.get("diagnosis")
        semantic_consistency = bool(
            parsed.get("evidence_bbox") is None
            and isinstance(diagnosis, str)
            and diagnosis.strip().lower() in {"uncertain", "abstain"}
        )
    task_compliance = bool(semantic_consistency and parsed == row.get("target"))
    return syntax_valid, schema_valid, semantic_consistency, task_compliance


def _message_text(row: dict[str, Any], roles: set[str]) -> str:
    chunks: list[str] = []
    for message in row.get("messages") or []:
        if message.get("role") not in roles:
            continue
        for item in message.get("content") or []:
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                chunks.append(item["text"])
    return "\n".join(chunks)


def _user_prompt(row: dict[str, Any]) -> str:
    question = row.get("question")
    if isinstance(question, str):
        return question
    return _message_text(row, {"user"})


def _flatten_metadata(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key in sorted(value):
            result.extend(_flatten_metadata(value[key], f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        return [f"{prefix}={json.dumps(value, ensure_ascii=False, sort_keys=True)}"]
    return [f"{prefix}={value}"]


def _view_text(row: dict[str, Any], view: str) -> str:
    if view not in VIEWS:
        raise ValueError(f"unknown probe view: {view}")
    prompt = _user_prompt(row)
    if view == "user_prompt":
        return prompt
    system = _message_text(row, {"system"})
    if view == "system_user_prompt":
        return f"{system}\n{prompt}"
    exposed = {
        key: row.get(key)
        for key in (
            "id",
            "image",
            "source",
            "split",
            "source_split",
            "task_type",
            "query_pest_id",
            "query_pest_name",
            "metadata",
        )
        if key in row
    }
    return "\n".join([system, prompt, *_flatten_metadata(exposed)])


def _tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def _fit_bernoulli_nb(rows: list[dict[str, Any]], view: str) -> dict[str, Any]:
    labels = [bool(row["target"]["evidence_present"]) for row in rows]
    if not labels or all(labels) or not any(labels):
        raise ValueError("probe training split must contain both labels")
    documents = [_tokens(_view_text(row, view)) for row in rows]
    vocabulary = sorted(set().union(*documents))
    class_counts = {False: labels.count(False), True: labels.count(True)}
    token_counts = {False: Counter(), True: Counter()}
    for label, document in zip(labels, documents):
        token_counts[label].update(document)
    base_log_scores: dict[bool, float] = {}
    token_deltas: dict[bool, dict[str, float]] = {False: {}, True: {}}
    total = len(rows)
    for label in (False, True):
        class_count = class_counts[label]
        base = math.log((class_count + 1) / (total + 2))
        for token in vocabulary:
            probability = (token_counts[label][token] + 1) / (class_count + 2)
            base += math.log(1 - probability)
            token_deltas[label][token] = math.log(probability) - math.log(1 - probability)
        base_log_scores[label] = base
    return {
        "vocabulary": vocabulary,
        "class_counts": class_counts,
        "token_counts": token_counts,
        "base_log_scores": base_log_scores,
        "token_deltas": token_deltas,
        "rows": len(rows),
    }


def _score_nb(model: dict[str, Any], text: str) -> float:
    present = _tokens(text)
    log_scores: dict[bool, float] = {}
    for label in (False, True):
        log_scores[label] = model["base_log_scores"][label] + sum(
            model["token_deltas"][label][token]
            for token in present
            if token in model["token_deltas"][label]
        )
    difference = log_scores[True] - log_scores[False]
    if difference >= 0:
        return 1 / (1 + math.exp(-min(difference, 700)))
    exp_value = math.exp(max(difference, -700))
    return exp_value / (1 + exp_value)


def _balanced_accuracy(labels: list[bool], predictions: list[bool]) -> tuple[float, dict[str, int]]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError("probe evaluation split must contain both labels")
    value = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    return value, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _auroc(labels: list[bool], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        raise ValueError("AUROC requires both labels")
    wins = 0.0
    for left in positive:
        for right in negative:
            wins += 1.0 if left > right else 0.5 if left == right else 0.0
    return wins / (len(positive) * len(negative))


def probe_view(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], view: str
) -> dict[str, Any]:
    model = _fit_bernoulli_nb(train_rows, view)
    labels = [bool(row["target"]["evidence_present"]) for row in eval_rows]
    scores = [_score_nb(model, _view_text(row, view)) for row in eval_rows]
    predictions = [score >= 0.5 for score in scores]
    balanced_accuracy, confusion = _balanced_accuracy(labels, predictions)
    return {
        "view": view,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "vocabulary_size": len(model["vocabulary"]),
        "balanced_accuracy": balanced_accuracy,
        "auroc": _auroc(labels, scores),
        "confusion": confusion,
        "score_min": min(scores),
        "score_max": max(scores),
    }


def _length_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "unique": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "unique": len(set(values)),
    }


def profile_records(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    prompts: dict[str, Counter[str]] = {"positive": Counter(), "null": Counter()}
    task_types: dict[str, Counter[str]] = defaultdict(Counter)
    target_strings: dict[str, list[str]] = {"positive": [], "null": []}
    field_orders: dict[str, Counter[tuple[str, ...]]] = {"positive": Counter(), "null": Counter()}
    overlaps: dict[str, int] = {}
    record_id_label_token_rows = 0
    prompt_path_exposure_rows = 0
    class_counts: dict[str, Counter[str]] = {"positive": Counter(), "null": Counter()}
    quality_names = (
        "syntax_valid",
        "schema_valid",
        "semantic_consistency",
        "task_compliance",
    )
    quality_counts = Counter()
    quality_total = 0
    image_id_to_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        split_counts = Counter(_label(row) for row in rows)
        counts[split] = {
            "positive": split_counts["positive"],
            "null": split_counts["null"],
            "total": len(rows),
        }
        image_ids = {"positive": set(), "null": set()}
        for row in rows:
            quality_total += 1
            for name, passed in zip(quality_names, _json_quality(row)):
                quality_counts[name] += int(passed)
            label = _label(row)
            prompt = _user_prompt(row)
            prompts[label][prompt[:64]] += 1
            task_types[str(row.get("task_type", "<missing>"))][label] += 1
            target_strings[label].append(_canonical_target(row))
            field_orders[label][tuple(row["target"].keys())] += 1
            image_id = (row.get("metadata") or {}).get("image_id")
            if image_id is not None:
                image_ids[label].add(str(image_id))
                image_id_to_splits[str(image_id)].add(split)
            lowered_id = str(row.get("id", "")).lower()
            if (label == "positive" and "positive" in lowered_id) or (
                label == "null" and "null" in lowered_id
            ):
                record_id_label_token_rows += 1
            image = str(row.get("image", ""))
            if image and (Path(image).name in prompt or image in prompt):
                prompt_path_exposure_rows += 1
            diagnosis = row["target"].get("diagnosis")
            if isinstance(diagnosis, dict):
                class_counts[label][str(diagnosis.get("pest_id", "<missing>"))] += 1
            elif row.get("query_pest_id") is not None:
                class_counts[label][str(row["query_pest_id"])] += 1
        overlaps[split] = len(image_ids["positive"] & image_ids["null"])
    target_length = {
        label: _length_summary([len(value) for value in values])
        for label, values in target_strings.items()
    }
    return {
        "counts": counts,
        "prompt_prefix_by_label": {
            label: dict(counter.most_common()) for label, counter in prompts.items()
        },
        "task_type_by_label": {
            task: {"positive": counter["positive"], "null": counter["null"]}
            for task, counter in sorted(task_types.items())
        },
        "positive_null_image_id_overlap": overlaps,
        "image_ids_crossing_splits": sorted(
            image_id for image_id, splits in image_id_to_splits.items() if len(splits) > 1
        ),
        "target_unique_count_by_label": {
            label: len(set(values)) for label, values in target_strings.items()
        },
        "target_length_by_label": target_length,
        "field_order_by_label": {
            label: list(counter.most_common(1)[0][0]) if counter else []
            for label, counter in field_orders.items()
        },
        "field_order_variants_by_label": {
            label: len(counter) for label, counter in field_orders.items()
        },
        "path_label_tokens": {
            "record_id_label_token_rows": record_id_label_token_rows,
            "prompt_path_exposure_rows": prompt_path_exposure_rows,
        },
        "class_counts_by_label": {
            label: dict(sorted(counter.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 10**9))
            for label, counter in class_counts.items()
        },
        "json_quality": {
            name: {
                "count": quality_counts[name],
                "total": quality_total,
                "rate": quality_counts[name] / quality_total,
            }
            for name in quality_names
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Task 9A Static QLoRA v1 Data Forensics", "", "## Counts", ""]
    for split, values in report["profile"]["counts"].items():
        lines.append(
            f"- {split}: {values['positive']} positive + {values['null']} null = {values['total']}"
        )
    lines.extend(["", "## Held-out text-only probes", ""])
    for split, views in report["probes"].items():
        for view, metrics in views.items():
            lines.append(
                f"- {split}/{view}: Balanced Accuracy={metrics['balanced_accuracy']:.6f}, "
                f"AUROC={metrics['auroc']:.6f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a forensic characterization of v1, not the v2 shortcut gate.",
            "- No Task 8 or locked-confirmatory input is accepted by the auditor.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output_dir}")
    rows_by_split = {split: load_split(path, split) for split, path in paths.items()}
    report = {
        "version": "task9a-v1-forensics-1",
        "scope": "static_qlora_v1_only",
        "profile": profile_records(rows_by_split),
        "probes": {
            split: {
                view: probe_view(rows_by_split["train"], rows_by_split[split], view)
                for view in VIEWS
            }
            for split in ("val", "test")
        },
    }
    input_manifest = {
        split: {"path": str(path), "sha256": sha256_file(path), "rows": len(rows_by_split[split])}
        for split, path in paths.items()
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "forensic_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "forensic_report.md").write_text(
            _markdown_report(report), encoding="utf-8"
        )
        (temporary / "input_manifest.json").write_text(
            json.dumps(input_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        status = {"state": "completed", "scope": report["scope"], "locked_set_read": False}
        (temporary / "run_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        files = [
            temporary / "forensic_report.json",
            temporary / "forensic_report.md",
            temporary / "input_manifest.json",
            temporary / "run_status.json",
        ]
        (temporary / "completion.sha256").write_text(
            "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 9A Static QLoRA v1 data forensics")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run_audit(
        {"train": args.train_jsonl, "val": args.val_jsonl, "test": args.test_jsonl},
        args.output_dir,
    )
    print(json.dumps({"counts": report["profile"]["counts"], "probes": report["probes"]}, indent=2))


if __name__ == "__main__":
    main()
