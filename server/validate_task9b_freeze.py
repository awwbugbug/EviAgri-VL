"""Fail-closed validator and 9C probe exporter for Task 9B."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from task9b_protocol import DIAGNOSIS_KEYS, OUTPUT_KEYS, validate_target_semantics
from task9b_transforms import DEV_TRANSFORMS, TRAIN_TRANSFORMS


OPAQUE_IMAGE = re.compile(r"^images/[0-9a-f]{64}\.(?:png|jpg|jpeg)$")
BANNED_VISIBLE = (
    "task_type",
    "positive sample",
    "null sample",
    "synthetic_null",
    "real_null",
    "source_image",
    "/root/",
    "\\datasets\\",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON syntax failure in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object in {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _message_parts(row: dict[str, Any]) -> tuple[str, str, str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system", "user", "assistant"
    ]:
        raise ValueError("message roles/order violate frozen protocol")
    try:
        system_text = messages[0]["content"][0]["text"]
        image_ref = messages[1]["content"][0]["image"]
        user_text = messages[1]["content"][1]["text"]
        assistant_text = messages[2]["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("message content violates frozen protocol") from exc
    if not all(isinstance(value, str) for value in (system_text, image_ref, user_text, assistant_text)):
        raise ValueError("message values must be strings")
    return system_text, user_text, image_ref, assistant_text


def validate_freeze(
    dataset_root: str | Path,
    *,
    locked_exclusion: dict[str, Any],
) -> dict[str, Any]:
    root = Path(dataset_root)
    probes_root = root / "private" / "9c_probes"
    outputs = (root / "freeze_report.json", root / "protocol_manifest.json", root / "completion.sha256")
    if probes_root.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("freeze outputs already exist")

    model_paths = {
        "train": root / "model" / "train.jsonl",
        "val": root / "model" / "val.jsonl",
        "dev": root / "dev_audit" / "model.jsonl",
    }
    model_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for split, path in model_paths.items():
        for row in _read_jsonl(path):
            if set(row) != {"id", "messages"}:
                raise ValueError("model-visible key allowlist violation")
            identifier = str(row["id"])
            if not re.fullmatch(r"[0-9a-f]{32}", identifier) or identifier in model_by_id:
                raise ValueError("model-visible ID is non-opaque or duplicated")
            system_text, user_text, image_ref, _ = _message_parts(row)
            visible_text = f"{system_text}\n{user_text}\n{image_ref}".lower()
            if any(token in visible_text for token in BANNED_VISIBLE):
                raise ValueError("prompt or non-image metadata leakage")
            if OPAQUE_IMAGE.fullmatch(image_ref) is None:
                raise ValueError("image path is not an opaque derived reference")
            if not (root / image_ref).is_file():
                raise ValueError("model-visible image reference is missing")
            model_by_id[identifier] = (split, row)

    provenance_path = root / "private" / "provenance.jsonl"
    provenance = _read_jsonl(provenance_path)
    private_by_id = {str(row.get("id")): row for row in provenance}
    if len(private_by_id) != len(provenance) or set(private_by_id) != set(model_by_id):
        raise ValueError("model/private ID bijection failure")

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    template_by_role: dict[str, Counter] = defaultdict(Counter)
    length_by_role: dict[str, Counter] = defaultdict(Counter)
    source_splits: dict[str, set[str]] = defaultdict(set)
    sha_splits: dict[str, set[str]] = defaultdict(set)
    component_splits: dict[str, set[str]] = defaultdict(set)
    quality_counts = Counter()
    probe_rows = {name: [] for name in (
        "user_prompt_only", "system_user_prompt", "prompt_nonimage_metadata"
    )}

    locked_ids = {str(value) for value in locked_exclusion.get("image_ids", [])}
    locked_sha = {str(value) for value in locked_exclusion.get("image_sha256", [])}
    for identifier, private in private_by_id.items():
        split, model = model_by_id[identifier]
        if private.get("split") != split:
            raise ValueError("private/model split disagreement")
        source_id = str(private.get("source_image_id", ""))
        source_sha = str(private.get("source_image_sha256", ""))
        if source_id in locked_ids or source_sha in locked_sha:
            raise ValueError("locked source overlap")
        source_splits[source_id].add(split)
        sha_splits[source_sha].add(split)
        component_splits[str(private.get("near_duplicate_component_id", ""))].add(split)
        families[str(private.get("family_id", ""))].append(private)
        role = str(private.get("role", ""))
        template_by_role[role][str(private.get("template_id"))] += 1
        length_by_role[role][int(private.get("length_bucket"))] += 1

        transform_id = private.get("transform_id")
        if role == "visual_counterfactual":
            allowed = DEV_TRANSFORMS if split == "dev" else TRAIN_TRANSFORMS
            if transform_id not in allowed:
                raise ValueError("transform registry collision")
        elif transform_id is not None:
            raise ValueError("non-counterfactual row exposes a transform")

        system_text, user_text, image_ref, assistant_text = _message_parts(model)
        quality_counts["syntax"] += 1
        try:
            target = json.loads(assistant_text)
        except json.JSONDecodeError as exc:
            raise ValueError("assistant JSON syntax failure") from exc
        if (
            not isinstance(target, dict)
            or tuple(target) != OUTPUT_KEYS
            or not isinstance(target.get("diagnosis"), dict)
            or tuple(target["diagnosis"]) != DIAGNOSIS_KEYS
        ):
            raise ValueError("assistant JSON schema failure")
        quality_counts["schema"] += 1
        try:
            validate_target_semantics(target)
        except ValueError as exc:
            raise ValueError("assistant JSON semantic consistency failure") from exc
        quality_counts["semantic"] += 1
        expected = bool(private.get("evidence_present"))
        if target["evidence_present"] is not expected:
            raise ValueError("assistant task compliance failure")
        quality_counts["compliance"] += 1
        if len(assistant_text) != int(private["length_bucket"]):
            raise ValueError("assistant target-length contract failure")

        base_probe = {
            "id": identifier,
            "family_id": private["family_id"],
            "split": split,
            "label": int(expected),
        }
        probe_rows["user_prompt_only"].append({**base_probe, "text": user_text})
        probe_rows["system_user_prompt"].append({**base_probe, "text": system_text + "\n" + user_text})
        metadata = {
            "record_id": identifier,
            "image_ref": image_ref,
            "message_roles": [item["role"] for item in model["messages"]],
            "content_types": [[part["type"] for part in item["content"]] for item in model["messages"]],
        }
        probe_rows["prompt_nonimage_metadata"].append(
            {**base_probe, "text": user_text + "\n" + json.dumps(metadata, sort_keys=True)}
        )

    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("source image overlap across splits")
    if any(len(splits) != 1 for splits in sha_splits.values()):
        raise ValueError("source SHA overlap across splits")
    if any(len(splits) != 1 for splits in component_splits.values()):
        raise ValueError("near-duplicate component overlap across splits")
    expected_roles = Counter({"positive": 1, "semantic_negative": 1, "visual_counterfactual": 1})
    if any(Counter(row["role"] for row in rows) != expected_roles for rows in families.values()):
        raise ValueError("family role imbalance")
    if len({tuple(sorted(counter.items())) for counter in template_by_role.values()}) != 1:
        raise ValueError("template distribution differs by role")
    if len({tuple(sorted(counter.items())) for counter in length_by_role.values()}) != 1:
        raise ValueError("target-length distribution differs by role")

    probe_files = {}
    probes_root.mkdir(parents=True, exist_ok=False)
    for name, rows in probe_rows.items():
        path = probes_root / f"{name}.jsonl"
        _write_jsonl(path, rows)
        probe_files[name] = str(path.resolve())

    total = len(provenance)
    report = {
        "version": "task9b-freeze-validator-1",
        "passed": True,
        "families": len(families),
        "rows": total,
        "json_quality": {
            "syntax_validity": quality_counts["syntax"] / total,
            "schema_validity": quality_counts["schema"] / total,
            "semantic_consistency": quality_counts["semantic"] / total,
            "task_compliance": quality_counts["compliance"] / total,
        },
        "zero_overlap": {"source_id": True, "source_sha256": True, "near_duplicate_component": True},
        "role_counts": dict(Counter(row["role"] for row in provenance)),
        "null_source_counts": {str(k): v for k, v in Counter(row["null_source"] for row in provenance).items()},
        "probe_files": probe_files,
    }
    report_path = root / "freeze_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    protected_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"protocol_manifest.json", "completion.sha256"}
    )
    manifest = {
        "version": "task9b-protocol-manifest-1",
        "files": {
            str(path.relative_to(root)).replace("\\", "/"): _file_sha256(path)
            for path in protected_files
        },
    }
    manifest_path = root / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    completion_files = [*protected_files, manifest_path]
    completion = root / "completion.sha256"
    completion.write_text(
        "".join(f"{_file_sha256(path)}  {str(path.relative_to(root)).replace(chr(92), '/')}\n" for path in sorted(completion_files)),
        encoding="utf-8",
    )
    return report
