"""Prepare a pinned, deterministic PlantDoc healthy-leaf real-null micro set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


REPOSITORY = "pratikkayal/PlantDoc-Dataset"
HEALTHY_CLASSES = (
    "Apple leaf",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Peach leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Strawberry leaf",
    "Tomato leaf",
    "grape leaf",
)
IMAGES_PER_CLASS = 4


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def deterministic_selection(
    entries: list[dict[str, Any]], *, commit: str, class_name: str, count: int
) -> list[dict[str, Any]]:
    images = [
        entry
        for entry in entries
        if entry.get("type") == "file"
        and str(entry.get("name", "")).lower().endswith((".jpg", ".jpeg", ".png"))
        and entry.get("sha")
    ]
    images.sort(
        key=lambda entry: hashlib.sha256(
            f"task11a2|{commit}|{class_name}|{entry['name']}".encode("utf-8")
        ).hexdigest()
    )
    if len(images) < count:
        raise ValueError(f"insufficient PlantDoc images: {class_name}")
    return images[:count]


def _read_task10b_hashes(feature_rows: Path) -> set[str]:
    return {
        str(json.loads(line).get("source_image_sha256"))
        for line in feature_rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _git(repository: Path, *arguments: str, text: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    return completed.stdout


def _clone_source(repository: Path) -> str:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-checkout",
            f"https://github.com/{REPOSITORY}.git",
            str(repository),
        ],
        check=True,
    )
    commit = str(_git(repository, "rev-parse", "HEAD", text=True)).strip()
    if len(commit) != 40:
        raise ValueError("invalid pinned PlantDoc commit")
    return commit


def _tree_entries(repository: Path, commit: str, class_name: str) -> list[dict[str, str]]:
    directory = f"test/{class_name}"
    payload = bytes(
        _git(repository, "ls-tree", "-r", "-z", commit, "--", directory)
    )
    entries = []
    for record in payload.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, object_type, object_sha = metadata.decode("ascii").split(" ")
        path = encoded_path.decode("utf-8", errors="strict")
        entries.append(
            {
                "type": "file" if object_type == "blob" else object_type,
                "name": Path(path).name,
                "path": path,
                "sha": object_sha,
                "mode": mode,
            }
        )
    return entries


def prepare_dataset(
    *,
    output_root: Path,
    task10b_feature_rows: Path,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    images_root = destination / "images"
    images_root.mkdir()
    source_repository = destination / "source_repository"
    commit = _clone_source(source_repository)
    task10b_hashes = _read_task10b_hashes(Path(task10b_feature_rows))
    rows = []
    panels = []
    for class_name in HEALTHY_CLASSES:
        entries = _tree_entries(source_repository, commit, class_name)
        chosen = deterministic_selection(
            entries, commit=commit, class_name=class_name, count=IMAGES_PER_CLASS
        )
        for entry in chosen:
            payload = bytes(_git(source_repository, "show", f"{commit}:{entry['path']}"))
            if git_blob_sha1(payload) != str(entry["sha"]):
                raise ValueError(f"Git blob SHA mismatch: {entry['name']}")
            file_sha256 = hashlib.sha256(payload).hexdigest()
            if file_sha256 in task10b_hashes:
                raise ValueError("PlantDoc/Task10B content overlap")
            suffix = Path(str(entry["name"])).suffix.lower()
            name = f"{file_sha256}{suffix}"
            path = images_root / name
            if path.exists():
                raise ValueError("duplicate selected PlantDoc content")
            path.write_bytes(payload)
            try:
                with Image.open(path) as loaded:
                    image = loaded.convert("RGB")
            except Exception as exc:
                raise ValueError(f"unreadable PlantDoc image: {entry['name']}") from exc
            rows.append(
                {
                    "id": hashlib.sha256(
                        f"task11a2|{commit}|{class_name}|{entry['name']}".encode("utf-8")
                    ).hexdigest()[:32],
                    "healthy_class": class_name,
                    "source_split": "test",
                    "source_name": str(entry["name"]),
                    "source_path": str(entry["path"]),
                    "source_url": (
                        f"https://github.com/{REPOSITORY}/blob/{commit}/"
                        + urllib.parse.quote(str(entry["path"]), safe="/")
                    ),
                    "git_blob_sha1": str(entry["sha"]),
                    "image_sha256": file_sha256,
                    "image": str(path),
                    "width": image.width,
                    "height": image.height,
                }
            )
            thumb = image.copy()
            thumb.thumbnail((220, 180))
            panel = Image.new("RGB", (240, 215), "white")
            panel.paste(thumb, ((240 - thumb.width) // 2, 28))
            ImageDraw.Draw(panel).text((8, 6), class_name, fill="black")
            panels.append(panel)
    if len(rows) != len(HEALTHY_CLASSES) * IMAGES_PER_CLASS:
        raise AssertionError("unexpected PlantDoc micro cardinality")
    license_payload = bytes(_git(source_repository, "show", f"{commit}:LICENSE.txt"))
    (destination / "LICENSE.txt").write_bytes(license_payload)
    with (destination / "manifest.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    columns = 5
    sheet = Image.new("RGB", (240 * columns, 215 * 8), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 240, (index // columns) * 215))
    sheet.save(destination / "audit_sheet.jpg", quality=94)
    report = {
        "version": "task11a2-plantdoc-null-report-1",
        "state": "completed",
        "repository": REPOSITORY,
        "commit": commit,
        "acquisition": "git shallow partial clone plus pinned blob reads",
        "license": "CC-BY-4.0",
        "source_split": "test",
        "healthy_classes": list(HEALTHY_CLASSES),
        "images_per_class": IMAGES_PER_CLASS,
        "image_count": len(rows),
        "unique_image_sha256": len({row["image_sha256"] for row in rows}),
        "task10b_content_overlap": 0,
        "visual_gate": "PENDING_MANUAL_AUDIT",
    }
    write_json_new(destination / "dataset_report.json", report)
    signed = ["manifest.jsonl", "LICENSE.txt", "audit_sheet.jpg", "dataset_report.json"] + [
        f"images/{Path(row['image']).name}" for row in rows
    ]
    with (destination / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Task 11A.2 PlantDoc real nulls")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task10b-feature-rows", type=Path, required=True)
    arguments = parser.parse_args()
    report = prepare_dataset(
        output_root=arguments.output_root,
        task10b_feature_rows=arguments.task10b_feature_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
