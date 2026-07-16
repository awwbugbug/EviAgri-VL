"""Build aligned text-only shortcut probes from actual Task 9D model inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VIEWS = ("user_prompt_only", "system_user_prompt", "prompt_nonimage_metadata")


def probe_views(envelope: dict[str, Any], *, split: str, index: int) -> dict[str, dict[str, Any]]:
    model = envelope.get("model", envelope)
    messages = model["messages"]
    system_text = str(messages[0]["content"][0]["text"])
    user_text = str(messages[1]["content"][1]["text"])
    role = str(envelope["role"])
    common = {
        "id": f"{split}:{index:08d}:{envelope['id']}",
        "family_id": str(envelope["family_id"]),
        "split": split,
        "label": 1 if role == "positive" else 0,
    }
    return {
        "user_prompt_only": {**common, "text": user_text},
        "system_user_prompt": {**common, "text": system_text + "\n" + user_text},
        "prompt_nonimage_metadata": {
            **common,
            "text": system_text + "\n" + user_text
                    + "\nroles=system,user; image_content=present; output_contract=evidence_first_v2",
        },
    }


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_shortcut_inputs(prepared_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    prepared_root, output_root = Path(prepared_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing existing Task 9D shortcut inputs: {output_root}")
    dev = [row for row in _read(prepared_root / "evaluation_protocol/manifest.jsonl")
           if row["prompt_view"] == "canonical"
           and row["condition"] in {"original", "semantic_null", "source_visual_null"}]
    summary = {}
    for variant in "ABC":
        splits = {
            "train": _read(prepared_root / f"variants/{variant}/train_schedule.jsonl"),
            "val": _read(prepared_root / f"variants/{variant}/val.jsonl"),
            "dev": dev,
        }
        destination = output_root / variant
        destination.mkdir(parents=True)
        handles = {view: (destination / f"{view}.jsonl").open("w", encoding="utf-8", newline="\n")
                   for view in VIEWS}
        try:
            counts = {}
            for split, envelopes in splits.items():
                labels = {0: 0, 1: 0}
                for index, envelope in enumerate(envelopes):
                    views = probe_views(envelope, split=split, index=index)
                    for view in VIEWS:
                        handles[view].write(json.dumps(views[view], ensure_ascii=False, separators=(",", ":")) + "\n")
                    labels[views["user_prompt_only"]["label"]] += 1
                if not all(labels.values()):
                    raise ValueError(f"variant {variant} split {split} lacks both shortcut labels")
                counts[split] = {str(key): value for key, value in labels.items()}
            summary[variant] = counts
        finally:
            for handle in handles.values():
                handle.close()
    (output_root / "report.json").write_text(
        json.dumps({"version": "task9d-shortcut-inputs-v1", "variants": summary,
                    "private_fields_in_text": False}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_shortcut_inputs(args.prepared_root, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
