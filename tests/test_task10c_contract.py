import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task10c_contract import (
    CLASS_IDS,
    EXPECTED_MANIFEST_SHA256,
    SYSTEM_PROMPT,
    TRAIN_PROMPT,
    build_task10c_protocol,
    canonical_pest_id,
    run_protocol,
    strict_parse_pest_json,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> list[dict]:
    rows = []
    quotas = {"train": 12, "val": 3, "dev": 5}
    for class_id in CLASS_IDS:
        band = "head" if class_id in {16, 22, 24, 45, 50, 101} else (
            "medium" if class_id in {10, 68, 71, 87, 99} else "tail"
        )
        index = 0
        for split, count in quotas.items():
            for _ in range(count):
                rows.append({
                    "id": f"row-{class_id:03d}-{index:02d}",
                    "class_id": class_id,
                    "class_band": band,
                    "split": split,
                    "image": f"/images/IP{class_id:03d}{index:06d}.jpg",
                    "source_image_id": f"IP{class_id:03d}{index:06d}",
                    "source_image_sha256": f"{class_id:02x}{index:062x}"[-64:],
                    "near_duplicate_component_id": f"component-{class_id}-{index}",
                })
                index += 1
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_contract_freezes_ids_prompts_and_strict_json():
    assert canonical_pest_id(9) == "IP009"
    with pytest.raises(ValueError, match="outside frozen"):
        canonical_pest_id(1)

    assert strict_parse_pest_json('{"pest_id":"IP009"}') == {
        "syntax_valid": True,
        "schema_valid": True,
        "pest_id": "IP009",
        "error": None,
    }
    assert strict_parse_pest_json('```json\n{"pest_id":"IP009"}\n```')["syntax_valid"] is False
    assert strict_parse_pest_json('{"pest_id":"IP009","name":"x"}')["schema_valid"] is False
    assert strict_parse_pest_json('{ "pest_id": "IP009" }')["schema_valid"] is False
    assert strict_parse_pest_json('{"pest_id":"IP001"}')["schema_valid"] is False


def test_protocol_reuses_exact_split_and_builds_deterministic_smoke_subsets():
    result = build_task10c_protocol(_rows(), EXPECTED_MANIFEST_SHA256)

    assert result["report"]["passed"] is True
    assert result["report"]["rows_by_split"] == {"dev": 80, "train": 192, "val": 48}
    assert result["report"]["source_overlap"] == 0
    assert result["report"]["component_overlap"] == 0
    assert len(result["smoke_train"]) == 64
    assert len(result["smoke_dev"]) == 16
    assert Counter(row["class_id"] for row in result["smoke_train"]) == {
        class_id: 4 for class_id in CLASS_IDS
    }
    assert Counter(row["class_id"] for row in result["smoke_dev"]) == {
        class_id: 1 for class_id in CLASS_IDS
    }

    envelope = result["train"][0]
    model = envelope["model"]
    assert [message["role"] for message in model["messages"]] == ["system", "user", "assistant"]
    assert model["messages"][0]["content"][0]["text"] == SYSTEM_PROMPT
    assert model["messages"][1]["content"][1]["text"] == TRAIN_PROMPT
    assert model["messages"][2]["content"][0]["text"].startswith('{"pest_id":"IP')
    prompt_text = " ".join(
        item["text"]
        for message in model["messages"][:2]
        for item in message["content"]
        if item["type"] == "text"
    )
    assert envelope["source_image_id"] not in prompt_text
    assert envelope["image"] not in prompt_text

    second = build_task10c_protocol(list(reversed(_rows())), EXPECTED_MANIFEST_SHA256)
    assert result == second


def test_protocol_blocks_wrong_hash_counts_and_cross_split_components():
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        build_task10c_protocol(_rows(), "0" * 64)

    missing = _rows()[:-1]
    with pytest.raises(ValueError, match="split row count mismatch"):
        build_task10c_protocol(missing, EXPECTED_MANIFEST_SHA256)

    overlapping = _rows()
    train = next(row for row in overlapping if row["split"] == "train")
    dev = next(row for row in overlapping if row["split"] == "dev")
    dev["near_duplicate_component_id"] = train["near_duplicate_component_id"]
    with pytest.raises(ValueError, match="cross-split near-duplicate"):
        build_task10c_protocol(overlapping, EXPECTED_MANIFEST_SHA256)


def test_run_protocol_hashes_model_writes_completion_and_refuses_overwrite(tmp_path):
    source = tmp_path / "manifest.jsonl"
    _write_jsonl(source, _rows())
    model = tmp_path / "Qwen2___5-VL-3B-Instruct"
    model.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"frozen-test-shard")
    output = tmp_path / "protocol"

    report = run_protocol(
        source,
        model,
        output,
        expected_manifest_sha256=_sha256(source),
    )

    assert report["state"] == "completed"
    assert len((output / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 192
    assert len((output / "smoke_train.jsonl").read_text(encoding="utf-8").splitlines()) == 64
    assert len((output / "smoke_dev.jsonl").read_text(encoding="utf-8").splitlines()) == 16
    model_hashes = json.loads((output / "model_files.sha256.json").read_text(encoding="utf-8"))
    assert model_hashes == [{
        "name": shard.name,
        "bytes": len(b"frozen-test-shard"),
        "sha256": hashlib.sha256(b"frozen-test-shard").hexdigest(),
    }]
    assert (output / "completion.sha256").is_file()
    assert not (output / "failure.json").exists()

    with pytest.raises(FileExistsError):
        run_protocol(source, model, output, expected_manifest_sha256=_sha256(source))


def test_run_protocol_records_block_without_partial_success(tmp_path):
    source = tmp_path / "manifest.jsonl"
    _write_jsonl(source, _rows()[:-1])
    model = tmp_path / "Qwen2___5-VL-3B-Instruct"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"x")
    output = tmp_path / "blocked"

    with pytest.raises(ValueError, match="split row count mismatch"):
        run_protocol(source, model, output, expected_manifest_sha256=_sha256(source))

    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["state"] == "blocked"
    assert (output / "failure.json").is_file()
    assert not (output / "completion.sha256").exists()
