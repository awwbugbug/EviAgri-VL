import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9a_v1_forensics import (
    assert_allowed_input,
    load_split,
    probe_view,
    profile_records,
)


def make_row(split: str, index: int, positive: bool, prompt: str | None = None) -> dict:
    image_id = f"img-{index // 2}"
    question = prompt or (
        "Identify the pest supported by visible evidence."
        if positive
        else "Is rice borer visibly present in this image?"
    )
    target = {
        "evidence_present": positive,
        "evidence_bbox": [1, 2, 3, 4] if positive else None,
        "visible_attributes": [],
        "diagnosis": {"pest_id": 7, "pest_name": "rice borer"} if positive else "uncertain",
        "reliability": "supported" if positive else "insufficient_visual_evidence",
    }
    return {
        "id": f"ip102det_{split}_{index}_{'positive' if positive else 'null_7'}",
        "image": f"/data/class-{7 if positive else 99}/{image_id}.jpg",
        "source": "ip102_detection",
        "split": split,
        "source_split": "trainval" if split != "test" else "test",
        "task_type": "pest_evidence_grounding" if positive else "prompt_conflict_null_evidence",
        "question": question,
        "query_pest_name": None if positive else "rice borer",
        "target": target,
        "metadata": {"image_id": image_id, "secret_label_hint": "pos" if positive else "neg"},
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Inspect pixels and return JSON."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"/data/class-7/{image_id}.jpg"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(target, separators=(",", ":"))}],
            },
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize(
    "component",
    ["task8", "formal_clean_v2", "locked_confirmatory", "task9_dev_audit"],
)
def test_scope_guard_rejects_confirmatory_and_development_paths(component):
    with pytest.raises(ValueError, match="forbidden audit input"):
        assert_allowed_input(Path("/root/experiments") / component / "train.jsonl")


def test_scope_guard_accepts_static_qlora_v1_path():
    assert_allowed_input(Path("/root/datasets/derived/static_qlora_v1/train.jsonl"))


def test_strict_loader_rejects_duplicate_id_and_split_mismatch(tmp_path):
    duplicate = [make_row("train", 1, True), make_row("train", 1, False)]
    duplicate[1]["id"] = duplicate[0]["id"]
    path = tmp_path / "static_qlora_v1" / "train.jsonl"
    write_jsonl(path, duplicate)
    with pytest.raises(ValueError, match="duplicate record id"):
        load_split(path, "train")

    wrong_split = [make_row("val", 2, True)]
    write_jsonl(path, wrong_split)
    with pytest.raises(ValueError, match="split mismatch"):
        load_split(path, "train")


def test_strict_loader_rejects_invalid_json_and_non_boolean_label(tmp_path):
    path = tmp_path / "static_qlora_v1" / "train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_split(path, "train")

    row = make_row("train", 3, True)
    row["target"]["evidence_present"] = "yes"
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="must be boolean"):
        load_split(path, "train")


def test_profile_exposes_template_output_metadata_and_image_reuse():
    train = [make_row("train", 0, True), make_row("train", 1, False)]
    report = profile_records({"train": train})

    assert report["counts"]["train"] == {"positive": 1, "null": 1, "total": 2}
    assert report["prompt_prefix_by_label"]["positive"] != report["prompt_prefix_by_label"]["null"]
    assert report["task_type_by_label"]["pest_evidence_grounding"] == {"positive": 1, "null": 0}
    assert report["task_type_by_label"]["prompt_conflict_null_evidence"] == {"positive": 0, "null": 1}
    assert report["positive_null_image_id_overlap"]["train"] == 1
    assert report["target_unique_count_by_label"]["null"] == 1
    assert report["target_length_by_label"]["positive"]["min"] != report["target_length_by_label"]["null"]["min"]
    assert report["path_label_tokens"]["record_id_label_token_rows"] == 2
    assert report["field_order_by_label"]["positive"] == report["field_order_by_label"]["null"]
    assert report["json_quality"]["syntax_valid"]["rate"] == pytest.approx(1.0)
    assert report["json_quality"]["schema_valid"]["rate"] == pytest.approx(1.0)
    assert report["json_quality"]["semantic_consistency"]["rate"] == pytest.approx(1.0)
    assert report["json_quality"]["task_compliance"]["rate"] == pytest.approx(1.0)


def test_json_quality_separates_syntax_schema_semantics_and_compliance():
    clean = make_row("train", 0, True)
    invalid_syntax = make_row("train", 1, False)
    invalid_syntax["messages"][-1]["content"][0]["text"] = "not-json"
    semantic_error = make_row("train", 3, False)
    semantic_error["target"]["diagnosis"] = {"pest_id": 7, "pest_name": "rice borer"}
    semantic_error["messages"][-1]["content"][0]["text"] = json.dumps(
        semantic_error["target"], separators=(",", ":")
    )

    quality = profile_records({"train": [clean, invalid_syntax, semantic_error]})["json_quality"]

    assert quality["syntax_valid"] == {"count": 2, "total": 3, "rate": pytest.approx(2 / 3)}
    assert quality["schema_valid"] == {"count": 2, "total": 3, "rate": pytest.approx(2 / 3)}
    assert quality["semantic_consistency"] == {"count": 1, "total": 3, "rate": pytest.approx(1 / 3)}
    assert quality["task_compliance"] == {"count": 1, "total": 3, "rate": pytest.approx(1 / 3)}


def test_profile_reports_exact_image_ids_crossing_splits():
    train = make_row("train", 0, True)
    test = make_row("test", 0, True)

    profile = profile_records({"train": [train], "test": [test]})

    assert profile["image_ids_crossing_splits"] == ["img-0"]


def test_text_only_probes_generalize_template_shortcut_to_held_out_rows():
    train = [
        make_row("train", 0, True),
        make_row("train", 1, False),
        make_row("train", 2, True),
        make_row("train", 3, False),
    ]
    test = [
        make_row("test", 10, True),
        make_row("test", 11, False),
        make_row("test", 12, True),
        make_row("test", 13, False),
    ]

    for view in ("user_prompt", "system_user_prompt", "prompt_metadata"):
        result = probe_view(train, test, view)
        assert result["balanced_accuracy"] == pytest.approx(1.0)
        assert result["auroc"] == pytest.approx(1.0)


def test_probe_constant_scores_have_chance_auroc_and_balanced_accuracy():
    train = [
        make_row("train", 0, True, prompt="same"),
        make_row("train", 1, False, prompt="same"),
    ]
    test = [
        make_row("test", 2, True, prompt="same"),
        make_row("test", 3, False, prompt="same"),
    ]

    result = probe_view(train, test, "user_prompt")

    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["auroc"] == pytest.approx(0.5)
