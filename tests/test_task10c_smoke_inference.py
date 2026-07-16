import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task10c_contract import CLASS_IDS, SYSTEM_PROMPT, TRAIN_PROMPT, UNSEEN_PROMPT
from run_task10c_smoke_inference import (
    build_smoke_conditions,
    generation_contract,
    inference_messages,
    verify_prediction_ids,
)


def _row(class_id: int) -> dict:
    target = json.dumps({"pest_id": f"IP{class_id:03d}"}, separators=(",", ":"))
    return {
        "id": f"row-{class_id:03d}",
        "image": f"/images/IP{class_id:03d}.jpg",
        "class_id": class_id,
        "class_band": "head",
        "split": "dev",
        "source_image_id": f"source-{class_id}",
        "source_image_sha256": f"{class_id:064x}",
        "near_duplicate_component_id": f"component-{class_id}",
        "model": {
            "id": f"row-{class_id:03d}",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": f"/images/IP{class_id:03d}.jpg"},
                    {"type": "text", "text": TRAIN_PROMPT},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ],
        },
    }


def test_build_conditions_is_exact_16_by_4_and_no_image_only_removes_pixels():
    conditions = build_smoke_conditions([_row(class_id) for class_id in CLASS_IDS])

    assert len(conditions) == 64
    assert Counter(row["condition"] for row in conditions) == {
        "image_train_prompt": 16,
        "image_unseen_prompt": 16,
        "no_image_train_prompt": 16,
        "no_image_unseen_prompt": 16,
    }
    source = f"source-{CLASS_IDS[0]}"
    same_source = {row["condition"]: row for row in conditions if row["source_image_id"] == source}
    image_train = same_source["image_train_prompt"]
    no_image_train = same_source["no_image_train_prompt"]
    assert [item["type"] for item in image_train["messages"][1]["content"]] == ["image", "text"]
    assert [item["type"] for item in no_image_train["messages"][1]["content"]] == ["text"]
    assert image_train["messages"][1]["content"][-1]["text"] == TRAIN_PROMPT
    assert no_image_train["messages"][1]["content"][-1]["text"] == TRAIN_PROMPT
    assert same_source["image_unseen_prompt"]["messages"][1]["content"][-1]["text"] == UNSEEN_PROMPT
    assert len({row["id"] for row in conditions}) == 64


def test_generation_contract_is_frozen():
    assert generation_contract() == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 32,
        "min_pixels": 200704,
        "max_pixels": 401408,
        "parser_version": "task10c-strict-pest-json-v1",
    }


def test_inference_messages_rejects_metadata_or_wrong_content_order():
    row = build_smoke_conditions([_row(class_id) for class_id in CLASS_IDS])[0]
    assert inference_messages(row) == row["messages"]

    wrong = json.loads(json.dumps(row))
    wrong["messages"][1]["content"].reverse()
    with pytest.raises(ValueError, match="condition is invalid"):
        inference_messages(wrong)

    wrong = json.loads(json.dumps(row))
    wrong["messages"].append({"role": "assistant", "content": []})
    with pytest.raises(ValueError, match="system/user"):
        inference_messages(wrong)


def test_verify_prediction_ids_requires_exact_unique_set():
    rows = build_smoke_conditions([_row(class_id) for class_id in CLASS_IDS])
    predictions = [{"id": row["id"]} for row in reversed(rows)]
    verify_prediction_ids(rows, predictions)

    with pytest.raises(ValueError, match="prediction ID mismatch"):
        verify_prediction_ids(rows, predictions[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        verify_prediction_ids(rows, predictions[:-1] + [predictions[0]])
