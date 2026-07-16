import copy
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9b_protocol import SYSTEM_PROMPT, build_prompt, build_target, serialize_target
from task9b_v21_repair import repair_records


def _messages(prompt, positive, class_id=None, class_name=None):
    target = build_target(positive, [1, 2, 9, 10] if positive else None,
                          class_id if positive else None, class_name if positive else None)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": "images/" + "a" * 64 + ".jpg"},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": serialize_target(target, 320)}]},
    ]


def _fixture():
    models, provenance = {}, []
    classes = [(0, "pest zero"), (0, "pest zero"), (1, "pest one"), (1, "pest one")]
    for index, (class_id, name) in enumerate(classes):
        family = f"fam{index}"
        for role in ("positive", "semantic_negative", "visual_counterfactual"):
            identifier = f"{index:02d}-{role}"
            query_id = class_id if role != "semantic_negative" else 1 - class_id
            query_name = name if role != "semantic_negative" else ("pest one" if class_id == 0 else "pest zero")
            models[identifier] = {
                "id": identifier,
                "messages": _messages(build_prompt("train_neutral_0", query_name), role == "positive", class_id, name),
            }
            provenance.append({
                "id": identifier, "family_id": family, "role": role,
                "evidence_present": role == "positive",
                "null_source": None if role == "positive" else ("real_null" if role == "semantic_negative" else "synthetic_null"),
                "split": "train", "source_image_id": f"img{index}",
                "source_image_sha256": f"{index+1:064x}",
                "near_duplicate_component_id": f"comp{index}",
                "derived_image_sha256": "a" * 64,
                "present_class_ids": [class_id], "query_class_id": query_id,
                "template_id": "train_neutral_0", "length_bucket": 320,
                "transform_id": None if role != "visual_counterfactual" else "train_blur",
            })
    return models, provenance


def test_repair_is_family_bijective_and_only_changes_semantic_queries():
    models, provenance = _fixture()
    original_models = copy.deepcopy(models)
    original_provenance = copy.deepcopy(provenance)
    repaired_models, repaired_provenance, report = repair_records(models, provenance)

    assert set(repaired_models) == set(original_models)
    assert [row["family_id"] for row in repaired_provenance] == [row["family_id"] for row in original_provenance]
    assert [(row["split"], row["template_id"]) for row in repaired_provenance] == [
        (row["split"], row["template_id"]) for row in original_provenance
    ]
    for old, new in zip(original_provenance, repaired_provenance):
        if old["role"] == "semantic_negative":
            assert new["query_class_id"] not in new["present_class_ids"]
        else:
            assert new == old
            assert repaired_models[new["id"]] == original_models[old["id"]]
    assert report["family_bijection"] is True
    assert report["changed_fields"] == ["semantic_negative.query_class_id", "semantic_negative.user_prompt"]


def test_repair_has_exact_positive_semantic_query_counts_per_stratum():
    models, provenance = _fixture()
    _, repaired, report = repair_records(models, provenance)
    pos = Counter(row["query_class_id"] for row in repaired if row["role"] == "positive")
    sem = Counter(row["query_class_id"] for row in repaired if row["role"] == "semantic_negative")
    assert pos == sem
    assert report["max_total_variation"] == 0.0
    assert report["family_count_before"] == report["family_count_after"] == 4
