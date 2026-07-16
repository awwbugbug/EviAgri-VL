import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task9d_v22_micro import build_micro_protocol


ROLES = ("positive", "semantic_negative", "visual_counterfactual")


def _model(identifier, class_id, role):
    present = role == "positive"
    target = {
        "evidence_present": present,
        "evidence_region": [1, 2, 3, 4] if present else None,
        "visible_attributes": [],
        "diagnosis": {
            "status": "supported" if present else "abstain",
            "pest_id": class_id if present else None,
            "pest_name": f"pest-{class_id}" if present else None,
            "species": None,
            "stage": None,
        },
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }
    return {"id": identifier, "messages": [
        {"role": "system", "content": [{"type": "text", "text": "s"}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"images/{identifier}.png"},
            {"type": "text", "text": f"query pest-{class_id}"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]},
    ]}


def _schedule(prefix, class_ids, families_per_class):
    rows = []
    for class_id in class_ids:
        for family_index in range(families_per_class):
            family = f"{prefix}-c{class_id}-f{family_index}"
            for role in ROLES:
                identifier = f"{family}-{role}"
                rows.append({
                    "id": identifier, "family_id": family, "role": role,
                    "schedule_index": len(rows), "template_id": "neutral",
                    "model": _model(identifier, class_id, role),
                })
    return rows


def _dev_manifest(class_ids):
    rows = []
    views = ("canonical", "native_0", "native_1", "native_2", "unseen_alpha", "unseen_beta")
    for class_id in class_ids:
        family = f"dev-c{class_id}"
        for view in views:
            rows.append({
                "id": f"{family}-positive-{view}", "family_id": family,
                "role": "positive", "condition": "original", "prompt_view": view,
                "query_class_id": class_id, "messages": _model("x", class_id, "positive")["messages"][:2],
            })
        for condition, role in (
            ("semantic_null", "semantic_negative"),
            ("source_visual_null", "visual_counterfactual"),
            ("blank", "visual_counterfactual"),
            ("blur", "visual_counterfactual"),
            ("shuffle", "visual_counterfactual"),
        ):
            rows.append({
                "id": f"{family}-{condition}", "family_id": family,
                "role": role, "condition": condition, "prompt_view": "canonical",
                "query_class_id": class_id, "messages": _model("x", class_id, role)["messages"][:2],
            })
    return rows


def test_builder_selects_stratified_disjoint_families_and_exact_exposures():
    class_ids = [1, 2, 3, 4, 5, 6]
    bands = {str(c): band for c, band in zip(class_ids, ["head", "head", "medium", "medium", "tail", "tail"])}
    result = build_micro_protocol(
        _schedule("train", class_ids, 2),
        _schedule("val", class_ids, 1),
        _dev_manifest(class_ids),
        bands,
        band_quotas={"head": 1, "medium": 1, "tail": 1},
        total_exposures=24,
    )
    assert len(result["selected_classes"]) == 3
    assert len(result["unique_train_rows"]) == 18
    assert len(result["train_schedule"]) == 24
    assert len(result["val_rows"]) == 9
    assert len(result["evaluation_rows"]) == 33
    assert set(result["report"]["role_exposures"]) == set(ROLES)
    assert sum(result["report"]["role_exposures"].values()) == 24
    assert result["report"]["split_family_overlap"] == {
        "train_val": 0, "train_dev": 0, "val_dev": 0,
    }


def test_builder_blocks_infeasible_strict_class_quota_and_split_overlap():
    class_ids = [1, 2, 3]
    bands = {"1": "head", "2": "medium", "3": "tail"}
    with pytest.raises(ValueError, match="infeasible strict class quota"):
        build_micro_protocol(
            _schedule("train", class_ids, 2), _schedule("val", class_ids, 1),
            _dev_manifest(class_ids), bands,
            band_quotas={"head": 2, "medium": 1, "tail": 1}, total_exposures=24,
        )

    train = _schedule("shared", class_ids, 2)
    val = _schedule("val", class_ids, 1)
    val[0]["family_id"] = train[0]["family_id"]
    with pytest.raises(ValueError, match="family overlap across split"):
        build_micro_protocol(
            train, val, _dev_manifest(class_ids), bands,
            band_quotas={"head": 1, "medium": 1, "tail": 1}, total_exposures=24,
        )
