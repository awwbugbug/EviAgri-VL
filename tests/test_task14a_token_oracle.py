import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task14a_oracle_protocol import BLOCKED, PASSED, select_protocol
from evaluate_task14a_token_oracle import decide, paired_bootstrap, validate_feature_row_order
from extract_task14a_oracle_tokens import (
    bbox_region_indices,
    mask_region_indices,
    pool_global_region,
)
from task10b_protocol import FROZEN_SELECTED_CLASS_IDS


def _protocol_inputs(per_class=8, plantseg_count=60):
    positive = []
    provenance = []
    selected = []
    prior = []
    for offset, class_id in enumerate(FROZEN_SELECTED_CLASS_IDS):
        band = ("head", "medium", "tail")[offset % 3]
        selected.append({"class_id": class_id, "class_band": band})
        prior.append(
            {
                "source_image_id": f"old-{class_id}",
                "source_image_sha256": f"{100000 + class_id:064x}"[-64:],
                "near_duplicate_component_id": f"old-component-{class_id}",
            }
        )
        for index in range(per_class):
            image_id = f"fresh-{class_id}-{index}"
            digest = f"{class_id * 1000 + index + 1:064x}"[-64:]
            positive.append(
                {
                    "image": f"/images/{image_id}.jpg",
                    "source_split": "trainval",
                    "metadata": {"image_id": image_id},
                    "target": {
                        "diagnosis": {"pest_id": class_id},
                        "evidence_bbox": [1, 2, 9, 10],
                    },
                }
            )
            provenance.append(
                {
                    "source_image_id": image_id,
                    "source_image_sha256": digest,
                    "near_duplicate_component_id": f"component-{class_id}-{index}",
                }
            )
    plantseg = [
        {
            "id": f"null-{index}",
            "image": f"/images/null-{index}.jpg",
            "image_sha256": f"{200000 + index:064x}"[-64:],
            "mask": f"/masks/null-{index}.png",
            "mask_sha256": f"{300000 + index:064x}"[-64:],
            "mask_ratio": (index + 1) / (plantseg_count + 1),
            "plant": "plant",
            "disease": "disease",
        }
        for index in range(plantseg_count)
    ]
    return positive, provenance, selected, prior, plantseg


def test_protocol_is_fresh_family_safe_and_has_frozen_counts():
    positive, provenance, selected, prior, plantseg = _protocol_inputs()
    result = select_protocol(
        positive_rows=positive,
        provenance_rows=provenance,
        selected_classes=selected,
        prior_positive_rows=prior,
        plantseg_rows=plantseg,
        prior_used_rows=[{"id": "null-0"}],
        locked_ids=set(),
        locked_sha256=set(),
    )
    assert result["status"] == PASSED
    rows = result["manifest"]
    counts = {split: sum(row["probe_split"] == split for row in rows) for split in ("probe_train", "probe_val", "probe_test", "null_test")}
    assert counts == {"probe_train": 64, "probe_val": 16, "probe_test": 32, "null_test": 32}
    assert "null-0" not in {row["id"] for row in rows}
    prior_components = {row["near_duplicate_component_id"] for row in prior}
    assert not prior_components & {row.get("near_duplicate_component_id") for row in rows}
    for class_id in FROZEN_SELECTED_CLASS_IDS:
        assert sum(row.get("class_id") == class_id and row["probe_split"] == "probe_train" for row in rows) == 4


def test_protocol_blocks_instead_of_approximating_class_quota():
    positive, provenance, selected, prior, plantseg = _protocol_inputs(per_class=6)
    result = select_protocol(
        positive_rows=positive,
        provenance_rows=provenance,
        selected_classes=selected,
        prior_positive_rows=prior,
        plantseg_rows=plantseg,
        prior_used_rows=[{"id": "unused"}],
        locked_ids=set(),
        locked_sha256=set(),
    )
    assert result["status"] == BLOCKED
    assert result["report"]["reason"] == "insufficient_fresh_positive_components"


def test_task14b_protocol_uses_six_new_components_per_class_without_validation():
    positive, provenance, selected, prior, plantseg = _protocol_inputs(per_class=13)
    task14a_prior = []
    for class_id in FROZEN_SELECTED_CLASS_IDS:
        for index in range(7):
            task14a_prior.append(
                {
                    "source_image_id": f"fresh-{class_id}-{index}",
                    "source_image_sha256": f"{class_id * 1000 + index + 1:064x}"[-64:],
                    "near_duplicate_component_id": f"component-{class_id}-{index}",
                }
            )
    result = select_protocol(
        positive_rows=positive,
        provenance_rows=provenance,
        selected_classes=selected,
        prior_positive_rows=[*prior, *task14a_prior],
        plantseg_rows=plantseg,
        prior_used_rows=[{"id": f"null-{index}"} for index in range(20)],
        locked_ids=set(),
        locked_sha256=set(),
        positive_quotas={"probe_train": 4, "probe_val": 0, "probe_test": 2},
        null_count=32,
        protocol_label="task14b",
    )
    assert result["status"] == PASSED
    rows = result["manifest"]
    assert result["report"]["rows_by_split"] == {
        "probe_train": 64,
        "probe_val": 0,
        "probe_test": 32,
        "null_test": 32,
    }
    assert len(rows) == 128
    assert not {row["source_image_id"] for row in task14a_prior} & {row["id"] for row in rows}


def test_bbox_token_mapping_uses_centers_and_overlap_fallback():
    indices = bbox_region_indices([0, 0, 50, 50], 100, 100, 2, 2)
    assert indices.tolist() == [0]
    fallback = bbox_region_indices([49, 49, 51, 51], 100, 100, 2, 2)
    assert len(fallback) == 1 and int(fallback[0]) in {0, 1, 2, 3}


def test_mask_token_mapping_and_pooling_are_finite_and_normalized():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:10, :10] = 255
    indices = mask_region_indices(mask, 2, 2)
    assert indices.tolist() == [0]
    tokens = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    global_vector, region_vector = pool_global_region(tokens, indices)
    assert global_vector.shape == region_vector.shape == (4,)
    assert np.isclose(np.linalg.norm(global_vector), 1.0)
    assert np.isclose(np.linalg.norm(region_vector), 1.0)


def test_oracle_decision_requires_gain_and_null_safety():
    supported = decide({"accuracy": 1 / 32, "true_probability": 0.01, "null_confidence": 0.0, "confidence_auroc": 0.01})
    assert supported["decision"] == "H2_ORACLE_SUPPORTED"
    assert supported["authorize_tiny_learned_selector"]
    unsafe = decide({"accuracy": 1 / 32, "true_probability": 0.01, "null_confidence": 0.01, "confidence_auroc": -0.01})
    assert unsafe["decision"] == "H2_ORACLE_UNSAFE"
    no_gain = decide({"accuracy": 0.0, "true_probability": 0.01, "null_confidence": -0.01, "confidence_auroc": 0.01})
    assert no_gain["decision"] == "H2_ORACLE_NO_GAIN"
    assert not no_gain["authorize_large_training"] and not no_gain["authorize_task8"]


def test_replication_decision_never_authorizes_selector():
    inconsistent = decide(
        {"accuracy": 1 / 32, "true_probability": 0.01, "null_confidence": 0.0, "confidence_auroc": 0.01},
        mode="replication",
    )
    assert inconsistent["decision"] == "H2_REPLICATION_INCONSISTENT"
    assert not inconsistent["authorize_tiny_learned_selector"]
    retired = decide(
        {"accuracy": 0.0, "true_probability": 0.01, "null_confidence": 0.01, "confidence_auroc": -0.01},
        mode="replication",
    )
    assert retired["decision"] == "H2_MEAN_REGION_RETIRED"


def test_bootstrap_is_paired_at_fresh_image_level():
    result = paired_bootstrap(lambda p, n: float(p.mean() - n.mean()), 4, 4, 100, 7)
    assert result["unit"] == "fresh_image" and result["repetitions"] == 100


def test_feature_order_validation_supports_128_row_replication():
    rows = [{"feature_index": index} for index in range(128)]
    validate_feature_row_order(rows, 128)


def test_task14a_shell_has_single_attempt_and_forbids_scope_expansion():
    text = (ROOT / "server" / "run_task14a_full_frame_token_oracle.sh").read_text(encoding="utf-8")
    assert "task14a_full_frame_token_oracle/2026-07-27/protocol_v1/attempt_01" in text
    assert "--repetitions 1000" in text
    assert "extract_task14a_oracle_tokens.py" in text
    for forbidden in ("shutdown", "poweroff", "7B", "SAM2", "train_qlora", "Task8"):
        assert forbidden not in text


def test_task14b_shell_is_disjoint_replication_and_keeps_scope_frozen():
    text = (ROOT / "server" / "run_task14b_independent_token_oracle.sh").read_text(encoding="utf-8")
    assert "task14b_independent_token_oracle/2026-07-27/protocol_v1/attempt_01" in text
    assert '--prior-positive "$TASK14A"' in text and '--prior-used "$TASK14A"' in text
    assert "--train-per-class 4 --val-per-class 0" in text
    assert "--decision-mode replication" in text and "--repetitions 1000" in text
    for forbidden in ("shutdown", "poweroff", "7B", "SAM2", "train_qlora", "Task8"):
        assert forbidden not in text


def test_task14b_retry_reuses_verified_features_without_scientific_change():
    text = (ROOT / "server" / "run_task14b_evaluation_retry.sh").read_text(encoding="utf-8")
    assert "protocol_v1/attempt_01" in text and "protocol_v1/attempt_02" in text
    assert '"scientific_protocol_changed":False' in text
    assert "extract_task14a_oracle_tokens.py" not in text
    assert "--decision-mode replication" in text and "--repetitions 1000" in text
