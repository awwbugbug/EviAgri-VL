import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task13a_presence_head import decide, positive_only_threshold, select_rows


def _base_rows():
    rows = []
    for class_id in range(16):
        for index in range(4):
            rows.append({"id": f"train-{class_id}-{index}", "class_id": class_id, "split": "train", "feature_index": len(rows), "near_duplicate_component_id": f"train-{class_id}-{index}", "class_band": "x"})
        rows.append({"id": f"val-{class_id}", "class_id": class_id, "split": "val", "feature_index": len(rows), "near_duplicate_component_id": f"val-{class_id}", "class_band": "x"})
    return rows


def test_selection_is_balanced_fresh_and_plantseg_is_not_fit():
    plantdoc = [
        {"id": f"doc-{group}-{index}", "healthy_class": f"h-{group}", "feature_index": group * 4 + index}
        for group in range(10)
        for index in range(4)
    ]
    plantseg = [
        {"id": f"seg-{index}", "mask_ratio": index / 1000, "feature_index": index, "disease": "d"}
        for index in range(100)
    ]
    rows = select_rows(_base_rows(), plantdoc, plantseg, [[{"id": "seg-0"}]])
    counts = {role: sum(row["role"] == role for row in rows) for role in {row["role"] for row in rows}}
    assert counts == {"positive_train": 32, "positive_val": 16, "positive_test": 32, "null_train": 20, "plantdoc_test": 20, "plantseg_test": 32}
    assert not any(row["source"] == "plantseg" and row["role"] == "null_train" for row in rows)
    assert "seg-0" not in {row["id"] for row in rows}


def test_positive_threshold_uses_only_requested_recall_rank():
    scores = np.arange(1, 17, dtype=float) / 16
    threshold = positive_only_threshold(scores, 0.90)
    assert threshold == 2 / 16
    assert float((scores >= threshold).mean()) == 15 / 16


def _seed_row(coverage=0.9, doc_fpr=0.05, seg_fpr=0.0625, auroc=0.95, supported=0.8):
    return {"positive_coverage": coverage, "supported_diagnosis": supported, "plantdoc": {"null_fpr": doc_fpr}, "plantseg": {"null_fpr": seg_fpr}, "combined": {"auroc": auroc}}


def test_decision_only_authorizes_tiny_fusion_when_all_gates_pass():
    passed = {str(seed): {"T0_taxonomy": _seed_row(seg_fpr=0.2, supported=0.81), "P1_presence": _seed_row(supported=0.8)} for seed in (17, 29, 43)}
    decision = decide(passed)
    assert decision["decision"] == "PASS_H3_FEASIBILITY"
    assert decision["authorize_tiny_gated_fusion"]
    assert not decision["authorize_large_training"] and not decision["authorize_task8"]
    failed = {str(seed): {"T0_taxonomy": _seed_row(), "P1_presence": _seed_row(seg_fpr=0.125)} for seed in (17, 29, 43)}
    assert decide(failed)["decision"] == "H3_BLOCK_H2_PRIORITY"


def test_shell_is_single_micro_attempt_and_forbids_scaleup():
    text = (ROOT / "server" / "run_task13a_frozen_presence_head.sh").read_text(encoding="utf-8")
    assert "task13a_frozen_presence_head/2026-07-27/protocol_v1/attempt_01" in text
    assert "--repetitions 1000" in text and text.count("--prior-rows") == 3
    assert "test ! -e \"$ROOT\"" in text and "sha256sum -c code_manifest.sha256" in text
    for forbidden in ("shutdown", "poweroff", "Task8", "7B", "SAM2", "qlora", "torchrun"):
        assert forbidden not in text
