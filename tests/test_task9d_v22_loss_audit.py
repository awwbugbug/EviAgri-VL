import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from audit_task9d_v22_loss_reduction import summarize_loss_reduction_audit


def _observations():
    return [
        {
            "id": "p1", "role": "positive", "inputs_equal": True,
            "control_active_tokens": 20, "taxmask_active_tokens": 12,
            "masked_fields": ["pest_id", "pest_name"], "null_labels_equal": None,
        },
        {
            "id": "s1", "role": "semantic_negative", "inputs_equal": True,
            "control_active_tokens": 18, "taxmask_active_tokens": 18,
            "masked_fields": [], "null_labels_equal": True,
        },
        {
            "id": "v1", "role": "visual_counterfactual", "inputs_equal": True,
            "control_active_tokens": 17, "taxmask_active_tokens": 17,
            "masked_fields": [], "null_labels_equal": True,
        },
    ]


def test_loss_reduction_audit_reports_role_weights_and_passes_equal_loss_mass():
    report = summarize_loss_reduction_audit(_observations(), gradient_accumulation_steps=8)
    assert report["passed"] is True
    assert report["reduction"] == "per_example_active_token_mean_then_batch_mean"
    assert report["arms"]["Control"]["positive"]["active_tokens"]["mean"] == 20
    assert report["arms"]["TaxMask"]["positive"]["active_tokens"]["mean"] == 12
    assert report["arms"]["Control"]["positive"]["mean_example_loss_weight"] == 1.0
    assert report["arms"]["TaxMask"]["positive"]["normalized_total_gradient_weight"] == 1 / 3
    assert report["invariants"]["all_inputs_equal"] is True
    assert report["invariants"]["all_null_labels_equal"] is True


def test_loss_reduction_audit_blocks_changed_null_or_missing_positive_mask():
    changed_null = _observations()
    changed_null[1]["null_labels_equal"] = False
    report = summarize_loss_reduction_audit(changed_null, gradient_accumulation_steps=8)
    assert report["passed"] is False
    assert any("null labels changed" in reason for reason in report["block_reasons"])

    missing_mask = _observations()
    missing_mask[0]["masked_fields"] = []
    report = summarize_loss_reduction_audit(missing_mask, gradient_accumulation_steps=8)
    assert report["passed"] is False
    assert any("positive taxonomy mask missing" in reason for reason in report["block_reasons"])
