import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT/"server"))
from build_task11c1_pair_manifests import align_rows
from evaluate_task11c1_paired_crop import decide, paired_bootstrap


def test_align_rows_preserves_pairs_and_identity_bytes(tmp_path):
    source=tmp_path/"source.jpg"; source.write_bytes(b"source")
    crop=tmp_path/"crop.jpg"; crop.write_bytes(b"crop")
    import hashlib
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest(); crop_sha=hashlib.sha256(crop.read_bytes()).hexdigest()
    crop_rows=[{"id":"p","kind":"ip102_bbox","class_id":3,"crop_mode":"effective_crop","model_image":str(crop),"crop_sha256":crop_sha,"source_image_sha256":source_sha},
        {"id":"n","kind":"plantseg_mask","class_id":None,"crop_mode":"identity_full_frame","model_image":str(source),"crop_sha256":source_sha,"source_image_sha256":source_sha}]
    full,local=align_rows(crop_rows,[{"id":"p","image":str(source)}],[{"id":"n","image":str(source)}])
    assert [r["id"] for r in full]==[r["id"] for r in local]==["p","n"]
    assert local[0]["image"]==str(crop) and local[1]["image"]==str(source)
    assert [r["target_type"] for r in local]==["positive","real_null"]


def test_paired_bootstrap_uses_pair_axis():
    values=np.asarray([[1.,1.],[3.,3.]])
    result=paired_bootstrap(values,lambda idx:float(values[idx].mean()),100,7)
    assert result["estimate"]==2.0 and result["unit"]=="paired_image"


def test_decision_pass_is_micro_only():
    summary={"positive_accuracy_delta":0.,"positive_supported_delta":0.,"positive_true_probability_delta":.01,"local_null_fpr":.05,"null_fpr_delta":0.,"null_mean_confidence_delta":-.01}
    invariants={"identity_feature_cosine_ge_0_99999":True,"identity_predictions_agree":True,"effective_median_feature_cosine_lt_0_999":True,"all_features_finite":True}
    boot={"positive_true_probability_delta":{"low":.001},"null_mean_confidence_delta":{"high":-.001}}
    result=decide(summary,invariants,boot)
    assert result["passed"] and result["strong_signal"] and not result["authorize_training"]


def test_decision_blocks_null_harm():
    summary={"positive_accuracy_delta":0.,"positive_supported_delta":0.,"positive_true_probability_delta":.01,"local_null_fpr":.125,"null_fpr_delta":.0625,"null_mean_confidence_delta":.01}
    invariants={"identity_feature_cosine_ge_0_99999":True,"identity_predictions_agree":True,"effective_median_feature_cosine_lt_0_999":True,"all_features_finite":True}
    boot={"positive_true_probability_delta":{"low":-.1},"null_mean_confidence_delta":{"high":.1}}
    assert not decide(summary,invariants,boot)["passed"]


def test_shell_is_single_attempt_and_forbids_large_methods():
    text=(ROOT/"server"/"run_task11c1_paired_crop_probe.sh").read_text(encoding="utf-8")
    assert "task11c1_paired_crop_probe/2026-07-27/protocol_v1/attempt_01" in text
    assert "features_full" in text and "features_local" in text and "--repetitions 1000" in text
    for forbidden in ("shutdown","poweroff","Task8","7B","SAM2","qlora"):
        assert forbidden not in text
