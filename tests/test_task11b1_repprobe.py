import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task11b1_repprobe import decide


def _values():
    v={"accuracy":.82,"forced_macro_f1":.81,"coverage":.80,"blank_fpr":.05,"blur_fpr":.05,"shuffle_fpr":.15,"synthetic_overall_fpr":.08,"plantdoc_fpr":.02,"plantseg_fpr":.08,"json_contract":1.0}
    q={**v,"accuracy":.83,"forced_macro_f1":.82,"plantseg_fpr":.05,"synthetic_overall_fpr":.07}
    paired={"positive":{"macro_f1":{"low":-.01}},"plantseg":{"high":-.005}}
    gates={"accuracy_delta_ge":-.03,"forced_macro_f1_delta_ge":-.03,"forced_macro_f1_bootstrap_low_ge":-.05,"query_coverage_ge":.70,"coverage_delta_ge":-.05,"blank_fpr_lt":.10,"blur_fpr_lt":.10,"shuffle_fpr_lt":.25,"synthetic_overall_fpr_delta_le":0,"plantdoc_fpr_lt":.10,"plantdoc_fpr_delta_le":.025,"plantseg_fpr_lt":.10,"plantseg_fpr_delta_le":-.02,"plantseg_fpr_bootstrap_high_lt":0}
    return {"vision":v,"query":q},paired,gates


def test_repprobe_pass_requires_positive_noninferiority_and_real_null_superiority():
    summary,paired,gates=_values(); result=decide(summary,paired,gates)
    assert result["passed"] and result["authorize_evidence_head_planning"]


def test_repprobe_blocks_when_plantseg_ci_crosses_zero():
    summary,paired,gates=_values(); paired["plantseg"]["high"]=.001
    result=decide(summary,paired,gates)
    assert not result["passed"] and not result["conditions"]["plantseg_ci_superior"]


def test_repprobe_blocks_positive_regression_even_with_null_gain():
    summary,paired,gates=_values(); summary["query"]["forced_macro_f1"]=.70
    assert not decide(summary,paired,gates)["passed"]


def test_shell_preserves_protocol_attempt_axes_and_forbids_large_methods():
    text=(ROOT/"server"/"run_task11b1_repprobe.sh").read_text(encoding="utf-8")
    assert "task11b1_repprobe/2026-07-27/protocol_v1/attempt_01" in text
    assert "query_base" in text and "query_stress" in text and "query_plantdoc" in text and "query_plantseg" in text
    assert "bash" not in text.splitlines()[0].lower() or text.startswith("#!/usr/bin/env bash")
    for forbidden in ("shutdown","poweroff","Task8","7B","SAM2"):
        assert forbidden not in text
