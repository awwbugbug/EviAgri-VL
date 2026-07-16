import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from validate_task9d_pretraining import decide_pretraining


def test_pretraining_decision_blocks_engineering_failures_but_not_three_step_quality():
    checks = {
        "protocol": True, "eval_protocol": True, "shortcut_gates": True,
        "schedule_and_schema": True, "smoke_training": True,
        "adapter_reload_generation": True, "three_step_generation_schema_valid": False,
    }
    result = decide_pretraining(checks)
    assert result["passed"] is True
    assert result["informational_risks"] == ["three_step_generation_schema_valid=false"]
    checks["shortcut_gates"] = False
    assert decide_pretraining(checks)["passed"] is False
