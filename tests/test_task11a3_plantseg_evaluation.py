import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task11a3_plantseg_null import bootstrap_fpr, decide, exact_interval


def test_task11a3_decision_uses_preregistered_gates():
    assert decide(.09, .24, 1.0)["decision"] == "PASS"
    assert decide(.10, .24, 1.0)["decision"] == "FAIL"
    assert decide(.09, .25, 1.0)["decision"] == "FAIL"
    assert decide(.09, .24, .99)["decision"] == "FAIL"


def test_task11a3_bootstrap_and_exact_interval_are_image_level():
    acceptance = {17: [False, True, False], 29: [False, False, False], 43: [False, False, False]}
    result = bootstrap_fpr(acceptance, 100)
    assert result["estimate"] == 1 / 9
    assert result["unit"] == "unique_external_image"
    exact = exact_interval(1, 3)
    assert exact["successes"] == 1 and exact["trials"] == 3
