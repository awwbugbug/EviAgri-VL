import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task11a2_plantdoc_null import (
    bootstrap_fpr,
    decide_real_null,
    exact_binomial_interval,
)


def test_bootstrap_is_deterministic_and_paired_by_external_image():
    values = {17: [False] * 39 + [True], 29: [False] * 39 + [True], 43: [False] * 39 + [True]}
    first = bootstrap_fpr(values, repetitions=1000)
    second = bootstrap_fpr(values, repetitions=1000)
    assert first == second
    assert first["estimate"] == 0.025
    assert first["unit"] == "external_image"


def test_real_null_decision_is_strict_at_boundaries():
    assert decide_real_null(0.075, 0.20, 0.20)["passed"] is True
    assert decide_real_null(0.10, 0.20, 0.20)["passed"] is False
    assert decide_real_null(0.075, 0.25, 0.20)["passed"] is False
    assert decide_real_null(0.075, 0.20, 0.25)["passed"] is False


def test_exact_interval_does_not_collapse_for_zero_events():
    interval = exact_binomial_interval(0, 40)
    assert interval["low"] == 0.0
    assert 0.08 < interval["high"] < 0.10
    assert interval["trials"] == 40
