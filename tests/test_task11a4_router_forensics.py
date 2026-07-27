import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from analyze_task11a4_router_false_positives import _entropy, disease_pattern


def test_disease_pattern_is_deterministic_and_metadata_only():
    assert disease_pattern("soybean frog eye leaf spot") == "spot_or_blotch"
    assert disease_pattern("blueberry rust") == "rust"
    assert disease_pattern("peach brown rot") == "rot"
    assert disease_pattern("unknown damage") == "other"


def test_entropy_is_finite_and_zero_for_certain_distribution():
    values = _entropy(np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64))
    assert np.isfinite(values).all()
    assert values[0] < 1e-9
    assert np.isclose(values[1], np.log(2.0))


def test_forensic_source_hard_codes_read_only_boundaries():
    text = (ROOT / "server" / "analyze_task11a4_router_false_positives.py").read_text(encoding="utf-8")
    assert '"read_only": True' in text
    assert '"task8_locked_set_read": False' in text
    assert '"task11b_started": False' in text
    assert '"threshold_changed": False' in text
