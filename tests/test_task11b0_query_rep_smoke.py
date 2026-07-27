import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task11b0_query_rep_smoke import decide_smoke, layer_output_index, unique_subsequence


def test_unique_subsequence_requires_exactly_one_hit():
    assert unique_subsequence([1, 2, 3, 4], [2, 3]) == (1, 3)
    with pytest.raises(ValueError): unique_subsequence([1, 2, 1, 2], [1, 2])


def test_layer_output_index_is_explicit_for_local_qwen():
    assert layer_output_index(36) == 27


def test_smoke_gate_passes_only_image_dependent_reproducible_features():
    gates = {"expected_layers": 36, "expected_hidden_size": 2048,
        "expected_layer_output_index": 27, "expected_query_tokens": 19,
        "duplicate_max_abs_le": 1e-6, "median_original_blank_cosine_distance_ge": 1e-4,
        "minimum_original_blank_l2_ge": 1e-3}
    result = decide_smoke(layers=36, hidden_size=2048, layer_index=27, query_tokens=19,
        duplicate_max_abs=0.0, cosine_distances=[.01] * 8, l2_distances=[.1] * 8, gates=gates)
    assert result["decision"] == "PASS"
    failed = decide_smoke(layers=36, hidden_size=2048, layer_index=27, query_tokens=19,
        duplicate_max_abs=0.0, cosine_distances=[0.0] * 8, l2_distances=[0.0] * 8, gates=gates)
    assert failed["decision"] == "FAIL"


def test_shell_uses_protocol_and_attempt_axes_and_bash_invocation():
    text = (ROOT / "server" / "run_task11b0_query_rep_smoke.sh").read_text(encoding="utf-8")
    assert "/protocol_v1/attempt_01" in text
    assert "task11b0_query_rep_smoke.py" in text
    assert "shutdown" not in text and "poweroff" not in text
