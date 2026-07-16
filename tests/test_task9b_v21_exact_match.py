import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9b_v21_exact_match import ExactMatchInfeasible, match_all_strata


def _family(fid, split, template, positive, present=None):
    return {
        "family_id": fid,
        "split": split,
        "template_id": template,
        "positive_query_class_id": positive,
        "present_class_ids": sorted(set(present if present is not None else [positive])),
    }


def test_exact_matching_preserves_every_family_and_each_stratum_marginal():
    families = [
        _family("a", "train", "t0", 0),
        _family("b", "train", "t0", 0),
        _family("c", "train", "t0", 1),
        _family("d", "train", "t0", 1),
        _family("e", "val", "t0", 2),
        _family("f", "val", "t0", 3),
    ]
    result = match_all_strata(families)
    assert set(result["assignment"]) == {row["family_id"] for row in families}
    assert result["family_count_before"] == result["family_count_after"] == len(families)
    by_stratum = defaultdict(list)
    for row in families:
        by_stratum[(row["split"], row["template_id"])].append(row)
        assert result["assignment"][row["family_id"]] not in row["present_class_ids"]
    for rows in by_stratum.values():
        expected = Counter(row["positive_query_class_id"] for row in rows)
        actual = Counter(result["assignment"][row["family_id"]] for row in rows)
        assert actual == expected
    assert all(item["total_variation"] == 0.0 for item in result["strata"])


def test_multiobject_forbidden_classes_are_never_assigned():
    families = [
        _family("a", "train", "t0", 0, [0, 2]),
        _family("b", "train", "t0", 1, [1]),
        _family("c", "train", "t0", 2, [2, 1]),
    ]
    result = match_all_strata(families)
    assert all(
        result["assignment"][row["family_id"]] not in row["present_class_ids"]
        for row in families
    )


def test_infeasible_stratum_blocks_with_explicit_deficit_and_no_partial_assignment():
    families = [
        _family("a", "train", "t0", 0, [0, 1]),
        _family("b", "train", "t0", 1, [0, 1]),
    ]
    with pytest.raises(ExactMatchInfeasible) as caught:
        match_all_strata(families)
    report = caught.value.report
    assert report["decision"] == "BLOCK"
    assert report["stratum"] == {"split": "train", "template_id": "t0"}
    assert report["required_flow"] == 2
    assert report["achieved_flow"] == 0
    assert report["deficit"] == 2
    assert report["class_quotas"] == {"0": 1, "1": 1}
    assert "assignment" not in report


def test_matching_is_deterministic_and_never_moves_split_or_template():
    families = [
        _family(f"{split}-{template}-{i}", split, template, i % 4)
        for split in ("train", "dev")
        for template in ("t0", "t1")
        for i in range(8)
    ]
    first = match_all_strata(families)
    second = match_all_strata(list(reversed(families)))
    assert first["assignment"] == second["assignment"]
    assert first["family_stratum"] == second["family_stratum"]
