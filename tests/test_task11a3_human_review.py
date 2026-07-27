import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_human_review_bundle import CRITERIA, review_decision, validate_review_rows


def _review(audit_id, reviewer="human-a", value="PASS"):
    return {"audit_id": audit_id, "reviewer_id": reviewer, **{key: value for key in CRITERIA}, "notes": ""}


def test_review_decision_fails_closed():
    assert review_decision(_review("a")) == "PASS"
    uncertain = _review("a")
    uncertain["mask_valid"] = "UNCERTAIN"
    assert review_decision(uncertain) == "UNCERTAIN"
    rejected = _review("a")
    rejected["no_visible_pest"] = "FAIL"
    assert review_decision(rejected) == "REJECT"


def test_review_requires_every_id_once_and_one_reviewer():
    rows = [_review("a"), _review("b")]
    assert validate_review_rows(rows, {"a", "b"})["counts"]["PASS"] == 2
    with pytest.raises(ValueError, match="every audit_id"):
        validate_review_rows(rows[:1], {"a", "b"})
    rows[1]["reviewer_id"] = "human-b"
    with pytest.raises(ValueError, match="one non-empty reviewer"):
        validate_review_rows(rows, {"a", "b"})


def test_review_rejects_blank_or_unrecognized_values():
    row = _review("a")
    row["lesion_visible"] = ""
    with pytest.raises(ValueError, match="invalid review value"):
        validate_review_rows([row], {"a"})
