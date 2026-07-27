import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_human_review_bundle import CRITERIA
from validate_task11a3_dual_review import validate_dual_review


def _write_review(path: Path, reviewer: str, rejected: set[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", "reviewer_id", *CRITERIA, "notes"])
        writer.writeheader()
        for audit_id in ("a", "b", "c"):
            failed = audit_id in rejected
            row = {"audit_id": audit_id, "reviewer_id": reviewer, "notes": "FAIL:no_collage" if failed else ""}
            row.update({criterion: "FAIL" if failed and criterion == "no_collage" else "PASS" for criterion in CRITERIA})
            writer.writerow(row)


def test_dual_review_reports_agreement_and_disagreements(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"audit_id": x}) + "\n" for x in ("a", "b", "c")), encoding="utf-8")
    reviewer_a, reviewer_b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_review(reviewer_a, "reviewer_A", {"a", "b"})
    _write_review(reviewer_b, "reviewer_B", {"b", "c"})

    report = validate_dual_review(
        manifest_path=manifest,
        reviewer_a_path=reviewer_a,
        reviewer_b_path=reviewer_b,
        output_root=tmp_path / "out",
    )

    assert report["agreement_counts"] == {
        "both_pass": 0, "both_reject": 1, "a_reject_b_pass": 1,
        "a_pass_b_reject": 1, "uncertain_involved": 0,
    }
    assert report["disagreement_rows"] == 2
    assert report["decision"] == "PENDING_DISAGREEMENT_ADJUDICATION"
    assert len((tmp_path / "out" / "adjudication_manifest.jsonl").read_text().splitlines()) == 2
