import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_human_review_bundle import CRITERIA
from finalize_task11a3_human_audit import finalize_human_audit


def _review(path: Path, reviewer: str, rejected: set[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", "reviewer_id", *CRITERIA, "notes"])
        writer.writeheader()
        for audit_id in ("a", "b", "c"):
            failed = audit_id in rejected
            row = {"audit_id": audit_id, "reviewer_id": reviewer, "notes": "FAIL:no_collage" if failed else ""}
            row.update({criterion: "FAIL" if failed and criterion == "no_collage" else "PASS" for criterion in CRITERIA})
            writer.writerow(row)


def test_final_audit_merges_agreement_and_adjudication(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"audit_id": x}) + "\n" for x in ("a", "b", "c")), encoding="utf-8")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _review(a, "A", {"a", "b"})
    _review(b, "B", {"b", "c"})
    adjudication = tmp_path / "adjudication.csv"
    with adjudication.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audit_id", "adjudicator_id", "final_decision", "failed_criteria", "notes"])
        writer.writerow(["a", "joint_AB", "KEEP", "", ""])
        writer.writerow(["c", "joint_AB", "EXCLUDE", "mask_valid", ""])

    report = finalize_human_audit(
        manifest_path=manifest, reviewer_a_path=a, reviewer_b_path=b,
        adjudication_path=adjudication, output_root=tmp_path / "out",
    )

    assert report["final_decision_counts"] == {"KEEP": 1, "EXCLUDE": 2, "UNCERTAIN": 0}
    assert report["strict_real_null_candidates"] == 1
    assert report["decision"] == "READY_FOR_FROZEN_ROUTER"
    assert len((tmp_path / "out" / "final_human_audit.jsonl").read_text().splitlines()) == 3
