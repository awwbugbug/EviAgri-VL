"""Read-only forensics for the 24 Task11A.3 PlantSeg router false positives."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from evaluate_task10b_probe import _classifier, _fit, _split_arrays
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new
from task11a_confidence_router import temperature_scale


BASE_FEATURE_SHA = "5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc"
BASE_ROWS_SHA = "2ad5192520a2fdbf1b1f058cfd987d6ad121985f62239e41234ae0d2d2a25ffd"
NULL_FEATURE_SHA = "e05f01467c70ec334656f1702e1e0ec8fd4c5d8a14a7dad1616b9d98fc62b618"
NULL_MANIFEST_SHA = "7196eb45259b851908c362dfbbb08e1b5b81f65f36dfe1a25d36692e63efc025"
PREDICTION_SHA = "41d605b8deae125452f4e9437396d494bb841f1a7174da05f61ec89f94d0db51"
REVIEW_MANIFEST_SHA = "10321ccf7fc9d6882c5fd63290977a6ff0eba9a9f52dd515e14c0ddd9dd59e0b"
TEMPERATURE = 0.18887372662036642
THRESHOLD = 0.63
EXPECTED_IMAGES = 290
EXPECTED_FALSE_POSITIVES = 24


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(median(array.tolist())),
        "minimum": float(array.min()), "maximum": float(array.max()),
    }


def disease_pattern(name: str) -> str:
    value = name.lower()
    groups = (
        ("rust", ("rust",)),
        ("spot_or_blotch", ("spot", "blotch", "scab")),
        ("blight", ("blight",)),
        ("mildew_or_mold", ("mildew", "mold")),
        ("rot", ("rot",)),
        ("virus_or_mosaic", ("virus", "mosaic")),
    )
    return next((label for label, words in groups if any(word in value for word in words)), "other")


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def _write_html(path: Path, cases: list[dict[str, Any]]) -> None:
    cards = []
    for row in cases:
        cards.append(f"""
<article><header><b>{html.escape(row['id'])}</b><span>conf={row['confidence']:.4f} · pest_id={row['forced_prediction']}</span></header>
<div class="pics"><figure><img src="assets/images/{row['id']}.jpg"><figcaption>原图</figcaption></figure>
<figure><img src="assets/overlays/{row['id']}.jpg"><figcaption>PlantSeg 病斑 mask</figcaption></figure></div>
<p>{html.escape(row['plant'])} / {html.escape(row['disease'])} · pattern={row['disease_pattern']} · mask={row['mask_ratio']:.3f}</p>
<p>margin={row['top1_top2_margin']:.4f} · entropy={row['entropy']:.4f} · nearest predicted-class IP102 cosine={row['nearest_predicted_class_cosine']:.4f}</p></article>""")
    path.write_text("""<!doctype html><meta charset="utf-8"><title>Task11A.4 false positives</title>
<style>body{font-family:system-ui;background:#eef1ea;color:#17231c;margin:0}main{max-width:1400px;margin:auto;padding:24px}h1{margin:0 0 8px}.note{background:#fff7d6;padding:14px;border-left:6px solid #c79a1b;margin-bottom:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}article{background:#fff;border:1px solid #aab3aa;box-shadow:5px 5px 0 #cbd1c9}header{display:flex;justify-content:space-between;padding:12px;background:#183d2d;color:#fff}.pics{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#999}figure{margin:0;background:#222;position:relative}img{width:100%;height:300px;object-fit:contain}figcaption{position:absolute;bottom:6px;left:6px;background:#000b;color:#fff;padding:4px 7px}p{margin:10px 12px;font-size:13px}</style>
<main><h1>Task11A.4：24 个误接收案例</h1><div class="note">只读法医页。红色区域为 PlantSeg 病斑，不是害虫框；不得用本页调 Task11A.3 的冻结阈值。</div><div class="grid">""" + "\n".join(cards) + "</div></main>", encoding="utf-8")


def analyze(*, task11a3_root: Path, base_feature_root: Path, review_root: Path, output_root: Path) -> dict[str, Any]:
    task11a3_root, base_feature_root, review_root = map(Path, (task11a3_root, base_feature_root, review_root))
    output_root = Path(output_root)
    ensure_new_directory(output_root)
    try:
        base_features_path = base_feature_root / "features.npy"
        base_rows_path = base_feature_root / "feature_rows.jsonl"
        null_features_path = task11a3_root / "features" / "features.npy"
        null_manifest_path = task11a3_root / "dataset" / "manifest.jsonl"
        review_manifest_path = review_root / "review_manifest.jsonl"
        expected = ((base_features_path, BASE_FEATURE_SHA), (base_rows_path, BASE_ROWS_SHA),
                    (null_features_path, NULL_FEATURE_SHA), (null_manifest_path, NULL_MANIFEST_SHA),
                    (review_manifest_path, REVIEW_MANIFEST_SHA))
        for path, digest in expected:
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"frozen input SHA mismatch: {path}")
        prediction_paths = [task11a3_root / "evaluation" / f"seed_{seed}_predictions.jsonl" for seed in (17, 29, 43)]
        if any(sha256_file(path) != PREDICTION_SHA for path in prediction_paths):
            raise ValueError("seed prediction SHA mismatch")
        prediction_rows = [_jsonl(path) for path in prediction_paths]
        if any(len(rows) != EXPECTED_IMAGES for rows in prediction_rows) or prediction_rows[0] != prediction_rows[1] or prediction_rows[0] != prediction_rows[2]:
            raise ValueError("seed predictions are not aligned and identical")

        base_features = np.load(base_features_path, allow_pickle=False)
        null_features = np.load(null_features_path, allow_pickle=False)
        base_rows, null_rows = _jsonl(base_rows_path), _jsonl(null_manifest_path)
        if base_features.shape != (320, 2048) or null_features.shape != (EXPECTED_IMAGES, 2048) or len(null_rows) != EXPECTED_IMAGES:
            raise ValueError("feature/manifest shape mismatch")
        x_train, y_train, train_rows = _split_arrays(base_features, base_rows, "train")
        classifier = _classifier(17)
        _fit(classifier, x_train, y_train)
        probabilities = temperature_scale(classifier.predict_proba(null_features), TEMPERATURE)
        order = np.argsort(probabilities, axis=1)
        top1_index, top2_index = order[:, -1], order[:, -2]
        forced = classifier.classes_[top1_index].astype(np.int64)
        confidence = probabilities[np.arange(EXPECTED_IMAGES), top1_index]
        margin = confidence - probabilities[np.arange(EXPECTED_IMAGES), top2_index]
        accepted = confidence >= THRESHOLD
        frozen = prediction_rows[0]
        if any(int(row["forced_prediction"]) != int(pred) or bool(row["accepted"]) != bool(keep) or not math.isclose(float(row["confidence"]), float(score), abs_tol=1e-12)
               for row, pred, keep, score in zip(frozen, forced, accepted, confidence, strict=True)):
            raise ValueError("frozen router reproduction mismatch")
        if int(accepted.sum()) != EXPECTED_FALSE_POSITIVES:
            raise ValueError("unexpected false-positive count")

        similarity = null_features @ x_train.T
        entropy = _entropy(probabilities)
        review_ids = {str(row["audit_id"]) for row in _jsonl(review_manifest_path)}
        all_cases, accepted_cases = [], []
        for index, (source, pred) in enumerate(zip(null_rows, frozen, strict=True)):
            row_id, class_id = str(source["id"]), int(forced[index])
            same = np.flatnonzero(y_train == class_id)
            nearest_all = int(similarity[index].argmax())
            nearest_same_local = int(same[similarity[index, same].argmax()])
            case = {
                "id": row_id, "accepted": bool(accepted[index]), "confidence": float(confidence[index]),
                "forced_prediction": class_id, "top2_prediction": int(classifier.classes_[top2_index[index]]),
                "top1_top2_margin": float(margin[index]), "entropy": float(entropy[index]),
                "plant": str(source["plant"]), "disease": str(source["disease"]),
                "disease_pattern": disease_pattern(str(source["disease"])), "mask_ratio": float(source["mask_ratio"]),
                "nearest_train_cosine": float(similarity[index, nearest_all]),
                "nearest_train_class": int(y_train[nearest_all]), "nearest_train_id": str(train_rows[nearest_all]["id"]),
                "nearest_predicted_class_cosine": float(similarity[index, nearest_same_local]),
                "nearest_predicted_class_id": str(train_rows[nearest_same_local]["id"]),
                "input_contract": "pixels_only", "posthoc_metadata_only": True,
            }
            all_cases.append(case)
            if case["accepted"]:
                if row_id not in review_ids:
                    raise ValueError(f"missing review row: {row_id}")
                accepted_cases.append(case)

        image_dir, overlay_dir = output_root / "assets" / "images", output_root / "assets" / "overlays"
        image_dir.mkdir(parents=True); overlay_dir.mkdir(parents=True)
        for case in accepted_cases:
            for source_dir, destination_dir in ((review_root / "images", image_dir), (review_root / "overlays", overlay_dir)):
                source = source_dir / f"{case['id']}.jpg"
                if not source.is_file(): raise ValueError(f"missing review asset: {source}")
                shutil.copyfile(source, destination_dir / source.name)

        accepted_group = [row for row in all_cases if row["accepted"]]
        rejected_group = [row for row in all_cases if not row["accepted"]]
        report = {
            "version": "task11a4-router-false-positive-forensics-1", "state": "completed",
            "protocol_identity": "task11a3_protocol_v1_attempt_03", "read_only": True,
            "task8_locked_set_read": False, "task11b_started": False, "threshold_changed": False,
            "image_count": EXPECTED_IMAGES, "false_positive_count": len(accepted_group),
            "false_positive_rate": len(accepted_group) / EXPECTED_IMAGES,
            "seed_outputs_identical": True,
            "accepted_prediction_counts": dict(sorted(Counter(str(row["forced_prediction"]) for row in accepted_group).items())),
            "accepted_disease_patterns": dict(sorted(Counter(row["disease_pattern"] for row in accepted_group).items())),
            "accepted_plants": dict(sorted(Counter(row["plant"] for row in accepted_group).items())),
            "accepted_vs_rejected": {
                key: {"accepted": _summary([float(row[key]) for row in accepted_group]),
                      "rejected": _summary([float(row[key]) for row in rejected_group])}
                for key in ("confidence", "top1_top2_margin", "entropy", "mask_ratio", "nearest_train_cosine", "nearest_predicted_class_cosine")
            },
            "observations": {
                "largest_predicted_class_fraction": max(Counter(row["forced_prediction"] for row in accepted_group).values()) / len(accepted_group),
                "metadata_pattern_is_posthoc_only": True,
                "plantseg_no_longer_confirmatory_for_representation_selection": True,
            },
            "next_single_variable_hypothesis": "Task11B-RepProbe: vision-pooled versus exact Qwen 3L/4 query-token representation only",
            "limitations": ["Disease names and masks are post-output forensic metadata and never model inputs.",
                            "Nearest-neighbor similarity is descriptive and does not establish causality.",
                            "PlantSeg cannot be reused as an untouched confirmatory set after this analysis."],
        }
        with (output_root / "forensic_cases.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in accepted_cases: handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        write_json_new(output_root / "forensic_report.json", report)
        lines = ["# Task11A.4 误接收法医", "", f"- 误接收：{len(accepted_group)}/{EXPECTED_IMAGES}。",
                 f"- 类别分布：{report['accepted_prediction_counts']}。", f"- 病种模式：{report['accepted_disease_patterns']}。",
                 "- 结论边界：只读描述，不改变 Task11A.3 PASS 或阈值。", f"- 下一单变量：{report['next_single_variable_hypothesis']}。"]
        (output_root / "forensic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _write_html(output_root / "review.html", accepted_cases)
        write_json_new(output_root / "run_summary.json", {"state": "completed", "false_positive_count": len(accepted_group),
            "base_features_sha256": BASE_FEATURE_SHA, "null_features_sha256": NULL_FEATURE_SHA,
            "null_manifest_sha256": NULL_MANIFEST_SHA, "prediction_sha256": PREDICTION_SHA})
        signed = ["forensic_cases.jsonl", "forensic_report.json", "forensic_report.md", "review.html", "run_summary.json"]
        signed += [f"assets/{kind}/{case['id']}.jpg" for kind in ("images", "overlays") for case in accepted_cases]
        with (output_root / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
            for name in signed: handle.write(f"{sha256_file(output_root / name)}  {name}\n")
        return report
    except Exception:
        if output_root.exists():
            (output_root / "status.json").write_text('{"state":"failed"}\n', encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task11a3-root", type=Path, required=True)
    parser.add_argument("--base-feature-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(task11a3_root=args.task11a3_root, base_feature_root=args.base_feature_root,
        review_root=args.review_root, output_root=args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
