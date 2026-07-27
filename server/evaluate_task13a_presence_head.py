"""Evaluate the frozen Task 13A evidence-presence head protocol."""

from __future__ import annotations

import argparse
import json
import math
import random
import traceback
import warnings
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new

SEEDS = (17, 29, 43)
EXPECTED_SHA256 = {
    "base_features": "5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc",
    "base_rows": "2ad5192520a2fdbf1b1f058cfd987d6ad121985f62239e41234ae0d2d2a25ffd",
    "plantdoc_features": "412815de2d6addd61b2863b9ec5227879888ae04250aabd5d736cce70159907a",
    "plantdoc_rows": "11f9a72a735b9f4b90634cd3dc3d8fe49ce4584f10f2bc89c04db7b33cfaf8f2",
    "plantseg_features": "e05f01467c70ec334656f1702e1e0ec8fd4c5d8a14a7dad1616b9d98fc62b618",
    "plantseg_rows": "7623b5fccd48650a15f6112e3c6564c0541f488c8bdb144dfb21552c8a48ae1c",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_completion(root: Path) -> None:
    for line in (root / "completion.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        target = root / relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"completion SHA256 mismatch: {target}")


def _selected(row: dict[str, Any], *, source: str, role: str) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "source": source,
        "role": role,
        "feature_index": int(row["feature_index"]),
        "class_id": None if row.get("class_id") is None else int(row["class_id"]),
        "group": row.get("healthy_class") or row.get("disease") or row.get("class_band"),
    }


def select_rows(
    base: list[dict[str, Any]],
    plantdoc: list[dict[str, Any]],
    plantseg: list[dict[str, Any]],
    prior_groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    used_ids = {str(row["id"]) for rows in prior_groups for row in rows}
    selected: list[dict[str, Any]] = []
    classes = sorted({int(row["class_id"]) for row in base})
    if len(classes) != 16:
        raise ValueError("expected 16 positive classes")
    for class_id in classes:
        fresh_train = sorted(
            (
                row
                for row in base
                if int(row["class_id"]) == class_id
                and row["split"] == "train"
                and str(row["id"]) not in used_ids
            ),
            key=lambda row: str(row["id"]),
        )
        fresh_val = sorted(
            (
                row
                for row in base
                if int(row["class_id"]) == class_id
                and row["split"] == "val"
                and str(row["id"]) not in used_ids
            ),
            key=lambda row: str(row["id"]),
        )
        if len(fresh_train) != 4 or len(fresh_val) != 1:
            raise ValueError(f"unexpected fresh positive pool for class {class_id}")
        selected.extend(
            _selected(row, source="ip102", role="positive_train")
            for row in fresh_train[:2]
        )
        selected.append(_selected(fresh_val[0], source="ip102", role="positive_val"))
        selected.extend(
            _selected(row, source="ip102", role="positive_test")
            for row in fresh_train[2:]
        )

    components: dict[str, set[str]] = {}
    base_by_id = {str(row["id"]): row for row in base}
    for role in ("positive_train", "positive_val", "positive_test"):
        components[role] = {
            str(base_by_id[row["id"]]["near_duplicate_component_id"])
            for row in selected
            if row["role"] == role
        }
    pairs = (
        ("positive_train", "positive_val"),
        ("positive_train", "positive_test"),
        ("positive_val", "positive_test"),
    )
    if any(components[left] & components[right] for left, right in pairs):
        raise ValueError("near-duplicate component crossed positive split")

    healthy_classes = sorted({str(row["healthy_class"]) for row in plantdoc})
    if len(healthy_classes) != 10:
        raise ValueError("expected 10 PlantDoc healthy classes")
    for healthy_class in healthy_classes:
        rows = sorted(
            (row for row in plantdoc if str(row["healthy_class"]) == healthy_class),
            key=lambda row: str(row["id"]),
        )
        if len(rows) != 4:
            raise ValueError(f"unexpected PlantDoc class size: {healthy_class}")
        selected.extend(
            _selected(row, source="plantdoc", role="null_train") for row in rows[:2]
        )
        selected.extend(
            _selected(row, source="plantdoc", role="plantdoc_test") for row in rows[2:]
        )

    remaining = sorted(
        (row for row in plantseg if str(row["id"]) not in used_ids),
        key=lambda row: (float(row["mask_ratio"]), str(row["id"])),
    )
    indices = np.linspace(0, len(remaining) - 1, 32, dtype=int).tolist()
    if len(set(indices)) != 32:
        raise ValueError("insufficient mechanism-untouched PlantSeg rows")
    selected.extend(
        _selected(remaining[index], source="plantseg", role="plantseg_test")
        for index in indices
    )

    expected = {
        "positive_train": 32,
        "positive_val": 16,
        "positive_test": 32,
        "null_train": 20,
        "plantdoc_test": 20,
        "plantseg_test": 32,
    }
    counts = {role: sum(row["role"] == role for row in selected) for role in expected}
    if counts != expected or len({row["id"] for row in selected}) != len(selected):
        raise ValueError(f"selection contract failure: {counts}")
    return selected


def positive_only_threshold(scores: np.ndarray, target_recall: float = 0.90) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("invalid positive validation scores")
    if not 0 < target_recall <= 1:
        raise ValueError("invalid target recall")
    required = int(math.ceil(target_recall * values.size))
    return float(np.sort(values)[values.size - required])


def _classifier(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )


def _fit(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model.fit(x, y)


def _source_metrics(
    positive_scores: np.ndarray, null_scores: np.ndarray, threshold: float
) -> dict[str, float]:
    positive_accept = positive_scores >= threshold
    null_accept = null_scores >= threshold
    truth = np.r_[np.ones(positive_scores.size), np.zeros(null_scores.size)]
    predicted = np.r_[positive_accept, null_accept].astype(np.int64)
    return {
        "positive_coverage": float(positive_accept.mean()),
        "null_fpr": float(null_accept.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "auroc": float(roc_auc_score(truth, np.r_[positive_scores, null_scores])),
    }


def decide(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    presence = [per_seed[str(seed)]["P1_presence"] for seed in SEEDS]
    baseline = [per_seed[str(seed)]["T0_taxonomy"] for seed in SEEDS]
    supported_delta = mean(
        current["supported_diagnosis"] - control["supported_diagnosis"]
        for current, control in zip(presence, baseline, strict=True)
    )
    gates = {
        "positive_coverage_ge_0_875_all_seeds": min(
            row["positive_coverage"] for row in presence
        )
        >= 0.875,
        "plantdoc_fpr_lt_0_10_all_seeds": max(
            row["plantdoc"]["null_fpr"] for row in presence
        )
        < 0.10,
        "plantseg_fpr_lt_0_10_all_seeds": max(
            row["plantseg"]["null_fpr"] for row in presence
        )
        < 0.10,
        "combined_auroc_ge_0_90_all_seeds": min(
            row["combined"]["auroc"] for row in presence
        )
        >= 0.90,
        "plantseg_fpr_not_worse_all_seeds": all(
            current["plantseg"]["null_fpr"] <= control["plantseg"]["null_fpr"]
            for current, control in zip(presence, baseline, strict=True)
        ),
        "mean_supported_diagnosis_delta_ge_minus_1_over_32": supported_delta
        >= -(1 / 32),
    }
    passed = all(gates.values())
    return {
        "decision": "PASS_H3_FEASIBILITY" if passed else "H3_BLOCK_H2_PRIORITY",
        "gates": gates,
        "mean_supported_diagnosis_delta": supported_delta,
        "authorize_tiny_gated_fusion": passed,
        "authorize_large_training": False,
        "authorize_task8": False,
    }


def _summary(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name in ("T0_taxonomy", "P1_presence"):
        metrics = {
            "positive_coverage": [per_seed[str(seed)][model_name]["positive_coverage"] for seed in SEEDS],
            "supported_diagnosis": [per_seed[str(seed)][model_name]["supported_diagnosis"] for seed in SEEDS],
            "plantdoc_fpr": [per_seed[str(seed)][model_name]["plantdoc"]["null_fpr"] for seed in SEEDS],
            "plantseg_fpr": [per_seed[str(seed)][model_name]["plantseg"]["null_fpr"] for seed in SEEDS],
            "combined_auroc": [per_seed[str(seed)][model_name]["combined"]["auroc"] for seed in SEEDS],
        }
        result[model_name] = {
            key: {
                "mean": mean(values),
                "std": pstdev(values),
                "worst": max(values) if key.endswith("fpr") else min(values),
            }
            for key, values in metrics.items()
        }
    return result


def _bootstrap(
    raw: dict[int, dict[str, dict[str, np.ndarray]]], repetitions: int
) -> dict[str, Any]:
    rng = random.Random(20260727)
    n_pos = raw[SEEDS[0]]["T0_taxonomy"]["positive_score"].size
    n_doc = raw[SEEDS[0]]["T0_taxonomy"]["plantdoc_score"].size
    n_seg = raw[SEEDS[0]]["T0_taxonomy"]["plantseg_score"].size

    def calculate(pos: np.ndarray, doc: np.ndarray, seg: np.ndarray) -> dict[str, float]:
        values: dict[str, list[float]] = {
            "positive_coverage_delta": [],
            "supported_diagnosis_delta": [],
            "plantdoc_fpr_delta": [],
            "plantseg_fpr_delta": [],
            "combined_fpr_delta": [],
            "combined_auroc_delta": [],
        }
        for seed in SEEDS:
            control = raw[seed]["T0_taxonomy"]
            current = raw[seed]["P1_presence"]
            values["positive_coverage_delta"].append(float((current["positive_accept"][pos] - control["positive_accept"][pos]).mean()))
            values["supported_diagnosis_delta"].append(float((current["positive_accept"][pos] * current["correct"][pos] - control["positive_accept"][pos] * control["correct"][pos]).mean()))
            values["plantdoc_fpr_delta"].append(float((current["plantdoc_accept"][doc] - control["plantdoc_accept"][doc]).mean()))
            values["plantseg_fpr_delta"].append(float((current["plantseg_accept"][seg] - control["plantseg_accept"][seg]).mean()))
            current_null = np.r_[current["plantdoc_score"][doc], current["plantseg_score"][seg]]
            control_null = np.r_[control["plantdoc_score"][doc], control["plantseg_score"][seg]]
            values["combined_fpr_delta"].append(float(np.r_[current["plantdoc_accept"][doc], current["plantseg_accept"][seg]].mean() - np.r_[control["plantdoc_accept"][doc], control["plantseg_accept"][seg]].mean()))
            truth = np.r_[np.ones(pos.size), np.zeros(doc.size + seg.size)]
            values["combined_auroc_delta"].append(float(roc_auc_score(truth, np.r_[current["positive_score"][pos], current_null]) - roc_auc_score(truth, np.r_[control["positive_score"][pos], control_null])))
        return {key: mean(items) for key, items in values.items()}

    original = calculate(np.arange(n_pos), np.arange(n_doc), np.arange(n_seg))
    samples = {key: [] for key in original}
    for _ in range(repetitions):
        result = calculate(
            np.asarray([rng.randrange(n_pos) for _ in range(n_pos)]),
            np.asarray([rng.randrange(n_doc) for _ in range(n_doc)]),
            np.asarray([rng.randrange(n_seg) for _ in range(n_seg)]),
        )
        for key, value in result.items():
            samples[key].append(value)
    return {
        key: {
            "estimate": original[key],
            "low": float(np.quantile(samples[key], 0.025)),
            "high": float(np.quantile(samples[key], 0.975)),
            "repetitions": repetitions,
            "unit": "image",
        }
        for key in original
    }


def run(
    *,
    base_root: Path,
    plantdoc_root: Path,
    plantseg_root: Path,
    prior_rows: list[Path],
    output_root: Path,
    repetitions: int = 1000,
) -> dict[str, Any]:
    root = Path(output_root)
    ensure_new_directory(root)
    (root / "status.json").write_text('{"state":"running"}\n', encoding="utf-8")
    try:
        roots = {"base": Path(base_root), "plantdoc": Path(plantdoc_root), "plantseg": Path(plantseg_root)}
        for feature_root in roots.values():
            verify_completion(feature_root)
        files = {
            "base_features": roots["base"] / "features.npy",
            "base_rows": roots["base"] / "feature_rows.jsonl",
            "plantdoc_features": roots["plantdoc"] / "features.npy",
            "plantdoc_rows": roots["plantdoc"] / "feature_rows.jsonl",
            "plantseg_features": roots["plantseg"] / "features.npy",
            "plantseg_rows": roots["plantseg"] / "feature_rows.jsonl",
        }
        for name, path in files.items():
            if sha256_file(path) != EXPECTED_SHA256[name]:
                raise ValueError(f"unexpected frozen input SHA256: {name}")
        matrices = {name: np.load(path, allow_pickle=False) for name, path in files.items() if name.endswith("features")}
        rows = {name.removesuffix("_rows"): read_jsonl(path) for name, path in files.items() if name.endswith("rows")}
        if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
            raise ValueError("non-finite frozen feature")
        selected = select_rows(rows["base"], rows["plantdoc"], rows["plantseg"], [read_jsonl(path) for path in prior_rows])
        with (root / "manifest.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in selected:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

        def values(role: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
            chosen = [row for row in selected if row["role"] == role]
            matrix = np.stack([matrices[f"{row['source']}_features" if row["source"] != "ip102" else "base_features"][row["feature_index"]] for row in chosen]).astype(np.float32)
            return matrix, chosen

        x_train, train_rows = values("positive_train")
        x_val, _ = values("positive_val")
        x_test, test_rows = values("positive_test")
        x_null_train, _ = values("null_train")
        x_doc, doc_rows = values("plantdoc_test")
        x_seg, seg_rows = values("plantseg_test")
        y_train = np.asarray([int(row["class_id"]) for row in train_rows])
        y_test = np.asarray([int(row["class_id"]) for row in test_rows])

        per_seed: dict[str, dict[str, Any]] = {}
        raw: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        predictions: list[dict[str, Any]] = []
        signed = ["manifest.jsonl"]
        for seed in SEEDS:
            taxonomy = _classifier(seed)
            presence = _classifier(seed)
            _fit(taxonomy, x_train, y_train)
            _fit(presence, np.concatenate([x_train, x_null_train]), np.r_[np.ones(x_train.shape[0]), np.zeros(x_null_train.shape[0])])
            taxonomy_prediction = taxonomy.classes_[taxonomy.predict_proba(x_test).argmax(axis=1)].astype(np.int64)
            correct = (taxonomy_prediction == y_test).astype(np.int64)
            score_sets = {
                "T0_taxonomy": {
                    "val": taxonomy.predict_proba(x_val).max(axis=1),
                    "positive": taxonomy.predict_proba(x_test).max(axis=1),
                    "plantdoc": taxonomy.predict_proba(x_doc).max(axis=1),
                    "plantseg": taxonomy.predict_proba(x_seg).max(axis=1),
                },
                "P1_presence": {
                    "val": presence.predict_proba(x_val)[:, list(presence.classes_).index(1.0)],
                    "positive": presence.predict_proba(x_test)[:, list(presence.classes_).index(1.0)],
                    "plantdoc": presence.predict_proba(x_doc)[:, list(presence.classes_).index(1.0)],
                    "plantseg": presence.predict_proba(x_seg)[:, list(presence.classes_).index(1.0)],
                },
            }
            seed_metrics: dict[str, Any] = {}
            raw[seed] = {}
            for model_name, scores in score_sets.items():
                threshold = positive_only_threshold(scores["val"])
                positive_accept = (scores["positive"] >= threshold).astype(np.int64)
                doc_accept = (scores["plantdoc"] >= threshold).astype(np.int64)
                seg_accept = (scores["plantseg"] >= threshold).astype(np.int64)
                seed_metrics[model_name] = {
                    "threshold": threshold,
                    "positive_val_recall": float((scores["val"] >= threshold).mean()),
                    "positive_coverage": float(positive_accept.mean()),
                    "taxonomy_accuracy": float(correct.mean()),
                    "supported_diagnosis": float((correct * positive_accept).mean()),
                    "plantdoc": _source_metrics(scores["positive"], scores["plantdoc"], threshold),
                    "plantseg": _source_metrics(scores["positive"], scores["plantseg"], threshold),
                    "combined": _source_metrics(scores["positive"], np.r_[scores["plantdoc"], scores["plantseg"]], threshold),
                }
                raw[seed][model_name] = {
                    "positive_score": scores["positive"],
                    "plantdoc_score": scores["plantdoc"],
                    "plantseg_score": scores["plantseg"],
                    "positive_accept": positive_accept,
                    "plantdoc_accept": doc_accept,
                    "plantseg_accept": seg_accept,
                    "correct": correct,
                }
                for role, role_rows, role_scores, accepted in (
                    ("positive_test", test_rows, scores["positive"], positive_accept),
                    ("plantdoc_test", doc_rows, scores["plantdoc"], doc_accept),
                    ("plantseg_test", seg_rows, scores["plantseg"], seg_accept),
                ):
                    for row, score, keep in zip(role_rows, role_scores, accepted, strict=True):
                        predictions.append({"seed": seed, "model": model_name, "id": row["id"], "source": row["source"], "role": role, "score": float(score), "accepted": bool(keep)})
            per_seed[str(seed)] = seed_metrics
            name = f"seed_{seed}_metrics.json"
            write_json_new(root / name, seed_metrics)
            signed.append(name)

        with (root / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in predictions:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        signed.append("predictions.jsonl")
        decision = decide(per_seed)
        report = {
            "version": "task13a-frozen-presence-head-1",
            "sample_counts": {role: sum(row["role"] == role for row in selected) for role in sorted({row["role"] for row in selected})},
            "per_seed": per_seed,
            "three_seed_summary": _summary(per_seed),
            "paired_bootstrap": _bootstrap(raw, repetitions),
            "decision": decision,
            "training": {"qwen_frozen": True, "linear_heads_only": True, "plantseg_used_for_fit": False, "null_used_for_threshold": False},
            "interpretation_limit": "Cross-source domain signal is not eliminated; this is feasibility evidence only.",
            "task8_locked_set_read": False,
        }
        write_json_new(root / "metrics.json", report)
        write_json_new(root / "decision_report.json", decision)
        write_json_new(root / "run_summary.json", {"state": "completed", "decision": decision["decision"], "manifest_sha256": sha256_file(root / "manifest.jsonl")})
        signed.extend(["metrics.json", "decision_report.json", "run_summary.json"])
        with (root / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
            for name in signed:
                handle.write(f"{sha256_file(root / name)}  {name}\n")
        (root / "status.json").write_text('{"state":"completed"}\n', encoding="utf-8")
        return report
    except Exception as exc:
        write_json_new(root / "failure.json", {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        (root / "status.json").write_text('{"state":"failed"}\n', encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--plantdoc-root", type=Path, required=True)
    parser.add_argument("--plantseg-root", type=Path, required=True)
    parser.add_argument("--prior-rows", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    report = run(base_root=args.base_root, plantdoc_root=args.plantdoc_root, plantseg_root=args.plantseg_root, prior_rows=args.prior_rows, output_root=args.output_root, repetitions=args.repetitions)
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
