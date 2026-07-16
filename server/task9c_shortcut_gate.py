"""Fail-closed text-only shortcut gate for the frozen Task 9B protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import FeatureUnion


REQUIRED_VIEWS = (
    "user_prompt_only",
    "system_user_prompt",
    "prompt_nonimage_metadata",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if set(row) != {"id", "family_id", "split", "label", "text"}:
                raise ValueError(f"unexpected probe schema at {path}:{line_number}")
            if row["split"] not in {"train", "val", "dev"} or row["label"] not in {0, 1}:
                raise ValueError(f"invalid split/label at {path}:{line_number}")
            if not isinstance(row["text"], str) or not row["text"]:
                raise ValueError(f"empty probe text at {path}:{line_number}")
            rows.append(row)
    return rows


def _vectorizer(max_features: int) -> FeatureUnion:
    if max_features < 20:
        raise ValueError("max_features must be at least 20")
    word_features = max(10, max_features // 3)
    char_features = max(10, max_features - word_features)
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    min_df=2,
                    max_features=word_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    min_df=2,
                    max_features=char_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def evaluate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    max_features: int = 60_000,
) -> dict[str, dict[str, Any]]:
    rows = list(rows)
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val", "dev")
    }
    if any(not values for values in by_split.values()):
        raise ValueError("train, val, and dev must all be non-empty")
    if any(len({row["label"] for row in values}) != 2 for values in by_split.values()):
        raise ValueError("each split must contain both labels")
    train_families = {row["family_id"] for row in by_split["train"]}
    if any(train_families & {row["family_id"] for row in by_split[split]} for split in ("val", "dev")):
        raise ValueError("family leakage between fit and evaluation splits")

    vectorizer = _vectorizer(max_features)
    train_text = [row["text"] for row in by_split["train"]]
    train_y = [int(row["label"]) for row in by_split["train"]]
    train_x = vectorizer.fit_transform(train_text)
    classifier = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    classifier.fit(train_x, train_y)

    metrics: dict[str, dict[str, Any]] = {}
    for split in ("val", "dev"):
        values = by_split[split]
        labels = [int(row["label"]) for row in values]
        features = vectorizer.transform([row["text"] for row in values])
        scores = classifier.predict_proba(features)[:, 1]
        predictions = (scores >= 0.5).astype(int)
        metrics[split] = {
            "n": len(values),
            "label_counts": {str(key): value for key, value in sorted(Counter(labels).items())},
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "auroc": float(roc_auc_score(labels, scores)),
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist(),
        }
    return metrics


def decide_gate(
    view_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    threshold: float = 0.55,
) -> dict[str, Any]:
    violations = []
    for view in sorted(view_metrics):
        for split in ("val", "dev"):
            for metric in ("balanced_accuracy", "auroc"):
                value = float(view_metrics[view][split][metric])
                if value > threshold:
                    violations.append({"view": view, "split": split, "metric": metric, "value": value})
    return {
        "threshold": threshold,
        "rule": "all views and evaluation splits require balanced_accuracy <= 0.55 and auroc <= 0.55",
        "decision": "BLOCK" if violations else "PASS",
        "training_allowed": not violations,
        "violations": violations,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_gate(
    view_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    seed: int = 20260715,
    max_features: int = 60_000,
    threshold: float = 0.55,
) -> dict[str, Any]:
    if set(view_paths) != set(REQUIRED_VIEWS):
        raise ValueError(f"exactly these views are required: {REQUIRED_VIEWS}")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing existing gate output: {output_dir}")
    normalized = {name: Path(path) for name, path in view_paths.items()}
    rows_by_view = {name: _read_jsonl(normalized[name]) for name in REQUIRED_VIEWS}
    signatures = {
        name: [(row["id"], row["family_id"], row["split"], row["label"]) for row in rows]
        for name, rows in rows_by_view.items()
    }
    if len({tuple(value) for value in signatures.values()}) != 1:
        raise ValueError("probe views are not aligned by id/family/split/label")

    metrics = {
        name: evaluate_rows(rows_by_view[name], seed=seed, max_features=max_features)
        for name in REQUIRED_VIEWS
    }
    decision = decide_gate(metrics, threshold=threshold)
    report = {
        "version": "task9c-shortcut-gate-1",
        "classifier": {
            "type": "word+char TF-IDF / class-balanced logistic regression",
            "word_ngrams": [1, 2],
            "char_ngrams": [3, 5],
            "max_features": max_features,
            "C": 1.0,
            "solver": "liblinear",
            "seed": seed,
            "fit_split": "train",
            "evaluation_splits": ["val", "dev"],
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        },
        "input_sha256": {name: _sha256(path) for name, path in normalized.items()},
        "metrics": metrics,
        "decision": decision,
    }
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    completion = output_dir / "completion.sha256"
    completion.write_text(f"{_sha256(metrics_path)}  metrics.json\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-features", type=int, default=60_000)
    parser.add_argument("--threshold", type=float, default=0.55)
    arguments = parser.parse_args()
    paths = {name: arguments.probe_dir / f"{name}.jsonl" for name in REQUIRED_VIEWS}
    report = run_gate(
        paths,
        arguments.output_dir,
        seed=arguments.seed,
        max_features=arguments.max_features,
        threshold=arguments.threshold,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
