import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9c_shortcut_gate import decide_gate, evaluate_rows, run_gate


def _rows(leaking: bool):
    rows = []
    for split in ("train", "val", "dev"):
        for index in range(20):
            label = index % 2
            text = ("definitely-positive" if label else "definitely-negative") if leaking else "same neutral prompt"
            rows.append({
                "id": f"{split}-{index}", "family_id": f"fam-{split}-{index}",
                "split": split, "label": label, "text": text,
            })
    return rows


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_leaking_text_exceeds_balanced_accuracy_and_auroc_gate():
    result = evaluate_rows(_rows(leaking=True), seed=17, max_features=200)
    assert result["val"]["balanced_accuracy"] > 0.55
    assert result["val"]["auroc"] > 0.55
    assert result["dev"]["balanced_accuracy"] > 0.55
    assert result["dev"]["auroc"] > 0.55


def test_identical_text_passes_both_metrics_at_chance():
    result = evaluate_rows(_rows(leaking=False), seed=17, max_features=200)
    for split in ("val", "dev"):
        assert result[split]["balanced_accuracy"] == 0.5
        assert result[split]["auroc"] == 0.5
    assert decide_gate({"view": result}, threshold=0.55)["decision"] == "PASS"


def test_any_view_or_split_above_threshold_blocks_training():
    metrics = {
        "safe": {"val": {"balanced_accuracy": 0.5, "auroc": 0.5},
                 "dev": {"balanced_accuracy": 0.5, "auroc": 0.5}},
        "leak": {"val": {"balanced_accuracy": 0.56, "auroc": 0.5},
                 "dev": {"balanced_accuracy": 0.5, "auroc": 0.5}},
    }
    decision = decide_gate(metrics, threshold=0.55)
    assert decision["decision"] == "BLOCK"
    assert decision["training_allowed"] is False
    assert decision["violations"] == [{
        "view": "leak", "split": "val", "metric": "balanced_accuracy", "value": 0.56
    }]


def test_run_gate_requires_three_aligned_views_and_refuses_overwrite(tmp_path):
    rows = _rows(leaking=False)
    paths = {}
    for name in ("user_prompt_only", "system_user_prompt", "prompt_nonimage_metadata"):
        path = tmp_path / f"{name}.jsonl"
        _write(path, rows)
        paths[name] = path
    output = tmp_path / "result"
    report = run_gate(paths, output, seed=19, max_features=200)
    assert report["decision"]["decision"] == "PASS"
    assert (output / "metrics.json").exists()
    assert (output / "completion.sha256").exists()
    try:
        run_gate(paths, output, seed=19, max_features=200)
    except FileExistsError:
        pass
    else:
        raise AssertionError("gate output must be immutable")


def test_run_gate_rejects_misaligned_probe_labels(tmp_path):
    rows = _rows(leaking=False)
    paths = {}
    for index, name in enumerate(("user_prompt_only", "system_user_prompt", "prompt_nonimage_metadata")):
        altered = [dict(row) for row in rows]
        if index == 2:
            altered[0]["label"] = 1 - altered[0]["label"]
        path = tmp_path / f"{name}.jsonl"
        _write(path, altered)
        paths[name] = path
    try:
        run_gate(paths, tmp_path / "result", seed=19, max_features=200)
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("misaligned views must be rejected")
