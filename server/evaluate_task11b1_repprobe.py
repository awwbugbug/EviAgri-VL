"""Paired Task11B.1 comparison of vision-pooled and query-token representations."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from evaluate_task10b_probe import _classifier, _fit, _split_arrays
from evaluate_task11a_confidence_router import _read_jsonl, _verify_completion, evaluate_seed
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new
from task11a_confidence_router import CONDITIONS, SEEDS, temperature_scale


EXPECTED_VISION = {
    "base": "5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc",
    "stress": "836527d652a860c1d6faf9c252414520209a43878973369aed848bdb094ee2e0",
    "plantdoc": "412815de2d6addd61b2863b9ec5227879888ae04250aabd5d736cce70159907a",
    "plantseg": "e05f01467c70ec334656f1702e1e0ec8fd4c5d8a14a7dad1616b9d98fc62b618",
}


def _load(root: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    _verify_completion(root)
    matrix = np.load(root / "features.npy", allow_pickle=False)
    rows = _read_jsonl(root / "feature_rows.jsonl")
    if matrix.ndim != 2 or matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid feature root: {root}")
    if [int(row["feature_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"feature alignment mismatch: {root}")
    return matrix, rows


def _external(classifier: Any, temperature: float, threshold: float,
              matrix: np.ndarray, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scaled = temperature_scale(classifier.predict_proba(matrix), temperature)
    confidence = scaled.max(axis=1)
    predictions = classifier.classes_[scaled.argmax(axis=1)].astype(np.int64)
    return [{"id": str(row["id"]), "accepted": bool(score >= threshold),
             "confidence": float(score), "forced_prediction": int(prediction)}
            for row, score, prediction in zip(rows, confidence, predictions, strict=True)]


def evaluate_representation(base: tuple[np.ndarray, list[dict[str, Any]]],
                            stress: tuple[np.ndarray, list[dict[str, Any]]],
                            plantdoc: tuple[np.ndarray, list[dict[str, Any]]],
                            plantseg: tuple[np.ndarray, list[dict[str, Any]]]) -> dict[str, Any]:
    base_matrix, base_rows = base; stress_matrix, stress_rows = stress
    outputs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        routed = evaluate_seed(base_matrix, base_rows, stress_matrix, stress_rows, seed)
        x_train, y_train, _ = _split_arrays(base_matrix, base_rows, "train")
        classifier = _classifier(seed); _fit(classifier, x_train, y_train)
        metrics = routed["metrics"]
        outputs[seed] = {"router": routed,
            "plantdoc": _external(classifier, float(metrics["temperature"]), float(metrics["threshold"]), *plantdoc),
            "plantseg": _external(classifier, float(metrics["temperature"]), float(metrics["threshold"]), *plantseg)}
    aggregate = {
        "accuracy": mean(outputs[s]["router"]["metrics"]["forced_original"]["accuracy"] for s in SEEDS),
        "forced_macro_f1": mean(outputs[s]["router"]["metrics"]["forced_original"]["macro_f1"] for s in SEEDS),
        "coverage": mean(outputs[s]["router"]["metrics"]["confidence_original"]["coverage"] for s in SEEDS),
        **{f"{c}_fpr": mean(outputs[s]["router"]["metrics"]["null"][f"{c}_fpr"] for s in SEEDS) for c in CONDITIONS},
        "synthetic_overall_fpr": mean(outputs[s]["router"]["metrics"]["null"]["overall_fpr"] for s in SEEDS),
        "plantdoc_fpr": mean(mean(float(r["accepted"]) for r in outputs[s]["plantdoc"]) for s in SEEDS),
        "plantseg_fpr": mean(mean(float(r["accepted"]) for r in outputs[s]["plantseg"]) for s in SEEDS),
        "json_contract": min(outputs[s]["router"]["metrics"]["json_contract"][k]
            for s in SEEDS for k in ("syntax_validity", "schema_validity", "semantic_consistency", "task_compliance")),
    }
    return {"seeds": outputs, "aggregate": aggregate}


def _bootstrap(values: Callable[[list[int]], float], count: int, repetitions: int, seed: int) -> dict[str, Any]:
    observed = values(list(range(count))); rng = random.Random(seed)
    samples = [values([rng.randrange(count) for _ in range(count)]) for _ in range(repetitions)]
    return {"estimate": observed, "low": float(np.quantile(samples, .025)),
            "high": float(np.quantile(samples, .975)), "repetitions": repetitions,
            "unit": "paired_source_image"}


def paired_positive(vision: dict[str, Any], query: dict[str, Any], repetitions: int) -> dict[str, Any]:
    first = [r for r in vision["seeds"][SEEDS[0]]["router"]["predictions"] if r["condition"] == "original"]
    ids = [str(r["source_image_id"]) for r in first]
    truth = np.asarray([int(r["class_id"]) for r in first]); labels = sorted(set(truth.tolist()))
    def arrays(rep: dict[str, Any], seed: int) -> np.ndarray:
        rows = {str(r["source_image_id"]): int(r["forced_prediction"]) for r in rep["seeds"][seed]["router"]["predictions"] if r["condition"] == "original"}
        return np.asarray([rows[i] for i in ids])
    vp={s:arrays(vision,s) for s in SEEDS}; qp={s:arrays(query,s) for s in SEEDS}
    def delta(indices: list[int], metric: str) -> float:
        idx=np.asarray(indices,dtype=np.int64)
        score=lambda p: float(accuracy_score(truth[idx],p[idx])) if metric=="accuracy" else float(f1_score(truth[idx],p[idx],labels=labels,average="macro",zero_division=0))
        return mean(score(qp[s])-score(vp[s]) for s in SEEDS)
    return {m:_bootstrap(lambda i,m=m:delta(i,m),len(ids),repetitions,20260727+(0 if m=="accuracy" else 1)) for m in ("accuracy","macro_f1")}


def paired_fpr(vision: dict[str, Any], query: dict[str, Any], key: str, repetitions: int) -> dict[str, Any]:
    def mapping(rep: dict[str, Any], seed: int) -> dict[str, bool]:
        if key in {"plantdoc", "plantseg"}:
            return {str(r["id"]): bool(r["accepted"]) for r in rep["seeds"][seed][key]}
        return {str(r["source_image_id"]): bool(r["accepted"]) for r in rep["seeds"][seed]["router"]["predictions"] if r["condition"]==key}
    ids=sorted(mapping(vision,SEEDS[0])); vm={s:mapping(vision,s) for s in SEEDS}; qm={s:mapping(query,s) for s in SEEDS}
    if any(sorted(vm[s])!=ids or sorted(qm[s])!=ids for s in SEEDS): raise ValueError("unpaired FPR rows")
    def delta(indices: list[int]) -> float:
        chosen=[ids[i] for i in indices]
        return mean(mean(float(qm[s][i])-float(vm[s][i]) for i in chosen) for s in SEEDS)
    return _bootstrap(delta,len(ids),repetitions,20260800+len(key))


def decide(summary: dict[str, Any], paired: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    v,q=summary["vision"],summary["query"]; delta=lambda k:q[k]-v[k]
    conditions={
        "accuracy_noninferior": delta("accuracy") >= gates["accuracy_delta_ge"],
        "macro_f1_noninferior": delta("forced_macro_f1") >= gates["forced_macro_f1_delta_ge"],
        "macro_f1_ci_noninferior": paired["positive"]["macro_f1"]["low"] >= gates["forced_macro_f1_bootstrap_low_ge"],
        "coverage_absolute": q["coverage"] >= gates["query_coverage_ge"],
        "coverage_noninferior": delta("coverage") >= gates["coverage_delta_ge"],
        "blank_fpr": q["blank_fpr"] < gates["blank_fpr_lt"],
        "blur_fpr": q["blur_fpr"] < gates["blur_fpr_lt"],
        "shuffle_fpr": q["shuffle_fpr"] < gates["shuffle_fpr_lt"],
        "synthetic_overall_nonworse": delta("synthetic_overall_fpr") <= gates["synthetic_overall_fpr_delta_le"],
        "plantdoc_absolute": q["plantdoc_fpr"] < gates["plantdoc_fpr_lt"],
        "plantdoc_nonworse": delta("plantdoc_fpr") <= gates["plantdoc_fpr_delta_le"],
        "plantseg_absolute": q["plantseg_fpr"] < gates["plantseg_fpr_lt"],
        "plantseg_improves_2pp": delta("plantseg_fpr") <= gates["plantseg_fpr_delta_le"],
        "plantseg_ci_superior": paired["plantseg"]["high"] < gates["plantseg_fpr_bootstrap_high_lt"],
        "json_contract": q["json_contract"] == 1.0,
    }
    passed=all(conditions.values())
    return {"conditions":conditions,"passed":passed,"decision":"PASS" if passed else "FAIL",
            "authorize_evidence_head_planning":passed,"next_if_fail":"Local-Crop/Mask single-variable micro experiment"}


def run(*, roots: dict[str, dict[str, Path]], config_path: Path, output_root: Path) -> dict[str, Any]:
    destination=Path(output_root); ensure_new_directory(destination)
    (destination/"status.json").write_text('{"state":"running","stage":"verify"}\n',encoding="utf-8")
    try:
        config=json.loads(Path(config_path).read_text(encoding="utf-8")); loaded={}
        for rep in ("vision","query"):
            loaded[rep]={name:_load(Path(path)) for name,path in roots[rep].items()}
        for name,digest in EXPECTED_VISION.items():
            if sha256_file(Path(roots["vision"][name])/"features.npy") != digest: raise ValueError(f"vision baseline hash drift: {name}")
        for name in ("base","stress","plantdoc","plantseg"):
            identity=lambda r:(str(r["id"]),str(r.get("split","")),str(r.get("condition","")),int(r.get("stress_seed",0)))
            if [identity(r) for r in loaded["vision"][name][1]] != [identity(r) for r in loaded["query"][name][1]]: raise ValueError(f"representation row mismatch: {name}")
        results={rep:evaluate_representation(loaded[rep]["base"],loaded[rep]["stress"],loaded[rep]["plantdoc"],loaded[rep]["plantseg"]) for rep in ("vision","query")}
        if abs(results["vision"]["aggregate"]["forced_macro_f1"]-.8094315406815407)>1e-12 or abs(results["vision"]["aggregate"]["plantseg_fpr"]-.08275862068965517)>1e-12: raise ValueError("vision baseline reproduction mismatch")
        reps=int(config["bootstrap_repetitions"])
        paired={"positive":paired_positive(results["vision"],results["query"],reps),
                **{key:paired_fpr(results["vision"],results["query"],key,reps) for key in (*CONDITIONS,"plantdoc","plantseg")}}
        compact={rep:results[rep]["aggregate"] for rep in results}; decision=decide(compact,paired,config["gates"])
        seed_metrics={rep:{str(seed):{
            "router_metrics":results[rep]["seeds"][seed]["router"]["metrics"],
            "router_decision":results[rep]["seeds"][seed]["router"]["decision"],
            "plantdoc_fpr":mean(float(r["accepted"]) for r in results[rep]["seeds"][seed]["plantdoc"]),
            "plantseg_fpr":mean(float(r["accepted"]) for r in results[rep]["seeds"][seed]["plantseg"]),
        } for seed in SEEDS} for rep in ("vision","query")}
        report={"version":"task11b1-repprobe-result-v1","summary":compact,"seed_metrics":seed_metrics,
                "paired_bootstrap":paired,"decision":decision,
                "task8_locked_set_read":False,"training_performed":False}
        signed=[]
        for rep in ("vision","query"):
            for seed in SEEDS:
                name=f"{rep}_seed_{seed}_predictions.jsonl"
                rows=results[rep]["seeds"][seed]["router"]["predictions"]+[{**r,"condition":"plantdoc"} for r in results[rep]["seeds"][seed]["plantdoc"]]+[{**r,"condition":"plantseg"} for r in results[rep]["seeds"][seed]["plantseg"]]
                with (destination/name).open("x",encoding="utf-8",newline="\n") as h:
                    for row in rows:h.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
                signed.append(name)
        write_json_new(destination/"metrics.json",report); write_json_new(destination/"decision_report.json",decision)
        write_json_new(destination/"run_summary.json",{"state":"completed","decision":decision["decision"],"bootstrap_repetitions":reps,"representations":["vision","query"]})
        signed += ["metrics.json","decision_report.json","run_summary.json"]
        with (destination/"completion.sha256").open("x",encoding="utf-8",newline="\n") as h:
            for name in signed:h.write(f"{sha256_file(destination/name)}  {name}\n")
        (destination/"status.json").write_text('{"state":"completed","stage":"done"}\n',encoding="utf-8")
        return report
    except Exception as exc:
        write_json_new(destination/"failure.json",{"state":"failed","error":str(exc),"traceback":traceback.format_exc()})
        (destination/"status.json").write_text('{"state":"failed","stage":"evaluation"}\n',encoding="utf-8"); raise


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True)
    for rep in ("vision","query"):
        for name in ("base","stress","plantdoc","plantseg"):p.add_argument(f"--{rep}-{name}",type=Path,required=True)
    a=p.parse_args(); roots={rep:{name:getattr(a,f"{rep}_{name}") for name in ("base","stress","plantdoc","plantseg")} for rep in ("vision","query")}
    print(json.dumps(run(roots=roots,config_path=a.config,output_root=a.output_root)["decision"],indent=2,sort_keys=True))


if __name__=="__main__":main()
