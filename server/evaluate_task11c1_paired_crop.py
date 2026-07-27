"""Evaluate a frozen paired full-image versus evidence-crop micro probe."""

from __future__ import annotations

import argparse, json, random, traceback
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

from evaluate_task10b_probe import _classifier, _fit, _split_arrays
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new
from task11a_confidence_router import SEEDS, temperature_scale

TEMPERATURE=0.18887372662036642
THRESHOLD=0.63


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_completion(root: Path) -> None:
    for line in (root/"completion.sha256").read_text(encoding="utf-8").splitlines():
        expected,relative=line.split(maxsplit=1); target=root/relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target)!=expected: raise ValueError(f"completion SHA256 mismatch: {target}")


def paired_bootstrap(values: np.ndarray, statistic: Callable[[np.ndarray],float], repetitions: int, seed: int) -> dict[str,Any]:
    if values.ndim<1 or len(values)==0 or repetitions<=0: raise ValueError("invalid bootstrap input")
    estimate=float(statistic(np.arange(len(values))))
    rng=random.Random(seed); samples=[]
    for _ in range(repetitions):
        idx=np.asarray([rng.randrange(len(values)) for _ in range(len(values))],dtype=np.int64); samples.append(float(statistic(idx)))
    return {"estimate":estimate,"low":float(np.quantile(samples,.025)),"high":float(np.quantile(samples,.975)),"repetitions":repetitions,"unit":"paired_image"}


def decide(summary: dict[str,float], invariants: dict[str,bool], bootstrap: dict[str,dict[str,Any]]) -> dict[str,Any]:
    gates={**invariants,
        "positive_accuracy_delta_ge_minus_one_of_16":summary["positive_accuracy_delta"]>=-0.0625,
        "positive_supported_delta_ge_minus_one_of_16":summary["positive_supported_delta"]>=-0.0625,
        "positive_true_probability_delta_gt_zero":summary["positive_true_probability_delta"]>0,
        "local_null_fpr_lt_0_10":summary["local_null_fpr"]<.10,
        "null_fpr_not_higher":summary["null_fpr_delta"]<=0,
        "null_mean_confidence_delta_lt_zero":summary["null_mean_confidence_delta"]<0}
    passed=all(gates.values())
    strong=passed and bootstrap["positive_true_probability_delta"]["low"]>0 and bootstrap["null_mean_confidence_delta"]["high"]<0
    return {"decision":"PASS_MICRO" if passed else "BLOCK","passed":passed,"strong_signal":strong,
        "authorize_larger_paired_validation":passed,"authorize_training":False,"gates":gates}


def run(*, base_root: Path, full_root: Path, local_root: Path, pair_root: Path, output_root: Path, repetitions: int=1000) -> dict[str,Any]:
    destination=Path(output_root); ensure_new_directory(destination); (destination/"status.json").write_text('{"state":"running"}\n',encoding="utf-8")
    try:
        for root in map(Path,(base_root,full_root,local_root,pair_root)): verify_completion(root)
        base=np.load(Path(base_root)/"features.npy",allow_pickle=False); full=np.load(Path(full_root)/"features.npy",allow_pickle=False); local=np.load(Path(local_root)/"features.npy",allow_pickle=False)
        base_rows=read_jsonl(Path(base_root)/"feature_rows.jsonl"); full_rows=read_jsonl(Path(full_root)/"feature_rows.jsonl"); local_rows=read_jsonl(Path(local_root)/"feature_rows.jsonl")
        if full.shape!=local.shape or full.shape!=(32,2048) or base.shape!=(320,2048): raise ValueError("unexpected feature shape")
        if [r["id"] for r in full_rows] != [r["id"] for r in local_rows] or [r["crop_mode"] for r in full_rows] != [r["crop_mode"] for r in local_rows]: raise ValueError("pair alignment mismatch")
        full_norm=full/np.linalg.norm(full,axis=1,keepdims=True); local_norm=local/np.linalg.norm(local,axis=1,keepdims=True); cos=np.sum(full_norm*local_norm,axis=1)
        identity=np.asarray([r["crop_mode"]=="identity_full_frame" for r in full_rows]); effective=~identity
        x_train,y_train,_=_split_arrays(base,base_rows,"train")
        positives=np.asarray([r["target_type"]=="positive" for r in full_rows]); nulls=~positives
        per_seed={}; all_deltas={"positive_accuracy":[],"positive_supported":[],"positive_true_probability":[],"null_fpr":[],"null_mean_confidence":[]}; identity_agreement=[]
        signed=[]
        for seed in SEEDS:
            classifier=_classifier(seed); _fit(classifier,x_train,y_train)
            condition={}
            prediction_rows=[]
            for name,matrix in (("full",full),("local",local)):
                probabilities=temperature_scale(classifier.predict_proba(matrix),TEMPERATURE); forced=classifier.classes_[probabilities.argmax(axis=1)].astype(np.int64); confidence=probabilities.max(axis=1); accepted=confidence>=THRESHOLD
                truth=np.asarray([int(r["class_id"]) if r["class_id"] is not None else -1 for r in full_rows]); class_index={int(c):i for i,c in enumerate(classifier.classes_)}
                true_probability=np.asarray([probabilities[i,class_index[int(truth[i])]] if positives[i] else np.nan for i in range(len(truth))])
                metrics={"positive_accuracy":float((forced[positives]==truth[positives]).mean()),"positive_supported":float(((forced[positives]==truth[positives])&accepted[positives]).mean()),
                    "positive_true_probability":float(np.nanmean(true_probability[positives])),"positive_coverage":float(accepted[positives].mean()),
                    "null_fpr":float(accepted[nulls].mean()),"null_mean_confidence":float(confidence[nulls].mean())}
                condition[name]={"metrics":metrics,"forced":forced,"confidence":confidence,"accepted":accepted,"true_probability":true_probability}
                for i,row in enumerate(full_rows): prediction_rows.append({"id":row["id"],"condition":name,"target_type":row["target_type"],"crop_mode":row["crop_mode"],"class_id":row["class_id"],"forced_prediction":int(forced[i]),"confidence":float(confidence[i]),"accepted":bool(accepted[i]),"correct":bool(positives[i] and forced[i]==truth[i]),"true_class_probability":None if not positives[i] else float(true_probability[i])})
            deltas={key:condition["local"]["metrics"][key]-condition["full"]["metrics"][key] for key in all_deltas}
            for key,value in deltas.items(): all_deltas[key].append(value)
            identity_agreement.append(bool(np.array_equal(condition["full"]["forced"][identity],condition["local"]["forced"][identity])))
            name=f"seed_{seed}_predictions.jsonl"; signed.append(name)
            with (destination/name).open("x",encoding="utf-8",newline="\n") as handle:
                for row in prediction_rows: handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
            per_seed[str(seed)]={"full":condition["full"]["metrics"],"local":condition["local"]["metrics"],"delta":deltas}
        summary={"positive_accuracy_delta":mean(all_deltas["positive_accuracy"]),"positive_supported_delta":mean(all_deltas["positive_supported"]),
            "positive_true_probability_delta":mean(all_deltas["positive_true_probability"]),"local_null_fpr":mean(per_seed[str(s)]["local"]["null_fpr"] for s in SEEDS),
            "null_fpr_delta":mean(all_deltas["null_fpr"]),"null_mean_confidence_delta":mean(all_deltas["null_mean_confidence"])}
        def pair_values(metric: str, mask: np.ndarray) -> np.ndarray:
            values=[]
            for seed in SEEDS:
                classifier=_classifier(seed); _fit(classifier,x_train,y_train)
                fp=temperature_scale(classifier.predict_proba(full),TEMPERATURE); lp=temperature_scale(classifier.predict_proba(local),TEMPERATURE)
                if metric=="true_probability":
                    indices=np.where(mask)[0]; class_index={int(c):i for i,c in enumerate(classifier.classes_)}; values.append(np.asarray([lp[i,class_index[int(full_rows[i]["class_id"])]]-fp[i,class_index[int(full_rows[i]["class_id"])]] for i in indices]))
                else: values.append(lp[mask].max(axis=1)-fp[mask].max(axis=1))
            return np.stack(values,axis=0).T
        pos_delta=pair_values("true_probability",positives); null_delta=pair_values("confidence",nulls)
        bootstrap={"positive_true_probability_delta":paired_bootstrap(pos_delta,lambda idx:float(pos_delta[idx].mean()),repetitions,20260727),
            "null_mean_confidence_delta":paired_bootstrap(null_delta,lambda idx:float(null_delta[idx].mean()),repetitions,20260728)}
        invariants={"identity_feature_cosine_ge_0_99999":bool((cos[identity]>=.99999).all()),"identity_predictions_agree":all(identity_agreement),
            "effective_median_feature_cosine_lt_0_999":float(np.median(cos[effective]))<.999,"all_features_finite":bool(np.isfinite(full).all() and np.isfinite(local).all())}
        decision=decide(summary,invariants,bootstrap)
        report={"version":"task11c1-paired-crop-probe-1","sample_counts":{"positive":int(positives.sum()),"real_null":int(nulls.sum()),"effective_crop":int(effective.sum()),"identity_full_frame":int(identity.sum())},
            "protocol":{"seeds":list(SEEDS),"temperature":TEMPERATURE,"threshold":THRESHOLD,"bootstrap_repetitions":repetitions,"training_performed":False,"task8_locked_set_read":False},
            "feature_cosine":{"identity_min":float(cos[identity].min()),"effective_median":float(np.median(cos[effective]))},"per_seed":per_seed,"mean_deltas":summary,"bootstrap":bootstrap,"decision":decision}
        write_json_new(destination/"metrics.json",report); write_json_new(destination/"decision_report.json",decision); write_json_new(destination/"run_summary.json",{"state":"completed","decision":decision["decision"],"full_features_sha256":sha256_file(Path(full_root)/"features.npy"),"local_features_sha256":sha256_file(Path(local_root)/"features.npy")})
        signed += ["metrics.json","decision_report.json","run_summary.json"]
        with (destination/"completion.sha256").open("x",encoding="utf-8",newline="\n") as handle:
            for name in signed: handle.write(f"{sha256_file(destination/name)}  {name}\n")
        (destination/"status.json").write_text('{"state":"completed"}\n',encoding="utf-8"); return report
    except Exception as exc:
        write_json_new(destination/"failure.json",{"state":"failed","error":str(exc),"traceback":traceback.format_exc()}); (destination/"status.json").write_text('{"state":"failed"}\n',encoding="utf-8"); raise


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--base-root",type=Path,required=True); p.add_argument("--full-root",type=Path,required=True); p.add_argument("--local-root",type=Path,required=True); p.add_argument("--pair-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--repetitions",type=int,default=1000); a=p.parse_args()
    report=run(base_root=a.base_root,full_root=a.full_root,local_root=a.local_root,pair_root=a.pair_root,output_root=a.output_root,repetitions=a.repetitions); print(json.dumps(report["decision"],indent=2,sort_keys=True))


if __name__=="__main__": main()
