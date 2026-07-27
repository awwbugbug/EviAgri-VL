"""Run the frozen G/L/GG/GL conditional-complementarity tournament."""

from __future__ import annotations

import argparse, json, random, traceback
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from evaluate_task10b_probe import _classifier, _fit
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new

SEEDS=(17,29,43)
EXPECTED_BASE_FEATURE_SHA256="5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc"
EXPECTED_PLANTSEG_FEATURE_SHA256="e05f01467c70ec334656f1702e1e0ec8fd4c5d8a14a7dad1616b9d98fc62b618"


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_completion(root: Path) -> None:
    for line in (root/"completion.sha256").read_text(encoding="utf-8").splitlines():
        expected,relative=line.split(maxsplit=1); target=root/relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target)!=expected: raise ValueError(f"completion SHA256 mismatch: {target}")


def bootstrap(statistic: Callable[[np.ndarray,np.ndarray],float], n_pos: int, n_null: int, repetitions: int, seed: int) -> dict[str,Any]:
    if min(n_pos,n_null,repetitions)<=0: raise ValueError("invalid bootstrap size")
    p=np.arange(n_pos); n=np.arange(n_null); estimate=float(statistic(p,n)); rng=random.Random(seed); samples=[]
    for _ in range(repetitions):
        pi=np.asarray([rng.randrange(n_pos) for _ in range(n_pos)]); ni=np.asarray([rng.randrange(n_null) for _ in range(n_null)]); samples.append(float(statistic(pi,ni)))
    return {"estimate":estimate,"low":float(np.quantile(samples,.025)),"high":float(np.quantile(samples,.975)),"repetitions":repetitions,"unit":"fresh_image"}


def branch(mean_delta: dict[str,float]) -> dict[str,Any]:
    gain=mean_delta["accuracy"]>=1/32 and mean_delta["true_probability"]>0
    safe=mean_delta["null_confidence"]<=0 and mean_delta["confidence_auroc"]>=0
    priority="H1_PRIORITY" if gain and safe else "H3_PRIORITY" if gain else "H2_PRIORITY"
    return {"conditional_local_gain":gain,"reliability_safe":safe,"priority":priority,"authorize_training":False,"authorize_task8":False}


def run(*, base_root: Path, plantseg_root: Path, local_root: Path, dataset_root: Path, output_root: Path, repetitions: int=1000) -> dict[str,Any]:
    out=Path(output_root); ensure_new_directory(out); (out/"status.json").write_text('{"state":"running"}\n',encoding="utf-8")
    try:
        for root in map(Path,(base_root,plantseg_root,local_root,dataset_root)): verify_completion(root)
        if sha256_file(Path(base_root)/"features.npy")!=EXPECTED_BASE_FEATURE_SHA256 or sha256_file(Path(plantseg_root)/"features.npy")!=EXPECTED_PLANTSEG_FEATURE_SHA256: raise ValueError("unexpected frozen global feature SHA256")
        local_summary=json.loads((Path(local_root)/"run_summary.json").read_text(encoding="utf-8"))
        if local_summary.get("manifest_sha256")!=sha256_file(Path(dataset_root)/"manifest.jsonl"): raise ValueError("local feature/dataset manifest mismatch")
        base=np.load(Path(base_root)/"features.npy",allow_pickle=False); null_global=np.load(Path(plantseg_root)/"features.npy",allow_pickle=False); local=np.load(Path(local_root)/"features.npy",allow_pickle=False)
        base_rows=read_jsonl(Path(base_root)/"feature_rows.jsonl"); null_rows=read_jsonl(Path(plantseg_root)/"feature_rows.jsonl"); local_rows=read_jsonl(Path(local_root)/"feature_rows.jsonl")
        if local.shape!=(144,2048) or len(local_rows)!=144: raise ValueError("unexpected local feature contract")
        if {str(r["id"]) for r in base_rows}&{str(r["id"]) for r in null_rows}: raise ValueError("global feature ID collision")
        global_map={str(r["id"]):base[int(r["feature_index"])] for r in base_rows}; global_map.update({str(r["id"]):null_global[int(r["feature_index"])] for r in null_rows})
        if any(str(r["id"]) not in global_map for r in local_rows): raise ValueError("missing aligned global feature")
        global_values=np.stack([global_map[str(r["id"])] for r in local_rows]).astype(np.float32); local_values=local.astype(np.float32)
        if not np.isfinite(global_values).all() or not np.isfinite(local_values).all(): raise ValueError("non-finite feature")
        masks={split:np.asarray([r["probe_split"]==split for r in local_rows]) for split in ("probe_train","probe_val","probe_test","null_test")}
        labels=np.asarray([int(r["class_id"]) if r["class_id"] is not None else -1 for r in local_rows]); all_labels=sorted(set(labels[masks["probe_train"]].tolist()))
        representations={"G":global_values,"L":local_values,"GG":np.concatenate([global_values,global_values],axis=1)/np.sqrt(2),"GL":np.concatenate([global_values,local_values],axis=1)/np.sqrt(2)}
        per_seed={}; prediction_payload={}; paired={"correct":[],"true_probability":[],"null_confidence":[],"gg_pos_confidence":[],"gl_pos_confidence":[],"gg_null_confidence":[],"gl_null_confidence":[]}
        signed=[]
        for seed in SEEDS:
            seed_result={}; seed_predictions={}
            for condition,matrix in representations.items():
                classifier=_classifier(seed); _fit(classifier,matrix[masks["probe_train"]],labels[masks["probe_train"]]); probabilities=classifier.predict_proba(matrix); forced=classifier.classes_[probabilities.argmax(axis=1)].astype(np.int64); confidence=probabilities.max(axis=1); class_index={int(c):i for i,c in enumerate(classifier.classes_)}
                test=masks["probe_test"]; null=masks["null_test"]; truth=labels[test]; pred=forced[test]; true_prob=np.asarray([probabilities[i,class_index[int(labels[i])]] for i in np.where(test)[0]])
                auroc=float(roc_auc_score(np.r_[np.ones(test.sum()),np.zeros(null.sum())],np.r_[confidence[test],confidence[null]]))
                seed_result[condition]={"accuracy":float(accuracy_score(truth,pred)),"macro_f1":float(f1_score(truth,pred,labels=all_labels,average="macro",zero_division=0)),"mean_true_probability":float(true_prob.mean()),"mean_null_confidence":float(confidence[null].mean()),"confidence_auroc":auroc,
                    "validation_accuracy":float(accuracy_score(labels[masks["probe_val"]],forced[masks["probe_val"]]))}
                seed_predictions[condition]={"forced":forced,"confidence":confidence,"true_probability":true_prob}
            gg=seed_predictions["GG"]; gl=seed_predictions["GL"]; test=masks["probe_test"]; null=masks["null_test"]
            paired["correct"].append(((gl["forced"][test]==labels[test]).astype(float)-(gg["forced"][test]==labels[test]).astype(float)))
            paired["true_probability"].append(gl["true_probability"]-gg["true_probability"]); paired["null_confidence"].append(gl["confidence"][null]-gg["confidence"][null])
            for key,value in (("gg_pos_confidence",gg["confidence"][test]),("gl_pos_confidence",gl["confidence"][test]),("gg_null_confidence",gg["confidence"][null]),("gl_null_confidence",gl["confidence"][null])): paired[key].append(value)
            gcorrect=seed_predictions["G"]["forced"][test]==labels[test]; lcorrect=seed_predictions["L"]["forced"][test]==labels[test]
            seed_result["G_L_complementarity"]={"G_wrong_L_right":int((~gcorrect&lcorrect).sum()),"G_right_L_wrong":int((gcorrect&~lcorrect).sum()),"both_right":int((gcorrect&lcorrect).sum()),"both_wrong":int((~gcorrect&~lcorrect).sum())}
            per_seed[str(seed)]=seed_result
            name=f"seed_{seed}_metrics.json"; write_json_new(out/name,seed_result); signed.append(name)
        arrays={k:np.stack(v,axis=1) for k,v in paired.items()}
        mean_delta={"accuracy":mean(per_seed[str(s)]["GL"]["accuracy"]-per_seed[str(s)]["GG"]["accuracy"] for s in SEEDS),"macro_f1":mean(per_seed[str(s)]["GL"]["macro_f1"]-per_seed[str(s)]["GG"]["macro_f1"] for s in SEEDS),
            "true_probability":float(arrays["true_probability"].mean()),"null_confidence":float(arrays["null_confidence"].mean()),
            "confidence_auroc":mean(per_seed[str(s)]["GL"]["confidence_auroc"]-per_seed[str(s)]["GG"]["confidence_auroc"] for s in SEEDS)}
        boot={"accuracy_delta":bootstrap(lambda p,n:float(arrays["correct"][p].mean()),32,32,repetitions,20260727),"true_probability_delta":bootstrap(lambda p,n:float(arrays["true_probability"][p].mean()),32,32,repetitions,20260728),
            "null_confidence_delta":bootstrap(lambda p,n:float(arrays["null_confidence"][n].mean()),32,32,repetitions,20260729),
            "confidence_auroc_delta":bootstrap(lambda p,n:float(roc_auc_score(np.r_[np.ones(len(p)*3),np.zeros(len(n)*3)],np.r_[arrays["gl_pos_confidence"][p].ravel(),arrays["gl_null_confidence"][n].ravel()])-roc_auc_score(np.r_[np.ones(len(p)*3),np.zeros(len(n)*3)],np.r_[arrays["gg_pos_confidence"][p].ravel(),arrays["gg_null_confidence"][n].ravel()])),32,32,repetitions,20260730)}
        decision=branch(mean_delta)
        report={"version":"task12a-conditional-complementarity-1","sample_counts":{k:int(v.sum()) for k,v in masks.items()},"representations":{"G":2048,"L":2048,"GG":4096,"GL":4096},"per_seed":per_seed,"primary_GL_minus_GG":mean_delta,"bootstrap":boot,"decision":decision,"training":{"qwen_frozen":True,"linear_probe_only":True,"null_used_for_fit":False},"task8_locked_set_read":False}
        write_json_new(out/"metrics.json",report); write_json_new(out/"decision_report.json",decision); write_json_new(out/"run_summary.json",{"state":"completed","priority":decision["priority"],"local_features_sha256":sha256_file(Path(local_root)/"features.npy")})
        signed += ["metrics.json","decision_report.json","run_summary.json"]
        with (out/"completion.sha256").open("x",encoding="utf-8",newline="\n") as handle:
            for name in signed: handle.write(f"{sha256_file(out/name)}  {name}\n")
        (out/"status.json").write_text('{"state":"completed"}\n',encoding="utf-8"); return report
    except Exception as exc:
        write_json_new(out/"failure.json",{"state":"failed","error":str(exc),"traceback":traceback.format_exc()}); (out/"status.json").write_text('{"state":"failed"}\n',encoding="utf-8"); raise


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--base-root",type=Path,required=True); p.add_argument("--plantseg-root",type=Path,required=True); p.add_argument("--local-root",type=Path,required=True); p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--repetitions",type=int,default=1000); a=p.parse_args()
    report=run(base_root=a.base_root,plantseg_root=a.plantseg_root,local_root=a.local_root,dataset_root=a.dataset_root,output_root=a.output_root,repetitions=a.repetitions); print(json.dumps(report["decision"],indent=2,sort_keys=True))


if __name__=="__main__": main()
