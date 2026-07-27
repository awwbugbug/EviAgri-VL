"""Build the fresh Task 12A local-evidence exploration dataset."""

from __future__ import annotations

import argparse, json, traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from build_task11c0_local_crop_smoke import expand_box, largest_component_box, primary_bbox
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_rows(base: list[dict[str,Any]], plantseg: list[dict[str,Any]], used_ids: set[str]) -> list[dict[str,Any]]:
    selected=[]
    plan=(("train","probe_train",4),("val","probe_val",1),("dev","probe_test",2))
    classes=sorted({int(r["class_id"]) for r in base})
    if len(classes)!=16: raise ValueError("expected 16 Task10B classes")
    for source_split,probe_split,count in plan:
        for class_id in classes:
            candidates=sorted((r for r in base if r["split"]==source_split and int(r["class_id"])==class_id and str(r["id"]) not in used_ids),key=lambda r:str(r["id"]))
            if len(candidates)<count: raise ValueError(f"insufficient {source_split} class {class_id}")
            selected.extend({**r,"probe_split":probe_split,"target_type":"positive"} for r in candidates[:count])
    remaining=sorted((r for r in plantseg if str(r["id"]) not in used_ids),key=lambda r:(float(r["mask_ratio"]),str(r["id"])))
    indices=np.linspace(0,len(remaining)-1,32,dtype=int).tolist()
    if len(set(indices))!=32: raise ValueError("insufficient fresh PlantSeg rows")
    selected.extend({**remaining[i],"probe_split":"null_test","target_type":"real_null","class_id":None,"class_band":None} for i in indices)
    if len(selected)!=144 or len({str(r["id"]) for r in selected})!=144: raise ValueError("Task12A selection cardinality failure")
    components={split:{str(r["near_duplicate_component_id"]) for r in selected if r["target_type"]=="positive" and r["probe_split"]==split} for split in ("probe_train","probe_val","probe_test")}
    if any(components[a]&components[b] for a,b in (("probe_train","probe_val"),("probe_train","probe_test"),("probe_val","probe_test"))): raise ValueError("near-duplicate component crossed probe split")
    return selected


def build(*, base_manifest: Path, plantseg_manifest: Path, prior_crop_manifest: Path, output_root: Path) -> dict[str,Any]:
    root=Path(output_root); ensure_new_directory(root); (root/"status.json").write_text('{"state":"running"}\n',encoding="utf-8")
    try:
        crops=root/"crops"; crops.mkdir()
        used={str(r["id"]) for r in read_jsonl(prior_crop_manifest)}
        selected=select_rows(read_jsonl(base_manifest),read_jsonl(plantseg_manifest),used); records=[]
        for row in selected:
            source=Path(str(row["image"])); image_sha=sha256_file(source)
            with Image.open(source) as loaded: image=loaded.convert("RGB")
            if row["target_type"]=="positive": evidence,annotation_size=primary_bbox(source); expansion=.20
            else:
                with Image.open(row["mask"]) as loaded: mask=np.asarray(loaded.convert("L"))>0
                if mask.shape!=(image.height,image.width): raise ValueError("mask/image size mismatch")
                evidence,_=largest_component_box(mask); annotation_size=image.size; expansion=.25
            if annotation_size!=image.size: raise ValueError("annotation/image size mismatch")
            crop_box=expand_box(evidence,image.width,image.height,expansion); crop=image.crop(crop_box); fraction=(crop.width*crop.height)/(image.width*image.height)
            mode="identity_full_frame" if fraction>=.95 else "effective_crop"
            if mode=="effective_crop":
                crop_path=crops/f"{row['id']}.jpg"; crop.save(crop_path,quality=95); model_image=crop_path; model_sha=sha256_file(crop_path)
            else: model_image=source; model_sha=image_sha
            records.append({"id":str(row["id"]),"image":str(model_image),"model_image_sha256":model_sha,"source_image_sha256":image_sha,
                "probe_split":str(row["probe_split"]),"target_type":str(row["target_type"]),"class_id":row.get("class_id"),"class_band":row.get("class_band"),
                "crop_mode":mode,"crop_area_fraction":fraction,"evidence_box":list(evidence),"crop_box":list(crop_box),"source_size":list(image.size)})
        with (root/"manifest.jsonl").open("x",encoding="utf-8",newline="\n") as handle:
            for row in records: handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        counts={split:sum(r["probe_split"]==split for r in records) for split in ("probe_train","probe_val","probe_test","null_test")}
        report={"version":"task12a-local-dataset-1","count":len(records),"split_counts":counts,
            "effective_crop_count":sum(r["crop_mode"]=="effective_crop" for r in records),"identity_fallback_count":sum(r["crop_mode"]=="identity_full_frame" for r in records),
            "prior_used_count":len(used),"base_manifest_sha256":sha256_file(base_manifest),"plantseg_manifest_sha256":sha256_file(plantseg_manifest),
            "prior_crop_manifest_sha256":sha256_file(prior_crop_manifest),"task8_locked_set_read":False}
        write_json_new(root/"dataset_report.json",report)
        signed=["manifest.jsonl","dataset_report.json"]+[str(p.relative_to(root)) for p in sorted(crops.iterdir())]
        with (root/"completion.sha256").open("x",encoding="utf-8",newline="\n") as handle:
            for name in signed: handle.write(f"{sha256_file(root/name)}  {name}\n")
        (root/"status.json").write_text('{"state":"completed"}\n',encoding="utf-8"); return report
    except Exception as exc:
        write_json_new(root/"failure.json",{"state":"failed","error":str(exc),"traceback":traceback.format_exc()}); (root/"status.json").write_text('{"state":"failed"}\n',encoding="utf-8"); raise


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--base-manifest",type=Path,required=True); p.add_argument("--plantseg-manifest",type=Path,required=True); p.add_argument("--prior-crop-manifest",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
    print(json.dumps(build(base_manifest=a.base_manifest,plantseg_manifest=a.plantseg_manifest,prior_crop_manifest=a.prior_crop_manifest,output_root=a.output_root),indent=2,sort_keys=True))


if __name__=="__main__": main()
