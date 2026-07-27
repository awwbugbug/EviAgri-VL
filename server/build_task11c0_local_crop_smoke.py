"""Build a fail-closed oracle local-evidence crop smoke bundle."""

from __future__ import annotations

import argparse, hashlib, html, json, traceback, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


def expand_box(box: tuple[int,int,int,int], width: int, height: int, fraction: float) -> tuple[int,int,int,int]:
    x1,y1,x2,y2=box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height): raise ValueError("invalid source box")
    dx=(x2-x1)*fraction; dy=(y2-y1)*fraction
    out=(max(0,int(np.floor(x1-dx))),max(0,int(np.floor(y1-dy))),min(width,int(np.ceil(x2+dx))),min(height,int(np.ceil(y2+dy))))
    if not (out[0] <= x1 and out[1] <= y1 and out[2] >= x2 and out[3] >= y2): raise ValueError("expanded box lost evidence")
    return out


def largest_component_box(mask: np.ndarray) -> tuple[tuple[int,int,int,int], int]:
    values=np.asarray(mask,dtype=bool)
    if values.ndim != 2 or not values.any(): raise ValueError("empty or invalid mask")
    labels,count=ndimage.label(values,structure=np.ones((3,3),dtype=np.uint8))
    sizes=np.bincount(labels.ravel()); sizes[0]=0; selected=int(sizes.argmax())
    ys,xs=np.where(labels==selected)
    return (int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)),int(sizes[selected])


def read_rows(path: Path) -> list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def primary_bbox(image_path: Path) -> tuple[tuple[int,int,int,int], tuple[int,int]]:
    xml=image_path.parent.parent/"Annotations"/(image_path.stem+".xml")
    root=ET.parse(xml).getroot(); size=root.find("size")
    width,height=int(size.find("width").text),int(size.find("height").text)
    boxes=[]
    for obj in root.findall("object"):
        b=obj.find("bndbox"); boxes.append(tuple(int(float(b.find(k).text)) for k in ("xmin","ymin","xmax","ymax")))
    if not boxes: raise ValueError(f"no bbox: {xml}")
    return max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1])),(width,height)


def fixed_samples(base: list[dict[str,Any]], plantseg: list[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    dev=[r for r in base if r["split"]=="dev"]; classes=sorted({int(r["class_id"]) for r in dev})
    positive=[min((r for r in dev if int(r["class_id"])==c),key=lambda r:str(r["id"])) for c in classes]
    ordered=sorted(plantseg,key=lambda r:(float(r["mask_ratio"]),str(r["id"])))
    indices=np.linspace(0,len(ordered)-1,16,dtype=int).tolist(); nulls=[ordered[i] for i in indices]
    if len(positive)!=16 or len({r["id"] for r in nulls})!=16: raise ValueError("smoke selection cardinality failure")
    return positive,nulls


def build(*, base_manifest: Path, plantseg_manifest: Path, output_root: Path) -> dict[str,Any]:
    root=Path(output_root); ensure_new_directory(root); (root/"status.json").write_text('{"state":"running"}\n',encoding="utf-8")
    try:
        crops=root/"crops"; overlays=root/"overlays"; crops.mkdir(); overlays.mkdir()
        positives,nulls=fixed_samples(read_rows(base_manifest),read_rows(plantseg_manifest)); records=[]
        for kind,rows in (("ip102_bbox",positives),("plantseg_mask",nulls)):
            for row in rows:
                source=Path(row["image"])
                with Image.open(source) as loaded:image=loaded.convert("RGB")
                if kind=="ip102_bbox": evidence,xml_size=primary_bbox(source)
                else:
                    with Image.open(row["mask"]) as loaded:mask=np.asarray(loaded.convert("L"))>0
                    if mask.shape != (image.height,image.width): raise ValueError("mask/image size mismatch")
                    evidence,_=largest_component_box(mask); xml_size=image.size
                if xml_size != image.size: raise ValueError("annotation/image size mismatch")
                fraction=.20 if kind=="ip102_bbox" else .25
                crop_box=expand_box(evidence,image.width,image.height,fraction)
                crop=image.crop(crop_box); opaque=hashlib.sha256(f"task11c0|{kind}|{row['id']}".encode()).hexdigest()[:16]
                crop_path=crops/f"{opaque}.jpg"; crop.save(crop_path,quality=95)
                overlay=image.copy(); draw=ImageDraw.Draw(overlay); draw.rectangle(evidence,outline="red",width=max(2,min(image.size)//150)); draw.rectangle(crop_box,outline="lime",width=max(2,min(image.size)//150))
                overlay_path=overlays/f"{opaque}.jpg"; overlay.save(overlay_path,quality=90)
                area=(crop.width*crop.height)/(image.width*image.height)
                records.append({"id":str(row["id"]),"kind":kind,"class_id":row.get("class_id"),"source_image_sha256":sha256_file(source),
                    "evidence_box":list(evidence),"crop_box":list(crop_box),"source_size":list(image.size),"crop_size":list(crop.size),
                    "crop_area_fraction":area,"crop":str(crop_path),"crop_sha256":sha256_file(crop_path),"overlay":str(overlay_path)})
        with (root/"manifest.jsonl").open("x",encoding="utf-8",newline="\n") as h:
            for r in records:h.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
        sections=[]
        for kind in ("ip102_bbox","plantseg_mask"):
            cards=[]
            for r in [x for x in records if x["kind"]==kind]:
                cards.append(f'<article><h3>{html.escape(r["id"])}</h3><img src="overlays/{Path(r["overlay"]).name}"><img src="crops/{Path(r["crop"]).name}"><p>crop fraction={r["crop_area_fraction"]:.3f}</p></article>')
            sections.append(f"<h2>{kind}</h2><main>{''.join(cards)}</main>")
        page='<!doctype html><meta charset="utf-8"><style>main{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}article{border:1px solid #aaa;padding:8px}img{width:48%;vertical-align:top;margin:1%}</style>'+''.join(sections)
        (root/"review.html").write_text(page,encoding="utf-8")
        fractions=[r["crop_area_fraction"] for r in records]
        gates={"counts_16_plus_16":len(records)==32,"all_evidence_contained":all(r["crop_box"][0]<=r["evidence_box"][0] and r["crop_box"][1]<=r["evidence_box"][1] and r["crop_box"][2]>=r["evidence_box"][2] and r["crop_box"][3]>=r["evidence_box"][3] for r in records),
            "all_crops_nonempty":all(min(r["crop_size"])>0 for r in records),"all_crops_strictly_local":all(r["crop_area_fraction"]<1 for r in records),"median_crop_fraction_lt_0_75":float(np.median(fractions))<.75}
        report={"version":"task11c0-local-crop-smoke-v1","decision":"PASS" if all(gates.values()) else "BLOCK","gates":gates,"count":len(records),
            "crop_fraction":{"min":min(fractions),"median":float(np.median(fractions)),"max":max(fractions)},"training_performed":False,"task8_locked_set_read":False}
        write_json_new(root/"smoke_report.json",report)
        signed=["manifest.jsonl","review.html","smoke_report.json"]+[str(p.relative_to(root)) for p in sorted(crops.iterdir())]+[str(p.relative_to(root)) for p in sorted(overlays.iterdir())]
        with (root/"completion.sha256").open("x",encoding="utf-8",newline="\n") as h:
            for name in signed:h.write(f"{sha256_file(root/name)}  {name}\n")
        (root/"status.json").write_text('{"state":"completed"}\n',encoding="utf-8"); return report
    except Exception as exc:
        write_json_new(root/"failure.json",{"state":"failed","error":str(exc),"traceback":traceback.format_exc()}); (root/"status.json").write_text('{"state":"failed"}\n',encoding="utf-8"); raise


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base-manifest",type=Path,required=True);p.add_argument("--plantseg-manifest",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(base_manifest=a.base_manifest,plantseg_manifest=a.plantseg_manifest,output_root=a.output_root),indent=2,sort_keys=True))
if __name__=="__main__":main()
