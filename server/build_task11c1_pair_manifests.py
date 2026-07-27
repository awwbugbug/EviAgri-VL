"""Materialize aligned full/local manifests from the passed Task 11C.0 bundle."""

from __future__ import annotations

import argparse, json, traceback
from pathlib import Path
from typing import Any

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_completion(root: Path) -> None:
    for line in (root / "completion.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        target = root / relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"completion SHA256 mismatch: {target}")


def align_rows(crop_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], plantseg_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]],list[dict[str, Any]]]:
    sources = {str(row["id"]): row for row in base_rows + plantseg_rows}
    full, local = [], []
    for index, crop in enumerate(crop_rows):
        pair_id = str(crop["id"])
        if pair_id not in sources:
            raise ValueError(f"missing source row: {pair_id}")
        source = sources[pair_id]
        source_path = Path(str(source["image"]))
        local_path = Path(str(crop["model_image"]))
        if sha256_file(source_path) != str(crop["source_image_sha256"]):
            raise ValueError(f"source image drift: {pair_id}")
        mode = str(crop["crop_mode"])
        if mode == "identity_full_frame":
            if sha256_file(local_path) != sha256_file(source_path):
                raise ValueError(f"identity fallback changed bytes: {pair_id}")
        elif mode == "effective_crop":
            if sha256_file(local_path) != str(crop["crop_sha256"]):
                raise ValueError(f"crop image drift: {pair_id}")
        else:
            raise ValueError(f"unexpected crop mode: {mode}")
        common={"id":pair_id,"pair_index":index,"target_type":"positive" if crop["kind"]=="ip102_bbox" else "real_null",
            "class_id":crop.get("class_id"),"crop_mode":mode,"source_kind":str(crop["kind"]),"source_image_sha256":str(crop["source_image_sha256"])}
        full.append({**common,"input_condition":"full","image":str(source_path)})
        local.append({**common,"input_condition":"local","image":str(local_path)})
    return full,local


def build(*, crop_root: Path, base_manifest: Path, plantseg_manifest: Path, output_root: Path) -> dict[str,Any]:
    root=Path(output_root); ensure_new_directory(root); (root/"status.json").write_text('{"state":"running"}\n',encoding="utf-8")
    try:
        verify_completion(Path(crop_root))
        report=json.loads((Path(crop_root)/"smoke_report.json").read_text(encoding="utf-8"))
        if report.get("version") != "task11c0-local-crop-smoke-v2" or report.get("decision") != "PASS":
            raise ValueError("Task11C.0 protocol v2 did not pass")
        crop_rows=read_jsonl(Path(crop_root)/"manifest.jsonl")
        if len(crop_rows) != 32 or len({str(row["id"]) for row in crop_rows}) != 32:
            raise ValueError("Task11C.0 requires 32 unique rows")
        full,local=align_rows(crop_rows,read_jsonl(base_manifest),read_jsonl(plantseg_manifest))
        for name,rows in (("full_manifest.jsonl",full),("local_manifest.jsonl",local)):
            with (root/name).open("x",encoding="utf-8",newline="\n") as handle:
                for row in rows: handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        contract={"version":"task11c1-pair-manifest-1","count":32,"positive_count":16,"real_null_count":16,
            "effective_crop_count":sum(r["crop_mode"]=="effective_crop" for r in local),
            "identity_fallback_count":sum(r["crop_mode"]=="identity_full_frame" for r in local),
            "crop_manifest_sha256":sha256_file(Path(crop_root)/"manifest.jsonl"),"base_manifest_sha256":sha256_file(base_manifest),
            "plantseg_manifest_sha256":sha256_file(plantseg_manifest),"task8_locked_set_read":False}
        write_json_new(root/"contract.json",contract)
        signed=["full_manifest.jsonl","local_manifest.jsonl","contract.json"]
        with (root/"completion.sha256").open("x",encoding="utf-8",newline="\n") as handle:
            for name in signed: handle.write(f"{sha256_file(root/name)}  {name}\n")
        (root/"status.json").write_text('{"state":"completed"}\n',encoding="utf-8"); return contract
    except Exception as exc:
        write_json_new(root/"failure.json",{"state":"failed","error":str(exc),"traceback":traceback.format_exc()}); (root/"status.json").write_text('{"state":"failed"}\n',encoding="utf-8"); raise


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--crop-root",type=Path,required=True); p.add_argument("--base-manifest",type=Path,required=True); p.add_argument("--plantseg-manifest",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
    print(json.dumps(build(crop_root=a.crop_root,base_manifest=a.base_manifest,plantseg_manifest=a.plantseg_manifest,output_root=a.output_root),indent=2,sort_keys=True))


if __name__=="__main__": main()
