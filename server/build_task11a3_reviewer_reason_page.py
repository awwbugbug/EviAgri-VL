"""Build a local-only reason-capture page for a confirmed independent reject list."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


CRITERIA = (
    ("real_photo", "非真实照片", "不是自然拍摄的真实植株照片"),
    ("lesion_visible", "病斑不可见", "看不到明确病斑或病害症状"),
    ("no_visible_pest", "存在虫体", "画面中存在可辨认的害虫虫体"),
    ("no_dominant_text", "文字或水印", "存在明显文字、Logo 或水印"),
    ("no_collage", "拼图", "多图拼接、分栏或信息图"),
    ("mask_valid", "红色标记不准", "红色区域未合理覆盖可见病斑"),
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_reason_page(*, pending_path: Path, manifest_path: Path, output_html: Path) -> dict:
    pending = json.loads(Path(pending_path).read_text(encoding="utf-8"))
    if pending.get("status") != "IDENTITY_CONFIRMED_PENDING_CRITERIA":
        raise ValueError("reviewer independence must be confirmed before reason capture")
    reject_ids = [str(value) for value in pending.get("reject_ids", [])]
    if len(reject_ids) != len(set(reject_ids)):
        raise ValueError("reject IDs must be unique")
    manifest = _read_jsonl(Path(manifest_path))
    all_ids = [str(row["audit_id"]) for row in manifest]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("manifest audit IDs must be unique")
    unknown = set(reject_ids) - set(all_ids)
    if unknown:
        raise ValueError(f"unknown reject IDs: {sorted(unknown)}")
    output_html = Path(output_html)
    if output_html.exists():
        raise FileExistsError(output_html)
    cards = []
    for index, audit_id in enumerate(reject_ids, 1):
        controls = "".join(
            f'<label class="reason"><input type="checkbox" data-id="{audit_id}" '
            f'data-criterion="{criterion}"><span><b>{html.escape(label)}</b>'
            f'<small>{html.escape(description)}</small></span></label>'
            for criterion, label, description in CRITERIA
        )
        cards.append(f"""
<article class="card" id="card-{audit_id}">
  <header><span class="index">{index:02d}</span><code>{audit_id}</code><span class="state">待补原因</span></header>
  <div class="visuals">
    <figure><img src="images/{audit_id}.jpg" alt="{audit_id} 原图"><figcaption>原图</figcaption></figure>
    <figure><img src="overlays/{audit_id}.jpg" alt="{audit_id} 红色标记"><figcaption>红色标记</figcaption></figure>
  </div>
  <div class="reasons">{controls}</div>
</article>""")
    criteria_json = json.dumps([item[0] for item in CRITERIA], ensure_ascii=False)
    all_ids_json = json.dumps(all_ids, ensure_ascii=False)
    reject_ids_json = json.dumps(reject_ids, ensure_ascii=False)
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reviewer B · 失败原因补录</title>
<style>
:root{{--paper:#f3efe5;--ink:#171713;--muted:#6d685e;--line:#c9c0ae;--red:#b3261e;--green:#246b4b;--panel:#fffdf7}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Noto Serif SC","Songti SC",Georgia,serif}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;background-image:repeating-linear-gradient(0deg,transparent,transparent 23px,rgba(80,70,50,.08) 24px)}}
.mast{{position:sticky;top:0;z-index:20;background:rgba(23,23,19,.96);color:#fff;padding:14px max(20px,calc((100vw - 1420px)/2));display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center;box-shadow:0 8px 30px #17171333}}
.mast h1{{font-size:20px;margin:0;letter-spacing:.08em}} .mast p{{margin:4px 0 0;color:#d6cfbf;font:13px/1.4 sans-serif}}
.progress{{font:700 15px ui-monospace,monospace;text-align:right}} progress{{width:220px;height:8px;accent-color:#d7493f}}
main{{max-width:1420px;margin:36px auto;padding:0 20px 120px}} .notice{{border:1px solid var(--line);background:#fff8dc;padding:18px 20px;margin-bottom:30px;line-height:1.7}}
.card{{background:var(--panel);border:1px solid var(--line);box-shadow:8px 8px 0 #d8cfbd;margin:0 0 34px}}
.card header{{display:grid;grid-template-columns:52px 1fr auto;align-items:center;border-bottom:1px solid var(--line);min-height:52px}}
.index{{height:100%;display:grid;place-items:center;background:var(--ink);color:#fff;font:700 14px monospace}} code{{padding:0 16px;font-size:16px}}
.state{{margin-right:16px;padding:6px 10px;border:1px solid var(--red);color:var(--red);font:700 12px sans-serif}}
.card.done .state{{border-color:var(--green);color:var(--green)}}
.visuals{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}} figure{{margin:0;background:#24231f;position:relative;min-height:280px;display:grid;place-items:center}}
figure img{{width:100%;height:min(62vh,620px);object-fit:contain}} figcaption{{position:absolute;left:12px;bottom:12px;background:#171713dd;color:#fff;padding:6px 10px;font:12px sans-serif;letter-spacing:.15em}}
.reasons{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:16px}} .reason{{border:1px solid var(--line);padding:12px;display:flex;gap:10px;align-items:flex-start;cursor:pointer;transition:.15s;background:#fff}}
.reason:hover{{transform:translateY(-2px);box-shadow:0 4px 0 #ded5c3}} .reason:has(input:checked){{border-color:var(--red);background:#fff0eb}}
.reason input{{margin-top:4px;accent-color:var(--red)}} .reason b{{display:block;font-size:14px}} .reason small{{display:block;color:var(--muted);font:12px/1.45 sans-serif;margin-top:3px}}
.actions{{position:fixed;right:24px;bottom:22px;z-index:30;display:flex;gap:10px}} button{{border:0;padding:14px 18px;background:var(--ink);color:#fff;font-weight:700;cursor:pointer;box-shadow:5px 5px 0 #b9ae99}} button:hover{{background:var(--red)}}
@media(max-width:850px){{.visuals,.reasons{{grid-template-columns:1fr}}.mast{{grid-template-columns:1fr}}progress{{width:100%}}figure img{{height:48vh}}}}
</style></head><body>
<div class="mast"><div><h1>Reviewer B · 失败原因补录</h1><p>只复核已判定“不合格”的 45 张；每张至少选择一个原因。未列出的 293 张自动保持六项 PASS。</p></div><div class="progress"><span id="count">0 / __REJECT_COUNT__</span><br><progress id="bar" max="__REJECT_COUNT__" value="0"></progress></div></div>
<main><div class="notice"><b>保持独立：</b>只依据当前原图与红色 overlay 勾选。一个样本可以选择多个失败原因。页面会在本机自动保存进度，不上传任何内容。</div>__CARDS__</main>
<div class="actions"><button id="next">跳到下一张未完成</button><button id="export">全部完成后导出 CSV</button></div>
<script>
const criteria=__CRITERIA_JSON__, allIds=__ALL_IDS_JSON__, rejectIds=__REJECT_IDS_JSON__;
const key='task11a3-reviewer-b-reasons-v1';
let saved=JSON.parse(localStorage.getItem(key)||'{{}}');
document.querySelectorAll('input[type=checkbox]').forEach(x=>{{x.checked=(saved[x.dataset.id]||[]).includes(x.dataset.criterion);x.addEventListener('change',save)}});
function selected(id){{return [...document.querySelectorAll(`input[data-id="${{id}}"]:checked`)].map(x=>x.dataset.criterion)}}
function save(){{saved={{}};rejectIds.forEach(id=>{{const s=selected(id);if(s.length)saved[id]=s}});localStorage.setItem(key,JSON.stringify(saved));render()}}
function render(){{let n=0;rejectIds.forEach(id=>{{const ok=selected(id).length>0;document.getElementById('card-'+id).classList.toggle('done',ok);document.querySelector('#card-'+id+' .state').textContent=ok?'已记录':'待补原因';if(ok)n++}});count.textContent=`${{n}} / __REJECT_COUNT__`;bar.value=n}}
function esc(v){{v=String(v);return /[",\\n]/.test(v)?'"'+v.replaceAll('"','""')+'"':v}}
next.onclick=()=>{{const id=rejectIds.find(id=>selected(id).length===0);(id?document.getElementById('card-'+id):document.querySelector('.actions')).scrollIntoView({{behavior:'smooth',block:'start'}})}};
export.onclick=()=>{{const missing=rejectIds.filter(id=>selected(id).length===0);if(missing.length){{alert(`还有 ${{missing.length}} 张没有选择失败原因。`);document.getElementById('card-'+missing[0]).scrollIntoView({{behavior:'smooth'});return}}
 const rejected=new Set(rejectIds);let rows=[['audit_id','reviewer_id',...criteria,'notes']];allIds.forEach(id=>{{const fails=new Set(saved[id]||[]);const vals=criteria.map(c=>fails.has(c)?'FAIL':'PASS');const notes=[...fails].map(c=>'FAIL:'+c).join(';');rows.push([id,'reviewer_B',...vals,notes])}});
 const csv='\\ufeff'+rows.map(r=>r.map(esc).join(',')).join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='reviewer_b_completed.csv';a.click();URL.revokeObjectURL(a.href)}};
render();
</script></body></html>"""
    page = (
        page.replace("__REJECT_COUNT__", str(len(reject_ids)))
        .replace("__CARDS__", "".join(cards))
        .replace("__CRITERIA_JSON__", criteria_json)
        .replace("__ALL_IDS_JSON__", all_ids_json)
        .replace("__REJECT_IDS_JSON__", reject_ids_json)
        .replace("{{", "{")
        .replace("}}", "}")
    )
    output_html.write_text(page, encoding="utf-8", newline="\n")
    return {"audit_rows": len(all_ids), "reject_rows": len(reject_ids), "output": str(output_html)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_reason_page(
        pending_path=args.pending,
        manifest_path=args.manifest,
        output_html=args.output_html,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
