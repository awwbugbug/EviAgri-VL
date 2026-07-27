"""Build a decision-first adjudication page for Task 11A.3 review disagreements."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


CRITERIA_LABELS = {
    "real_photo": "非真实照片",
    "lesion_visible": "病斑不可见",
    "no_visible_pest": "存在虫体",
    "no_dominant_text": "文字或水印",
    "no_collage": "拼图",
    "mask_valid": "红色标记不准",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_adjudication_page(*, disagreement_path: Path, output_html: Path) -> dict:
    rows = _read_jsonl(Path(disagreement_path))
    ids = [str(row["audit_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("disagreements must be non-empty with unique audit IDs")
    output_html = Path(output_html)
    if output_html.exists():
        raise FileExistsError(output_html)
    cards = []
    for index, row in enumerate(rows, 1):
        audit_id = row["audit_id"]
        opinion_a = ", ".join(CRITERIA_LABELS[x] for x in row["reviewer_a_failures"]) or "保留"
        opinion_b = ", ".join(CRITERIA_LABELS[x] for x in row["reviewer_b_failures"]) or "保留"
        reasons = "".join(
            f'<label><input type="checkbox" data-id="{audit_id}" data-criterion="{criterion}">'
            f'<span>{html.escape(label)}</span></label>'
            for criterion, label in CRITERIA_LABELS.items()
        )
        cards.append(f"""
<article class="case" id="case-{audit_id}">
  <header><span>{index:02d}</span><code>{audit_id}</code><b class="status">待裁决</b></header>
  <div class="visuals"><figure><img src="../11A3_reviewer_b_blind_bundle/images/{audit_id}.jpg"><figcaption>原图</figcaption></figure><figure><img src="../11A3_reviewer_b_blind_bundle/overlays/{audit_id}.jpg"><figcaption>红色标记</figcaption></figure></div>
  <div class="decision"><button data-id="{audit_id}" data-decision="KEEP">保留</button><button data-id="{audit_id}" data-decision="EXCLUDE">排除</button><button data-id="{audit_id}" data-decision="UNCERTAIN">仍存疑</button></div>
  <div class="criteria" data-for="{audit_id}"><p>若排除，至少选择一个原因：</p>{reasons}</div>
  <details><summary>作出判断后再查看两位审查者的原始意见</summary><p>意见 1：{html.escape(opinion_a)}</p><p>意见 2：{html.escape(opinion_b)}</p></details>
</article>""")
    ids_json = json.dumps(ids, ensure_ascii=False)
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task11A.3 分歧裁决</title>
<style>
:root{{--ink:#14201b;--paper:#eef0e8;--cream:#fffdf5;--line:#9aa397;--keep:#176b46;--drop:#a82b23;--maybe:#9a6b16}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);font-family:"Noto Serif SC","Songti SC",serif}}.top{{position:sticky;top:0;z-index:10;background:#14201bf2;color:white;padding:16px max(20px,calc((100vw - 1380px)/2));display:grid;grid-template-columns:1fr auto;gap:18px;align-items:center}}h1{{font-size:20px;margin:0}}.top p{{font:13px sans-serif;margin:5px 0 0;color:#cdd6ce}}.meta{{display:flex;gap:12px;align-items:center;font:13px sans-serif}}.meta input{{padding:9px 11px;border:1px solid #86958b;background:#fff;color:#111}}progress{{accent-color:#e1b953;width:180px}}main{{max-width:1380px;margin:32px auto;padding:0 20px 120px}}.intro{{background:#fff7d8;border-left:7px solid #d0a733;padding:18px 20px;line-height:1.65;margin-bottom:28px}}.case{{background:var(--cream);border:1px solid var(--line);margin-bottom:34px;box-shadow:7px 7px 0 #c9cec4}}.case header{{display:grid;grid-template-columns:54px 1fr auto;align-items:center;border-bottom:1px solid var(--line)}}.case header>span{{background:var(--ink);color:#fff;height:52px;display:grid;place-items:center;font:bold 14px monospace}}code{{padding:0 16px;font-size:16px}}.status{{margin-right:15px;color:var(--drop);font:12px sans-serif}}.case.complete .status{{color:var(--keep)}}.visuals{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}}figure{{margin:0;background:#202520;position:relative;display:grid;place-items:center}}figure img{{width:100%;height:min(62vh,600px);object-fit:contain}}figcaption{{position:absolute;left:10px;bottom:10px;background:#14201bdc;color:#fff;padding:6px 10px;font:12px sans-serif}}.decision{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:16px}}button{{padding:13px;border:1px solid var(--line);background:#fff;cursor:pointer;font-weight:700}}button.active[data-decision=KEEP]{{background:var(--keep);color:#fff}}button.active[data-decision=EXCLUDE]{{background:var(--drop);color:#fff}}button.active[data-decision=UNCERTAIN]{{background:var(--maybe);color:#fff}}.criteria{{padding:0 16px 16px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.criteria p{{grid-column:1/-1;margin:0 0 3px;font:13px sans-serif;color:#596159}}.criteria label{{border:1px solid var(--line);padding:10px;background:#fff;font:13px sans-serif}}.criteria label:has(input:checked){{border-color:var(--drop);background:#fff0ec}}details{{border-top:1px dashed var(--line);padding:14px 16px;color:#596159;font:13px/1.5 sans-serif}}.footer{{position:fixed;right:22px;bottom:20px;z-index:20;display:flex;gap:10px}}.footer button{{background:var(--ink);color:#fff;box-shadow:5px 5px 0 #9da79e}}@media(max-width:800px){{.top,.visuals,.decision,.criteria{{grid-template-columns:1fr}}.criteria p{{grid-column:auto}}.meta{{flex-wrap:wrap}}}}
</style></head><body><div class="top"><div><h1>Task 11A.3 · 双审分歧裁决</h1><p>只处理 17 张分歧；先独立看图作答，必要时再展开双方意见。</p></div><div class="meta"><label>裁决者 <input id="adjudicator" placeholder="例如 adjudicator_C"></label><span id="count">0 / __COUNT__</span><progress id="bar" max="__COUNT__" value="0"></progress></div></div>
<main><div class="intro"><b>裁决规则：</b>保留＝进入 strict real-null 候选；排除＝至少违反一项冻结标准；仍存疑＝不进入 strict 池但保留在不确定集合。不要因为某位审查者更严格而机械取并集或交集。</div>__CARDS__</main>
<div class="footer"><button id="next">下一张未完成</button><button id="download">导出裁决 CSV</button></div>
<script>
const ids=__IDS__, criteria=__CRITERIA__;
const key='task11a3-adjudication-v1';
const countEl=document.getElementById('count'),barEl=document.getElementById('bar'),nameEl=document.getElementById('adjudicator');
function load(){{try{{return JSON.parse(localStorage.getItem(key)||'{{}}')}}catch(e){{return {{}}}}}}let state=load();
function persist(){{try{{localStorage.setItem(key,JSON.stringify(state))}}catch(e){{}}}}
nameEl.value=state.adjudicator||'';nameEl.oninput=()=>{{state.adjudicator=nameEl.value;persist()}};
function selectedCriteria(id){{return [...document.querySelectorAll(`input[data-id="${{id}}"]:checked`)].map(x=>x.dataset.criterion)}}
document.querySelectorAll('button[data-decision]').forEach(b=>b.onclick=()=>{{state[b.dataset.id]=state[b.dataset.id]||{{}};state[b.dataset.id].decision=b.dataset.decision;persist();render()}});
document.querySelectorAll('input[data-criterion]').forEach(x=>x.onchange=()=>{{state[x.dataset.id]=state[x.dataset.id]||{{}};state[x.dataset.id].criteria=selectedCriteria(x.dataset.id);persist();render()}});
function complete(id){{const s=state[id]||{{}};return !!s.decision&&(s.decision!=='EXCLUDE'||selectedCriteria(id).length>0)}}
function render(){{let n=0;ids.forEach(id=>{{const s=state[id]||{{}};document.querySelectorAll(`button[data-id="${{id}}"]`).forEach(b=>b.classList.toggle('active',b.dataset.decision===s.decision));document.querySelectorAll(`input[data-id="${{id}}"]`).forEach(x=>{{x.checked=(s.criteria||[]).includes(x.dataset.criterion);x.disabled=s.decision!=='EXCLUDE'}});const ok=complete(id);document.getElementById('case-'+id).classList.toggle('complete',ok);document.querySelector('#case-'+id+' .status').textContent=ok?'已完成':'待裁决';if(ok)n++}});countEl.textContent=`${{n}} / __COUNT__`;barEl.value=n}}
document.getElementById('next').onclick=()=>{{const id=ids.find(x=>!complete(x));(id?document.getElementById('case-'+id):document.querySelector('.footer')).scrollIntoView({{behavior:'smooth'}})}};
function esc(v){{v=String(v);return /[",\\n]/.test(v)?'"'+v.replaceAll('"','""')+'"':v}}
document.getElementById('download').onclick=()=>{{if(!nameEl.value.trim()){{alert('请先填写裁决者 ID。');return}}const missing=ids.filter(x=>!complete(x));if(missing.length){{alert(`还有 ${{missing.length}} 张未完成。`);document.getElementById('case-'+missing[0]).scrollIntoView({{behavior:'smooth'}});return}}const rows=[['audit_id','adjudicator_id','final_decision','failed_criteria','notes']];ids.forEach(id=>{{const s=state[id];const failed=s.decision==='EXCLUDE'?(s.criteria||[]).join(';'):'';rows.push([id,nameEl.value.trim(),s.decision,failed,''])}});const csv='\\ufeff'+rows.map(r=>r.map(esc).join(',')).join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='task11a3_adjudication.csv';a.click();URL.revokeObjectURL(a.href)}};
render();
</script></body></html>"""
    page = (
        page.replace("__COUNT__", str(len(ids)))
        .replace("__CARDS__", "".join(cards))
        .replace("__IDS__", ids_json)
        .replace("__CRITERIA__", json.dumps(list(CRITERIA_LABELS), ensure_ascii=False))
        .replace("{{", "{")
        .replace("}}", "}")
    )
    output_html.write_text(page, encoding="utf-8", newline="\n")
    return {"disagreement_rows": len(ids), "output": str(output_html)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disagreements", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_adjudication_page(
        disagreement_path=args.disagreements, output_html=args.output_html
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
