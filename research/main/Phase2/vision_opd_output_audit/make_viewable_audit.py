#!/usr/bin/env python3
import argparse, csv, html, json
from pathlib import Path

def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def esc(x):
    return html.escape(str(x))

def badge(answer, ref):
    ok = answer == ref
    return '<span class="badge {}">{}：{}（参考：{}）</span>'.format("ok" if ok else "bad", "正确" if ok else "错误", esc(answer), esc(ref))

def label(a):
    return {"choice_with_explanation":"相关推理/证据（先给答案，随后说明）", "reasoning_then_choice":"相关推理/证据（先判断，最后给答案）", "direct_only":"只有答案"}.get(a["output_class"], "有内容但与推理无关")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--true", required=True); p.add_argument("--false", required=True)
    p.add_argument("--audit", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    tr = {str(x["question_id"]):x for x in rows(a.true)}
    fr = {str(x["question_id"]):x for x in rows(a.false)}
    with Path(a.audit).open(encoding="utf-8", newline="") as f:
        ar = {x["question_id"]:x for x in csv.DictReader(f)}
    cards = []
    for qid, audit in ar.items():
        t, f = tr[qid], fr[qid]
        query = esc(t["query"]).replace("\n", "<br>")
        image_name = Path(t["images"][0]).name
        cards.append(f"""<article class="card"><div class="head"><b>样本 {esc(qid)}</b><strong>{esc(label(audit))}</strong></div>
<div class="question"><div class="image"><img src="assets/{esc(image_name)}" alt="{esc(image_name)}"><small>{esc(image_name)}</small></div><div><h3>题目与选项</h3><p>{query}</p></div></div>
<div class="meta">参考答案：<b>{esc(audit["reference"])}</b>　{badge(audit["true_answer"], audit["reference"])}　{badge(audit["false_answer"], audit["reference"])}　答案先于推理：{esc(audit["answer_before_reasoning"])}　与决策相关：{esc(audit["reasoning_decision_relevance"])}</div>
<div class="outputs"><section><h3>Thinking=True</h3><pre>{esc(t.get("answer",""))}</pre></section><section><h3>Thinking=False</h3><pre>{esc(f.get("answer",""))}</pre></section></div>
<p class="note">视觉证据摘要：{esc(audit["claimed_visual_evidence"])}；人工图像复核：{esc(audit["human_image_verification"])}</p></article>""")
    total = len(ar); wrong = sum(x["correctness"] != "both_correct" for x in ar.values())
    css = "body{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f4f6f8;color:#1f2937;margin:0}main{max-width:1180px;margin:auto;padding:32px 24px 60px}h1{margin-bottom:6px}.sub{color:#64748b}.summary,.card{background:#fff;border:1px solid #d8dee7;border-radius:12px;padding:18px;margin:18px 0;box-shadow:0 2px 8px #1522380d}.head{display:flex;justify-content:space-between;border-bottom:1px solid #e5e7eb;padding-bottom:10px}.head strong{color:#1256a0}.question{display:grid;grid-template-columns:150px 1fr;gap:18px;margin:16px 0}.image{min-height:105px;border:2px dashed #b8c2cf;border-radius:8px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:#637083;background:#f8fafc;overflow:hidden}.image img{display:block;max-width:100%;max-height:180px;object-fit:contain}.image small{font-size:11px}.meta{font-size:13px;line-height:2}.badge{padding:5px 8px;border-radius:6px;font-weight:bold}.ok{background:#dcfce7;color:#166534}.bad{background:#fee2e2;color:#991b1b}.outputs{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.outputs h3{background:#eef2f7;padding:8px;margin:0}.outputs pre{white-space:pre-wrap;word-break:break-word;background:#111827;color:#e5e7eb;padding:12px;margin:0;line-height:1.45;font-size:12px;max-height:260px;overflow:auto}.note{color:#64748b;font-size:12px}@media(max-width:760px){.question,.outputs{grid-template-columns:1fr}}@media print{body{background:#fff}.card{box-shadow:none;break-inside:avoid}.outputs pre{max-height:none;color:#111;background:#f3f4f6}}"
    doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Vision-OPD 样本与输出可视化审计</title><style>{css}</style></head><body><main>
<h1>Vision-OPD：样本与输出可视化审计</h1><p class="sub">Run 59944935 · HRBench-4K · 论文式样本面板（图像来自服务器缩略图）</p>
<div class="summary"><h2>结果摘要</h2><p>共展示 <b>{total}</b> 条样本；两种设置均为 <b>10/{total}</b> 正确。输出归类：<b>只有答案 0</b>，<b>相关推理/证据 {total}</b>，<b>与推理无关 0</b>；其中 <b>{wrong}</b> 条是“有相关推理但最终答错”（样本 393、127）。</p><p>绿色=正确，红色=错误；蓝色标签=导师要求的输出类型。</p></div>
<h2>逐样本展示</h2>{''.join(cards)}<p class="sub">说明：页面使用服务器生成并经 GitHub 中转的低分辨率缩略图；原始高分辨率 benchmark 图像不纳入仓库。</p></main></body></html>"""
    Path(a.output).write_text(doc, encoding="utf-8")

if __name__ == "__main__":
    main()

