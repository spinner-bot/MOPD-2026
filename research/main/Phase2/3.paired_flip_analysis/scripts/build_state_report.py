#!/usr/bin/env python3
"""Eight-state bilingual analysis with denominator-aware task rates and case evidence."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

STATES = ["000", "001", "010", "011", "100", "101", "110", "111"]
STATE_ZH = {
    "000": "始终错误", "001": "仅 S3 修正", "010": "仅 S1 修正", "011": "共同修正",
    "100": "共同损失", "101": "仅 S1 损失", "110": "仅 S3 损失", "111": "始终正确",
}
STATE_EN = {
    "000": "all wrong", "001": "S3 only recovers", "010": "S1 only recovers", "011": "both recover",
    "100": "both lose", "101": "S1 only loses", "110": "S3 only loses", "111": "all correct",
}
STATE_COLOR = {
    "000": "#9ba9b5", "001": "#6bb5ad", "010": "#7e9dd0", "011": "#2e9073",
    "100": "#ba5364", "101": "#df9c64", "110": "#d87973", "111": "#3c6f9d",
}


def ci(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    lo = 0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta.ppf(.975, k+1, n-k))
    return [lo, hi]


def pct(k: int, n: int) -> str:
    return f"{100*k/n:.1f}%" if n else "—"


def figure(path: Path, alt: str, caption_zh: str, caption_en: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'
            f'<figcaption>{caption_zh}<span class="en">{caption_en}</span></figcaption></figure>')


def para(zh: str, en: str) -> str:
    return f'<p>{zh}<span class="en">{en}</span></p>'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root / "derived/run-60011513"
    work = args.work_root
    derived = work / "derived/run-60011513"
    figures = work / "figures/run-60011513"
    report = work / "report/paired-flip-analysis-report.html"
    figures.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(source / "token_per_sample.csv")
    d = d[d.condition.isin(["S0", "S1", "S3"])].copy()
    assert len(d) == 6000 and not d.duplicated(["sample_id", "condition"]).any()
    by_condition = {c: d[d.condition == c].set_index("sample_id").sort_index() for c in ["S0", "S1", "S3"]}
    ids = by_condition["S0"].index
    assert len(ids) == 2000 and all(x.index.equals(ids) for x in by_condition.values())
    correct = pd.DataFrame({c: by_condition[c].correct.astype(str).str.lower().eq("true") for c in by_condition}, index=ids)
    meta = by_condition["S0"][["task", "source", "split", "gold_answer", "response_chars"]].copy()
    state = correct.astype(int).astype(str).agg("".join, axis=1).rename("state")
    states = pd.concat([state, meta[["task", "source", "split"]]], axis=1)
    assert set(state.unique()) == set(STATES)
    states.to_csv(derived / "eight_state_cases.csv", encoding="utf-8")
    counts = state.value_counts().reindex(STATES, fill_value=0).astype(int)
    n_wrong = int((~correct.S0).sum())
    n_right = int(correct.S0.sum())
    assert n_wrong + n_right == 2000
    groups = {
        "baseline_wrong": n_wrong, "baseline_correct": n_right,
        "both_recover": int(counts["011"]), "either_recover": int(counts[["001", "010", "011"]].sum()),
        "both_lose": int(counts["100"]), "either_lose": int(counts[["100", "101", "110"]].sum()),
        "s1_gain": int(counts[["010", "011"]].sum()), "s3_gain": int(counts[["001", "011"]].sum()),
        "s1_loss": int(counts[["100", "101"]].sum()), "s3_loss": int(counts[["100", "110"]].sum()),
        "s1_s3_disagree": int(counts[["001", "010", "101", "110"]].sum()),
    }
    assert groups["s1_gain"] == 298 and groups["s3_gain"] == 262
    task_counts = states.groupby(["task", "state"]).size().unstack(fill_value=0).reindex(columns=STATES, fill_value=0)
    task_counts.to_csv(derived / "task_by_state.csv", encoding="utf-8")
    task_rows = []
    for task, row in task_counts.iterrows():
        total = int(row.sum()); wrong = int(row[STATES[:4]].sum()); right = total-wrong
        for label, keys, denom in [
            ("either_recover", ["001", "010", "011"], wrong),
            ("both_recover", ["011"], wrong),
            ("either_lose", ["100", "101", "110"], right),
            ("both_lose", ["100"], right),
            ("s1_gain", ["010", "011"], wrong),
            ("s3_gain", ["001", "011"], wrong),
            ("s1_loss", ["100", "101"], right),
            ("s3_loss", ["100", "110"], right),
        ]:
            k = int(row[keys].sum())
            task_rows.append({"task": task, "measure": label, "count": k, "denominator": denom,
                              "rate": k/denom if denom else None, "ci95": ci(k, denom)})
    task_stats = pd.DataFrame(task_rows)
    task_stats.to_csv(derived / "task_conditional_rates.csv", index=False, encoding="utf-8")

    # Sample-level lengths, preserving all three measured outputs for every state.
    lengths = pd.DataFrame({c: by_condition[c].completion_tokens for c in ["S0", "S1", "S3"]}, index=ids)
    lengths["state"] = state
    lengths["S1_minus_S0"] = lengths.S1-lengths.S0
    lengths["S3_minus_S0"] = lengths.S3-lengths.S0
    lengths.to_csv(derived / "state_token_lengths.csv", encoding="utf-8")
    length_summary = []
    for code in STATES:
        x = lengths[lengths.state == code]
        length_summary.append({"state": code, "n": len(x), **{
            f"{c}_median_tokens": float(x[c].median()) for c in ["S0", "S1", "S3"]},
            "s1_delta_median": float(x.S1_minus_S0.median()),
            "s3_delta_median": float(x.S3_minus_S0.median())})

    result = {
        "schema": "paired-flip-eight-state-v1", "sample_size": 2000,
        "state_order": "S0,S1,S3; 1=correct, 0=incorrect",
        "state_counts": {s: int(counts[s]) for s in STATES},
        "state_rates_all": {s: float(counts[s]/2000) for s in STATES},
        "groups": groups,
        "group_ci95": {
            "either_recover_given_s0_wrong": ci(groups["either_recover"], n_wrong),
            "both_recover_given_s0_wrong": ci(groups["both_recover"], n_wrong),
            "either_lose_given_s0_correct": ci(groups["either_lose"], n_right),
            "both_lose_given_s0_correct": ci(groups["both_lose"], n_right)},
        "task_stats": task_rows, "length_summary": length_summary,
    }
    (derived / "eight_state_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 12, "axes.titlesize": 14,
                         "figure.facecolor": "white", "axes.unicode_minus": False})
    # Figure A: all eight mutually exclusive states with exact denominators.
    fig, ax = plt.subplots(figsize=(12, 6.2))
    positions = np.arange(8)
    ax.bar(positions, counts.values, color=[STATE_COLOR[s] for s in STATES], width=.70)
    for i, s in enumerate(STATES):
        ax.text(i, counts[s]+14, f"{counts[s]}\n{pct(int(counts[s]),2000)}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(positions, [f"{s}\n{STATE_EN[s]}" for s in STATES])
    ax.set_ylabel("Samples / 样本数"); ax.set_ylim(0, 1320)
    ax.set_title("八种正确性状态 / Eight correctness states (S0 / S1 / S3)")
    ax.axvline(3.5, color="#d0d8df", linewidth=1.5)
    ax.text(1.5, 1260, "S0 wrong: 768", ha="center", color="#536879")
    ax.text(5.5, 1260, "S0 correct: 1,232", ha="center", color="#536879")
    ax.grid(axis="y", alpha=.18); fig.tight_layout()
    fig.savefig(figures / "eight_states.png", dpi=220); plt.close(fig)

    # Figure B: conditional recovery/loss and all-sample shares as two distinct denominators.
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    panels = [
        (axes[0], ["Both recover", "S1 only", "S3 only", "Neither"],
         [counts["011"], counts["010"], counts["001"], counts["000"]], n_wrong,
         "S0 错题内部 / Within S0-wrong (n=768)", [STATE_COLOR[s] for s in ["011", "010", "001", "000"]]),
        (axes[1], ["Both lose", "S1 only", "S3 only", "Neither"],
         [counts["100"], counts["101"], counts["110"], counts["111"]], n_right,
         "S0 对题内部 / Within S0-correct (n=1,232)", [STATE_COLOR[s] for s in ["100", "101", "110", "111"]]),
    ]
    for ax, labels, values, denom, title, colors in panels:
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        for bar, v in zip(bars, values[::-1]):
            ax.text(v+7, bar.get_y()+bar.get_height()/2, f"{v} ({pct(int(v),denom)})", va="center", fontsize=11)
        ax.set_xlim(0, max(values)*1.30); ax.set_title(title); ax.set_xlabel("Samples / 样本数")
        ax.grid(axis="x", alpha=.15)
    fig.tight_layout(); fig.savefig(figures / "conditional_outcomes.png", dpi=220); plt.close(fig)

    # Figure C: four in-group task compositions. Labels carry both count and share.
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.4))
    task_colors = {t: c for t, c in zip(sorted(task_counts.index), ["#4c78a8", "#72b7b2", "#f58518", "#b279a2"])}
    groups_for_pie = [
        ("S1 gains", ["010", "011"]), ("S1 losses", ["100", "101"]),
        ("S3 gains", ["001", "011"]), ("S3 losses", ["100", "110"]),
    ]
    for ax, (title, keys) in zip(axes.flat, groups_for_pie):
        vals = task_counts[keys].sum(axis=1)
        labels = [f"{t}: {int(v)} ({pct(int(v),int(vals.sum()))})" for t,v in vals.items()]
        ax.pie(vals, labels=labels, colors=[task_colors[t] for t in vals.index], startangle=90,
               wedgeprops={"width": .38, "edgecolor":"white"}, labeldistance=1.08, textprops={"fontsize": 10})
        ax.text(0,0,f"n={int(vals.sum())}",ha="center",va="center",fontsize=12,fontweight="bold")
        ax.set_title(title)
    fig.suptitle("翻转组内部任务构成 / Task composition within each flip group", fontsize=15)
    fig.tight_layout(); fig.savefig(figures / "flip_task_composition.png", dpi=220); plt.close(fig)

    # Figure D: task risk with the correct task-specific baseline denominator.
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.7), sharey=False)
    tasks = list(task_counts.index)
    for ax, measures, title, denom_label in [
        (axes[0], ["s1_gain", "s3_gain"], "按任务的修正率 / Recovery among S0-wrong", "该任务 S0 错题 / S0-wrong in task"),
        (axes[1], ["s1_loss", "s3_loss"], "按任务的损失率 / Loss among S0-correct", "该任务 S0 对题 / S0-correct in task"),
    ]:
        x = np.arange(len(tasks)); width=.35
        for j, measure in enumerate(measures):
            subset=task_stats.set_index(["task","measure"])
            rates=np.array([subset.loc[(t,measure),"rate"] for t in tasks])*100
            ax.bar(x+(j-.5)*width, rates, width, label=measure.split("_")[0].upper(),
                   color=["#7e9dd0","#6bb5ad"][j])
        denominators=[int(subset.loc[(t,measures[0]),"denominator"]) for t in tasks]
        ax.set_xticks(x,[f"{t}\nn={n}" for t,n in zip(tasks,denominators)])
        ax.set_ylabel("Within-task rate (%)"); ax.set_title(title)
        ax.legend(frameon=False); ax.grid(axis="y",alpha=.18)
        ax.set_xlabel(denom_label)
    fig.tight_layout(); fig.savefig(figures / "task_conditional_rates.png", dpi=220); plt.close(fig)

    # Figure E: token medians by state.
    fig, ax=plt.subplots(figsize=(12,6))
    x=np.arange(8); width=.24
    for j,c in enumerate(["S0","S1","S3"]):
        med=[r[f"{c}_median_tokens"] for r in length_summary]
        ax.bar(x+(j-1)*width,med,width,label=c,color=["#4c78a8","#f58518","#54a24b"][j])
    ax.set_xticks(x,STATES); ax.set_xlabel("Correctness state / 正确性状态")
    ax.set_ylabel("Median completion tokens / 输出 token 中位数")
    ax.set_title("八种状态的输出长度 / Output length across eight states")
    ax.legend(frameon=False);ax.grid(axis="y",alpha=.18)
    fig.tight_layout();fig.savefig(figures / "tokens_by_state.png",dpi=220);plt.close(fig)

    # Case evidence: deterministic high-reduction examples plus a typical example for six flip states.
    old = pd.read_csv(derived / "flip_cases.csv")
    old = old[old.target == "S1"].set_index("sample_id").sort_index()
    assert old.index.equals(ids)
    cases=[]
    selected=[]
    flip_states=["001","010","011","100","101","110"]
    for code in flip_states:
        subset=lengths[lengths.state==code].copy()
        subset["largest_reduction"]=(subset[["S1_minus_S0","S3_minus_S0"]].min(axis=1))
        top=subset.sort_values(["largest_reduction","S0"],ascending=[True,False]).index[0]
        typical=subset.iloc[(subset["largest_reduction"]-subset["largest_reduction"].median()).abs().argsort()].index[0]
        selected.extend([(code, top,"large_reduction"),(code,typical,"typical")])
    assert len({sample_id for _,sample_id,_ in selected})==len(selected)
    for i,(code,sample_id,selection) in enumerate(selected,1):
        row=old.loc[sample_id]
        source_rows={c:by_condition[c].loc[sample_id] for c in ["S0","S1","S3"]}
        statuses=" / ".join("✓" if correct.loc[sample_id,c] else "✗" for c in ["S0","S1","S3"])
        token_line=" → ".join(str(int(lengths.loc[sample_id,c])) for c in ["S0","S1","S3"])
        choices=row["choices"]
        try: choices=json.loads(choices.replace("'",'"')) if isinstance(choices,str) else choices
        except (ValueError,TypeError): choices=[]
        choice_items="".join(f"<li>({chr(65+j)}) {html.escape(str(v))}</li>" for j,v in enumerate(choices))
        outputs=""
        for c in ["S0","S1","S3"]:
            rr=source_rows[c]
            status="正确 / Correct" if correct.loc[sample_id,c] else "错误 / Incorrect"
            outputs+=(f'<div class="response"><h4>{c} · {status} · {int(rr.completion_tokens)} tokens</h4>'
                      f'<p>解析选项 / Parsed option: <b>{html.escape(str(rr.parsed_answer))}</b></p>'
                      f'<pre>{html.escape(str(rr.response)).replace(chr(10), "&#10;")}</pre></div>')
        cases.append(f'<details class="case"><summary>案例 {i:02d} / Case {i:02d} · {code} {STATE_ZH[code]} / {STATE_EN[code]} · '
                     f'{html.escape(str(row["task"]))} · S0/S1/S3 {statuses} · tokens {token_line}</summary>'
                     f'<div class="casebody"><p><b>问题 / Question:</b> {html.escape(str(row["question"]))}</p>'
                     f'<ol>{choice_items}</ol><p><b>标准答案 / Gold:</b> {html.escape(str(row["gold_answer"]))}</p>'
                     f'<p><b>长度变化 / Token change:</b> S1−S0 {int(lengths.loc[sample_id,"S1_minus_S0"]):+d}; '
                     f'S3−S0 {int(lengths.loc[sample_id,"S3_minus_S0"]):+d}. '
                     f'该状态中按{"最大长度缩减" if selection=="large_reduction" else "接近长度变化中位数"}选取。'
                     f'<span class="en">Selection: {"largest reduction" if selection=="large_reduction" else "near median reduction"} within this state.</span></p>'
                     f'<div class="responses">{outputs}</div><small>sample_id: {html.escape(sample_id)}</small></div></details>')
    # Keep the exact selection and its provenance separately from the presentation.
    case_manifest=[{"case_index":i,"state":code,"sample_id":sample_id,"selection":selection} for i,(code,sample_id,selection) in enumerate(selected,1)]
    (derived/"case_manifest.json").write_text(json.dumps(case_manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    task_text=[]
    for task in tasks:
        task_total=int(task_counts.loc[task].sum()); baseline_wrong=int(task_counts.loc[task,STATES[:4]].sum())
        recovery=int(task_counts.loc[task,["001","010","011"]].sum())
        baseline_right=task_total-baseline_wrong
        loss=int(task_counts.loc[task,["100","101","110"]].sum())
        task_text.append(f'<li>{html.escape(task)}：{task_total} 题；S0 错题 {baseline_wrong} 中至少一个条件修正 {recovery}（{pct(recovery,baseline_wrong)}）；'
                         f'S0 对题 {baseline_right} 中至少一个条件损失 {loss}（{pct(loss,baseline_right)}）。'
                         f'<span class="en">{task}: {task_total} questions; {recovery}/{baseline_wrong} baseline errors recovered by at least one condition '
                         f'({pct(recovery,baseline_wrong)}); {loss}/{baseline_right} baseline correct answers lost by at least one condition '
                         f'({pct(loss,baseline_right)}).</span></li>')

    state_list="".join(f'<li><code>{s}</code> · {STATE_ZH[s]} / {STATE_EN[s]}：<b>{int(counts[s])}</b> / 2000 '
                       f'({pct(int(counts[s]),2000)})</li>' for s in STATES)
    section_a=figure(figures/"eight_states.png","Eight mutually exclusive correctness states","图 1：八种状态覆盖全部 2000 题。","Figure 1: Eight states partition all 2,000 questions.")
    section_b=figure(figures/"conditional_outcomes.png","Conditional recovery and loss","图 2：基线错题和基线对题采用不同分母。","Figure 2: Baseline errors and correct answers have distinct denominators.")
    section_c=figure(figures/"flip_task_composition.png","Task composition in each flip group","图 3：翻转组内部任务构成；四个环形图分别以本组样本数为分母。","Figure 3: Task composition within each flip group; each donut uses its own group total.")
    section_d=figure(figures/"task_conditional_rates.png","Conditional flip rates by task","图 4：任务内部翻转率，以该任务的 S0 错题或对题数为分母。","Figure 4: Within-task rates use baseline-wrong or baseline-correct samples in that task.")
    section_e=figure(figures/"tokens_by_state.png","Median output tokens across states","图 5：按八种状态比较三条件输出 token 中位数。","Figure 5: Median completion tokens for all three conditions in each state.")
    nav="".join(f'<li><a href="#{key}">{label}</a></li>' for key,label in [
        ("summary","摘要 / Summary"),("states","八种状态 / Eight states"),
        ("branches","逻辑合并 / Logical groups"),("composition","任务构成 / Task composition"),
        ("rates","任务内部概率 / Within-task rates"),("length","长度与翻转 / Length and flips"),
        ("cases","精选输出长度压缩案例 / Selected output-compression cases"),("limits","研究判断 / Interpretation")])
    style="""
*{box-sizing:border-box}html{scroll-behavior:smooth}body{max-width:1150px;margin:0 auto;padding:30px 30px 80px;background:#fff;color:#233648;font:16px/1.72 Arial,"Microsoft YaHei",sans-serif}h1{margin:0;padding:26px 30px;background:linear-gradient(135deg,#18324b,#2f6f9f);border-radius:12px;color:white;font-size:30px}h2{margin:56px 0 20px;padding:10px 14px;border-left:5px solid #2f6f9f;border-bottom:1px solid #dce4eb;color:#18324b;background:#f8fafc;font-size:23px}h3{color:#244d6d;margin-top:30px}.author{color:#647689;font-size:14px;margin:10px 2px 18px}.lead{font-size:18px;max-width:940px}.en{display:block;color:#586b7b;font-size:.92em;margin-top:4px;line-height:1.6}.toc{border:1px solid #d7e1ea;padding:18px 22px;border-radius:10px;margin:26px 0;background:#fff}.toc ol{columns:2;margin:8px 0;padding-left:22px}.toc a{color:#246b9d;text-decoration:none}.toc a:hover{text-decoration:underline}.key{font-weight:700;color:#18324b}.callout{border-left:4px solid #4c78a8;background:#f7fafc;padding:16px 20px;margin:24px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0}.metric{border:1px solid #d6e0e8;border-radius:9px;padding:16px 18px;background:#fff}.metric strong{font-size:26px;color:#1e567c}.metric .en{margin-top:0}.case{margin:13px 0;border:1px solid #d7e1ea;border-radius:8px;background:#fff}.case summary{cursor:pointer;padding:14px 16px;font-weight:700;color:#244d6d}.casebody{padding:0 18px 18px}.casebody ol{padding-left:24px}.case small{display:block;color:#8793a0;font-size:12px;margin-top:10px;word-break:break-all}.responses{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.response{border:1px solid #dce4eb;border-radius:7px;padding:10px;background:#fff;min-width:0}.response h4{margin:0;color:#244d6d}.response p{margin:4px 0;font-size:13px}.response pre{white-space:pre-wrap;word-break:break-word;background:#f7f9fb;padding:10px;font:13px/1.55 Consolas,monospace;max-height:420px;overflow:auto}.state-list{columns:2}.state-list li{break-inside:avoid;margin:5px 0}.task-list li{margin:8px 0}.source-note{color:#5a6c7e;font-size:14px}@media(max-width:800px){body{padding:15px}.metrics{grid-template-columns:1fr}.toc ol,.state-list{columns:1}.responses{grid-template-columns:1fr}h1{font-size:23px;padding:20px}}
figure{width:100%;max-width:1000px;margin:26px auto 34px;overflow:hidden}figure img{display:block;width:100%;max-width:100%;height:auto;object-fit:contain}figure figcaption{padding:12px 16px;text-align:left;color:#38556b;font-weight:600}figure figcaption .en{font-weight:400}
"""
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>S0/S1/S3 题目级配对翻转分析 / Paired flip analysis</title><style>{style}</style></head><body>
<h1>S0 / S1 / S3 题目级配对翻转分析<br><span style="font-size:.62em;font-weight:400">Question-level paired flip analysis</span></h1>
<p class="author">作者 / Author: <a href="https://github.com/spinner-bot">spinner-bot</a> · 正式作业 / Formal run: 60011513</p>
<p class="lead">同一道视觉选择题在三种输入条件下，什么时候从错变对，什么时候从对变错？<span class="en">For the same visual multiple-choice question, when does a prompt turn an incorrect answer into a correct one, and when does it reverse a correct answer?</span></p>
<nav class="toc"><b>目录 / Contents</b><ol>{nav}</ol></nav>
<section id="summary"><h2>摘要 / Summary</h2>
<div class="metrics"><div class="metric"><strong>{groups['either_recover']}/{n_wrong}</strong><br>至少一种提示修正 S0 错题<br><span class="en">Baseline errors recovered by S1 or S3 · {pct(groups['either_recover'],n_wrong)}</span></div><div class="metric"><strong>{groups['either_lose']}/{n_right}</strong><br>至少一种提示损失 S0 对题<br><span class="en">Baseline correct answers lost by S1 or S3 · {pct(groups['either_lose'],n_right)}</span></div><div class="metric"><strong>{groups['both_recover']}/{groups['either_recover']}</strong><br>被修正题目中两提示共同修正<br><span class="en">Both prompts recover among any recovered · {pct(groups['both_recover'],groups['either_recover'])}</span></div></div>
{para('总体收益的结构：S1 有 298 次修正、45 次损失，净增加 253 题；S3 有 262 次修正、61 次损失，净增加 201 题。相对 2000 题分别提高 12.65 和 10.05 个百分点。','S1 recovers 298 baseline errors and loses 45 correct answers, for a net gain of 253 questions. S3 recovers 262 and loses 61, for a net gain of 201. These correspond to +12.65 and +10.05 percentage points over 2,000 questions.')}
{para('这批样本中，S1 与 S3 的正确性结果不同的题目有 186 题（9.3%）。因此两种提示不能仅用相近的总体准确率代替逐题比较。','S1 and S3 disagree in correctness on 186 questions (9.3%). Similar aggregate accuracy does not capture where the two prompts behave differently.')}
</section>
<section id="states"><h2>一、八种互斥状态 / I. Eight mutually exclusive states</h2>
{para('按 S0、S1、S3 的顺序，以 1 表示答对、0 表示答错。例如 011 表示 S0 错、S1 和 S3 均对。八组恰好分割全部 2000 题。','The three digits follow S0, S1, S3; 1 means correct and 0 means incorrect. For example, 011 means S0 is wrong while both S1 and S3 are correct. The eight groups partition all 2,000 questions.')}
{section_a}<ul class="state-list">{state_list}</ul>
{para('最值得研究的是 011（共同修正，214 题）与 100（共同损失，26 题），因为两种提示在这些题目上给出同向翻转；001、010、101、110 则直接暴露两种提示的差异。','The 011 group (214 jointly recovered questions) and 100 group (26 jointly lost questions) show same-direction flips under both prompts. The 001, 010, 101, and 110 groups reveal their differences.')}</section>
<section id="branches"><h2>二、按基线状态合并 / II. Logical groups by baseline state</h2>
{para(f'S0 原本答错 {n_wrong} 题：至少一种提示修正 {groups["either_recover"]} 题（{pct(groups["either_recover"],n_wrong)}），其中两者共同修正 {groups["both_recover"]} 题（占 S0 错题 {pct(groups["both_recover"],n_wrong)}）。另有 {int(counts["000"])} 题三组均错。',f'S0 is wrong on {n_wrong} questions. At least one prompt recovers {groups["either_recover"]} ({pct(groups["either_recover"],n_wrong)}), and both recover {groups["both_recover"]} ({pct(groups["both_recover"],n_wrong)} of S0 errors). All three remain wrong on {int(counts["000"])} questions.')}
{para(f'S0 原本答对 {n_right} 题：至少一种提示造成损失 {groups["either_lose"]} 题（{pct(groups["either_lose"],n_right)}），其中两者共同损失 {groups["both_lose"]} 题（占 S0 对题 {pct(groups["both_lose"],n_right)}）。另有 {int(counts["111"])} 题三组均对。',f'S0 is correct on {n_right} questions. At least one prompt loses {groups["either_lose"]} ({pct(groups["either_lose"],n_right)}), and both lose {groups["both_lose"]} ({pct(groups["both_lose"],n_right)} of S0 correct answers). All three are correct on {int(counts["111"])} questions.')}
{section_b}
{para('以上“至少一种提示”的联合数字描述潜在可恢复题目的覆盖范围，不能当作一个预先固定提示的部署准确率，因为事先不知道每题应选 S1 还是 S3。','The “at least one prompt” counts describe the coverage of potentially recoverable questions. They are not the accuracy of a deployable fixed-prompt system: choosing S1 or S3 for each question in advance is unresolved.')}</section>
<section id="composition"><h2>三、翻转组内部是什么任务 / III. Task composition inside flip groups</h2>
{para('下面四个环形图分别问：在 S1 修正、S1 损失、S3 修正、S3 损失的题目内部，Count、Depth、Distance、Relation 各占多少。每个环的分母都是该翻转组自己的题目数。','The four donuts show the mix of Count, Depth, Distance, and Relation within S1 gains, S1 losses, S3 gains, and S3 losses. Each donut uses its own flip-group total as denominator.')}
{section_c}
{para('Distance 和 Depth 在修正组中有较多样本，但它们本来就在总体数据中占较大比例。仅凭组内占比，不能断言这两类题目更容易被提示修正；需要看同类题目中的翻转率。','Distance and Depth contribute many recovered questions, but they also make up a large share of the full dataset. Within-group composition alone cannot show whether prompts are more likely to recover these tasks; that requires within-task rates.')}</section>
<section id="rates"><h2>四、各任务自身的翻转概率 / IV. Flip rates within each task</h2>
{para('修正率以“该任务中 S0 原本答错的题目”为分母；损失率以“该任务中 S0 原本答对的题目”为分母。这样可以区分样本量效应与任务特异行为。','Recovery rates use baseline-wrong questions within each task as denominator; loss rates use baseline-correct questions within that task. This separates task prevalence from task-specific behavior.')}
{section_d}<ul class="task-list">{''.join(task_text)}</ul>
{para('最突出的差异是：Count 的 S0 错题仅 21/306（6.9%）被至少一种提示修正；Relation 为 97/119（81.5%），Depth 为 90/119（75.6%），Distance 为 138/224（61.6%）。另一方面，Distance 的 S0 对题有 42/292（14.4%）在至少一种提示下损失，Relation 只有 8/426（1.9%）。这些值都是同类任务内部的条件比例。','The largest contrast is that only 21/306 baseline-wrong Count questions (6.9%) are recovered by either prompt, versus 97/119 Relation (81.5%), 90/119 Depth (75.6%), and 138/224 Distance (61.6%). Among baseline-correct questions, at least one prompt loses 42/292 Distance questions (14.4%) but only 8/426 Relation questions (1.9%). These are conditional rates within each task.')}
{para('这些任务层数字用于定位值得检查的题目簇。任务、图像来源与 2D/3D 划分并非独立分配，本轮无法将任务效应与数据来源效应分离；这些细分差异属于探索性发现。','These task-level figures locate question clusters worth inspecting. Task, image source, and 2D/3D split were not independently assigned, so this run cannot separate task effects from source effects; subgroup differences remain exploratory.')}</section>
<section id="length"><h2>五、长度变化是否解释翻转 / V. Does length explain the flips?</h2>
{para('按八种状态比较三条件的输出 token 中位数。S1 和 S3 通常比 S0 更短；但“共同修正”和“共同损失”都存在于短输出条件下，因此长度压缩本身无法解释翻转方向。','Median completion tokens are shown for all eight states. S1 and S3 are usually shorter than S0, yet both joint recovery and joint loss occur under these shorter-output conditions. Compression alone does not determine flip direction.')}
{section_e}
{para('在 S1 的修正题与损失题中，相对 S0 的平均长度变化分别约为 −77.4 与 −77.1 tokens；两类几乎相同。S3 的对应变化约为 −85.5 与 −57.3 tokens。后者提示值得继续检查输出内容与题目属性，但仍不能作为因果结论。','For S1, mean changes from S0 are about −77.4 tokens for gains and −77.1 for losses, nearly identical. For S3, the corresponding changes are about −85.5 and −57.3 tokens. The latter invites closer inspection of responses and task properties, but remains an association.')}</section>
<section id="cases"><h2>六、精选输出长度压缩案例 / VI. Selected output-compression cases</h2>
{para('每个翻转状态选取一个大幅缩短输出的案例和一个接近该状态典型长度变化的案例，共 12 个。每个案例给出题目、选项、标准答案、三条件正确性、token 数及完整原始输出。点击条目展开。','Each flip state contributes one large-reduction case and one case near its median length change, for 12 cases total. Each includes the question, choices, gold answer, correctness, token counts, and complete raw responses under all three conditions. Open a case to inspect it.')}
{''.join(cases)}
{para('上述案例按预定的长度规则选出，用于展示行为差异；不能把这些个例的语言模式直接当作整个状态组的典型机制。','Cases are selected by a fixed length-based rule to illustrate behavior. Their wording alone does not establish a mechanism for an entire state group.')}</section>
<section id="limits"><h2>七、研究判断与边界 / VII. Interpretation and next questions</h2>
{para('题目级结果比总体准确率多揭示两件事：一是两种提示共同修正了大量 S0 错题；二是它们也存在共同损失和彼此不一致的题目。后续最值得做的是人工审阅 011、100 和四类单条件翻转，归纳可观察的错误模式，再用独立样本检验这些模式。','Question-level results add two findings beyond aggregate accuracy: the prompts jointly recover many baseline errors, and they also share losses and disagree on some questions. The next useful step is to review 011, 100, and the four single-prompt flip states, define observable error patterns, and test those patterns on independent samples.')}
{para('正确性依据既有解析结果；本报告不重新判题。输出 token 数来自正式 vLLM usage。任务组成、长度关联及所选案例都属于描述性或探索性证据，不能证明模型内部推理机制。','Correctness uses the existing parsed results; this report does not regrade answers. Output token counts come from the formal vLLM usage records. Task composition, length associations, and selected cases provide descriptive or exploratory evidence, not proof of an internal reasoning mechanism.')}
<p class="source-note">数据与复现 / Data and reproduction: <code>derived/run-60011513/</code>, <code>figures/run-60011513/</code>, <code>scripts/build_state_report.py</code>. 分析条件 / Conditions: S0, S1, S3.</p></section></body></html>'''
    report.write_text(doc,encoding="utf-8")
    print(json.dumps({"states":result["state_counts"],"cases":len(selected),"report":str(report)},ensure_ascii=False))


if __name__ == "__main__":
    main()
