#!/usr/bin/env python3
"""Render the Work5 score-source diagnostic report as a self-contained HTML file.

Figures are embedded as base64 data URIs so the deliverable needs no side files.
"""
from __future__ import annotations

import argparse
import base64
import pathlib

FIGURES = (
    ("f1_three_sources.png", "图1", "S0→S3 三口径效应对比",
     "只有官方口径上升（+10.68 pp）；两条不依赖输出格式的口径均下降。配对样本各 32,162，区间为 95% 置信区间。"),
    ("f2_forest.png", "图2", "跨评测集效应森林图",
     "官方效应的 I² = 95.3%——六评测集几乎无共同效应，不可合并为单一估计。判定（58.9%）与抽取（46.2%）虽亦有异质性，但方向一致为负。"),
    ("f3_nesting.png", "图3", "三口径的嵌套结构",
     "官方判对的记录，判定必然也判对（仅左对 = 0），构成严格子集关系。官方指标在结构上只可能少计。"),
    ("f4_waterfall.png", "图4", "官方增益的分解",
     "Δ官方 + Δ丢弃 = Δ判定，在六个评测集上逐条精确成立。官方的每一次上升，都等量对应被格式规则丢弃条数的下降。"),
    ("f5_mechanism.png", "图5", "机制：非二元率是难度信号",
     "判官在难判定/有争议的题目上展开推理，与正确率强负相关（r = −0.917）。输出长度非机制原因（r = −0.155）。"),
    ("f6_category_gap.png", "图6", "格式惩罚的类别差异",
     "同一模型、同一批数据，官方相对真实判定的少计幅度从 0.9 pp 到 59.1 pp，几乎完全由类别决定。"),
    ("f7_zoombench.png", "图7", "zoombench 结构性异常",
     "845/845 判官输出均为非二元，官方规则无法计任何一条为对；终行诊断同期给出 57.9%–61.2%。该评测集官方分数应作废。"),
)

CSS = """
body{font-family:-apple-system,"Segoe UI",Arial,"Microsoft YaHei",sans-serif;line-height:1.65;
     color:#1c2026;background:#fff;margin:0 auto;padding:34px;max-width:1000px}
h1{font-size:1.7rem;margin:0 0 .3rem}
h2{font-size:1.15rem;margin:2.4rem 0 .7rem;border-bottom:1px solid #d9dee4;padding-bottom:.3rem}
h3{font-size:1rem;margin:1.5rem 0 .4rem}
.sub{color:#66707c;font-size:.9rem;margin-bottom:1.6rem}
figure{margin:1.8rem 0;padding:0}
figure img{width:100%;height:auto;border:1px solid #e2e6ea;border-radius:6px;background:#fff}
figcaption{font-size:.86rem;color:#4a5560;margin-top:.55rem}
figcaption b{color:#1c2026}
table{border-collapse:collapse;width:100%;margin:.9rem 0;font-size:.88rem}
th,td{border:1px solid #dfe4e9;padding:5px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f4f7fa;font-weight:600}
.key{background:#f0f6fb;border-left:4px solid #2f6f9f;padding:12px 16px;margin:1.2rem 0;border-radius:0 6px 6px 0}
.caveat{background:#fdf6ec;border-left:4px solid #c98a2e;padding:12px 16px;margin:1.2rem 0;border-radius:0 6px 6px 0}
code{background:#f2f4f6;padding:1px 5px;border-radius:3px;font-size:.88em}
.mono{font-variant-numeric:tabular-nums}
"""


def table(headers, rows):
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td class='mono'>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()

    figs = []
    for name, label, title, caption in FIGURES:
        data = base64.b64encode((args.figures / name).read_bytes()).decode("ascii")
        figs.append(
            f'<figure><img alt="{title}" src="data:image/png;base64,{data}">'
            f"<figcaption><b>{label}　{title}</b> —— {caption}</figcaption></figure>"
        )

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Work5 评分口径诊断报告 / Score-source diagnostics</title><style>{CSS}</style></head><body>

<h1>工作5 评分口径诊断报告</h1>
<div class="sub">Five-condition full-benchmark evaluation · score-source diagnostics ·
数据：160,810 条条件—样本记录 · 六评测集 × 五条件 · 2026-10-17</div>

<div class="key"><b>一句话结论。</b>
三条评分来源构成严格嵌套——<b>官方是判定的子集，判定是抽取的子集</b>。
官方分数相对真实判定的缺口，来自判官在难判定/有争议的题目上展开推理（输出非二元文本）而被原规则计为错误。
该缺口规模随评测集与类别从 0.9 pp 变化到 59.1 pp，导致官方效应跨评测集高度异质（I2 = 95.3%），
<b>不可合并报告</b>；按不依赖输出格式的两条口径，提示强度增加并未提升答题正确率。</div>

<h2>1　三种评分来源的定义</h2>
<table><thead><tr><th>来源</th><th>判定规则</th><th>是否依赖输出格式</th></tr></thead><tbody>
<tr><td><b>官方原生</b></td><td>Vision-OPD <code>cal_acc.py</code>：判官原始字段去首尾空白后<b>严格等于 Yes</b> 才计为正确</td><td><b>是</b></td></tr>
<tr><td>裁定终行</td><td>只读判官输出最后一行的精确 Yes/No（诊断性解析）</td><td>否</td></tr>
<tr><td>答案抽取</td><td>Qwen3.8-27B 独立文本抽取，据逐字证据判定</td><td>否</td></tr>
</tbody></table>

<h2>2　三口径在 S0→S3 上方向相反</h2>
{figs[0]}
{table(["口径","S0","S3","Δ","95% CI"],
       [["官方原生","47.12%","57.80%","<b>+10.68 pp</b>","+10.15 ～ +11.23"],
        ["裁定终行","74.65%","72.09%","<b>-2.56 pp</b>","-2.93 ～ -2.19"],
        ["答案抽取","70.76%","69.79%","<b>-0.97 pp</b>","-1.32 ～ -0.61"]])}
<p>三条口径对<b>同一批模型输出</b>作答，模型输出不因评分来源而改变（配对 n = 32,162）。</p>

<h2>3　嵌套结构与全量构成</h2>
{figs[2]}
{table(["类别","条数","占全量"],
       [["官方计为正确（格式 = yes）","86,090","53.5%"],
        ["<b>判官判对但被格式规则丢弃</b>","<b>31,715</b>","<b>19.7%</b>"],
        ["判官判错且输出非二元","42,951","26.7%"],
        ["判官判定 other","54","0.03%"]])}
<p>"仅左对 = 0"表示官方判对的记录判定必然也判对，故三条口径构成<b>严格嵌套</b>，官方指标在结构上只可能少计。</p>

<h2>4　官方增益的精确分解</h2>
{figs[3]}
{table(["评测集","Δ官方","Δ判定（真实）","Δ丢弃","恒等式"],
       [["vstar","-2","-3","-1","✓"],["zoombench","0","-28","-28","✓"],
        ["hrbench-4k","+123","-8","-131","✓"],["hrbench-8k","+114","-12","-126","✓"],
        ["mme-realworld","+2,781","-543","-3,324","✓"],
        ["mme-realworld-cn","+420","-228","-648","✓"]])}
<p><code>Δ官方 + Δ丢弃 = Δ判定</code>，六评测集逐条精确成立。这是算术恒等式，非统计估计。</p>

<h2>5　异质性：官方效应不可合并</h2>
{figs[1]}
{table(["口径","合并效应","Cochran Q (df=5)","I2","各评测集效应区间"],
       [["官方","+10.62 pp","<b>105.4</b>","<b>95.3%</b>","-1.05 ～ +15.38"],
        ["裁定终行","-2.47 pp","12.2","58.9%","-3.85 ～ -1.00"],
        ["答案抽取","-0.93 pp","9.3","46.2%","-3.31 ～ -0.38"]])}
<p>官方效应的离散几乎全部来自评测集间差异而非抽样误差，因此"官方分数 S3 提升 10.62 pp"这一合并陈述具误导性。</p>

<h2>6　机制：非二元率是难度信号</h2>
{figs[4]}
{table(["相关（30 个 评测集×条件 单元）","Pearson r"],
       [["非二元率 ↔ 判定正确率","<b>-0.917</b>"],["输出 token ↔ 非二元率","-0.155"],
        ["输出 token ↔ 判定正确率","+0.154"],["非二元率 ↔ 官方分数","-1.000（恒等式）"]])}
<div class="caveat"><b>对前序报告的修订。</b>报告 2232 曾提出"输出变短使判官更易输出裸 Yes"作为机制；
本分析不支持该解释（r = -0.155）。真实机制是判官在<b>难判定或存在争议</b>的题目上才展开推理，
故非二元率与正确率强负相关。<b>输出长度只是伴生现象，不是原因。</b></div>

<h2>7　格式惩罚由类别主导</h2>
{figs[5]}
<p>同一模型、同一批数据，官方相对真实判定的少计幅度从 0.9 pp（<code>direct_attributes</code>）
到 59.1 pp（zoombench 无类别记录），几乎完全由类别决定。</p>

<h2>8　健全性告警：zoombench</h2>
{figs[6]}
<div class="caveat"><b>zoombench 的官方分数应作废。</b>其 845 条记录在全部五个条件下判官输出均为非二元，
官方规则无法计任何一条为对，准确率恒为 0.00%；同期终行诊断给出 57.9%–61.2%。二者相差一个数量级，
0% 是评分路径与输出形态不兼容的产物，不是模型表现。
<b>除 zoombench 外，未发现其他"官方 0% 或 100%"类结构性故障。</b></div>

<h2>9　敏感性：主结论不稳健</h2>
<p>若采用官方口径，"提示强度提升准确率"成立（+10.68 pp）；换用任一不依赖输出格式的口径，效应方向即翻转为负。
因此该结论<b>依赖评分口径的选择</b>，不能作为答题能力的结论报告。</p>

<h2>10　方法学边界</h2>
<ol>
<li><b>终行诊断不是金标准。</b>它是对判官输出的另一种解析，与被审计对象同源；本分析依赖"终行解析能正确读出判官结论"。其与独立文本抽取 97.0% 的一致率为该假设提供旁证，但不构成独立验证。</li>
<li><b>答案抽取亦非独立验证。</b>抽取由 Qwen3.8-27B 完成，与图像辅助审计（报告 2225）使用同一权重。</li>
<li><b>判定与抽取的高度一致不排除共同偏差</b>，二者可能对同类错误同样敏感。</li>
<li><b>相关性不等于因果</b>；第 6 节的相关为描述性。</li>
<li><b>类别标签来自各 benchmark 官方字段</b>，跨 benchmark 的类别名不完全可比。</li>
<li><b>zoombench 的 0% 未经独立复现</b>，本文依据其判官输出 100% 非二元这一事实推断为格式不兼容。</li>
</ol>

<h2>11　复现</h2>
<p>全部数字与图表由以下脚本从冻结的逐样本表生成，均随本报告一并版本管理：</p>
<ul>
<li><code>scripts/plot_score_source_analysis.py</code> —— 本报告全部图表</li>
<li>输入：<code>derived/full-statistics-v4/sample_level_merged.csv</code>（160,810 行，唯一键完全对齐）</li>
<li>中文字体：<code>simhei.ttf</code>（服务器另行提供，不随仓库分发）</li>
</ul>
<p style="color:#66707c;font-size:.86rem;margin-top:2rem">
本报告为工作5 评分口径的诊断性分析交付件，对应进展报告 2233。原始实验数据、统计产物与审阅材料另行归档。</p>
</body></html>"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"PASS: {args.output} ({len(html.encode('utf-8')):,} bytes, {len(FIGURES)} 张内嵌图)")


if __name__ == "__main__":
    main()
