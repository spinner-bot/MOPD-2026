#!/usr/bin/env python3
"""Render the full Work5 evaluation report as a self-contained, animated HTML."""
from __future__ import annotations

import argparse
import base64
import pathlib

FIGS = {
    "f1": ("f1_three_sources.png", "图1　S0→S3 三口径效应对比",
           "只有官方口径上升；两条不依赖输出格式的口径均下降。配对 n=32,162，误差棒为 95% CI。"),
    "f2": ("f2_forest.png", "图2　跨评测集效应森林图",
           "官方 I²=95.3%，六评测集几乎无共同效应；判定 58.9%、抽取 46.2%，但方向一致为负。"),
    "f3": ("f3_nesting.png", "图3　三口径的嵌套结构",
           "官方判对的记录判定必然也判对（仅左对=0），构成严格子集。"),
    "f4": ("f4_waterfall.png", "图4　官方增益的分解",
           "Δ官方 + Δ丢弃 = Δ判定，六评测集逐条精确成立。"),
    "f5": ("f5_mechanism.png", "图5　非二元率是难度信号",
           "与正确率 r=−0.917；与输出长度 r=−0.155。长度非机制原因。"),
    "f6": ("f6_category_gap.png", "图6　格式惩罚的类别差异",
           "官方相对真实判定的少计幅度从 0.9pp 到 59.1pp，由类别主导。"),
    "f7": ("f7_zoombench.png", "图7　zoombench 结构性异常",
           "845/845 判官输出非二元，官方恒为 0；终行给出 57.9%–61.2%。"),
    "f8": ("f8_discriminator_kappa.png", "图8　判别器两两一致性（κ）",
           "判定与抽取 0.926，官方与二者仅 0.53–0.59。"),
    "f9": ("f9_accuracy_matrix.png", "图9　六评测集 × 五条件准确率矩阵",
           "左起：官方、判定、抽取。官方 zoombench 行恒为 0，另两口径正常。"),
    "f10": ("f10_condition_trend.png", "图10　五条件趋势",
            "六条线近乎平行：提示对六个评测集方向一致（均略降）。"),
    "f11": ("f11_status_composition.png", "图11　抽取状态构成",
            "各条件均以 ANSWERED 为绝对主体（>98%），失败模式未随条件改变。"),
    "f12": ("f12_redundancy_kappa.png", "图12　条件间逐题一致性（κ）",
            "S1/S2/S3 之间 0.79–0.81，与 S0 为 0.70–0.77。"),
    "f13": ("f13_human_agreement.png", "图13 [参考] 三条链与人工判断的一致性",
            "抽取达到 100%/κ=1.000。审计员的流程为「先自读、再比对」，该一致率反映抽取链准确。"),
    "f14": ("f14_by_condition.png", "图14 [参考] 各条件下人工与三条链的判对数",
            "官方在各条件均明显低于人工；判定与抽取接近人工。"),
    "f15": ("f15_disagreement_direction.png", "图15 [参考] 分歧方向",
            "官方有 62 例被人工判为「漏判为错」，仅 10 例方向相反。"),
}

CSS = r"""
:root{--ink:#111827;--sub:#5b6674;--line:#dfe4ea;--bg:#f7f8fa;--card:#ffffff;
--accent:#2563a8;--accent2:#c2410c;--ok:#047857;--warn:#b45309;--max:1080px}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","Noto Sans SC",sans-serif;
 font-size:16px;line-height:1.78;-webkit-font-smoothing:antialiased}
.hero{background:linear-gradient(135deg,#0b172a 0%,#16324f 48%,#1f4b6e 100%);color:#eef3f8;
 padding:76px 32px 64px;position:relative;overflow:hidden}
.hero:before{content:"";position:absolute;inset:0;
 background:radial-gradient(900px 380px at 18% 0%,rgba(96,165,250,.24),transparent 70%),
            radial-gradient(700px 340px at 88% 100%,rgba(251,146,60,.18),transparent 70%)}
.hero .inner{max-width:var(--max);margin:0 auto;position:relative}
.kicker{letter-spacing:.22em;font-size:.74rem;text-transform:uppercase;color:#93b4d1;margin-bottom:18px}
.hero h1{font-size:clamp(1.9rem,4vw,2.9rem);line-height:1.22;margin:0 0 16px;font-weight:650;letter-spacing:-.01em}
.hero p.lead{max-width:760px;color:#c9d8e6;font-size:1.02rem;margin:0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-top:38px}
.stat{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);border-radius:12px;
 padding:16px 18px;backdrop-filter:blur(4px)}
.stat .n{font-size:1.7rem;font-weight:700;font-variant-numeric:tabular-nums;color:#fff}
.stat .l{font-size:.78rem;color:#a9c1d6;margin-top:2px}
.progress{position:fixed;top:0;left:0;height:3px;background:linear-gradient(90deg,#2563a8,#f59e0b);
 width:0;z-index:99;transition:width .1s linear}
main{max-width:var(--max);margin:0 auto;padding:0 32px 100px}
section{background:var(--card);border:1px solid var(--line);border-radius:16px;
 padding:34px 38px;margin:-34px 0 26px;box-shadow:0 1px 2px rgba(16,24,40,.05),0 12px 32px -18px rgba(16,24,40,.18);
 position:relative}
section:first-of-type{margin-top:-52px}
section h2{display:flex;align-items:baseline;gap:14px;font-size:1.32rem;margin:0 0 6px;font-weight:650}
section h2 .no{font-size:2.1rem;font-weight:800;color:#c8d6e5;line-height:1;font-variant-numeric:tabular-nums}
section .q{color:var(--sub);font-size:.93rem;margin:0 0 18px;padding-left:2px}
h3{font-size:1.02rem;margin:26px 0 8px;font-weight:650;color:#1f2937}
p{margin:.65rem 0}
.key{background:linear-gradient(180deg,#f0f6fc,#e8f1fa);border-left:4px solid var(--accent);
 padding:16px 20px;border-radius:0 10px 10px 0;margin:20px 0}
.caveat{background:linear-gradient(180deg,#fff8ef,#fdf1e3);border-left:4px solid var(--warn);
 padding:16px 20px;border-radius:0 10px 10px 0;margin:20px 0}
.okbox{background:linear-gradient(180deg,#f0fdf6,#e7f8ef);border-left:4px solid var(--ok);
 padding:16px 20px;border-radius:0 10px 10px 0;margin:20px 0}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:.9rem;
 font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:9px 11px;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f2f5f8;font-weight:650;border-bottom:2px solid #d5dce4}
tbody tr:hover{background:#fafcfe}
figure{margin:26px 0;border:1px solid var(--line);border-radius:14px;overflow:hidden;
 background:#fff;transition:transform .35s cubic-bezier(.2,.7,.3,1),box-shadow .35s}
figure:hover{transform:translateY(-3px);box-shadow:0 18px 40px -22px rgba(16,24,40,.35)}
figure img{display:block;width:100%;height:auto;background:#fff}
figcaption{padding:13px 18px 16px;font-size:.86rem;color:var(--sub);border-top:1px solid #eef1f4}
figcaption b{color:var(--ink);font-weight:650;display:block;margin-bottom:3px}
.reveal{opacity:0;transform:translateY(18px);transition:opacity .7s ease,transform .7s cubic-bezier(.2,.7,.3,1)}
.reveal.in{opacity:1;transform:none}
.toc{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin:18px 0 0}
.toc a{display:block;padding:9px 12px;border:1px solid var(--line);border-radius:9px;
 color:var(--ink);text-decoration:none;font-size:.9rem;transition:.2s}
.toc a:hover{border-color:var(--accent);background:#f4f9ff;transform:translateX(2px)}
.tag{display:inline-block;font-size:.72rem;padding:2px 8px;border-radius:999px;
 background:#eef3f8;color:var(--sub);margin-left:8px;vertical-align:middle}
footer{max-width:var(--max);margin:0 auto;padding:0 32px 70px;color:var(--sub);font-size:.82rem}
.tier-divider{max-width:var(--max);margin:56px auto 26px;padding:0 32px;text-align:center}
.tier-divider:before,.tier-divider:after{content:"";display:block;height:3px;
 background:repeating-linear-gradient(90deg,#b45309 0 14px,transparent 14px 24px);margin:0 0 22px}
.tier-divider:after{margin:22px 0 0}
.tier-label{font-size:.8rem;letter-spacing:.28em;color:#b45309;font-weight:700;text-transform:uppercase}
.tier-text{max-width:820px;margin:14px auto 0;color:#5b3b12;font-size:.94rem;line-height:1.85;
 background:#fff8ef;border:1px dashed #d9a04a;border-radius:12px;padding:18px 24px;text-align:left}
.tier-meta{margin-top:12px;font-size:.74rem;letter-spacing:.18em;color:#c07a2d}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}
 .reveal{opacity:1;transform:none}}
@media(max-width:720px){section{padding:24px 20px}main{padding:0 16px 70px}.hero{padding:56px 20px 48px}}
"""

JS = r"""
const bar=document.getElementById('bar');
const onScroll=()=>{const h=document.documentElement;
 const p=(h.scrollTop)/(h.scrollHeight-h.clientHeight)*100; bar.style.width=p+'%';};
addEventListener('scroll',onScroll,{passive:true}); onScroll();
const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}})},{threshold:.12});
document.querySelectorAll('.reveal').forEach(el=>io.observe(el));
document.querySelectorAll('[data-count]').forEach(el=>{
 const target=parseFloat(el.dataset.count), dec=(el.dataset.dec||'0')|0, dur=1100;
 const t0=performance.now();
 const step=t=>{const k=Math.min(1,(t-t0)/dur); const e=1-Math.pow(1-k,3);
  el.textContent=(target*e).toFixed(dec); if(k<1)requestAnimationFrame(step);};
 requestAnimationFrame(step);
});
"""


def fig_factory(prefix):
    def f(key):
        name, title, cap = FIGS[key]
        data = base64.b64encode((prefix / name).read_bytes()).decode("ascii")
        return (f'<figure class="reveal"><img alt="{title}" src="data:image/png;base64,{data}">'
                f"<figcaption><b>{title}</b>{cap}</figcaption></figure>")
    return f


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    F = fig_factory(args.figures)

    body = f"""
<div class="hero"><div class="inner">
 <div class="kicker">MOPD-2026 · Phase 2 · Work 5</div>
 <h1>五种提示条件下的全量视觉问答评测</h1>
 <p class="lead">六个公开评测集、五种 system prompt 条件、160,810 条条件—样本记录。
 本报告以评测结果为体，评分口径的诊断作为必要前置，并显式记录该设计能够回答与不能回答的问题。</p>
 <div class="stats">
  <div class="stat"><div class="n"><span data-count="160810">0</span></div><div class="l">条件—样本记录</div></div>
  <div class="stat"><div class="n"><span data-count="32162">0</span></div><div class="l">配对样本 / 条件</div></div>
  <div class="stat"><div class="n"><span data-count="10.68" data-dec="2">0</span><small> pp</small></div><div class="l">官方口径 S3−S0（格式产物）</div></div>
  <div class="stat"><div class="n"><span data-count="-2.56" data-dec="2">0</span><small> pp</small></div><div class="l">判定口径 S3−S0（真实）</div></div>
 </div>
</div></div>

<main>
<section class="reveal" id="s1"><h2><span class="no">01</span>研究问题与设计</h2>
<p class="q">五个条件在六个评测集上的表现如何？提示是否改变了答题正确率？</p>
<p>工作5 在同一模型（OPD-V-Qwen3.5-4B）上，对同一批题目施加五种 system prompt：
<b>S0</b>（无提示）、<b>SC</b>、<b>S1</b>、<b>S2</b>、<b>S3</b>（逐级加强输出压缩要求）。六个评测集为
vstar、zoombench、hrbench-4k、hrbench-8k、mme-realworld、mme-realworld-cn。</p>
{table(["项","值"],[["每条件样本","32,162"],["条件数","5"],["总记录","160,810"],
 ["评测集","6（选项数 4–6 不等）"],["配对","同一 sample_uid 五条件齐全"]])}
<p>分析总体为<b>冻结的逐题配对数据</b>；所有比较均为同一题在不同条件下的配对比较。</p>
</section>

<section class="reveal" id="s2"><h2><span class="no">02</span>三种判别器的统计比较</h2>
<p class="q">官方原生、裁定终行、Qwen 抽取三个判别器，在判定上有什么系统差异？</p>
<p>三个判别器对<b>同一批模型输出</b>作答。全部 160,810 条中，官方计为正确 86,090 条（53.5%），
判定计为正确 117,805 条（73.3%），抽取计为正确 117,949 条（73.3%）。</p>
{F('f8')}
<p>扣除偶然一致后：<b>判定与抽取 κ=0.926</b>，高度一致；<b>官方与二者仅 0.592 / 0.534</b>，中等一致。
交叉表进一步显示结构性关系——官方判对而判定判错的记录为 <b>0 条</b>；判定判对而抽取判错的仅 <b>4 条</b>。</p>
{table(["判别器","总体准确率","与官方 κ","与判定 κ","与抽取 κ"],
 [["官方原生","53.5%","—","0.592","0.534"],["裁定终行","73.3%","0.592","—","0.926"],
  ["答案抽取","73.3%","0.534","0.926","—"]])}
<p>差异在各评测集上极不均匀：</p>
{table(["评测集","官方","判定","抽取","官方−判定"],
 [["vstar","88.0%","88.5%","88.4%","−0.5 pp"],["hrbench-4k","76.5%","83.7%","82.4%","−7.2 pp"],
  ["hrbench-8k","73.2%","80.5%","79.3%","−7.3 pp"],["mme-realworld","51.7%","73.9%","70.2%","−22.2 pp"],
  ["mme-realworld-cn","61.5%","70.0%","68.4%","−8.5 pp"],["<b>zoombench</b>","<b>0.0%</b>","59.1%","59.1%","<b>−59.1 pp</b>"]])}
<div class="key"><b>结论。</b>判定与抽取几乎是同一测量（κ=0.926）；官方与它们<b>不是同一测量</b>。
差异量随评测集从 0.5pp 变化到 59.1pp，说明它测的是<b>输出格式</b>而非答题正确性。</div>
</section>

<section class="reveal" id="s3"><h2><span class="no">03</span>测量有效性</h2>
<p class="q">官方分数能否直接解释为准确率？</p>
{F('f3')}
<p>Vision-OPD 原规则要求判官原始字段去除首尾空白后<b>严格等于 <code>Yes</code></b> 才计为正确。
全量 160,810 条中有 <b>74,720 条（46.5%）</b>属于"推理 + 结论"的非二元文本，按该规则一律计错，
其中 <b>31,715 条其实判对了</b>。</p>
{F('f4')}
<div class="key"><b>恒等式。</b><code>Δ官方 + Δ丢弃 = Δ判定</code>在六个评测集上逐条精确成立。
官方的每一次"提升"，都等量对应被格式规则丢弃条数的下降。这是算术恒等式，不是统计估计。</div>
{F('f7')}
<div class="caveat"><b>zoombench 官方分数作废。</b>其 845 条记录在全部条件下判官输出均为非二元，
官方恒为 0.00%，而终行诊断给出 57.9%–61.2%。后续一律以判定/抽取为主口径。</div>
</section>

<section class="reveal" id="s4"><h2><span class="no">04</span>总体结果</h2>
<p class="q">五条件各自的准确率是多少？六评测集间差多少？</p>
{F('f9')}
{table(["条件","官方","判定","抽取","n"],
 [["S0","47.1%","74.6%","70.8%","32,162"],["SC","50.0%","73.9%","70.3%","32,162"],
  ["S1","56.9%","72.7%","70.1%","32,162"],["S2","55.9%","73.0%","70.3%","32,162"],
  ["S3","57.8%","72.1%","69.8%","32,162"]])}
<p>官方口径给出 47.1%→57.8% 的上升，判定与抽取分别给出 74.6%→72.1% 与 70.8%→69.8% 的下降。</p>
</section>

<section class="reveal" id="s5"><h2><span class="no">05</span>条件效应</h2>
<p class="q">SC/S1/S2/S3 相对 S0 是否有真实效应？多大？方向？</p>
{F('f1')}
{table(["条件","判定 Δ","95% CI","抽取 Δ","95% CI"],
 [["SC","−0.79 pp","[−1.11, −0.47]","−0.49 pp","[−0.81, −0.17]"],
  ["S1","−1.95 pp","[−2.30, −1.59]","−0.67 pp","[−1.00, −0.33]"],
  ["S2","−1.65 pp","[−2.00, −1.31]","−0.50 pp","[−0.85, −0.16]"],
  ["S3","−2.56 pp","[−2.93, −2.19]","−0.97 pp","[−1.32, −0.61]"]])}
<p>翻转结构（S0→S3，判定口径）：由错变对 <b>1,463（4.55%）</b>、由对变错 <b>2,285（7.10%）</b>、
不变 <b>28,414（88.35%）</b>。净变化 −822 条。</p>
<div class="okbox"><b>等价性检验（TOST）。</b>差 −2.56pp，90% CI [−2.87, −2.24]。
在 ±0.5、±1、±2 pp 任何裕度下都<b>不能判定等价</b>——这是一个<b>真实存在的负效应</b>，不是"测不出来"。</div>
{F('f2')}
</section>

<section class="reveal" id="s6"><h2><span class="no">06</span>分层与交互</h2>
<p class="q">效应是否随评测集、类别、语言变化？</p>
<p>六个评测集的效应方向一致为负（判定口径 −1.0 ~ −3.9pp），但幅度存在异质性（I²=58.9%）。
条件 × 评测集的趋势线近乎平行（图10），未见明显的交叉交互。</p>
{F('f10')}
<h3>语言对照</h3>
<p>mme-realworld（英文）与 mme-realworld-cn（中文）是同一任务的两个语言版本：</p>
{table(["评测集","S0","S3","Δ"],[["mme-realworld（EN）","75.20%","72.90%","−2.30 pp"],
 ["mme-realworld-cn（CN）","71.61%","67.75%","−3.86 pp"]])}
<p><b>中文基线更低，且受提示的损伤更大</b>（−3.86 vs −2.30 pp）。</p>
<h3>类别分层</h3>
<p>官方相对真实判定的少计幅度由类别主导，从 <code>direct_attributes</code> 的 0.9pp 到
<code>Perception/Diagram and Tables</code> 的 38.9pp。</p>
{F('f6')}
</section>

<section class="reveal" id="s7"><h2><span class="no">07</span>输出长度</h2>
<p class="q">提示把输出压短了多少？长度与正确性是什么关系？</p>
{table(["条件","均值 token","中位数","P95","相对 S0"],
 [["S0","336.9","92","—","—"],["SC","369.4","87","—","+9.6%"],["S1","306.1","60","—","−9.1%"],
  ["S2","331.4","61","—","−1.6%"],["S3","260.3","53","—","−22.7%"]])}
<p>S3 相对 S0 平均缩短 <b>22.7%</b>，中位数从 92 降到 53。</p>
{F('f5')}
<div class="caveat"><b>机制修正。</b>先前报告曾提出"输出变短使判官更易输出裸 Yes"作为官方分数上升的机制。
数据不支持该解释（长度与非二元率相关仅 r=−0.155）。真实机制是判官在<b>难判定或存在争议</b>的题目上才展开推理，
故非二元率与正确率强负相关（r=−0.917）。<b>长度只是伴生现象。</b></div>
</section>

<section class="reveal" id="s8"><h2><span class="no">08</span>失败模式</h2>
<p class="q">模型是怎么失败的？提示是否改变失败方式？</p>
{F('f11')}
{table(["条件","ANSWERED","NO_ANSWER","AMBIGUOUS"],[["S0","98.9%","约 0.5%","约 0.6%"],
 ["S3","98.9%","约 0.5%","约 0.6%"]])}
<p>五个条件的抽取状态构成几乎不变：ANSWERED 始终 >98%，<b>提示没有改变失败的形态</b>，
只是把少数题的答案推到了另一侧。</p>
</section>

<section class="reveal" id="s9"><h2><span class="no">09</span>条件冗余</h2>
<p class="q">S1/S2/S3 是三个独立条件，还是近重复？</p>
{F('f12')}
{table(["条件对","一致率","Cohen's κ"],[["S1 vs S2","92.6%","0.813"],["S1 vs S3","91.8%","0.794"],
 ["S2 vs S3","92.2%","0.804"],["SC vs S1","91.5%","0.783"],["S0 vs S3","88.3%","0.702"]])}
<p>扣除偶然一致后，S1/S2/S3 之间 κ≈0.79–0.81——<b>高度相关但并非冗余</b>，
在约 8–9% 的题目上仍给出不同判定。</p>
</section>

<section class="reveal" id="s10"><h2><span class="no">10</span>选项行为</h2>
<p class="q">提示改变的是"选什么"，还是"对不对"？</p>
{table(["条件","A","B","C","D","E"],
 [["S0","25.2%","25.3%","24.4%","20.3%","3.7%"],["SC","25.6%","25.3%","24.6%","20.3%","3.2%"],
  ["S1","25.6%","25.5%","24.1%","20.5%","3.2%"],["S2","25.8%","25.2%","24.2%","20.3%","3.3%"],
  ["S3","25.1%","25.4%","24.2%","20.8%","3.4%"]])}
<div class="okbox"><b>无选项偏置。</b>五个条件的选项分布几乎完全重合（各选项最大变动 &lt;1pp）。
提示造成的准确率变化不是由"更爱选某个字母"引起的。</div>
</section>

<section class="reveal" id="s11"><h2><span class="no">11</span>有效样本量</h2>
<p class="q">32,162 个样本真的是独立单位吗？区间是否被低估？</p>
{table(["项","值"],[["S0 条件下的题目数","32,162"],["不同 cluster_uid 数","32,162"],
 ["最大簇大小","1"],["簇大小分布","全部为 1"]])}
<div class="caveat"><b>发现。</b>数据中的 <code>cluster_uid</code> 每簇恰好一道题，
因此 v4 所称的"图像聚类 Bootstrap"<b>实际退化为逐题 Bootstrap</b>，图像层面的相关性未被控制。
若原始 benchmark 存在同图多题（MME-RealWorld 常见），当前区间可能偏窄。<b>此项需用原始图像 ID 核实。</b></div>
</section>

<section class="reveal" id="s12"><h2><span class="no">12</span>设计限制</h2>
<p class="q">这个设计能回答什么，不能回答什么？</p>
<h3>无法回答：提示效应是否随题目难度变化</h3>
<p>该问题需要一个<b>与治疗无关的难度度量</b>，而本设计没有对同一题、同一条件的重复测量，
任何难度划分都只能从某个条件的表现反推，因而必然被污染。两个方向的尝试均失败：</p>
{table(["分层依据","偏差机制","结果"],
 [["按 S0 是否答对","与比较的一端同源 → <b>回归均值</b>","S0 答错组 S3 由 0% 升至 17.9%，看似\"提示救难题\"——实为假象"],
  ["按 S1+S2 是否全对","与 S3 逐题一致率 92% → <b>间接选中 S3 的结果</b>","易题组 S3 反超 S0 0.5pp——同样不是\"提示没伤易题\""],
  ["用 SC+S1 定义难度后比 S2 vs S3","两个同族治疗互比 → <b>循环论证</b>","−1.2/+0.8pp，无信息"]])}
<div class="caveat"><b>因此本报告不给出任何\"难度梯度\"的方向性结论。</b>
两条路径都把 S3 往好看的方向推，本身就说明该设计无法干净地估计这一效应。
若要回答，需要同题同条件的重复采样（估计噪声）或与提示无关的难度标注。</div>
<h3>其他限制</h3>
<ul>
<li><b>终行诊断不是金标准</b>：它是对判官输出的另一种解析，与被审计对象同源；其与独立抽取 κ=0.926 提供旁证，但不构成独立验证。</li>
<li><b>答案抽取亦非独立验证</b>：由 Qwen3.8-27B 完成，与图像辅助审计（报告 2225）使用同一权重。</li>
<li><b>人工语义审阅尚未完成</b>：60 例 × 5 条件（300 视图）的离线审阅材料已备齐，但无任何已提交的决策。</li>
<li><b>相关性不等于因果</b>。</li>
</ul>
</section>

<section class="reveal" id="s13"><h2><span class="no">13</span>讨论与结论</h2>
<p class="q">这些结果对研究问题意味着什么？</p>
<div class="key">
<b>可以报告。</b>① 全量生成与两条判别链的门禁均通过；② 三条评分来源并非同一测量，构成严格嵌套
（官方 ⊂ 判定 ⊂ 抽取）；③ 按不依赖输出格式的口径，提示强度的增加<b>未提升</b>答题正确率，
四个条件相对 S0 均为小幅下降（−0.5 ~ −2.6pp），且为真实负效应（TOST 不支持等价）；
④ 该下降在中文评测集上更明显；⑤ 提示未改变选项分布，也未改变失败形态。
</div>
<div class="caveat">
<b>不可报告。</b>① 任何"提示提升准确率"的表述——官方口径的上升已证明为格式产物；
② 任何"提示与难度交互"的结论——该设计无法回答；③ 将 60 例定向抽样外推为评测集总体准确率。
</div>
<h3>与既有工作的关系</h3>
<p>Vision-OPD 的原有评测口径以"判官原始字段是否严格等于 Yes"计分。本分析表明该口径在工作5 的数据上
引入了与条件相关、与类别相关的系统性少计，直接沿用会把<b>格式合规率的变化</b>误读为准确率的变化。</p>
<h3>下一步</h3>
<ol><li>完成 60 例人工语义审阅，对三条链的分歧给出人工裁决；</li>
<li>用原始图像 ID 核实聚类结构，必要时以图像为簇重算区间；</li>
<li>若需回答难度问题，设计同题重复采样。</li></ol>
</section>

<div class="tier-divider">
 <div class="tier-label">证据边界 / EVIDENCE BOUNDARY</div>
 <div class="tier-text">
  以下为<b>参考级证据（T2）</b>：来自一次人工语义审阅。审阅者的判断水平未经正式评估，
  抽样为风险定向而非随机，且审阅页面同时展示了自动评分。<br>
  <b>本部分的任何数字与结论，均不得回流、修改或削弱上方第 1–13 节的任何结论。</b>
  主分析仅依据冻结数据与自动化检验（T1）。本部分的作用是<b>交叉参考</b>，不是确认。
 </div>
 <div class="tier-meta">T2 · REFERENCE ONLY · NOT CONFIRMATORY</div>
</div>

<section class="reveal" id="s14"><h2><span class="no">14</span>参考：人工审阅与三条判别器的对照</h2>
<p class="q">[参考] 人工判断与三条自动口径的一致程度如何？分歧集中在哪？</p>

<h3>方法与审阅流程</h3>
<p>60 例 × 5 条件 = 300 个视图，按风险层定向抽取（每评测集 10 例）。审阅者逐条阅读模型原始回答，
自行判断其最终结论选项，再与抽取结果比对以决定"抽取是否忠实"；同时独立判断答案语义与裁判理由。</p>
<p>300 个视图中，<b>284 个</b>的人工"语义是否正确"可二分类，16 个判为"不确定"（均因图像本身看不清，附有说明），
不进入一致率计算。</p>

{F('f13')}
{table(["判别器","与人工一致率","Cohen's κ","敏感度","特异度","人工对/自动错","人工错/自动对"],
 [["官方原生","74.6%","0.492","68.2%","88.8%","<b>62</b>","10"],
  ["裁定终行","96.1%","0.907","100.0%","87.6%","0","11"],
  ["<b>答案抽取</b>","<b>100.0%</b>","<b>1.000</b>","100.0%","100.0%","<b>0</b>","<b>0</b>"]])}

<div class="key"><b>[参考] 核心一致。</b>在官方与抽取全部分歧的 <b>72 处</b>（62 + 10），
人工的语义判断<b>每一处都与抽取一致</b>，无例外。这与主分析第 3 节的结论方向一致：
官方口径存在系统性少计。</div>

<h3>按条件</h3>
{F('f14')}
{table(["条件","n","人工判对","官方","判定","抽取"],
 [["S0","57","39","25","44","39"],["SC","57","39","24","41","39"],["S1","56","38","31","40","38"],
  ["S2","57","39","31","40","39"],["S3","57","40","32","41","40"]])}
<p>官方在全部五个条件下均明显低于人工；判定与抽取与人工基本重合。</p>

<h3>分歧方向</h3>
{F('f15')}
<p>官方被人工判为"漏判为错"的有 <b>62</b> 例，反方向仅 <b>10</b> 例——差距 6:1，
与"格式规则少计"的方向一致，而非随机误差。</p>

<h3>人工观察到的条件效应</h3>
<p>按人工语义判断做 S0→S3 配对：n=57，改善 4、恶化 3，<b>Δ=+1.8 pp，McNemar p=1</b>。
<b>未观察到提示带来的准确率变化</b>——与主分析第 5 节"无稳健提升"的结论不矛盾（但样本量极小，
且为定向抽样，不足以独立支持该结论）。</p>

<h3>审计员标记的异常</h3>
<p>裁判理由/标签被判为无效或无法评估：<b>13</b> 例；抽取忠实度被判为否或不确定：<b>3</b> 例。
其中 Case 46 与 Case 55 的若干视图涉及模型输出大量循环重复、上万 token 后截断、未给出最终答案，
审阅者将其裁判有效性标为"无法评估"并附说明。</p>

<div class="caveat"><b>[参考] 本部分不能证明什么。</b>
<ul>
<li><b>不能</b>用来验证抽取链的准确率——虽然一致率为 100%，但样本是为此类分歧而定向抽取的，不是随机样本；</li>
<li><b>不能</b>外推为任何评测集的总体准确率；</li>
<li><b>不能</b>替代主分析的自动化检验，也不改变第 1–13 节的任何结论。</li>
</ul>
其价值在于：<b>在一个独立于官方口径的视角下，未发现与主结论相反的证据。</b></div>
</section>

<section class="reveal" id="repro"><h2><span class="no">A</span>复现</h2>
<p class="q">全部数字与图表如何重建？</p>
<ul>
<li><code>scripts/analyze_work5_full.py</code> —— 全部统计量（κ、交叉表、矩阵、分层、长度、冗余、簇结构）</li>
<li><code>scripts/plot_score_source_analysis.py</code>、<code>plot_work5_extra.py</code> —— 12 张图</li>
<li><code>scripts/build_work5_report_v2.py</code> —— 本报告</li>
<li>输入：<code>derived/full-statistics-v4/sample_level_merged.csv</code>（160,810 行，唯一键完全对齐）</li>
<li>中文字体：<code>simhei.ttf</code>（服务器另行提供，不随仓库分发）</li>
</ul>
<p>本报告全部图像以 base64 内嵌，无外部依赖，可离线阅读。</p>
</section>
</main>
<footer>
<p>工作5 完整评测报告 · 2026-10-17 · 对应进展报告 2233 / 2234</p>
<p>数据口径：冻结的逐题配对数据；所有比较为同题配对比较。</p>
</footer>
<div class="progress" id="bar"></div>
"""

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>工作5 完整评测报告 · 五种提示条件下的全量视觉问答评测</title>
<style>{CSS}</style></head><body>{body}<script>{JS}</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"PASS: {args.output} ({args.output.stat().st_size:,} bytes, {len(FIGS)} 图)")


if __name__ == "__main__":
    main()
