#!/usr/bin/env python3
"""Render the full Work5 evaluation report as a self-contained, animated HTML.

Every section is bilingual, English first. The English is the original working
language of the analysis (canonical terminology from the v4 report); the Chinese
follows as a parallel rendering.
"""
from __future__ import annotations

import argparse
import base64
import pathlib

# key -> (file, english title, chinese title, english caption, chinese caption)
FIGS = {
    "f1": ("f1_three_sources.png",
           "Figure 1. S0 to S3 effect under the three score sources",
           "图1　S0→S3 三口径效应对比",
           "Only the official metric rises; both format-independent metrics fall. Paired n = 32,162; error bars are 95% CIs.",
           "只有官方口径上升；两条不依赖输出格式的口径均下降。配对 n=32,162，误差棒为 95% CI。"),
    "f2": ("f2_forest.png",
           "Figure 2. Forest plot of the effect across benchmarks",
           "图2　跨评测集效应森林图",
           "Official I² = 95.3%: the six benchmarks share almost no common effect; terminal 58.9% and extraction 46.2%, but all directions are negative.",
           "官方 I²=95.3%，六评测集几乎无共同效应；判定 58.9%、抽取 46.2%，但方向一致为负。"),
    "f3": ("f3_nesting.png",
           "Figure 3. Nested structure of the three score sources",
           "图3　三口径的嵌套结构",
           "A record the official scorer counts correct is always counted correct by the terminal diagnostic (0 exceptions): a strict subset.",
           "官方判对的记录判定必然也判对（仅左对=0），构成严格子集。"),
    "f4": ("f4_waterfall.png",
           "Figure 4. Decomposition of the official-score gain",
           "图4　官方增益的分解",
           "Delta official + delta discarded = delta terminal, exactly, on every benchmark.",
           "Δ官方 + Δ丢弃 = Δ判定，六评测集逐条精确成立。"),
    "f5": ("f5_mechanism.png",
           "Figure 5. The non-binary rate is a difficulty signal",
           "图5　非二元率是难度信号",
           "r = -0.917 with accuracy; r = -0.155 with output length. Length is not the mechanism.",
           "与正确率 r=−0.917；与输出长度 r=−0.155。长度非机制原因。"),
    "f6": ("f6_category_gap.png",
           "Figure 6. Category differences in the format penalty",
           "图6　格式惩罚的类别差异",
           "Under-counting relative to true decisions ranges from 0.9 pp to 59.1 pp and is driven by category.",
           "官方相对真实判定的少计幅度从 0.9pp 到 59.1pp，由类别主导。"),
    "f7": ("f7_zoombench.png",
           "Figure 7. The structural anomaly in ZoomBench",
           "图7　zoombench 结构性异常",
           "845/845 judge outputs are non-binary, so the official score is identically 0; the terminal diagnostic gives 57.9%-61.2%.",
           "845/845 判官输出非二元，官方恒为 0；终行给出 57.9%–61.2%。"),
    "f8": ("f8_discriminator_kappa.png",
           "Figure 8. Pairwise agreement between discriminators (kappa)",
           "图8　判别器两两一致性（κ）",
           "Terminal vs extraction 0.926; official vs either only 0.53-0.59.",
           "判定与抽取 0.926，官方与二者仅 0.53–0.59。"),
    "f9": ("f9_accuracy_matrix.png",
           "Figure 9. Accuracy matrix: six benchmarks by five conditions",
           "图9　六评测集 × 五条件准确率矩阵",
           "Left to right: official, terminal, extraction. The official ZoomBench row is identically 0; the other two are normal.",
           "左起：官方、判定、抽取。官方 zoombench 行恒为 0，另两口径正常。"),
    "f10": ("f10_condition_trend.png",
            "Figure 10. Condition trends",
            "图10　五条件趋势",
            "The six lines are nearly parallel: prompts move every benchmark in the same direction (slightly down).",
            "六条线近乎平行：提示对六个评测集方向一致（均略降）。"),
    "f11": ("f11_status_composition.png",
            "Figure 11. Composition of extraction status",
            "图11　抽取状态构成",
            "ANSWERED dominates every condition (>98%); the failure mode does not change with the condition.",
            "各条件均以 ANSWERED 为绝对主体（>98%），失败模式未随条件改变。"),
    "f12": ("f12_redundancy_kappa.png",
            "Figure 12. Per-item agreement between conditions (kappa)",
            "图12　条件间逐题一致性（κ）",
            "0.79-0.81 among S1/S2/S3; 0.70-0.77 against S0.",
            "S1/S2/S3 之间 0.79–0.81，与 S0 为 0.70–0.77。"),
    "f13": ("f13_human_agreement.png",
            "Figure 13 [reference]. Agreement of each score source with the human review",
            "图13 [参考] 三条链与人工判断的一致性",
            "Extraction reaches 100% / kappa = 1.000. The auditor read the output first and compared second; the agreement therefore reflects the extraction chain's accuracy.",
            "抽取达到 100%/κ=1.000。审计员的流程为「先自读、再比对」，该一致率反映抽取链准确。"),
    "f14": ("f14_by_condition.png",
            "Figure 14 [reference]. Correct counts by condition: human versus the three sources",
            "图14 [参考] 各条件下人工与三条链的判对数",
            "The official scorer is clearly below the human in every condition; terminal and extraction track the human.",
            "官方在各条件均明显低于人工；判定与抽取接近人工。"),
    "f15": ("f15_disagreement_direction.png",
            "Figure 15 [reference]. Direction of disagreement",
            "图15 [参考] 分歧方向",
            "62 records the human calls correct but the official scorer calls wrong; only 10 in the opposite direction.",
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
.hero h1{font-size:clamp(1.9rem,4vw,2.9rem);line-height:1.22;margin:0 0 10px;font-weight:650;letter-spacing:-.01em}
.hero h1 .zh{display:block;font-size:.52em;font-weight:500;color:#b9cee0;letter-spacing:.02em;margin-top:12px}
.hero p.lead{max-width:790px;color:#c9d8e6;font-size:1.02rem;margin:0}
.hero p.lead.zh{color:#9fb6ca;font-size:.92rem;margin-top:8px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-top:38px}
.stat{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);border-radius:12px;
 padding:16px 18px;backdrop-filter:blur(4px)}
.stat .n{font-size:1.7rem;font-weight:700;font-variant-numeric:tabular-nums;color:#fff}
.stat .l{font-size:.78rem;color:#c2d5e4;margin-top:2px}
.stat .l .zh{display:block;color:#93aec5;font-size:.72rem;margin-top:1px}
.progress{position:fixed;top:0;left:0;height:3px;background:linear-gradient(90deg,#2563a8,#f59e0b);
 width:0;z-index:99;transition:width .1s linear}
main{max-width:var(--max);margin:0 auto;padding:0 32px 100px}
section{background:var(--card);border:1px solid var(--line);border-radius:16px;
 padding:34px 38px;margin:-34px 0 26px;box-shadow:0 1px 2px rgba(16,24,40,.05),0 12px 32px -18px rgba(16,24,40,.18);
 position:relative}
section:first-of-type{margin-top:-52px}
section h2{display:flex;align-items:baseline;gap:14px;font-size:1.32rem;margin:0 0 6px;font-weight:650}
section h2 .no{font-size:2.1rem;font-weight:800;color:#c8d6e5;line-height:1;font-variant-numeric:tabular-nums}
section h2 .t{display:block}
section h2 .t .zh{display:block;font-size:.72em;font-weight:500;color:var(--sub);margin-top:1px;letter-spacing:.01em}
section .q{color:var(--sub);font-size:.93rem;margin:0 0 18px;padding-left:2px}
section .q .zh{display:block;font-size:.92em;color:#7a8492;margin-top:1px}
h3{font-size:1.02rem;margin:28px 0 4px;font-weight:650;color:#1f2937}
h3 .zh{font-weight:500;color:var(--sub);font-size:.86em;margin-left:8px}
p{margin:.65rem 0}
/* bilingual body blocks: English first, Chinese second */
.bi{margin:14px 0}
.bi .en{margin:.55rem 0}
.bi .zh{margin:.2rem 0 .55rem;padding-left:13px;border-left:2px solid #e5eaf0;
 color:#55606e;font-size:.94em;line-height:1.76}
.bi .zh p{margin:.45rem 0}
.key,.caveat,.okbox{margin:20px 0}
.key .bi,.caveat .bi,.okbox .bi{margin:0}
.key .bi .zh,.caveat .bi .zh,.okbox .bi .zh{border-left:none;padding-left:0;color:inherit;opacity:.92}
.key{background:linear-gradient(180deg,#f0f6fc,#e8f1fa);border-left:4px solid var(--accent);
 padding:16px 20px;border-radius:0 10px 10px 0}
.caveat{background:linear-gradient(180deg,#fff8ef,#fdf1e3);border-left:4px solid var(--warn);
 padding:16px 20px;border-radius:0 10px 10px 0}
.okbox{background:linear-gradient(180deg,#f0fdf6,#e7f8ef);border-left:4px solid var(--ok);
 padding:16px 20px;border-radius:0 10px 10px 0}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:.9rem;
 font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:9px 11px;text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}
thead th{background:#f2f5f8;font-weight:650;border-bottom:2px solid #d5dce4}
thead th .zh{display:block;font-weight:400;font-size:.8em;color:var(--sub);margin-top:1px}
td .zh{display:block;font-weight:400;font-size:.9em;color:var(--sub);margin-top:2px;line-height:1.5}
tbody tr:hover{background:#fafcfe}
table.tight{max-width:780px}
table.content th,table.content td{text-align:left}
table.content td{line-height:1.62}
figure{margin:26px 0;border:1px solid var(--line);border-radius:14px;overflow:hidden;
 background:#fff;transition:transform .35s cubic-bezier(.2,.7,.3,1),box-shadow .35s}
figure:hover{transform:translateY(-3px);box-shadow:0 18px 40px -22px rgba(16,24,40,.35)}
figure img{display:block;width:100%;height:auto;background:#fff}
figcaption{padding:14px 18px 16px;font-size:.86rem;color:var(--sub);border-top:1px solid #eef1f4}
figcaption b{color:var(--ink);font-weight:650;display:block;margin-bottom:4px}
figcaption .cap-en{display:block;line-height:1.6}
figcaption .cap-zh{display:block;margin-top:10px;padding-top:10px;border-top:1px dashed #e6eaef;
 color:#6b7684;font-size:.96em;line-height:1.7}
figcaption .cap-zh b{display:block;font-size:1em;color:#5b6674;font-weight:600;margin-bottom:2px}
.reveal{opacity:0;transform:translateY(18px);transition:opacity .7s ease,transform .7s cubic-bezier(.2,.7,.3,1)}
.reveal.in{opacity:1;transform:none}
.tag{display:inline-block;font-size:.72rem;padding:2px 8px;border-radius:999px;
 background:#eef3f8;color:var(--sub);margin-left:8px;vertical-align:middle}
footer{max-width:var(--max);margin:0 auto;padding:0 32px 70px;color:var(--sub);font-size:.82rem}
footer .zh{display:block;color:#8592a1;font-size:.96em;margin-top:2px}
.tier-divider{max-width:var(--max);margin:56px auto 26px;padding:0 32px;text-align:center}
.tier-divider:before,.tier-divider:after{content:"";display:block;height:3px;
 background:repeating-linear-gradient(90deg,#b45309 0 14px,transparent 14px 24px);margin:0 0 22px}
.tier-divider:after{margin:22px 0 0}
.tier-label{font-size:.8rem;letter-spacing:.28em;color:#b45309;font-weight:700;text-transform:uppercase}
.tier-label .zh{display:block;letter-spacing:.18em;font-size:.86em;margin-top:4px;color:#c07a2d}
.tier-text{max-width:860px;margin:14px auto 0;color:#5b3b12;font-size:.94rem;line-height:1.8;
 background:#fff8ef;border:1px dashed #d9a04a;border-radius:12px;padding:18px 24px;text-align:left}
.tier-text .zh{display:block;margin-top:10px;padding-top:10px;border-top:1px dashed #e3c08a;font-size:.95em}
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
/* threshold 0 + rootMargin: a section taller than the viewport can never reach
   an intersection ratio of 0.12, which left the long review section invisible. */
const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}});},
 {threshold:0, rootMargin:"0px 0px -8% 0px"});
document.querySelectorAll('.reveal').forEach(el=>io.observe(el));
/* safety net: reveal anything already on screen if the observer misses it */
addEventListener('load',()=>setTimeout(()=>{
 document.querySelectorAll('.reveal:not(.in)').forEach(el=>{
  const r=el.getBoundingClientRect(); if(r.top<innerHeight*1.2) el.classList.add('in');});},600));
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
        name, en_t, zh_t, en_c, zh_c = FIGS[key]
        data = base64.b64encode((prefix / name).read_bytes()).decode("ascii")
        label = en_t.split(". ")[0] + "."
        return (f'<figure class="reveal"><img alt="{en_t}" src="data:image/png;base64,{data}">'
                f'<figcaption><b>{label}</b><span class="cap-en">{en_c}</span>'
                f'<span class="cap-zh"><b>{zh_t}</b>{zh_c}</span></figcaption></figure>')
    return f


def bi(en: str, zh: str) -> str:
    """Bilingual block: English first, Chinese second."""
    return f'<div class="bi"><div class="en" lang="en">{en}</div><div class="zh" lang="zh-CN">{zh}</div></div>'


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cls = ' class="tight"' if len(head) <= 2 else ""
    return f"<table{cls}><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    F = fig_factory(args.figures)

    body = f"""
<div class="hero"><div class="inner">
 <div class="kicker">MOPD-2026 · Phase 2 · Work 5</div>
 <h1>Full-benchmark evaluation under five prompt conditions<span class="zh">五种提示条件下的全量视觉问答评测</span></h1>
 <p class="lead">Six public benchmarks, five system-prompt conditions, 160,810 condition-sample records.
 The report is organised around the evaluation results; the score-source diagnosis is treated as a necessary precondition,
 and the questions this design can and cannot answer are stated explicitly.</p>
 <p class="lead zh" lang="zh-CN">六个公开评测集、五种 system prompt 条件、160,810 条条件—样本记录。
 本报告以评测结果为体，评分口径的诊断作为必要前置，并显式记录该设计能够回答与不能回答的问题。</p>
 <div class="stats">
  <div class="stat"><div class="n"><span data-count="160810">0</span></div><div class="l">Condition-sample records<span class="zh">条件—样本记录</span></div></div>
  <div class="stat"><div class="n"><span data-count="32162">0</span></div><div class="l">Paired samples per condition<span class="zh">配对样本 / 条件</span></div></div>
  <div class="stat"><div class="n"><span data-count="10.68" data-dec="2">0</span><small> pp</small></div><div class="l">Official S3 minus S0 (format artefact)<span class="zh">官方口径 S3−S0（格式产物）</span></div></div>
  <div class="stat"><div class="n"><span data-count="-2.56" data-dec="2">0</span><small> pp</small></div><div class="l">Terminal S3 minus S0 (real)<span class="zh">判定口径 S3−S0（真实）</span></div></div>
 </div>
</div></div>

<main>
<section class="reveal" id="s1"><h2><span class="no">01</span><span class="t">Research question and design<span class="zh">研究问题与设计</span></span></h2>
<p class="q">How do the five conditions perform across the six benchmarks, and does prompting change answer correctness?<span class="zh">五个条件在六个评测集上的表现如何？提示是否改变了答题正确率？</span></p>
{bi('Work 5 applies five system prompts to the same model (OPD-V-Qwen3.5-4B) on the same item set: <b>S0</b> (no prompt), <b>SC</b>, <b>S1</b>, <b>S2</b>, <b>S3</b> (progressively stronger output-compression requirements). The six benchmarks are V*, ZoomBench, HRBench-4K, HRBench-8K, MME-RealWorld and MME-RealWorld-CN.',
    '工作5 在同一模型（OPD-V-Qwen3.5-4B）上，对同一批题目施加五种 system prompt：<b>S0</b>（无提示）、<b>SC</b>、<b>S1</b>、<b>S2</b>、<b>S3</b>（逐级加强输出压缩要求）。六个评测集为 vstar、zoombench、hrbench-4k、hrbench-8k、mme-realworld、mme-realworld-cn。')}
{table(["Item<span class='zh'>项</span>","Value<span class='zh'>值</span>"],[["Samples per condition<span class='zh'>每条件样本</span>","32,162"],["Conditions<span class='zh'>条件数</span>","5"],["Total records<span class='zh'>总记录</span>","160,810"],
 ["Benchmarks<span class='zh'>评测集</span>","6 (4-6 options each)"],["Pairing<span class='zh'>配对</span>","Same sample_uid present under all five conditions"]])}
{bi('The analysis population is the <b>frozen per-item paired table</b>; every comparison pairs the same item across conditions.',
    '分析总体为<b>冻结的逐题配对数据</b>；所有比较均为同一题在不同条件下的配对比较。')}
</section>

<section class="reveal" id="s2"><h2><span class="no">02</span><span class="t">Statistical comparison of the three discriminators<span class="zh">三种判别器的统计比较</span></span></h2>
<p class="q">What systematic differences exist between the official native, terminal-line and Qwen-extraction discriminators?<span class="zh">官方原生、裁定终行、Qwen 抽取三个判别器，在判定上有什么系统差异？</span></p>
{bi('All three discriminators score <b>the same model outputs</b>. Of the 160,810 records, the official scorer counts 86,090 as correct (53.5%), the terminal-line diagnostic 117,805 (73.3%), and answer extraction 117,949 (73.3%).',
    '三个判别器对<b>同一批模型输出</b>作答。全部 160,810 条中，官方计为正确 86,090 条（53.5%），判定计为正确 117,805 条（73.3%），抽取计为正确 117,949 条（73.3%）。')}
{F('f8')}
{bi('After correcting for chance: <b>terminal and extraction agree at kappa = 0.926</b>, very high; <b>official agrees with either at only 0.592 / 0.534</b>, moderate. The cross-tables show the structural relation: <b>no record</b> is counted correct by official but wrong by terminal, and only <b>4 records</b> are counted correct by terminal but wrong by extraction.',
    '扣除偶然一致后：<b>判定与抽取 κ=0.926</b>，高度一致；<b>官方与二者仅 0.592 / 0.534</b>，中等一致。交叉表进一步显示结构性关系——官方判对而判定判错的记录为 <b>0 条</b>；判定判对而抽取判错的仅 <b>4 条</b>。')}
{table(["Discriminator<span class='zh'>判别器</span>","Overall accuracy<span class='zh'>总体准确率</span>","kappa vs official<span class='zh'>与官方 κ</span>","kappa vs terminal<span class='zh'>与判定 κ</span>","kappa vs extraction<span class='zh'>与抽取 κ</span>"],
 [["Official native<span class='zh'>官方原生</span>","53.5%","—","0.592","0.534"],["Terminal diagnostic<span class='zh'>裁定终行</span>","73.3%","0.592","—","0.926"],
  ["Answer extraction<span class='zh'>答案抽取</span>","73.3%","0.534","0.926","—"]])}
{bi('The gap is very uneven across benchmarks:', '差异在各评测集上极不均匀：')}
{table(["Benchmark<span class='zh'>评测集</span>","Official<span class='zh'>官方</span>","Terminal<span class='zh'>判定</span>","Extraction<span class='zh'>抽取</span>","Official minus terminal<span class='zh'>官方−判定</span>"],
 [["V*<span class='zh'>vstar</span>","88.0%","88.5%","88.4%","−0.5 pp"],["HRBench-4K<span class='zh'>hrbench-4k</span>","76.5%","83.7%","82.4%","−7.2 pp"],
  ["HRBench-8K<span class='zh'>hrbench-8k</span>","73.2%","80.5%","79.3%","−7.3 pp"],["MME-RealWorld<span class='zh'>mme-realworld</span>","51.7%","73.9%","70.2%","−22.2 pp"],
  ["MME-RealWorld-CN<span class='zh'>mme-realworld-cn</span>","61.5%","70.0%","68.4%","−8.5 pp"],["<b>ZoomBench</b><span class='zh'>zoombench</span>","<b>0.0%</b>","59.1%","59.1%","<b>−59.1 pp</b>"]])}
<div class="key">{bi('<b>Conclusion.</b> Terminal and extraction are effectively the same measurement (kappa = 0.926); the official scorer is <b>not the same measurement</b>. The gap ranges from 0.5 pp to 59.1 pp across benchmarks, which shows that it measures <b>output format</b>, not answer correctness.',
    '<b>结论。</b>判定与抽取几乎是同一测量（κ=0.926）；官方与它们<b>不是同一测量</b>。差异量随评测集从 0.5pp 变化到 59.1pp，说明它测的是<b>输出格式</b>而非答题正确性。')}</div>
</section>

<section class="reveal" id="s3"><h2><span class="no">03</span><span class="t">Measurement validity<span class="zh">测量有效性</span></span></h2>
<p class="q">Can the official score be read directly as accuracy?<span class="zh">官方分数能否直接解释为准确率？</span></p>
{F('f3')}
{bi("Vision-OPD's original rule counts a record correct only when the raw judge field, after trimming, is <b>exactly <code>Yes</code></b>. Of the 160,810 records, <b>74,720 (46.5%)</b> are non-binary 'reasoning + conclusion' text and are counted wrong under that rule; <b>31,715 of them were in fact correct</b>.",
    'Vision-OPD 原规则要求判官原始字段去除首尾空白后<b>严格等于 <code>Yes</code></b> 才计为正确。全量 160,810 条中有 <b>74,720 条（46.5%）</b>属于"推理 + 结论"的非二元文本，按该规则一律计错，其中 <b>31,715 条其实判对了</b>。')}
{F('f4')}
<div class="key">{bi('<b>Identity.</b> <code>Delta official + delta discarded = delta terminal</code> holds exactly on all six benchmarks. Every point the official score gains corresponds to an equal drop in records discarded by the format rule. This is an arithmetic identity, not a statistical estimate.',
    '<b>恒等式。</b><code>Δ官方 + Δ丢弃 = Δ判定</code>在六个评测集上逐条精确成立。官方的每一次"提升"，都等量对应被格式规则丢弃条数的下降。这是算术恒等式，不是统计估计。')}</div>
{F('f7')}
<div class="caveat">{bi('<b>The ZoomBench official score is void.</b> All 845 of its records have non-binary judge output under every condition, so the official score is identically 0.00%, while the terminal diagnostic gives 57.9%-61.2%. All later analyses use terminal/extraction as the primary score sources.',
    '<b>zoombench 官方分数作废。</b>其 845 条记录在全部条件下判官输出均为非二元，官方恒为 0.00%，而终行诊断给出 57.9%–61.2%。后续一律以判定/抽取为主口径。')}</div>
</section>

<section class="reveal" id="s4"><h2><span class="no">04</span><span class="t">Overall results<span class="zh">总体结果</span></span></h2>
<p class="q">What is the accuracy under each condition, and how far apart are the six benchmarks?<span class="zh">五条件各自的准确率是多少？六评测集间差多少？</span></p>
{F('f9')}
{table(["Condition<span class='zh'>条件</span>","Official<span class='zh'>官方</span>","Terminal<span class='zh'>判定</span>","Extraction<span class='zh'>抽取</span>","n"],
 [["S0","47.1%","74.6%","70.8%","32,162"],["SC","50.0%","73.9%","70.3%","32,162"],
  ["S1","56.9%","72.7%","70.1%","32,162"],["S2","55.9%","73.0%","70.3%","32,162"],
  ["S3","57.8%","72.1%","69.8%","32,162"]])}
{bi('The official score rises from 47.1% to 57.8%, while terminal and extraction fall from 74.6% to 72.1% and from 70.8% to 69.8% respectively.',
    '官方口径给出 47.1%→57.8% 的上升，判定与抽取分别给出 74.6%→72.1% 与 70.8%→69.8% 的下降。')}
</section>

<section class="reveal" id="s5"><h2><span class="no">05</span><span class="t">Condition effects<span class="zh">条件效应</span></span></h2>
<p class="q">Do SC, S1, S2 and S3 have a real effect relative to S0? How large, and in which direction?<span class="zh">SC/S1/S2/S3 相对 S0 是否有真实效应？多大？方向？</span></p>
{F('f1')}
{table(["Condition<span class='zh'>条件</span>","Terminal delta<span class='zh'>判定 Δ</span>","95% CI","Extraction delta<span class='zh'>抽取 Δ</span>","95% CI"],
 [["SC","−0.79 pp","[−1.11, −0.47]","−0.49 pp","[−0.81, −0.17]"],
  ["S1","−1.95 pp","[−2.30, −1.59]","−0.67 pp","[−1.00, −0.33]"],
  ["S2","−1.65 pp","[−2.00, −1.31]","−0.50 pp","[−0.85, −0.16]"],
  ["S3","−2.56 pp","[−2.93, −2.19]","−0.97 pp","[−1.32, −0.61]"]])}
{bi('Flip structure (S0 to S3, terminal): wrong to correct <b>1,463 (4.55%)</b>, correct to wrong <b>2,285 (7.10%)</b>, unchanged <b>28,414 (88.35%)</b>. Net change: 822 records.',
    '翻转结构（S0→S3，判定口径）：由错变对 <b>1,463（4.55%）</b>、由对变错 <b>2,285（7.10%）</b>、不变 <b>28,414（88.35%）</b>。净变化 −822 条。')}
<div class="okbox">{bi('<b>Equivalence test (TOST).</b> The difference is -2.56 pp, 90% CI [-2.87, -2.24]. At any margin of +/-0.5, +/-1 or +/-2 pp the test <b>cannot declare equivalence</b>: this is a <b>real negative effect</b>, not a failure to detect one.',
    '<b>等价性检验（TOST）。</b>差 −2.56pp，90% CI [−2.87, −2.24]。在 ±0.5、±1、±2 pp 任何裕度下都<b>不能判定等价</b>——这是一个<b>真实存在的负效应</b>，不是"测不出来"。')}</div>
{F('f2')}
</section>

<section class="reveal" id="s6"><h2><span class="no">06</span><span class="t">Stratification and interaction<span class="zh">分层与交互</span></span></h2>
<p class="q">Does the effect vary by benchmark, category or language?<span class="zh">效应是否随评测集、类别、语言变化？</span></p>
{bi('The effect is negative in all six benchmarks under the terminal score (-1.0 to -3.9 pp), but its magnitude is heterogeneous (I² = 58.9%). The condition-by-benchmark trend lines are nearly parallel (Figure 10), with no visible crossover interaction.',
    '六个评测集的效应方向一致为负（判定口径 −1.0 ~ −3.9pp），但幅度存在异质性（I²=58.9%）。条件 × 评测集的趋势线近乎平行（图10），未见明显的交叉交互。')}
{F('f10')}
<h3>Language contrast<span class="zh">语言对照</span></h3>
{bi('MME-RealWorld (English) and MME-RealWorld-CN (Chinese) are two language versions of the same task:',
    'mme-realworld（英文）与 mme-realworld-cn（中文）是同一任务的两个语言版本：')}
{table(["Benchmark<span class='zh'>评测集</span>","S0","S3","Delta<span class='zh'>Δ</span>"],[["MME-RealWorld (EN)<span class='zh'>英文</span>","75.20%","72.90%","−2.30 pp"],
 ["MME-RealWorld-CN (CN)<span class='zh'>中文</span>","71.61%","67.75%","−3.86 pp"]])}
{bi('<b>The Chinese baseline is lower and is damaged more by prompting</b> (-3.86 vs -2.30 pp).',
    '<b>中文基线更低，且受提示的损伤更大</b>（−3.86 vs −2.30 pp）。')}
<h3>Category stratification<span class="zh">类别分层</span></h3>
{bi('Under-counting relative to the true decision is dominated by category, from 0.9 pp for <code>direct_attributes</code> to 38.9 pp for <code>Perception/Diagram and Tables</code>.',
    '官方相对真实判定的少计幅度由类别主导，从 <code>direct_attributes</code> 的 0.9pp 到 <code>Perception/Diagram and Tables</code> 的 38.9pp。')}
{F('f6')}
</section>

<section class="reveal" id="s7"><h2><span class="no">07</span><span class="t">Output length<span class="zh">输出长度</span></span></h2>
<p class="q">How much do the prompts shorten the output, and how is length related to correctness?<span class="zh">提示把输出压短了多少？长度与正确性是什么关系？</span></p>
{table(["Condition<span class='zh'>条件</span>","Mean tokens<span class='zh'>均值 token</span>","Median<span class='zh'>中位数</span>","P95","Relative to S0<span class='zh'>相对 S0</span>"],
 [["S0","336.9","92","—","—"],["SC","369.4","87","—","+9.6%"],["S1","306.1","60","—","−9.1%"],
  ["S2","331.4","61","—","−1.6%"],["S3","260.3","53","—","−22.7%"]])}
{bi('S3 is <b>22.7%</b> shorter than S0 on average, and the median falls from 92 to 53 tokens.',
    'S3 相对 S0 平均缩短 <b>22.7%</b>，中位数从 92 降到 53。')}
{F('f5')}
<div class="caveat">{bi('<b>Mechanism correction.</b> An earlier report proposed that shorter outputs make the judge more likely to emit a bare Yes, which would explain the rise in the official score. The data do not support this (correlation between length and the non-binary rate is only r = -0.155). The real mechanism is that the judge reasons only on <b>hard or contested</b> items, so the non-binary rate is strongly negatively correlated with accuracy (r = -0.917). <b>Length is a by-product.</b>',
    '<b>机制修正。</b>先前报告曾提出"输出变短使判官更易输出裸 Yes"作为官方分数上升的机制。数据不支持该解释（长度与非二元率相关仅 r=−0.155）。真实机制是判官在<b>难判定或存在争议</b>的题目上才展开推理，故非二元率与正确率强负相关（r=−0.917）。<b>长度只是伴生现象。</b>')}</div>
</section>

<section class="reveal" id="s8"><h2><span class="no">08</span><span class="t">Failure modes<span class="zh">失败模式</span></span></h2>
<p class="q">How does the model fail, and do the prompts change the way it fails?<span class="zh">模型是怎么失败的？提示是否改变失败方式？</span></p>
{F('f11')}
{table(["Condition<span class='zh'>条件</span>","ANSWERED","NO_ANSWER","AMBIGUOUS"],[["S0","98.9%","~0.5%","~0.6%"],
 ["S3","98.9%","~0.5%","~0.6%"]])}
{bi('The extraction-status composition is almost unchanged across the five conditions: ANSWERED stays above 98%. <b>Prompting does not change the shape of failure</b>; it only pushes a few items onto the other side.',
    '五个条件的抽取状态构成几乎不变：ANSWERED 始终 >98%，<b>提示没有改变失败的形态</b>，只是把少数题的答案推到了另一侧。')}
</section>

<section class="reveal" id="s9"><h2><span class="no">09</span><span class="t">Condition redundancy<span class="zh">条件冗余</span></span></h2>
<p class="q">Are S1, S2 and S3 three independent conditions, or near-duplicates?<span class="zh">S1/S2/S3 是三个独立条件，还是近重复？</span></p>
{F('f12')}
{table(["Condition pair<span class='zh'>条件对</span>","Agreement<span class='zh'>一致率</span>","Cohen's kappa"],
 [["S1 vs S2","92.6%","0.813"],["S1 vs S3","91.8%","0.794"],
  ["S2 vs S3","92.2%","0.804"],["SC vs S1","91.5%","0.783"],["S0 vs S3","88.3%","0.702"]])}
{bi('After chance correction, S1/S2/S3 agree at kappa about 0.79-0.81: <b>highly correlated but not redundant</b>, still giving different decisions on about 8-9% of items.',
    '扣除偶然一致后，S1/S2/S3 之间 κ≈0.79–0.81——<b>高度相关但并非冗余</b>，在约 8–9% 的题目上仍给出不同判定。')}
</section>

<section class="reveal" id="s10"><h2><span class="no">10</span><span class="t">Option behaviour<span class="zh">选项行为</span></span></h2>
<p class="q">Do the prompts change which option is chosen, or whether the choice is right?<span class="zh">提示改变的是"选什么"，还是"对不对"？</span></p>
{table(["Condition<span class='zh'>条件</span>","A","B","C","D","E"],
 [["S0","25.2%","25.3%","24.4%","20.3%","3.7%"],["SC","25.6%","25.3%","24.6%","20.3%","3.2%"],
  ["S1","25.6%","25.5%","24.1%","20.5%","3.2%"],["S2","25.8%","25.2%","24.2%","20.3%","3.3%"],
  ["S3","25.1%","25.4%","24.2%","20.8%","3.4%"]])}
<div class="okbox">{bi('<b>No option bias.</b> The option distributions of the five conditions almost coincide (no option moves by more than 1 pp). The accuracy change is not caused by a stronger preference for any letter.',
    '<b>无选项偏置。</b>五个条件的选项分布几乎完全重合（各选项最大变动 &lt;1pp）。提示造成的准确率变化不是由"更爱选某个字母"引起的。')}</div>
</section>

<section class="reveal" id="s11"><h2><span class="no">11</span><span class="t">Effective sample size<span class="zh">有效样本量</span></span></h2>
<p class="q">Are the 32,162 samples really independent units, and are the intervals too narrow?<span class="zh">32,162 个样本真的是独立单位吗？区间是否被低估？</span></p>
{table(["Item<span class='zh'>项</span>","Value<span class='zh'>值</span>"],[["Items under S0<span class='zh'>S0 条件下的题目数</span>","32,162"],["Distinct cluster_uid values<span class='zh'>不同 cluster_uid 数</span>","32,162"],
 ["Largest cluster size<span class='zh'>最大簇大小</span>","1"],["Cluster-size distribution<span class='zh'>簇大小分布</span>","All equal to 1"]])}
<div class="caveat">{bi('<b>Finding.</b> Every <code>cluster_uid</code> in the data contains exactly one item, so the "image-cluster bootstrap" described in v4 <b>degenerates to a per-item bootstrap</b>; image-level correlation is not controlled. If the original benchmarks contain multiple questions per image (common in MME-RealWorld), the current intervals may be too narrow. <b>This must be checked against the original image IDs.</b>',
    '<b>发现。</b>数据中的 <code>cluster_uid</code> 每簇恰好一道题，因此 v4 所称的"图像聚类 Bootstrap"<b>实际退化为逐题 Bootstrap</b>，图像层面的相关性未被控制。若原始 benchmark 存在同图多题（MME-RealWorld 常见），当前区间可能偏窄。<b>此项需用原始图像 ID 核实。</b>')}</div>
</section>

<section class="reveal" id="s12"><h2><span class="no">12</span><span class="t">Design limitations<span class="zh">设计限制</span></span></h2>
<p class="q">What can this design answer, and what can it not?<span class="zh">这个设计能回答什么，不能回答什么？</span></p>
<h3>Unanswerable: whether the prompt effect depends on item difficulty<span class="zh">无法回答：提示效应是否随题目难度变化</span></h3>
{bi('This question needs a <b>difficulty measure that is independent of the treatment</b>, and this design has no repeated measurement of the same item under the same condition. Any difficulty split can only be inferred from performance under some condition and is therefore confounded. Both directions were tried and both failed:',
    '该问题需要一个<b>与治疗无关的难度度量</b>，而本设计没有对同一题、同一条件的重复测量，任何难度划分都只能从某个条件的表现反推，因而必然被污染。两个方向的尝试均失败：')}
{table(["Stratifier<span class='zh'>分层依据</span>","Bias mechanism<span class='zh'>偏差机制</span>","Result<span class='zh'>结果</span>"],
 [["Correct under S0?<span class='zh'>按 S0 是否答对</span>","Shares a term with one side of the comparison, so <b>regression to the mean</b><span class='zh'>与比较的一端同源 → <b>回归均值</b></span>","In the S0-wrong group S3 rises from 0% to 17.9%, which looks like 'prompting rescues hard items' but is an artefact"],
  ["All correct under S1+S2?<span class='zh'>按 S1+S2 是否全对</span>","92% per-item agreement with S3, so <b>S3 outcomes are indirectly selected</b><span class='zh'>与 S3 逐题一致率 92% → <b>间接选中 S3 的结果</b></span>","Easy group: S3 overtakes S0 by 0.5 pp, which is equally not 'prompting leaves easy items unharmed'"],
  ["Define difficulty by SC+S1, compare S2 vs S3<span class='zh'>用 SC+S1 定义难度后比 S2 vs S3</span>","Two treatments from the same family compared, so <b>circular</b><span class='zh'>两个同族治疗互比 → <b>循环论证</b></span>","-1.2 / +0.8 pp, uninformative"]])}
<div class="caveat">{bi('<b>The report therefore makes no directional claim about a difficulty gradient.</b> Both routes push S3 in the favourable direction, which itself shows that the design cannot estimate this effect cleanly. Answering it would require repeated sampling of the same item under the same condition (to estimate noise) or difficulty labels independent of the prompts.',
    '<b>因此本报告不给出任何"难度梯度"的方向性结论。</b>两条路径都把 S3 往好看的方向推，本身就说明该设计无法干净地估计这一效应。若要回答，需要同题同条件的重复采样（估计噪声）或与提示无关的难度标注。')}</div>
<h3>Other limitations<span class="zh">其他限制</span></h3>
<div class="bi"><div class="en" lang="en"><ul>
<li><b>The terminal diagnostic is not a gold standard</b>: it is another parse of the same judge output and shares its origin with the audited object; its kappa of 0.926 against independent extraction is corroboration, not independent validation.</li>
<li><b>Answer extraction is not independent validation either</b>: it is produced by Qwen3.8-27B, the same weights used for the image-assisted audit (report 2225).</li>
<li><b>The human semantic review was not yet complete at the time of the main analysis</b>: material for 60 cases x 5 conditions (300 views) was prepared, but no decisions had been committed.</li>
<li><b>Correlation is not causation.</b></li>
</ul></div><div class="zh" lang="zh-CN"><ul>
<li><b>终行诊断不是金标准</b>：它是对判官输出的另一种解析，与被审计对象同源；其与独立抽取 κ=0.926 提供旁证，但不构成独立验证。</li>
<li><b>答案抽取亦非独立验证</b>：由 Qwen3.8-27B 完成，与图像辅助审计（报告 2225）使用同一权重。</li>
<li><b>人工语义审阅尚未完成</b>：60 例 × 5 条件（300 视图）的离线审阅材料已备齐，但无任何已提交的决策。</li>
<li><b>相关性不等于因果</b>。</li>
</ul></div></div>
</section>

<section class="reveal" id="s13"><h2><span class="no">13</span><span class="t">Discussion and conclusions<span class="zh">讨论与结论</span></span></h2>
<p class="q">What do these results mean for the research question?<span class="zh">这些结果对研究问题意味着什么？</span></p>
<div class="key">{bi('<b>Reportable.</b> (1) Full generation and both discrimination chains passed their gates. (2) The three score sources are not the same measurement and form a strict nesting (official is a subset of terminal, terminal a subset of extraction). (3) Under the metrics that do not depend on output format, increasing prompt strength <b>did not improve</b> answer correctness; all four conditions are slightly below S0 (-0.5 to -2.6 pp), and this is a real negative effect (TOST does not support equivalence). (4) The decline is larger on the Chinese benchmark. (5) Prompting changed neither the option distribution nor the shape of failure.',
    '<b>可以报告。</b>① 全量生成与两条判别链的门禁均通过；② 三条评分来源并非同一测量，构成严格嵌套（官方 ⊂ 判定 ⊂ 抽取）；③ 按不依赖输出格式的口径，提示强度的增加<b>未提升</b>答题正确率，四个条件相对 S0 均为小幅下降（−0.5 ~ −2.6pp），且为真实负效应（TOST 不支持等价）；④ 该下降在中文评测集上更明显；⑤ 提示未改变选项分布，也未改变失败形态。')}</div>
<div class="caveat">{bi('<b>Not reportable.</b> (1) Any statement that prompting improves accuracy: the rise in the official score is shown to be a format artefact. (2) Any conclusion about a prompt-by-difficulty interaction: the design cannot answer it. (3) Extrapolating the 60-case targeted sample to the population accuracy of any benchmark.',
    '<b>不可报告。</b>① 任何"提示提升准确率"的表述——官方口径的上升已证明为格式产物；② 任何"提示与难度交互"的结论——该设计无法回答；③ 将 60 例定向抽样外推为评测集总体准确率。')}</div>
<h3>Relation to prior work<span class="zh">与既有工作的关系</span></h3>
{bi('Vision-OPD scores a record by asking whether the raw judge field is exactly Yes. This analysis shows that, on the Work 5 data, that rule introduces systematic under-counting that depends on both the condition and the category. Reusing it directly would make <b>a change in format compliance</b> look like a change in accuracy.',
    'Vision-OPD 的原有评测口径以"判官原始字段是否严格等于 Yes"计分。本分析表明该口径在工作5 的数据上引入了与条件相关、与类别相关的系统性少计，直接沿用会把<b>格式合规率的变化</b>误读为准确率的变化。')}
<h3>Next steps<span class="zh">下一步</span></h3>
<div class="bi"><div class="en" lang="en"><ol><li>Complete the 60-case human semantic review and obtain human adjudication of the disagreements between the three chains.</li>
<li>Verify the cluster structure with the original image IDs and, if necessary, recompute intervals with images as clusters.</li>
<li>To answer the difficulty question, design a repeated-sampling study on the same items.</li></ol></div>
<div class="zh" lang="zh-CN"><ol><li>完成 60 例人工语义审阅，对三条链的分歧给出人工裁决；</li>
<li>用原始图像 ID 核实聚类结构，必要时以图像为簇重算区间；</li>
<li>若需回答难度问题，设计同题重复采样。</li></ol></div></div>
</section>

<section class="reveal" id="repro"><h2><span class="no">A</span><span class="t">Reproducibility<span class="zh">复现</span></span></h2>
<p class="q">How can every number and figure be rebuilt?<span class="zh">全部数字与图表如何重建？</span></p>
<div class="bi"><div class="en" lang="en"><ul>
<li><code>scripts/analyze_work5_full.py</code>: all statistics in the main analysis</li>
<li><code>scripts/analyze_human_review.py</code>: reference-tier review comparison</li>
<li><code>scripts/plot_score_source_analysis.py</code>, <code>plot_work5_extra.py</code>, <code>plot_human_review.py</code>: all figures</li>
<li><code>scripts/build_work5_report_v2.py</code>: this report</li>
<li>Input: <code>derived/full-statistics-v4/sample_level_merged.csv</code> (160,810 rows, unique key fully aligned)</li>
<li>CJK font: <code>simhei.ttf</code> (provided separately on the server, not distributed with the repository)</li>
</ul>
<p>All images are embedded as base64; the report has no external dependencies and can be read offline.</p></div>
<div class="zh" lang="zh-CN"><ul>
<li><code>scripts/analyze_work5_full.py</code> —— 主分析全部统计量</li>
<li><code>scripts/analyze_human_review.py</code> —— 参考级审阅对照</li>
<li><code>scripts/plot_score_source_analysis.py</code>、<code>plot_work5_extra.py</code>、<code>plot_human_review.py</code> —— 全部图表</li>
<li><code>scripts/build_work5_report_v2.py</code> —— 本报告</li>
<li>输入：<code>derived/full-statistics-v4/sample_level_merged.csv</code>（160,810 行，唯一键完全对齐）</li>
<li>中文字体：<code>simhei.ttf</code>（服务器另行提供，不随仓库分发）</li>
</ul>
<p>本报告全部图像以 base64 内嵌，无外部依赖，可离线阅读。</p></div></div>
</section>

<div class="tier-divider">
 <div class="tier-label">Evidence boundary<span class="zh">证据边界</span></div>
 <div class="tier-text">
  <span lang="en">Everything below is <b>reference-tier evidence (T2)</b>: it comes from one human semantic review. The reviewer's competence has not been formally assessed, the sample is risk-targeted rather than random, and the review page displayed the automatic scores alongside the material.<br>
  <b>No number or conclusion in this part may flow back into, alter or weaken any conclusion in sections 1-13 above.</b> The main analysis rests only on frozen data and automated tests (T1). This part is a <b>cross-reference</b>, not confirmation.</span>
  <span class="zh" lang="zh-CN">以下为<b>参考级证据（T2）</b>：来自一次人工语义审阅。审阅者的判断水平未经正式评估，抽样为风险定向而非随机，且审阅页面同时展示了自动评分。<br>
  <b>本部分的任何数字与结论，均不得回流、修改或削弱上方第 1–13 节的任何结论。</b>主分析仅依据冻结数据与自动化检验（T1）。本部分的作用是<b>交叉参考</b>，不是确认。</span>
 </div>
 <div class="tier-meta">T2 · REFERENCE ONLY · NOT CONFIRMATORY</div>
</div>

<section class="reveal" id="s14"><h2><span class="no">14</span><span class="t">Reference: human review against the three discriminators<span class="zh">参考：人工审阅与三条判别器的对照</span></span></h2>
<p class="q">[Reference] How closely does human judgement agree with the three automatic score sources, and where do they diverge?<span class="zh">[参考] 人工判断与三条自动口径的一致程度如何？分歧集中在哪？</span></p>

<h3>What this audit is<span class="zh">这次审计是什么</span></h3>
{bi('All three score sources in the main analysis are automatic. At several records they disagree, <b>and no fourth source says which of them is wrong</b>. The human review introduces exactly that external view.',
    '主分析的三条评分来源都是自动化的。它们在若干处给出不同结论，<b>但没有第四条信息源说明哪一条错了</b>。人工审阅就是引入这样一个外部视角。')}
<table class="content"><thead><tr><th>Item<span class="zh">项</span></th><th>Content<span class="zh">内容</span></th></tr></thead><tbody>
<tr><td>Material<span class="zh">材料</span></td><td>Offline self-contained review page: original image, question and options, reference answer, raw model responses under all five conditions, the three automatic scores with their evidence<span class="zh">离线自包含审阅页：原始图像、题目选项、参考答案、五条件模型原始回答、三种自动评分及证据</span></td></tr>
<tr><td>Sampling<span class="zh">抽样</span></td><td>10 cases per benchmark, 60 cases in total; five conditions per case, <b>300 views</b><span class="zh">六评测集各 10 例，共 60 例；每例五条件，共 <b>300 个视图</b></span></td></tr>
<tr><td>Sampling design<span class="zh">抽样方式</span></td><td><b>Risk-targeted, not random</b>: official/extraction disagreements, official/terminal disagreements, ambiguous or no-answer cases and the longest outputs per benchmark were prioritised; an "all three agree" control stratum was added<span class="zh"><b>风险定向</b>，非随机：优先纳入官方/抽取分歧、官方/终行分歧、歧义或无答案、以及各集最长输出，另设"三项一致"对照层</span></td></tr>
<tr><td>Judgements<span class="zh">判断项</span></td><td>Extraction faithful / answer semantically correct / judge rationale and label valid (yes, no, unclear each)<span class="zh">抽取是否忠实 / 答案语义是否正确 / 裁判理由与标签是否有效（各为 是·否·不确定）</span></td></tr>
</tbody></table>
<h3>Who did what<span class="zh">谁做了什么</span></h3>
<table class="content"><thead><tr><th>Role<span class="zh">执行者</span></th><th>Work performed<span class="zh">工作内容</span></th></tr></thead><tbody>
<tr><td><b>External auditor</b><span class="zh">外部审计员</span></td><td>Completed the three judgements for <b>293</b> views<span class="zh">完成 <b>293</b> 个视图的三项判断</span></td></tr>
<tr><td><b>Project owner (experiment lead)</b><span class="zh">项目所有者（实验负责人）</span></td><td>Completed <b>7</b> views (Case 1·SC, Case 51·SC, all five conditions of Case 55); changed the judge-validity verdict for Case 46·S0 from "no" to "unclear" with a note; personally spot-checked and found no extraction errors<span class="zh">补全 <b>7</b> 个视图（Case 1·SC、Case 51·SC、Case 55 全部五条件）；将 Case 46·S0 的裁判有效性由"否"改为"无法判断"并附说明；亲自抽检未发现抽取错误</span></td></tr>
</tbody></table>
<h3>Audit workflow, end to end<span class="zh">审计工作流（端到端）</span></h3>
<table class="content"><thead><tr><th>Step<span class="zh">步骤</span></th><th>Content<span class="zh">内容</span></th><th>By<span class="zh">执行者</span></th></tr></thead><tbody>
<tr><td>1 · Sampling<span class="zh">抽样</span></td><td>Deterministically draw 60 cases from the 32,162 items by risk stratum (fixed seed 20261016), keeping all five conditions per case<span class="zh">按风险层从 32,162 题中确定性抽取 60 例（固定随机种子 20261016），每例保留五条件</span></td><td>Automated<span class="zh">自动化</span></td></tr>
<tr><td>2 · Material generation<span class="zh">材料生成</span></td><td>Render images, questions, reference answers, the five raw model responses and the three scoring evidence sets into a single offline HTML file<span class="zh">将图像、题目、参考答案、五条件模型回答与三种评分证据渲染为单文件离线 HTML</span></td><td>Automated<span class="zh">自动化</span></td></tr>
<tr><td>3 · Review distribution<span class="zh">分发审阅</span></td><td>The reviewer opens the page and works card by card: <b>image, then question and options, then reference answer, then the raw model response for that condition, then the three scores and their evidence</b><span class="zh">审阅者在浏览器打开页面，逐卡查看：<b>原图 → 题目与选项 → 参考答案 → 该条件的模型原始回答 → 三种评分及其证据</b></span></td><td>Auditor<span class="zh">审计员</span></td></tr>
<tr><td>4 · Item judgements<span class="zh">逐项判断</span></td><td>Three judgements per view: extraction faithful, semantics correct, judge valid; <b>any no or unclear required a written evidence note</b><span class="zh">对每个视图回答三项：抽取是否忠实、语义是否正确、裁判是否有效；<b>任一"否/不确定"须写证据说明</b></span></td><td>Auditor<span class="zh">审计员</span></td></tr>
<tr><td>5 · Export<span class="zh">导出</span></td><td>Export <code>work5-manual-review-decisions-v1.json</code> with the page button (browser-local state is not an archive; it must be exported)<span class="zh">点击页面按钮导出 <code>work5-manual-review-decisions-v1.json</code>（浏览器本地状态不是归档，必须导出）</span></td><td>Auditor<span class="zh">审计员</span></td></tr>
<tr><td>6 · Completion<span class="zh">补全</span></td><td>The 7 views the auditor left blank were completed by the experiment lead under the same standard; the Case 46·S0 judge verdict was revised on review<span class="zh">审计员未填的 7 个视图由实验负责人按同一标准补全；Case 46·S0 的裁判判定经复核修正</span></td><td>Experiment lead<span class="zh">实验负责人</span></td></tr>
<tr><td>7 · Normalisation<span class="zh">规范化</span></td><td>Unify value formats and repair the <code>case_title</code> and <code>condition</code> fields damaged by the translation tool (no judgement content changed)<span class="zh">统一取值格式、修复被翻译工具损坏的 <code>case_title</code> 与 <code>condition</code> 字段（不改变任何判断内容）</span></td><td>Automated<span class="zh">自动化</span></td></tr>
<tr><td>8 · Validation<span class="zh">校验</span></td><td><code>validate_manual_review.py</code> checks 300-row coverage, legal field values, notes on negative judgements, and hashes of the three files<span class="zh"><code>validate_manual_review.py</code> 核对 300 行覆盖、字段合法、否定项附说明、三份文件哈希</span></td><td>Automated<span class="zh">自动化</span></td></tr>
<tr><td>9 · Archiving<span class="zh">归档</span></td><td>Delivery version, raw export, validation record and provenance note committed together<span class="zh">交付版本、原始导出、校验记录与来源说明一并入库</span></td><td>Automated<span class="zh">自动化</span></td></tr>
</tbody></table>
{bi('<b>Key design.</b> The "any negative judgement needs a note" rule from step 4 was narrowed after step 7: only <b>extraction faithfulness</b> and <b>judge validity</b> require notes. <code>semantic_correct = no</code> is treated as a directly observable fact (the answer does not match the reference) and no longer requires a note per case. The amendment is on record.',
    '<b>关键设计</b>：第 4 步的"任一否定须附说明"规则在第 7 步后被收窄——仅<b>抽取忠实度</b>与<b>裁判有效性</b>强制要求说明，<code>semantic_correct = 否</code> 视为可直接观察的事实（答案与参考答案不符），不再逐条要求说明。该修订已记录在案。')}
{bi("<b>The auditor's reading procedure</b>: read the raw model output, decide the final option independently, then compare with the extraction; if they agree, mark 'extraction faithful = yes'. <b>Independent judgement first, comparison second</b>, so the agreement rate reflects the actual accuracy of the extraction chain rather than a copy of the score shown on the page.",
    '<b>审计员的判读流程</b>：阅读模型原始输出 → 自行判断其最终结论选项 → 与抽取结果比对 → 一致则记"抽取忠实 = 是"。<b>先独立判断、再比对</b>，因此一致率反映抽取链的实际准确度，而不是照抄页面上的评分。')}
{bi('Of the 300 views, <b>284</b> support a binary semantic judgement; 16 are marked "unclear", all with an image-quality note ("the original is illegible", "very dark" and similar), and are excluded from the agreement calculation.',
    '300 个视图中，<b>284 个</b>的"语义是否正确"可二分类；16 个判为"不确定"，均附有图像质量说明（"原图看不清楚""天色黑很"等），不进入一致率计算。')}

{F('f13')}
{table(["Discriminator<span class='zh'>判别器</span>","Agreement with human<span class='zh'>与人工一致率</span>","Cohen's kappa","Sensitivity<span class='zh'>敏感度</span>","Specificity<span class='zh'>特异度</span>","Human correct / auto wrong<span class='zh'>人工对/自动错</span>","Human wrong / auto correct<span class='zh'>人工错/自动对</span>"],
 [["Official native<span class='zh'>官方原生</span>","74.6%","0.492","68.2%","88.8%","<b>62</b>","10"],
  ["Terminal diagnostic<span class='zh'>裁定终行</span>","96.1%","0.907","100.0%","87.6%","0","11"],
  ["<b>Answer extraction</b><span class='zh'>答案抽取</span>","<b>100.0%</b>","<b>1.000</b>","100.0%","100.0%","<b>0</b>","<b>0</b>"]])}

<div class="key">{bi('<b>[Reference] Central agreement.</b> In all <b>72</b> official-extraction disagreements (62 + 10), the human semantic judgement <b>sides with extraction every time</b>, without exception. This is consistent with the direction of section 3: the official scorer systematically under-counts.',
    '<b>[参考] 核心一致。</b>在官方与抽取全部分歧的 <b>72 处</b>（62 + 10），人工的语义判断<b>每一处都与抽取一致</b>，无例外。这与主分析第 3 节的结论方向一致：官方口径存在系统性少计。')}</div>

<h3>By condition<span class="zh">按条件</span></h3>
{F('f14')}
{table(["Condition<span class='zh'>条件</span>","n","Human correct<span class='zh'>人工判对</span>","Official<span class='zh'>官方</span>","Terminal<span class='zh'>判定</span>","Extraction<span class='zh'>抽取</span>"],
 [["S0","57","39","25","44","39"],["SC","57","39","24","41","39"],["S1","56","38","31","40","38"],
  ["S2","57","39","31","40","39"],["S3","57","40","32","41","40"]])}
{bi('The official scorer is clearly below the human in all five conditions; terminal and extraction essentially coincide with the human.',
    '官方在全部五个条件下均明显低于人工；判定与抽取与人工基本重合。')}

<h3>Direction of disagreement<span class="zh">分歧方向</span></h3>
{F('f15')}
{bi('The official scorer is judged by the human to have <b>missed 62 correct answers</b>, with only <b>10</b> in the opposite direction - a 6:1 gap. This is consistent with under-counting by the format rule rather than random error.',
    '官方被人工判为"漏判为错"的有 <b>62</b> 例，反方向仅 <b>10</b> 例——差距 6:1，与"格式规则少计"的方向一致，而非随机误差。')}

<h3>Condition effect observed by the human reviewer<span class="zh">人工观察到的条件效应</span></h3>
{bi('Pairing the human semantic judgements from S0 to S3: n = 57, 4 improved and 3 worsened, <b>delta = +1.8 pp, McNemar p = 1</b>. <b>No accuracy change from prompting was observed</b> - not inconsistent with the "no robust improvement" conclusion of section 5, though the sample is very small and risk-targeted, so it cannot independently support that conclusion.',
    '按人工语义判断做 S0→S3 配对：n=57，改善 4、恶化 3，<b>Δ=+1.8 pp，McNemar p=1</b>。<b>未观察到提示带来的准确率变化</b>——与主分析第 5 节"无稳健提升"的结论不矛盾（但样本量极小，且为定向抽样，不足以独立支持该结论）。')}

<h3>Anomalies flagged by the auditor<span class="zh">审计员标记的异常</span></h3>
{bi('Judge rationale or label judged invalid or not assessable: <b>13</b> cases. Extraction faithfulness judged no or unclear: <b>3</b> cases. Several views of Case 46 and Case 55 involve model outputs that looped extensively, ran to tens of thousands of tokens and were truncated without a final answer; the reviewer marked judge validity "not assessable" with a note.',
    '裁判理由/标签被判为无效或无法评估：<b>13</b> 例；抽取忠实度被判为否或不确定：<b>3</b> 例。其中 Case 46 与 Case 55 的若干视图涉及模型输出大量循环重复、上万 token 后截断、未给出最终答案，审阅者将其裁判有效性标为"无法评估"并附说明。')}

<h3>In depth: five questions raised by the audit and their answers<span class="zh">深入分析：审计提出的五个问题与结论</span></h3>
<table class="content"><thead><tr><th>Question<span class="zh">问题</span></th><th>Answer<span class="zh">结论</span></th></tr></thead><tbody>
<tr><td>Are extraction errors concentrated in table tasks?<span class="zh">抽取错误是否集中于表格任务</span></td><td><b>No.</b> Three audited cases happen to be in HRBench, but of all 4,860 "terminal correct / extraction wrong" records, <b>88% are in MME-RealWorld</b> (3.62% of that benchmark); HRBench accounts for only 2.0%.<span class="zh"><b>否</b>。审计中 3 例恰在 hrbench，但全量"判定对/抽取错"的 4,860 例中 <b>88% 在 mme-realworld</b>（3.62%），hrbench 仅占 2.0%。</span></td></tr>
<tr><td>Do judge and extraction failures share the same origin?<span class="zh">裁判与抽取失败是否同源</span></td><td><b>Yes.</b> <b>98.2%</b> of the disagreements are "extraction reads an explicit option that does not match the reference, while the judge still says Yes": systematic judge over-counting.<span class="zh"><b>是</b>。分歧中 <b>98.2%</b> 属于"抽取读出明确选项、与标准答案不符，而裁判仍判 Yes"——裁判系统性过度判对。</span></td></tr>
<tr><td>Are the "unclear" cases objectively unjudgeable?<span class="zh">判"不确定"的案例是否客观不可判</span></td><td><b>Partly.</b> The 16 "unclear" views concentrate in 4 cases: one image with luminance 17/255 (objectively very dark), two 36-39 MP images (the review preview necessarily loses detail), and one normal image where the judgement is subjective.<span class="zh"><b>部分成立</b>。16 个"不确定"集中在 4 例：1 例图像亮度仅 17/255（客观极暗）、2 例为 36–39 MP 超大图（审阅预览必失细节）、1 例图像正常而判断主观。</span></td></tr>
<tr><td>Is the English MME significantly harder?<span class="zh">英文 MME 是否显著更难</span></td><td><b>No.</b> The 24% vs 84% in the audited sample comes from <b>different stratum distributions</b> (the English sample has 10 more hard-stratum cases and 15 fewer control cases). By secondary category, the Chinese version is generally lower or equal.<span class="zh"><b>否</b>。审计样本的 24% vs 84% 源于<b>抽样层分布不同</b>（英文样本多 10 例硬层、少 15 例对照层）。按二级类别比较，中文版本普遍更低或持平。</span></td></tr>
<tr><td>How widespread are image-quality problems?<span class="zh">图像质量问题的普遍程度</span></td><td>100 images sampled per benchmark: very dark images are about 1%, not systematic; <b>HRBench-8K has a median resolution of 38.8 MP and a maximum of 59 MP</b>, the direct source of its review difficulty.<span class="zh">各集抽 100 图：极暗图像约 1%，非系统性问题；<b>hrbench-8k 分辨率中位 38.8 MP、最大 59 MP</b>，是其审阅困难的直接来源。</span></td></tr>
</tbody></table>

<h3>Two audit-derived findings that can be verified independently<span class="zh">两项由审计引出、可独立核验的发现</span></h3>
<div class="key">{bi('<b>(1) The S0-to-S3 drop under the terminal score is inflated.</b> The judge over-count rate varies by condition: S0 <b>3.89%</b> to S3 <b>2.30%</b>. Correcting for it gives <code>70.76% to 69.79% (delta = -0.97 pp)</code>, exactly matching answer extraction; the raw terminal score gives <code>74.65% to 72.09% (delta = -2.56 pp)</code>. <b>The direction is unchanged (no improvement), but the effect size should be taken as -0.97 pp.</b>',
    '<b>① 判定口径的 S0→S3 降幅被夸大。</b>裁判的过度判对率随条件变化：S0 <b>3.89%</b> → S3 <b>2.30%</b>。扣除后：<code>70.76% → 69.79%（Δ = −0.97pp）</code>，与答案抽取口径完全一致；而原始判定口径给出的是 <code>74.65% → 72.09%（Δ = −2.56pp）</code>。<b>方向不变（无提升），但效应量应以 −0.97pp 为准。</b>')}</div>
<div class="key">{bi('<b>(2) The official rule selects precisely the least reliable subset of judge decisions.</b> Splitting the over-count rate by judge output form: <b>bare <code>Yes</code></b> is <b>5.55%</b> (4,776/86,090), <b>non-binary output containing reasoning</b> only <b>0.26%</b> (84/31,715) - a <b>21-fold difference</b>. Because the official rule accepts only bare <code>Yes</code>, it does not merely measure format: it <b>systematically selects the subset with the highest misjudgement rate</b>. This gives a deeper mechanism for the "the official score is a format artefact" conclusion of section 3.',
    '<b>② 官方口径恰好筛出了裁判最不可靠的那批判定。</b>按裁判输出形态拆分过度判对率：<b>裸 <code>Yes</code></b> 为 <b>5.55%</b>（4,776/86,090），<b>含推理的非二元输出</b> 仅 <b>0.26%</b>（84/31,715）——<b>差 21 倍</b>。官方规则只承认裸 <code>Yes</code>，因此它不只是"测格式"，而是<b>系统性地选中了误判率最高的子集</b>。这为第 3 节的"官方是格式产物"提供了更深一层的机制解释。')}</div>

<div class="caveat">{bi('<b>[Reference] What this part cannot show.</b><ul><li><b>It cannot</b> validate the accuracy of the extraction chain: although the agreement rate is 100%, the sample was drawn specifically around such disagreements and is not random.</li><li><b>It cannot</b> be extrapolated to the population accuracy of any benchmark.</li><li><b>It cannot</b> replace the automated tests of the main analysis, and it does not change any conclusion in sections 1-13.</li></ul>Its value is this: <b>from a viewpoint independent of the official score, no evidence contrary to the main conclusions was found.</b>',
    '<b>[参考] 本部分不能证明什么。</b><ul><li><b>不能</b>用来验证抽取链的准确率——虽然一致率为 100%，但样本是为此类分歧而定向抽取的，不是随机样本；</li><li><b>不能</b>外推为任何评测集的总体准确率；</li><li><b>不能</b>替代主分析的自动化检验，也不改变第 1–13 节的任何结论。</li></ul>其价值在于：<b>在一个独立于官方口径的视角下，未发现与主结论相反的证据。</b>')}</div>
</section>

<section class="reveal" id="closing"><h2><span class="no">∞</span><span class="t">Closing remarks<span class="zh">结束语</span></span></h2>
<p class="q">What does this report ultimately establish?<span class="zh">这份报告最终确认了什么？</span></p>
{bi('The Work 5 computation chain is fully closed: 160,810 condition-sample records, both automatic discrimination chains passing 30/30 gates, and the v4 statistics with 127 source-hash checks complete. On that basis the report answers, through diagnostic decomposition, why the three score sources disagree, and introduces one human review at the reference tier as an external cross-check.',
    '工作5 的计算链已完整闭环：160,810 条条件—样本记录、两条自动判别链 30/30 通过门禁、v4 统计与 127 项源哈希校验完成。在此基础上，本报告用诊断性分解回答了"三种评分来源为何分歧"，并在参考层面引入了一次人工审阅作为外部对照。')}
{bi('<b>Three statements can be made safely:</b>', '<b>可以稳妥地说的是三句话：</b>')}
<div class="bi"><div class="en" lang="en"><ol>
<li><b>The official score cannot be read directly as accuracy.</b> It accepts only bare <code>Yes</code>, and bare <code>Yes</code> has a misjudgement rate <b>21 times</b> that of outputs containing reasoning; 46.5% of records are counted wrong on format grounds, and many of the correct ones among them are "recovered" under S3, producing the <b>+10.68 pp artefact</b>.</li>
<li><b>Increasing prompt strength did not improve answer correctness.</b> Both format-independent metrics agree on a negative effect: terminal -2.56 pp, extraction -0.97 pp; after audit correction the cleanest estimate is <b>-0.97 pp</b>. TOST rules out equivalence to no difference.</li>
<li><b>The unresolved questions matter just as much.</b> The difficulty gradient cannot be answered under this design (no repeated measurement of the same item under the same condition); the image resolution actually reaching the model has not been verified - if the 8K images of HRBench-8K are downsampled in preprocessing, their "high-definition" property is unreachable for the model.</li>
</ol></div>
<div class="zh" lang="zh-CN"><ol>
<li><b>官方口径不能直接读作准确率。</b>它只承认裸 <code>Yes</code>，而裸 <code>Yes</code> 的误判率是含推理输出的 <b>21 倍</b>；其 46.5% 的记录因格式被计为错误，其中的正确项在 S3 下被大量"找回"，形成了 <b>+10.68pp 的假象</b>。</li>
<li><b>提示强度的增加没有提升答题正确率。</b>两条不依赖输出格式的口径一致为负：判定 −2.56pp、抽取 −0.97pp；经审计校正后，最干净的估计是 <b>−0.97pp</b>。TOST 检验排除了"等价于无差异"。</li>
<li><b>仍未解决的问题同样重要。</b>难度梯度在该设计下无法回答（无同题同条件重复测量）；模型实际接收的图像分辨率尚未核实——若 hrbench-8k 的 8K 图像在预处理中被降采样，其"高清"属性对模型不可达。</li>
</ol></div></div>
{bi('All main conclusions of this report (sections 1-13) rest only on frozen data and automated tests. Section 14 and everything after the evidence boundary are <b>for reference only and do not enter the conclusions</b>.',
    '本报告的所有主结论（第 1–13 节）仅依据冻结数据与自动化检验。第 14 节及此后位于证据边界之下的内容，<b>仅作参考，不参与结论</b>。')}
</section>

</main>
<footer>
<p>Work 5 full evaluation report · 2026-10-17 · corresponds to progress reports 2233 / 2234<span class="zh">工作5 完整评测报告 · 2026-10-17 · 对应进展报告 2233 / 2234</span></p>
<p>Data basis: the frozen per-item paired table; every comparison is a paired comparison of the same item.<span class="zh">数据口径：冻结的逐题配对数据；所有比较为同题配对比较。</span></p>
</footer>
<div class="progress" id="bar"></div>
"""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Work 5 full evaluation report · Full-benchmark evaluation under five prompt conditions / 工作5 完整评测报告</title>
<style>{CSS}</style></head><body>{body}<script>{JS}</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"PASS: {args.output} ({args.output.stat().st_size:,} bytes, {len(FIGS)} 图)")


if __name__ == "__main__":
    main()
