import re, os

base = 'demo/分部/module3'
out_path = os.path.join(base, 'module3.html')

# ── Extract all sections from each part ──
all_sections = {}
for i in range(11):
    path = os.path.join(base, f'part{i}', f'part{i}.html')
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = re.findall(r'<section[^>]*>.*?</section>', content, re.DOTALL)
    all_sections[i] = sections

# ── Part labels ──
part_labels = {
    0: ("核心问题与研究动机", "三大瓶颈 · OPD核心公式 · 特权信息", "#e17055"),
    1: ("多模态OPD范式", "文本vs多模态 · 四大范式 · 时间线", "#7c5ce7"),
    2: ("视觉域：VLM的OPD/OPSD", "VOLD · Vision-OPD · ViGOS · V-Zero · PeRL-VL", "#a855f7"),
    3: ("视频域：时序OPD", "Video-OPD · PF-OPSD · TVDF", "#fdcb6e"),
    4: ("音频/语音域：跨模态OPD", "X-OPD · CORD · SightSound-R1 · Nemotron Audex", "#00cec9"),
    5: ("Agent与具身智能", "VLA-OPD · M2Distill · ACGD", "#e17055"),
    6: ("RL+OPD混合系统", "PRISM · PTD-PO · HDPO · Distill-R1", "#fdcb6e"),
    7: ("扩散与生成", "D-OPSD · LiveTalk · 通用分布匹配", "#00cec9"),
    8: ("新兴方向", "H-OPD · RAL · 世界模型+OPD · OmniOPSD", "#a855f7"),
    9: ("工业落地与基准", "工业部署总览 · 主要基准", "#7c5ce7"),
    10: ("开放问题与未来方向", "理论/工程/前沿 · 论文索引", "#e17055"),
}

# ── CSS ──
css = '''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>Module 3 · 多模态OPD研究全景</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.min.css"><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/black.min.css">
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]}};</script>
<style>
:root{--bg:#030206;--surface:rgba(18,16,28,0.7);--border:rgba(124,92,231,0.15);--text:#e0dce8;--text2:rgba(210,200,230,0.7);--text3:rgba(160,145,190,0.45);--core:#7c5ce7;--flare:#e17055;--fuse:#a855f7;--beam:#fdcb6e;--nova:#00cec9}
#bg-canvas{position:fixed;inset:0;z-index:0;pointer-events:none}.reveal{background:transparent}.reveal .slides{text-align:center;width:100%!important;left:50%!important;margin:0!important}.reveal .slides section{padding:28px 48px;height:100%;display:flex!important;flex-direction:column;align-items:center;justify-content:center}.reveal .controls{right:12px!important;bottom:12px!important}#page-indicator{position:fixed;left:24px;bottom:20px;z-index:100;font-size:13px;color:rgba(255,255,255,0.35);font-family:'JetBrains Mono','Fira Code','Consolas',monospace;letter-spacing:1px;pointer-events:none}
.core-card{position:relative;z-index:2;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:36px 46px;max-width:860px;box-shadow:0 8px 32px rgba(0,0,0,0.6)}.core-card::before{content:'';height:2px;background:linear-gradient(90deg,transparent,var(--core),var(--flare),var(--nova),transparent);display:block;margin-bottom:12px}.hero-title{font-size:46px;font-weight:800;background:linear-gradient(135deg,var(--core),var(--flare),var(--nova));background-size:300%300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:gS 5s ease infinite}@keyframes gS{0%,100%{background-position:0%50%}50%{background-position:100%50%}}.core-bar{width:100px;height:2px;background:linear-gradient(90deg,transparent,var(--core),var(--flare),var(--nova),transparent);margin:14px auto}
.struct-content{font-size:13px;color:var(--text2);line-height:2;text-align:left;max-width:780px}.struct-content strong{color:#fff}
.two-col{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;max-width:940px}.two-col>div{flex:1;min-width:260px}.tri-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;max-width:940px}
.concept-block{background:rgba(20,18,30,0.5);border:1px solid var(--border);border-radius:10px;padding:16px 14px;text-align:left}.concept-block h4{font-size:13px;color:var(--core);margin:0 0 5px}.concept-block p{font-size:11px;color:var(--text2);line-height:1.7;margin:0}
.def-box{background:rgba(124,92,231,0.06);border:1px solid rgba(124,92,231,0.2);border-left:3px solid var(--core);border-radius:0 10px 10px 0;padding:14px 20px;text-align:left;max-width:750px;margin:8px 0}.def-box .label{font-size:11px;color:var(--core);font-weight:700;margin-bottom:4px}.def-box .content{font-size:12px;color:var(--text);line-height:1.8}
.warn-box{background:rgba(225,112,85,0.06);border:1px solid rgba(225,112,85,0.2);border-left:3px solid var(--flare);border-radius:0 10px 10px 0;padding:14px 20px;text-align:left;max-width:750px;margin:8px 0}.warn-box .label{font-size:11px;color:var(--flare);font-weight:700;margin-bottom:4px}.warn-box .content{font-size:12px;color:var(--text);line-height:1.8}
.green-box{background:rgba(0,206,201,0.06);border:1px solid rgba(0,206,201,0.2);border-left:3px solid var(--nova);border-radius:0 10px 10px 0;padding:14px 20px;text-align:left;max-width:750px;margin:8px 0}.green-box .label{font-size:11px;color:var(--nova);font-weight:700;margin-bottom:4px}.green-box .content{font-size:12px;color:var(--text);line-height:1.8}
.formula-box{background:rgba(8,6,15,0.8);border:1px solid rgba(124,92,231,0.2);border-radius:10px;padding:18px 24px;max-width:780px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.5);margin:8px 0}.formula-box .math-block{font-size:15px;color:#d4c8b0}
.summary-table{width:100%;max-width:820px;border-collapse:collapse;font-size:11px;border-radius:10px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,0.5)}.summary-table thead{background:linear-gradient(135deg,rgba(124,92,231,0.25),rgba(225,112,85,0.12))}.summary-table th{padding:9px 12px;font-weight:700;color:var(--beam);font-size:10px}.summary-table td{padding:8px 12px;border-bottom:1px solid var(--border);color:var(--text2)}.summary-table tbody tr{background:rgba(20,18,30,0.4)}.summary-table .pc{font-weight:700;color:var(--beam)}.summary-table .hl{color:var(--nova);font-weight:600}.summary-table .wd{color:var(--flare);font-weight:600}
.page-badge{display:inline-flex;padding:4px 14px;border-radius:14px;font-size:10px;font-weight:700;background:rgba(124,92,231,0.08);color:var(--core);border:1px solid rgba(124,92,231,0.2);letter-spacing:1px}.cite{font-size:10px;color:var(--core);font-family:monospace}
.timeline-v{position:relative;padding-left:28px;text-align:left;max-width:820px}.timeline-v::before{content:'';position:absolute;left:10px;top:4px;bottom:4px;width:2px;background:linear-gradient(180deg,var(--flare),var(--beam),var(--core),var(--nova));border-radius:1px}.timeline-item{position:relative;margin-bottom:8px;padding:8px 14px;background:rgba(20,18,30,0.4);border-radius:8px;border:1px solid var(--border)}.timeline-item::before{content:'';position:absolute;left:-23px;top:12px;width:9px;height:9px;background:var(--beam);border-radius:50%;border:2px solid rgba(0,0,0,0.6)}.timeline-item .yr{font-weight:800;color:var(--flare);font-size:10px;letter-spacing:1px}.timeline-item .ev{font-size:11px;color:var(--text2);line-height:1.6}.timeline-item .ev strong{color:#fff}
.step-flow{display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;max-width:900px;margin:8px 0}.step{background:rgba(20,18,30,0.5);border:1px solid var(--border);border-radius:8px;padding:10px 14px;text-align:center;min-width:80px;flex:1}.step .sn{font-size:9px;color:var(--core);font-weight:700;margin-bottom:3px}.step .st{font-size:11px;color:#fff;font-weight:600}.step .sd{font-size:9.5px;color:var(--text2);line-height:1.5}.step-arrow{color:var(--text3);font-size:16px;font-weight:700}
.paradigm-card{background:linear-gradient(135deg,rgba(124,92,231,0.06),rgba(0,206,201,0.03));border:1px solid rgba(124,92,231,0.18);border-radius:10px;padding:18px 20px;text-align:left;margin-bottom:10px}.paradigm-card .n{font-size:11px;color:var(--fuse);font-weight:800;margin-bottom:4px}.paradigm-card .t{font-size:14px;font-weight:700;color:#fff;margin:4px 0}.paradigm-card .d{font-size:11px;color:var(--text2);line-height:1.7}.paradigm-card .papers{font-size:10px;color:var(--nova);margin-top:4px;font-family:monospace}
.paper-card{background:linear-gradient(135deg,rgba(124,92,231,0.05),rgba(0,206,201,0.03));border:1px solid rgba(124,92,231,0.15);border-radius:10px;padding:16px 18px;text-align:left;margin-bottom:8px}.paper-card .tag{font-size:9px;font-weight:800;padding:2px 8px;border-radius:8px;letter-spacing:1px;display:inline-block;margin-bottom:6px}.paper-card .t{font-size:14px;font-weight:700;color:#fff;margin:4px 0}.paper-card .d{font-size:11px;color:var(--text2);line-height:1.7}
.bottleneck-card{background:linear-gradient(135deg,rgba(225,112,85,0.08),rgba(124,92,231,0.04));border:1px solid rgba(225,112,85,0.2);border-radius:10px;padding:18px 20px;text-align:left;margin-bottom:10px}.bottleneck-card .n{font-size:11px;color:var(--flare);font-weight:800;margin-bottom:4px}.bottleneck-card .t{font-size:14px;font-weight:700;color:#fff;margin:4px 0}.bottleneck-card .d{font-size:11px;color:var(--text2);line-height:1.7}
.privilege-card{background:linear-gradient(135deg,rgba(0,206,201,0.06),rgba(168,85,247,0.04));border:1px solid rgba(0,206,201,0.18);border-radius:10px;padding:18px 20px;text-align:left;margin-bottom:10px}.privilege-card .lvl{font-size:10px;color:var(--nova);font-weight:700;margin-bottom:3px}.privilege-card .name{font-size:14px;font-weight:700;color:#fff;margin:3px 0}.privilege-card .desc{font-size:11px;color:var(--text2);line-height:1.7}.privilege-card .risk{font-size:10px;color:var(--flare);margin-top:4px}
.quad-grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:10px;max-width:800px;height:320px}.quad-cell{background:rgba(20,18,30,0.5);border:1px solid var(--border);border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;transition:all 0.25s}.quad-cell:hover{border-color:rgba(124,92,231,0.4)}.quad-cell .ql{font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:4px}.quad-cell .qt{font-size:13px;font-weight:700;color:#fff;margin-bottom:3px}.quad-cell .qd{font-size:9.5px;color:var(--text2);line-height:1.5}
.reveal .slides section .fragment.fade-up{opacity:0;transform:translateY(30px);transition:all 0.7s}.reveal .slides section .fragment.fade-up.visible{opacity:1;transform:translateY(0)}
@media(max-width:768px){.hero-title{font-size:28px}.tri-grid{grid-template-columns:1fr}.two-col{flex-direction:column}.quad-grid{grid-template-columns:1fr;height:auto}.step-flow{flex-direction:column}.step-arrow{transform:rotate(90deg)}}
</style></head><body><canvas id="bg-canvas"></canvas><div id="page-indicator"></div><div class="reveal"><div class="slides">
'''

# ── Module title page ──
title_page = '''<!-- ═══════════ MODULE 3 标题页 ═══════════ -->
<section data-transition="zoom"><div class="core-card fragment fade-up"><p class="cite">MODULE 3</p><h1 class="hero-title">多模态 OPD 研究全景</h1><p style="font-size:20px;color:var(--text2);">Multimodal Online Policy Distillation · 2025 → 2026</p><div class="core-bar"></div><p style="font-size:14px;color:var(--text3);">11 个 Part · 113 页 · 从三大瓶颈到开放问题</p></div></section>
'''

# ── TOC ──
toc_rows = []
for i in range(11):
    name, desc, color = part_labels[i]
    toc_rows.append(f'<div><strong style="color:{color};">Part {i}</strong> · {name}<br/><span style="font-size:9.5px;color:var(--text3);">{desc}</span></div>')

toc_page = f'''<!-- ═══════════ 目录页 ═══════════ -->
<section data-transition="slide"><div class="core-card fragment fade-up" style="max-width:900px;"><h2 style="font-size:28px;font-weight:700;color:#fff;margin-bottom:16px;">📋 模块目录</h2><div class="core-bar"></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:12px 36px;text-align:left;font-size:12.5px;color:var(--text2);line-height:2;">
{''.join(toc_rows)}
</div></div></section>
'''

# ── Full DAG-m3 ──
# I'll generate a large SVG showing all 11 parts connected
dag_rows_top = []
dag_rows_bot = []
for i in range(6):
    name, _, color = part_labels[i]
    x = 20 + i * 153
    dag_rows_top.append(f'<g><rect x="{x}" y="52" width="143" height="58" rx="8" fill="{color}1a" stroke="{color}33"/><text x="{x+72}" y="76" text-anchor="middle" fill="{color}" font-size="11" font-weight="700" font-family="sans-serif">Part {i}</text><text x="{x+72}" y="94" text-anchor="middle" fill="rgba(255,255,255,0.45)" font-size="8.5" font-family="sans-serif">{name}</text></g>')

for i in range(6, 11):
    name, _, color = part_labels[i]
    x = 20 + (i-6) * 180
    dag_rows_bot.append(f'<g><rect x="{x}" y="162" width="170" height="58" rx="8" fill="{color}1a" stroke="{color}33"/><text x="{x+85}" y="186" text-anchor="middle" fill="{color}" font-size="11" font-weight="700" font-family="sans-serif">Part {i}</text><text x="{x+85}" y="204" text-anchor="middle" fill="rgba(255,255,255,0.45)" font-size="8.5" font-family="sans-serif">{name}</text></g>')

full_dag = f'''<!-- ═══════════ Module 3 整体 DAG ═══════════ -->
<section data-transition="slide"><h2 style="font-size:24px;font-weight:700;color:var(--core);margin-bottom:6px;">Module 3 整体知识图谱</h2><p style="font-size:11px;color:var(--text3);margin:0 0 8px;">11 个 Part · 从核心问题到未来方向</p>
<svg viewBox="0 0 940 280" xmlns="http://www.w3.org/2000/svg" style="max-width:940px;">
<rect x="300" y="5" width="340" height="28" rx="14" fill="rgba(124,92,231,0.08)" stroke="rgba(124,92,231,0.2)"/>
<text x="470" y="24" text-anchor="middle" fill="#fff" font-size="13" font-weight="700" font-family="sans-serif">多模态 OPD 研究全景 · 2025 → 2026</text>
{''.join(dag_rows_top)}
{''.join(dag_rows_bot)}
<!-- 箭头 -->
<line x1="470" y1="110" x2="470" y2="155" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" marker-end="url(#am3)"/>
<defs><marker id="am3" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4 Z" fill="rgba(255,255,255,0.25)"/></marker></defs>
<rect x="200" y="235" width="540" height="28" rx="10" fill="rgba(8,6,15,0.6)" stroke="rgba(124,92,231,0.1)"/>
<text x="470" y="254" text-anchor="middle" fill="rgba(255,255,255,0.5)" font-size="10" font-family="sans-serif">六大域（视觉·视频·音频·Agent·扩散·新兴）→ 工业落地 → 开放问题</text>
</svg></section>
'''

# ── Generate DAG-hl slides for each part ──
dag_hl_slides = []
for hl_i in range(11):
    rows = []
    # Top row: parts 0-5
    for i in range(6):
        name, _, color = part_labels[i]
        x = 20 + i * 153
        if i == hl_i:
            rows.append(f'<g><rect x="{x}" y="52" width="143" height="58" rx="8" fill="{color}22" stroke="{color}" stroke-width="2.5"/><text x="{x+72}" y="74" text-anchor="middle" fill="#fff" font-size="11" font-weight="800" font-family="sans-serif">✦ Part {i} ✦</text><text x="{x+72}" y="94" text-anchor="middle" fill="{color}" font-size="9" font-weight="700" font-family="sans-serif">{name}</text></g>')
        else:
            rows.append(f'<g opacity="0.25"><rect x="{x}" y="52" width="143" height="58" rx="8" fill="{color}10" stroke="{color}22"/><text x="{x+72}" y="78" text-anchor="middle" fill="rgba(255,255,255,0.4)" font-size="10" font-family="sans-serif">Part {i}</text><text x="{x+72}" y="96" text-anchor="middle" fill="rgba(255,255,255,0.25)" font-size="8" font-family="sans-serif">{name}</text></g>')
    # Bottom row: parts 6-10
    for i in range(6, 11):
        name, _, color = part_labels[i]
        x = 20 + (i-6) * 180
        if i == hl_i:
            rows.append(f'<g><rect x="{x}" y="162" width="170" height="58" rx="8" fill="{color}22" stroke="{color}" stroke-width="2.5"/><text x="{x+85}" y="184" text-anchor="middle" fill="#fff" font-size="11" font-weight="800" font-family="sans-serif">✦ Part {i} ✦</text><text x="{x+85}" y="204" text-anchor="middle" fill="{color}" font-size="9" font-weight="700" font-family="sans-serif">{name}</text></g>')
        else:
            rows.append(f'<g opacity="0.25"><rect x="{x}" y="162" width="170" height="58" rx="8" fill="{color}10" stroke="{color}22"/><text x="{x+85}" y="188" text-anchor="middle" fill="rgba(255,255,255,0.4)" font-size="10" font-family="sans-serif">Part {i}</text><text x="{x+85}" y="206" text-anchor="middle" fill="rgba(255,255,255,0.25)" font-size="8" font-family="sans-serif">{name}</text></g>')

    pname, pdesc, pcolor = part_labels[hl_i]
    hl_slide = f'''<!-- ═══════════ Part {hl_i} DAG 高亮 ═══════════ -->
<section data-transition="slide"><h2 style="font-size:22px;font-weight:700;color:{pcolor};margin-bottom:4px;">📍 Part {hl_i} · {pname}</h2><p style="font-size:11px;color:var(--text3);margin:0 0 6px;">{pdesc}</p>
<svg viewBox="0 0 940 250" xmlns="http://www.w3.org/2000/svg" style="max-width:940px;">
<defs><marker id="am{hl_i}" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4 Z" fill="rgba(255,255,255,0.25)"/></marker></defs>
{''.join(rows)}
<rect x="200" y="232" width="540" height="14" rx="7" fill="rgba(8,6,15,0.4)"/>
</svg></section>
'''
    dag_hl_slides.append(hl_slide)

# ── Module summary ──
summary_page = '''<!-- ═══════════ 模块总结 ═══════════ -->
<section data-transition="slide"><div class="core-card fragment fade-up"><p class="page-badge">模块总结</p><h2 style="font-size:28px;font-weight:700;color:var(--core);margin:8px 0;">🎯 多模态 OPD：从问题到方法到前沿</h2><div class="core-bar"></div>
<div class="struct-content">
<p>本模块系统梳理了多模态在线策略蒸馏（OPD/OPSD）的<strong>完整研究版图</strong>——从 2025 年底到 2026 年 7 月，9 个月内涌现的 50+ 篇论文。</p>
<p><strong>三大模块级规律：</strong></p>
<p style="margin-left:16px;">🔹 <strong>特权信息的进化</strong>：从"给答案"→"给线索"→"用自身尺度"→"连答案都可以不要"（V-Zero）→"优化看哪里而非说什么"（RAL）<br/>
🔹 <strong>范式的融合</strong>：OPD 从独立范式走向与 RLVR 深度集成（PRISM、PTD-PO）→ 逐渐成为多模态后训练的<strong>标准组件</strong><br/>
🔹 <strong>从输出空间到内部表示空间</strong>：OPD 的优化对象从 token 分布 → 注意力权重 → 教师组合 → 世界状态 → 推理链</p>
<p><strong>六大域的核心张力：</strong>视觉域最成熟（15+论文）、Agent域最独特（Reverse KL确立）、扩散域最具颠覆性（MSE替代KL）、新兴方向最前沿（注意力OPD化）。</p>
<p style="text-align:center;color:var(--nova);margin-top:8px;">💡 最终启示：<strong>OPD 正在成为多模态模型后训练的「操作系统」——一套管理各种监督信号、特权信息、教师组合的通用框架。</strong></p>
</div></div></section>
'''

# ── End page ──
end_page = '''<!-- ═══════════ 模块结束页 ═══════════ -->
<section data-transition="zoom"><div class="core-card fragment fade-up"><p class="cite">MODULE 3 · END</p><h1 class="hero-title">多模态 OPD 研究全景</h1><p style="font-size:20px;color:var(--text2);">2025 → 2026 · 9 个月的爆发</p><div class="core-bar"></div><p style="font-size:14px;color:var(--text3);">感谢阅读 · 保持好奇 · 继续探索</p></div></section>
'''

# ── JS footer ──
js_footer = '''</div></div>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.min.js"></script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>(function(){const c=document.getElementById('bg-canvas'),ctx=c.getContext('2d');let W,H;function rs(){W=c.width=window.innerWidth;H=c.height=window.innerHeight}window.addEventListener('resize',rs);rs();const NUM=71,Z_NEAR=100,Z_FAR=2200,R=180,MAX_A=0.40,BLUR=14;let frame=0;function lp(){ctx.clearRect(0,0,W,H);const t=frame*0.016,F=Math.min(W,H)*0.75,CX=W/2,CY=H/2;const sx=Math.sin(t*0.22)*55+Math.cos(t*0.31)*25;const sy=Math.cos(t*0.28)*45+Math.sin(t*0.19)*30;for(let i=NUM-1;i>=0;i--){const p=i/(NUM-1),z=Z_NEAR+p*(Z_FAR-Z_NEAR);const rot=t*0.35-z*0.0025;const px=(-sx*F)/z+CX,py=(-sy*F)/z+CY,pr=(R*F)/z;const alpha=MAX_A*Math.pow(1-p,0.90);const lw=Math.max(0.08,4.8*Math.pow(Z_NEAR/z,2.35));if(alpha<0.005)continue;const hue=((p*280+t*15)%360+360)%360;ctx.save();ctx.translate(px,py);ctx.beginPath();for(let v=0;v<5;v++){const a=rot+(v*2*Math.PI)/5;v===0?ctx.moveTo(pr*Math.cos(a),pr*Math.sin(a)):ctx.lineTo(pr*Math.cos(a),pr*Math.sin(a))}ctx.closePath();ctx.strokeStyle='hsla('+hue+',100%,50%,'+alpha+')';ctx.lineWidth=lw;ctx.shadowColor='hsla('+hue+',100%,50%,'+(alpha*0.4)+')';ctx.shadowBlur=BLUR;ctx.stroke();ctx.restore()}frame++;requestAnimationFrame(lp)}lp()})();</script>
<script>Reveal.initialize({hash:true,center:true,transition:'slide',transitionSpeed:'slow',backgroundTransition:'zoom',slideNumber:false,progress:false,controls:true,controlsLayout:'bottom-right',navigationMode:'linear',width:1200,height:700,margin:0,minScale:0.2,maxScale:2.0});(function(){const e=document.getElementById('page-indicator');function u(){const t=Reveal.getTotalSlides(),c=Reveal.getSlidePastCount()+1;e.textContent=c+' / '+t}Reveal.on('ready',u);Reveal.on('slidechanged',u)})();</script></body></html>
'''

# ── Assemble ──
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(css)
    f.write(title_page)
    f.write(toc_page)
    f.write(full_dag)
    # For each part: DAG-hl + part sections
    for i in range(11):
        f.write(dag_hl_slides[i])
        for s in all_sections[i]:
            f.write(s + '\n')
    f.write(summary_page)
    f.write(end_page)
    f.write(js_footer)

total_slides = 4 + 11 + sum(len(v) for v in all_sections.values()) + 2
print(f'Written: {out_path}')
print(f'Module-level pages: 4 (title+TOC+DAG+summary/end=5 actually)')
print(f'DAG-hl slides: 11')
print(f'Content slides: {sum(len(v) for v in all_sections.values())}')
print(f'Total sections: ~{total_slides}')
# Actually: title + TOC + DAG + 11*DAG-hl + sum(content) + summary + end = 3 + 11 + 113 + 2 = 129
print(f'File size: {os.path.getsize(out_path)/1024:.0f} KB')
