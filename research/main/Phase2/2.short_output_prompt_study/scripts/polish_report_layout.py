#!/usr/bin/env python3
"""Apply a readable, spacious layout to the self-contained HTML report."""

from __future__ import annotations

import argparse
from pathlib import Path


STYLE = r"""
/* layout refinement */
:root { --ink:#243447; --muted:#5b6875; --line:#d7e0e8; --panel:#ffffff; --soft:#f4f7fa; --accent:#2f6f9f; }
* { box-sizing:border-box; }
body { max-width:1400px; margin:0 auto; padding:42px 34px 80px; background:#eef2f6; color:var(--ink); line-height:1.72; font-family:Arial,"Microsoft YaHei",sans-serif; }
body > h1 { margin:0 0 1.4rem; padding:1.5rem 1.8rem; background:linear-gradient(135deg,#18324b,#2f6f9f); color:#fff; border-radius:14px; letter-spacing:.01em; }
body > p.note { margin:0 0 2.5rem; }
h2 { margin:3.2rem 0 1.25rem; padding:.8rem 1rem; border-left:6px solid var(--accent); border-bottom:1px solid var(--line); background:#f8fafc; color:#18324b; font-size:1.45rem; line-height:1.35; }
h3 { margin:2rem 0 .8rem; color:#244d6d; font-size:1.12rem; }
p { margin:.85rem 0; }
ul,ol { margin:.8rem 0 1.3rem; padding-left:2rem; }
li { margin:.45rem 0; }
.note { margin:1.2rem 0 1.7rem; padding:1rem 1.2rem; border-left:5px solid var(--accent); border-radius:8px; background:#f8fbfd; color:#354656; }
table { display:block; overflow-x:auto; width:100%; margin:1.4rem 0 2.4rem; border:1px solid var(--line); border-radius:9px; background:var(--panel); box-shadow:0 2px 8px rgba(28,52,73,.04); border-collapse:separate; border-spacing:0; }
th,td { min-width:108px; padding:.75rem .85rem; border-right:1px solid var(--line); border-bottom:1px solid var(--line); vertical-align:top; line-height:1.55; }
th { background:#eaf1f7; color:#18324b; font-weight:700; white-space:nowrap; }
tr:last-child td { border-bottom:0; } th:last-child,td:last-child { border-right:0; }
td:last-child { max-width:none; white-space:pre-wrap; word-break:break-word; }
pre { margin:.65rem 0 1rem; padding:.9rem 1rem; border:1px solid #e1e8ee; border-radius:7px; background:#f7f9fb; white-space:pre-wrap; word-break:break-word; line-height:1.55; font-size:.88rem; }
.figures { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:1.4rem; margin:1.4rem 0 2rem; }
figure { margin:0; padding:.85rem; border:1px solid var(--line); border-radius:10px; background:var(--panel); box-shadow:0 2px 8px rgba(28,52,73,.05); }
figure img { display:block; width:100%; height:auto; border-radius:6px; }
figcaption { padding:.7rem .25rem .15rem; color:#39566d; text-align:center; font-weight:700; line-height:1.4; }
.input-grid,.mode-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:1.2rem; margin:1.2rem 0 2rem; }
.input-card,.mode-card { min-width:0; padding:1rem; border:1px solid var(--line); border-radius:9px; background:var(--panel); box-shadow:0 2px 7px rgba(28,52,73,.04); }
.input-card h3,.mode-card h4 { margin:.1rem 0 .75rem; padding-bottom:.45rem; border-bottom:1px solid var(--line); color:#18324b; }
section { margin:2rem 0; padding:0 1.2rem 1.4rem; border:1px solid #e0e7ed; border-radius:12px; background:rgba(255,255,255,.72); }
section h2 { margin-left:-1.2rem; margin-right:-1.2rem; border-radius:12px 12px 0 0; }
.case { margin:3.2rem 0; padding:1.5rem; border:1px solid #b9c9d6; border-top:6px solid #4c78a8; border-radius:12px; background:var(--panel); box-shadow:0 4px 14px rgba(28,52,73,.07); }
.case h3 { margin:.1rem 0 1.2rem; font-size:1.2rem; }
.case-image { display:block; max-width:100%; max-height:560px; margin:0 auto 1.5rem; padding:.35rem; object-fit:contain; border:1px solid var(--line); border-radius:8px; background:#f2f4f6; }
.question { margin:0 0 1.5rem; padding:1rem 1.2rem; border:1px solid #dbe6ef; border-radius:8px; background:#f7fafc; }
.question p { margin:.55rem 0; }
.mode-grid { align-items:start; }
.mode-card { border-top:4px solid #7fa8c4; }
@media (max-width:760px) { body { padding:20px 14px 50px; } body > h1 { padding:1.1rem; font-size:1.35rem; } h2 { margin-top:2.3rem; font-size:1.2rem; } .input-grid,.mode-grid,.figures { grid-template-columns:1fr; } .case { padding:1rem; } }
.toc { margin:0 0 2.7rem; padding:1.2rem 1.4rem; border:1px solid #cbd9e4; border-radius:12px; background:#fff; box-shadow:0 3px 10px rgba(28,52,73,.05); }
.toc-title { margin:0 0 .7rem; color:#18324b; font-size:1.05rem; font-weight:700; }
.toc ol { margin:0; padding-left:1.5rem; columns:2; column-gap:3rem; }
.toc li { break-inside:avoid; margin:.25rem 0; }
.toc a { color:#2f6f9f; text-decoration:none; }
.toc a:hover { text-decoration:underline; }
.author { margin:-.7rem 0 1.4rem; padding-left:.35rem; color:#5b6875; font-size:.95rem; }
@media (max-width:760px) { .toc ol { columns:1; } }
"""

AUTHOR = '<p class="author">作者 / Author: <a href="https://github.com/spinner-bot" target="_blank" rel="noopener">spinner-bot</a>（GitHub）</p>'

TOC = r'''
<nav class="toc" id="report-toc" aria-label="Table of contents">
<div class="toc-title">目录 / Table of contents</div>
<ol id="report-toc-list"></ol>
</nav>
<script>
document.addEventListener('DOMContentLoaded', () => {
  const list = document.getElementById('report-toc-list');
  const headings = [...document.querySelectorAll('h2, h3')];
  let h2 = 0, h3 = 0;
  headings.forEach((heading) => {
    if (heading.closest('#report-toc')) return;
    if (!heading.id) {
      if (heading.tagName === 'H2') { h2 += 1; h3 = 0; heading.id = `section-${h2}`; }
      else { h3 += 1; heading.id = `section-${h2}-${h3}`; }
    }
    const item = document.createElement('li');
    if (heading.tagName === 'H3') item.style.marginLeft = '1rem';
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    item.appendChild(link);
    list.appendChild(item);
  });
});
</script>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    text = args.report.read_text(encoding="utf-8")
    if "/* layout refinement */" in text:
        raise SystemExit("layout refinement already present")
    text = text.replace("</style>", STYLE + "</style>", 1)
    if 'class="author"' not in text:
        text = text.replace("</h1>", "</h1>" + AUTHOR, 1)
    marker = "</p>\n<h2>"
    if marker not in text:
        raise SystemExit("top insertion marker not found")
    text = text.replace(marker, "</p>" + TOC + "\n<h2>", 1)
    args.report.write_text(text, encoding="utf-8")
    print(f"updated {args.report} ({args.report.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
