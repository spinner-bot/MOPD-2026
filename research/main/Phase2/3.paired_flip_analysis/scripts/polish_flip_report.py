#!/usr/bin/env python3
"""Apply the readable bilingual-report layout to the flip analysis HTML."""

from __future__ import annotations

import argparse
from pathlib import Path


STYLE = r"""
*{box-sizing:border-box}body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1400px;margin:0 auto;padding:42px 34px 80px;background:#fff;color:#243447;line-height:1.72}h1{margin:0 0 1.2rem;padding:1.4rem 1.6rem;background:linear-gradient(135deg,#18324b,#2f6f9f);color:#fff;border-radius:14px;letter-spacing:.01em}h2{margin:3rem 0 1.2rem;padding:.75rem 1rem;border-left:6px solid #2f6f9f;border-bottom:1px solid #d7e0e8;background:#f8fafc;color:#18324b;font-size:1.4rem}h3{margin:2rem 0 .7rem;color:#244d6d}.author{margin:-.5rem 0 1.6rem;color:#5b6875;font-size:.95rem}.toc{margin:0 0 2.5rem;padding:1.1rem 1.3rem;border:1px solid #d7e0e8;border-radius:10px;background:#fff}.toc-title{margin-bottom:.6rem;color:#18324b;font-weight:700}.toc ol{margin:0;padding-left:1.5rem;columns:2}.toc li{break-inside:avoid;margin:.25rem 0}.toc a{color:#2f6f9f;text-decoration:none}.toc a:hover{text-decoration:underline}.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1.3rem;margin:1.3rem 0 2rem}figure{margin:0;padding:.8rem;border:1px solid #d7e0e8;border-radius:9px;background:#fff}figure img{display:block;width:100%;height:auto}figcaption{text-align:center;font-weight:700;padding:.6rem .2rem .1rem;color:#39566d}table{display:block;overflow-x:auto;width:100%;margin:1.3rem 0 2.2rem;border:1px solid #d7e0e8;border-radius:8px;border-collapse:separate;border-spacing:0;background:#fff;font-size:.88rem}th,td{padding:.65rem .75rem;border-right:1px solid #d7e0e8;border-bottom:1px solid #d7e0e8;vertical-align:top;white-space:nowrap}th{background:#eaf1f7;color:#18324b}tr:last-child td{border-bottom:0}th:last-child,td:last-child{border-right:0}details{margin:1rem 0;padding:.9rem 1rem;border:1px solid #d7e0e8;border-radius:8px;background:#fff}summary{cursor:pointer;font-weight:700;color:#244d6d}pre{margin:.7rem 0 1rem;padding:.9rem 1rem;border:1px solid #e1e8ee;border-radius:6px;background:#f7f9fb;white-space:pre-wrap;word-break:break-word;line-height:1.55}@media(max-width:760px){body{padding:20px 14px 50px}h1{padding:1.1rem;font-size:1.35rem}.toc ol{columns:1}.figures{grid-template-columns:1fr}}
"""

TOC = r'''<nav class="toc" id="report-toc"><div class="toc-title">目录 / Table of contents</div><ol id="report-toc-list"></ol></nav><script>document.addEventListener('DOMContentLoaded',()=>{const list=document.getElementById('report-toc-list');let n=0;document.querySelectorAll('h2').forEach(h=>{h.id=h.id||`section-${++n}`;const li=document.createElement('li'),a=document.createElement('a');a.href=`#${h.id}`;a.textContent=h.textContent;li.appendChild(a);list.appendChild(li)});});</script>'''
AUTHOR = '<p class="author">作者 / Author: <a href="https://github.com/spinner-bot" target="_blank" rel="noopener">spinner-bot</a>（GitHub）</p>'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    text = args.report.read_text(encoding="utf-8")
    if "id=\"report-toc\"" not in text:
        text = text.replace("</h1>", "</h1>" + AUTHOR + TOC, 1)
    if "/* flip report layout */" not in text:
        text = text.replace("</style>", "/* flip report layout */" + STYLE + "</style>", 1)
    args.report.write_text(text, encoding="utf-8")
    print(f"updated {args.report} ({args.report.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
