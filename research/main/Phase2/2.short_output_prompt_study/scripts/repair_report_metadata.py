#!/usr/bin/env python3
"""Repair the interactive table of contents and add report author metadata."""

from __future__ import annotations

import argparse
from pathlib import Path


AUTHOR = '<p class="author">作者 / Author: <a href="https://github.com/spinner-bot" target="_blank" rel="noopener">spinner-bot</a>（GitHub）</p>'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    text = args.report.read_text(encoding="utf-8")
    text = text.replace("(() => {", "document.addEventListener('DOMContentLoaded', () => {", 1)
    text = text.replace("})();", "});", 1)
    if "class=\"author\"" not in text:
        text = text.replace("</h1>", "</h1>" + AUTHOR, 1)
    if ".author {" not in text:
        text = text.replace("</style>", ".author { margin:-.7rem 0 1.4rem; padding-left:.35rem; color:#5b6875; font-size:.95rem; }\n</style>", 1)
    args.report.write_text(text, encoding="utf-8")
    print(f"updated {args.report} ({args.report.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
