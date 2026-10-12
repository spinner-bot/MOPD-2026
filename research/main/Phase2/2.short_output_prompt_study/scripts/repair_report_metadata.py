#!/usr/bin/env python3
"""Repair the interactive table of contents and add report author metadata."""

from __future__ import annotations

import argparse
from pathlib import Path


AUTHOR = '<p class="author">作者 / Author: <a href="https://github.com/spinner-bot" target="_blank" rel="noopener">spinner-bot</a>（GitHub）</p>'

OLD_HARMONY = r"""
/* harmony override */
body { background:#fff; color:#222; }
body > h1 { margin:0 0 1.1rem; padding:0; background:transparent; color:#18324b; border-radius:0; }
section { background:transparent; border-color:transparent; box-shadow:none; }
section h2 { background:transparent; }
.case { background:#fff; }
"""

HARMONY = r"""
/* harmony override */
body { background:#fff; }
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    text = args.report.read_text(encoding="utf-8")
    text = text.replace(OLD_HARMONY, HARMONY)
    text = text.replace("(() => {", "document.addEventListener('DOMContentLoaded', () => {", 1)
    text = text.replace("})();", "});", 1)
    if "class=\"author\"" not in text:
        text = text.replace("</h1>", "</h1>" + AUTHOR, 1)
    if ".author {" not in text:
        text = text.replace("</style>", ".author { margin:-.7rem 0 1.4rem; padding-left:.35rem; color:#5b6875; font-size:.95rem; }\n</style>", 1)
    if "/* harmony override */" not in text:
        text = text.replace("</style>", HARMONY + "</style>", 1)
    args.report.write_text(text, encoding="utf-8")
    print(f"updated {args.report} ({args.report.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
