#!/usr/bin/env python3
"""Repair the two structural defects of the local-LLM translation output.

1. `scores` / `extraction_evidence` — the source block is already bilingual, so a
   literal translation duplicates the label ("官方精确标签 / 官方精确标签：…").
   These blocks are structured, so they are rebuilt deterministically instead.
2. `stratum` / `flags` — snake_case codes must be rendered with the project's
   Chinese vocabulary (recovered from reports/translation-work/case-001..024).

The English extraction (extract_review_english.py) supplies the source for (1);
only fields whose rebuild is confirmed are touched.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# --- 码表：取自 Codex 已完成的前 24 例（报告 2230/2231 的产出） ---
STRATUM_ZH = {
    "three_score_agreement": "三项评分一致",
    "native_extraction_disagreement": "原生抽取不一致",
    "native_terminal_disagreement": "原生终行判定不一致",
    "non_answer_extraction": "未抽取答案",
    "long_tail_output": "长尾输出",
    "deterministic_random_remainder": "确定性随机剩余样本",
}
SOURCE_ZH = {
    "llm": "大语言模型",
    "first letter": "首字母",
    "keyword": "关键词",
}
STATUS_ZH = {
    "ANSWERED": "已回答",
    "NO_ANSWER": "无答案",
    "AMBIGUOUS": "歧义",
}

SCORE_LABELS = (
    ("official", "Official exact-label / 官方精确标签："),
    ("terminal", "Terminal diagnostic / 终行诊断："),
    ("extraction", "Extraction / 抽取："),
    ("source", "Native judge source / 原生裁判来源："),
)


def split_score_block(english: str) -> dict[str, str]:
    """Cut the concatenated bilingual block back into its four fields."""
    positions = []
    for key, label in SCORE_LABELS:
        i = english.find(label)
        if i < 0:
            raise ValueError(f"score label not found: {label}")
        positions.append((i + len(label), key))
    out = {}
    for n, (start, key) in enumerate(positions):
        end = positions[n + 1][0] - len(SCORE_LABELS[n + 1][1]) if n + 1 < len(positions) else len(english)
        out[key] = english[start:end].strip()
    return out


def yes_no(zh: bool) -> str:
    return "判对" if zh else "未判对"


def rebuild_scores(english: str) -> str:
    f = split_score_block(english)
    official = yes_no("Correct" in f["official"] and "Not correct" not in f["official"])
    term = re.match(r"\s*(yes|no)", f["terminal"], flags=re.I)
    if not term:
        raise ValueError(f"terminal decision unparsed: {f['terminal']!r}")
    accepted = term.group(1).lower() == "yes"
    terminal = f"{'是' if accepted else '否'}（{yes_no(accepted)}）"

    m = re.match(r"\s*([A-Z_]+)\s*·\s*(.+)", f["extraction"])
    if not m:
        raise ValueError(f"extraction field unparsed: {f['extraction']!r}")
    status = STATUS_ZH.get(m.group(1), m.group(1))
    judged = yes_no("Not correct" not in m.group(2))
    extraction = f"{status} · {judged}"

    src = f["source"].strip().lower()
    source = SOURCE_ZH.get(src, f["source"].strip())

    return "\n".join(
        (
            f"官方精确标签：{official}",
            f"终行诊断：{terminal}",
            f"抽取：{extraction}",
            f"原生裁判来源：{source}",
        )
    )


def rebuild_evidence(english: str) -> str:
    m = re.search(r"Extracted answer[^：]*：\s*(.*?)\s*Evidence[^：]*：\s*(.*)$", english, flags=re.S)
    if not m:
        raise ValueError(f"evidence block unparsed: {english[:60]!r}")
    return f"抽取答案：{m.group(1).strip()}\n证据：{m.group(2).strip()}"


def render_codes(raw: str) -> str:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return "、".join(STRATUM_ZH.get(p, p) for p in parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-dir", type=Path, required=True)
    ap.add_argument("--zh-dir", type=Path, required=True)
    ap.add_argument("--first", type=int, default=25)
    ap.add_argument("--last", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fixed = 0
    for path in sorted(args.zh_dir.glob("case-*.json")):
        zh = json.loads(path.read_text(encoding="utf-8"))
        idx = zh["case_index"]
        if not (args.first <= idx <= args.last):
            continue
        en = json.loads((args.en_dir / f"en-case-{idx:03d}.json").read_text(encoding="utf-8"))
        if len(en["conditions"]) != len(zh["conditions"]):
            raise SystemExit(f"case {idx}: condition count mismatch")

        zh["stratum"] = STRATUM_ZH.get(en["stratum"], en["stratum"])
        zh["flags"] = render_codes(en["flags"])
        if en.get("uid"):
            zh["flags"] = zh["flags"] + "。UID：" + en["uid"] if zh["flags"] else "UID：" + en["uid"]

        for ce, cz in zip(en["conditions"], zh["conditions"]):
            cz["scores"] = rebuild_scores(ce["scores"])
            if ce["extraction_evidence"].strip():
                cz["extraction_evidence"] = rebuild_evidence(ce["extraction_evidence"])

        if not args.dry_run:
            path.write_text(json.dumps(zh, ensure_ascii=False, indent=2), encoding="utf-8")
        fixed += 1

    print(f"{'[dry-run] ' if args.dry_run else ''}修正 {fixed} 例")


if __name__ == "__main__":
    main()
