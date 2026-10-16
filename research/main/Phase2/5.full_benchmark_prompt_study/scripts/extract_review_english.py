#!/usr/bin/env python3
"""Extract the English source fields of the manual-review packet, per case.

Emits one JSON per case holding the *English* text that must be translated.
The translation step consumes this file and writes the Chinese counterpart in
the format expected by add_manual_review_bilingual_layer.py.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

CONDITIONS = ("S0", "SC", "S1", "S2", "S3")
TAG = re.compile(r"<[^>]+>")


def text_of(fragment: str) -> str:
    return html.unescape(TAG.sub("", fragment)).strip()


def split_cases(document: str) -> list[str]:
    return re.split(r'<section class="case"', document)[1:]


def parse_case(block: str, index: int) -> dict:
    h3 = re.search(r"<h3>(.*?)</h3>", block, flags=re.S)
    raw_h3 = text_of(h3.group(1)) if h3 else ""
    case_title = raw_h3.split("selected stratum:")[0].strip()
    stratum = raw_h3.split("selected stratum:")[1].strip() if "selected stratum:" in raw_h3 else ""

    flags_m = re.search(r'<p class="flags">(.*?)</p>', block, flags=re.S)
    raw_flags = text_of(flags_m.group(1)) if flags_m else ""
    layer_m = re.search(r"命中审计层：\s*(.*?)\s*·\s*UID\s*[:：]\s*(\S+)", raw_flags)
    if layer_m:
        flags = layer_m.group(1).strip()
        uid = layer_m.group(2).strip()
    else:
        flags, uid = raw_flags, ""

    q_m = re.search(r'<div class="question-grid">.*?<pre[^>]*>(.*?)</pre>', block, flags=re.S)
    question = text_of(q_m.group(1)) if q_m else ""

    ref_m = re.search(r"Reference answer[^<]*</b>\s*(.*?)</p>", block, flags=re.S)
    reference_answer = text_of(ref_m.group(1)) if ref_m else ""

    conditions = []
    for art in re.findall(r'<article class="condition">(.*?)</article>', block, flags=re.S):
        cond_m = re.search(r"<h4>\s*(S0|SC|S1|S2|S3)", art)
        if not cond_m:
            continue
        scores_m = re.search(r'<div class="scores">(.*?)</div>', art, flags=re.S)
        scores = text_of(scores_m.group(1)) if scores_m else ""
        ev_m = re.search(r"<p>(<b>Extracted answer.*?)</p>", art, flags=re.S)
        evidence = text_of(ev_m.group(1)) if ev_m else ""
        resp = judge = ""
        for summ, body in re.findall(
            r"<details[^>]*><summary>(.*?)</summary><pre[^>]*>(.*?)</pre></details>", art, flags=re.S
        ):
            s = text_of(summ).lower()
            if s.startswith("model response"):
                resp = text_of(body)
            elif "native judge" in s:
                judge = text_of(body)
        conditions.append(
            {
                "condition": cond_m.group(1),
                "scores": scores,
                "extraction_evidence": evidence,
                "model_response": resp,
                "judge_output": judge,
            }
        )
    return {
        "case_index": index,
        "case_title": case_title,
        "stratum": stratum,
        "flags": flags,
        "uid": uid,
        "question": question,
        "reference_answer": reference_answer,
        "conditions": conditions,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--first", type=int, default=1)
    p.add_argument("--last", type=int, default=60)
    args = p.parse_args()

    document = args.packet.read_text(encoding="utf-8", errors="replace")
    blocks = split_cases(document)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for i, block in enumerate(blocks, 1):
        if not (args.first <= i <= args.last):
            continue
        case = parse_case(block, i)
        if len(case["conditions"]) != len(CONDITIONS):
            raise SystemExit(f"case {i}: expected 5 conditions, got {len(case['conditions'])}")
        missing = [k for k in ("question", "stratum", "flags") if not case[k]]
        for c in case["conditions"]:
            missing += [f"{c['condition']}.{k}" for k in ("scores", "model_response", "judge_output") if not c[k]]
        if missing:
            raise SystemExit(f"case {i}: empty fields {missing}")
        path = args.out_dir / f"en-case-{i:03d}.json"
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
        chars = len(case["question"]) + sum(
            len(c["scores"]) + len(c["extraction_evidence"]) + len(c["model_response"]) + len(c["judge_output"])
            for c in case["conditions"]
        )
        total += chars
        print(f"case {i:3d}: {chars:>8,} chars")
    print(f"\n合计 {total:,} 字符（case {args.first}-{args.last}）")


if __name__ == "__main__":
    main()
