#!/usr/bin/env python3
"""Add Codex-authored Chinese translations to the frozen manual-review HTML.

This utility does not translate text. It accepts a complete, human-authored
translation manifest and inserts every translation using one identical HTML
component. Original source text is only used as an insertion anchor and is
never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Callable


EXPECTED_CASES = 60
EXPECTED_CONDITIONS = 5
CONDITION_ORDER = ("S0", "SC", "S1", "S2", "S3")
DETAIL = (
    '<details class="zh-translation"><summary>中文直译</summary>'
    '<pre>{}</pre></details>'
)


def require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty translation at {path}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Require a uniform, complete 60 × 5 translation structure."""
    if manifest.get("schema") != "manual-review-bilingual-translations-v1":
        raise ValueError("unsupported translation manifest schema")
    for field in ("intro", "notice", "summary"):
        require_text(manifest.get("page", {}).get(field), f"page.{field}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError(f"expected exactly {EXPECTED_CASES} translated cases")
    for index, case in enumerate(cases, 1):
        prefix = f"cases[{index - 1}]"
        if case.get("case_index") != index:
            raise ValueError(f"{prefix}.case_index must be {index}")
        for field in ("question", "stratum", "flags", "reference_answer"):
            require_text(case.get(field), f"{prefix}.{field}")
        conditions = case.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITIONS:
            raise ValueError(f"{prefix}: expected five conditions")
        if tuple(row.get("condition") for row in conditions) != CONDITION_ORDER:
            raise ValueError(f"{prefix}: condition order must be {CONDITION_ORDER}")
        for condition_index, row in enumerate(conditions):
            cond_prefix = f"{prefix}.conditions[{condition_index}]"
            for field in ("scores", "extraction_evidence", "model_response", "judge_output"):
                require_text(row.get(field), f"{cond_prefix}.{field}")
    return cases


def translation_detail(text: str) -> str:
    return DETAIL.format(html.escape(text, quote=False))


def replace_once(source: str, pattern: str, replacement: str | Callable[[re.Match[str]], str], label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"expected one insertion anchor for {label}; found {count}")
    return updated


def add_after(pattern: str, text: str, translation: str, label: str) -> str:
    return replace_once(text, pattern, lambda match: match.group(0) + translation_detail(translation), label)


def render_bilingual(source: str, manifest: dict[str, Any]) -> str:
    cases = validate_manifest(manifest)
    sections = list(re.finditer(r'<section class="case".*?</section>', source, re.DOTALL))
    if len(sections) != EXPECTED_CASES:
        raise ValueError(f"source packet has {len(sections)} cases; expected {EXPECTED_CASES}")
    if len(re.findall(r'<details><summary>Model response\b', source)) != EXPECTED_CASES * EXPECTED_CONDITIONS:
        raise ValueError("source packet does not contain exactly 300 model responses")
    if len(re.findall(r'<details><summary>Full native judge output\b', source)) != EXPECTED_CASES * EXPECTED_CONDITIONS:
        raise ValueError("source packet does not contain exactly 300 native judge outputs")

    # Work from the end so every match remains anchored to the unchanged source.
    rendered = source
    for case_match, case in reversed(list(zip(sections, cases))):
        original_section = case_match.group(0)
        section = original_section
        i = case["case_index"]
        section = add_after(
            r'<h3>Case\s+\d+.*?</h3>\s*<p class="flags">.*?</p>', section,
            "分层：" + case["stratum"] + "\n审计标记：" + case["flags"], f"case {i} metadata")
        section = add_after(
            r'<h4>Original benchmark input[^<]*</h4>\s*<pre>.*?</pre>', section,
            case["question"], f"case {i} question")
        section = add_after(
            r'<p><b>Reference answer[^<]*</b>.*?</p>', section,
            case["reference_answer"], f"case {i} reference answer")

        articles = list(re.finditer(r'<article class="condition">.*?</article>', section, re.DOTALL))
        if len(articles) != EXPECTED_CONDITIONS:
            raise ValueError(f"case {i} has {len(articles)} condition cards")
        for article_match, condition in reversed(list(zip(articles, case["conditions"]))):
            article = article_match.group(0)
            article = add_after(
                r'<div class="scores">.*?</div>\s*<p>.*?</p>', article,
                "评分与来源：\n" + condition["scores"] + "\n\n抽取答案与证据：\n" + condition["extraction_evidence"],
                f"case {i} {condition['condition']} scores/evidence")
            article = add_after(
                r'<details><summary>Model response\b.*?</details>', article,
                condition["model_response"], f"case {i} {condition['condition']} model response")
            article = add_after(
                r'<details><summary>Full native judge output\b.*?</details>', article,
                condition["judge_output"], f"case {i} {condition['condition']} judge output")
            section = section[:article_match.start()] + article + section[article_match.end():]

        rendered = rendered[:case_match.start()] + section + rendered[case_match.end():]

    page = manifest["page"]
    rendered = add_after(r'<h1>.*?</h1>\s*<p>.*?</p>', rendered, page["intro"], "page intro")
    rendered = add_after(r'<div class="notice">.*?</div>', rendered, page["notice"], "page notice")
    rendered = add_after(r'<p>Cases:\s*60\s*·\s*paired condition views:\s*300.*?</p>', rendered, page["summary"], "page summary")

    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    rendered = rendered.replace("</head>", f'<meta name="translation-source-sha256" content="{digest}"></head>', 1)
    rendered = rendered.replace("</style>", ".zh-translation pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:480px;overflow:auto}\n</style>", 1)

    expected_panels = 3 + EXPECTED_CASES * (3 + EXPECTED_CONDITIONS * 3)
    if rendered.count('<details class="zh-translation">') != expected_panels:
        raise AssertionError("translation panel count invariant failed")
    if rendered.count("<select") != source.count("<select") or rendered.count("<textarea") != source.count("<textarea"):
        raise AssertionError("review controls changed during translation insertion")
    if rendered.count('data:image/') != source.count('data:image/'):
        raise AssertionError("embedded image count changed")
    if rendered.count("pickArchive") != source.count("pickArchive"):
        raise AssertionError("archive interaction code changed")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.source.open(encoding="utf-8", newline="") as handle:
        source = handle.read()
    manifest = json.loads(args.translations.read_text(encoding="utf-8"))
    result = render_bilingual(source, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(result)
    print(f"PASS: wrote bilingual packet ({len(result.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
