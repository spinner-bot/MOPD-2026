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


def require_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected string at {path}")
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


def resolve_translation_files(manifest: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Expand retained long-translation text files before complete validation."""
    base = base_dir.resolve()
    fields = ("scores", "extraction_evidence", "model_response", "judge_output")
    for case in manifest.get("cases", []):
        for condition in case.get("conditions", []):
            for field in fields:
                file_key = field + "_file"
                files_key = field + "_files"
                repetition_key = field + "_repetition_file"
                if repetition_key in condition:
                    recipe_path = (base / condition.pop(repetition_key)).resolve()
                    try:
                        recipe_path.relative_to(base)
                    except ValueError as exc:
                        raise ValueError(f"translation file escapes manifest directory: {recipe_path}") from exc
                    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
                    if recipe.get("schema") != "verbatim-repetition-translation-v1":
                        raise ValueError(f"unsupported repetition-translation recipe at {recipe_path}")
                    source_prefix = require_string(recipe.get("source_prefix"), "source_prefix")
                    source_unit = require_text(recipe.get("source_unit"), "source_unit")
                    source_suffix = require_string(recipe.get("source_suffix"), "source_suffix")
                    source_separator = recipe.get("source_separator")
                    translated_prefix = require_string(recipe.get("translated_prefix"), "translated_prefix")
                    translated_unit = require_text(recipe.get("translated_unit"), "translated_unit")
                    translated_suffix = require_string(recipe.get("translated_suffix"), "translated_suffix")
                    translated_separator = recipe.get("translated_separator")
                    repetitions = recipe.get("repetitions")
                    if not isinstance(source_separator, str) or not isinstance(translated_separator, str):
                        raise ValueError("repetition separators must be strings")
                    if not isinstance(repetitions, int) or repetitions < 1:
                        raise ValueError("repetitions must be a positive integer")
                    source_text = source_prefix + source_separator.join([source_unit] * repetitions) + source_suffix
                    condition[field] = (
                        translated_prefix
                        + translated_separator.join([translated_unit] * repetitions)
                        + translated_suffix
                    )
                    condition[field + "_source_sha256"] = hashlib.sha256(
                        source_text.encode("utf-8")
                    ).hexdigest()
                if files_key in condition:
                    relative_paths = condition.pop(files_key)
                    if not isinstance(relative_paths, list) or not relative_paths:
                        raise ValueError(f"{files_key} must be a non-empty ordered list")
                    chunks = []
                    for relative_path in relative_paths:
                        path = (base / relative_path).resolve()
                        try:
                            path.relative_to(base)
                        except ValueError as exc:
                            raise ValueError(f"translation file escapes manifest directory: {path}") from exc
                        with path.open(encoding="utf-8", newline="") as handle:
                            chunks.append(handle.read())
                    # Chunk files split only at paragraph boundaries; each
                    # file ends with one newline, so this separator restores
                    # the blank line between adjacent translated paragraphs.
                    condition[field] = "\n".join(chunks)
                if file_key not in condition:
                    continue
                path = (base / condition.pop(file_key)).resolve()
                try:
                    path.relative_to(base)
                except ValueError as exc:
                    raise ValueError(f"translation file escapes manifest directory: {path}") from exc
                with path.open(encoding="utf-8", newline="") as handle:
                    condition[field] = handle.read()
    return manifest


def translation_detail(text: str) -> str:
    return DETAIL.format(html.escape(text, quote=False))


def replace_once(source: str, pattern: str, replacement: str | Callable[[re.Match[str]], str], label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"expected one insertion anchor for {label}; found {count}")
    return updated


def add_after(pattern: str, text: str, translation: str, label: str) -> str:
    return replace_once(text, pattern, lambda match: match.group(0) + translation_detail(translation), label)


def model_length_zh(summary: str) -> str:
    match = re.search(r"\((\d+) chars, (\d+) tokens\)", summary)
    if not match:
        raise ValueError(f"unrecognized model-response length metadata: {summary}")
    chars, tokens = match.groups()
    return f"输出长度：{chars}个字符，{tokens}个词元"


def judge_format_zh(summary: str) -> str:
    match = re.search(r"\(([^()]*)\)\s*$", summary)
    if not match:
        raise ValueError(f"unrecognized native-judge format metadata: {summary}")
    formats = {
        "yes": "原生输出类型：单一肯定标签（是）",
        "non_binary": "原生输出类型：非二元（非单一是/否标签）",
    }
    try:
        return formats[match.group(1)]
    except KeyError as exc:
        raise ValueError(f"untranslated native-judge format: {match.group(1)}") from exc


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
            model_pattern = r'<details><summary>(Model response[^<]*)</summary><pre>(.*?)</pre></details>'
            model_match = re.search(model_pattern, article, re.DOTALL)
            if not model_match:
                raise ValueError(f"missing model-response summary in case {i}")
            source_hash = condition.get("model_response_source_sha256")
            if source_hash:
                source_response = html.unescape(model_match.group(2))
                actual_hash = hashlib.sha256(source_response.encode("utf-8")).hexdigest()
                if actual_hash != source_hash:
                    raise ValueError(f"source text does not match repetition recipe in case {i}")
            model_translation = model_length_zh(model_match.group(1)) + "\n\n" + condition["model_response"]
            article = add_after(model_pattern, article, model_translation,
                                f"case {i} {condition['condition']} model response")
            judge_pattern = r'<details><summary>(Full native judge output[^<]*)</summary><pre>.*?</pre></details>'
            judge_match = re.search(judge_pattern, article, re.DOTALL)
            if not judge_match:
                raise ValueError(f"missing native-judge summary in case {i}")
            judge_translation = judge_format_zh(judge_match.group(1)) + "\n\n" + condition["judge_output"]
            article = add_after(judge_pattern, article, judge_translation,
                                f"case {i} {condition['condition']} judge output")
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
    manifest = resolve_translation_files(manifest, args.translations.parent)
    result = render_bilingual(source, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(result)
    print(f"PASS: wrote bilingual packet ({len(result.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
