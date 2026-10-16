#!/usr/bin/env python3
"""Translate a complete English ordinal-bullet sequence without dropping rows.

This is a deterministic linguistic helper, not a model translation service.
It changes only the English ordinal/list wording, preserving every line,
number, order, blank line, and non-list sentence.
"""
from __future__ import annotations

import re
import argparse
import html
from pathlib import Path


CARDINAL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
ORDINAL = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
    "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80,
    "ninetieth": 90, "hundredth": 100, "thousandth": 1000,
}


def english_number_value(phrase: str) -> int:
    """Parse standard English cardinal or ordinal number words through millions."""
    words = phrase.casefold().replace("-", " ").split()
    total = current = 0
    for word in words:
        if word == "and":
            continue
        if word in {"hundred", "hundredth"}:
            current = max(current, 1) * 100
        elif word in {"thousand", "thousandth"}:
            total += max(current, 1) * 1000
            current = 0
        else:
            value = ORDINAL.get(word, CARDINAL.get(word))
            if value is None:
                raise ValueError(f"unrecognized number word {word!r} in {phrase!r}")
            current += value
    return total + current


def translate_ordinal_bullet_list(text: str) -> str:
    """Translate every balcony-count bullet and retain all surrounding lines."""
    translated: list[str] = []
    rows = 0
    for line in text.splitlines():
        if line == "- Top balcony (with green window and black railing)":
            translated.append("- 第1个阳台（最上方，带绿色窗户和黑色栏杆）")
            rows += 1
            continue
        match = re.fullmatch(r"- (.+?) balcony down(?: \((.*?)\))?", line)
        if not match:
            translated.append(line)
            continue
        ordinal_phrase, parenthetical = match.groups()
        number = english_number_value(ordinal_phrase)
        suffix = "（从上往下）"
        if parenthetical:
            translated_parenthetical = {
                "same style": "样式相同",
            }.get(parenthetical)
            if translated_parenthetical is None:
                raise ValueError(f"untranslated parenthetical {parenthetical!r}")
            suffix = f"（从上往下，{translated_parenthetical}）"
        translated.append(f"- 第{number}个阳台{suffix}")
        rows += 1
    if rows == 0:
        raise ValueError("no ordinal-list rows found")
    return "\n".join(translated)


def translate_balcony_counting_response(text: str) -> str:
    """Translate all prose and every row in the complete balcony-counting answer."""
    lines = text.splitlines()
    expected_intro = [
        "Looking at the image, we can identify a building with red text on its facade. On the right side of this building, there are multiple balconies with black railings.",
        "",
        "Let’s count them:",
    ]
    if lines[:3] != expected_intro:
        raise ValueError("unexpected introductory text; refuse partial/ambiguous translation")
    translated = [
        "观察图像，可以辨认出一栋外立面上有红色文字的建筑。在这栋建筑右侧，可以看到多个带黑色栏杆的阳台。",
        "",
        "我们来数一数：",
    ]
    translated.extend(translate_ordinal_bullet_list("\n".join(lines[3:])).splitlines())
    if len(translated) != len(lines):
        raise AssertionError("translation changed the source line count")
    return "\n".join(translated)


def translate_numbered_color_response(text: str) -> str:
    """Translate every numbered yellow/white item while retaining malformed tails."""
    lines = text.splitlines()
    intros = {
        ("The painted curb in the image shows alternating color blocks: yellow and white.",
         "", "Counting them from left to right:"):
            ("图中的涂漆路缘石呈现出黄色和白色交替的色块。", "", "从左向右数："),
        ("The image shows a painted curb with alternating color blocks. From left to right, we can identify:",):
            ("图中是一段涂漆路缘石，上面有交替排列的色块。从左向右可以辨认出：",),
        ("The image shows a painted curb with alternating color blocks. From left to right, the visible blocks are:",):
            ("图中是一段涂漆路缘石，上面有交替排列的色块。从左向右可见的色块为：",),
    }
    for original_intro, translated_intro in intros.items():
        if tuple(lines[:len(original_intro)]) == original_intro:
            translated = list(translated_intro) + lines[len(original_intro):]
            break
    else:
        raise ValueError("unrecognized introduction; refuse partial/ambiguous translation")

    for index in range(len(translated_intro), len(translated)):
        line = translated[index]
        match = re.fullmatch(r"(\d+)\. (Yellow|White)( block)?", line)
        if match:
            number, color, block = match.groups()
            zh_color = "黄色" if color == "Yellow" else "白色"
            translated[index] = f"{number}. {zh_color}{'色块' if block else ''}"
        elif re.fullmatch(r"\d+\.?", line) or not line:
            # Preserve numeric-only entries, malformed numeric tails, and blanks.
            continue
        else:
            raise ValueError(f"untranslated or unexpected list content at line {index + 1}: {line!r}")

    if len(translated) != len(lines):
        raise AssertionError("translation changed the source line count")
    return "\n".join(translated)


def extract_response(source: str, case_index: int, condition_index: int) -> str:
    case_match = re.search(
        rf'<section class="case" id="case-{case_index}">(.*?)</section>', source, re.S
    )
    if not case_match:
        raise ValueError(f"case {case_index} not found")
    articles = re.findall(r'<article class="condition">(.*?)</article>', case_match.group(1), re.S)
    if not 0 <= condition_index < len(articles):
        raise ValueError(f"condition index {condition_index} out of range")
    response = re.search(
        r'<details><summary>Model response[^<]*</summary><pre>(.*?)</pre></details>',
        articles[condition_index], re.S,
    )
    if not response:
        raise ValueError("model response block not found")
    return html.unescape(response.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-html", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--condition-index", type=int, required=True,
                        help="zero-based index in S0, SC, S1, S2, S3 order")
    parser.add_argument("--kind", choices=("balcony", "color"), default="balcony")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_html.read_text(encoding="utf-8")
    original = extract_response(source, args.case_index, args.condition_index)
    translator = (translate_balcony_counting_response if args.kind == "balcony"
                  else translate_numbered_color_response)
    translation = translator(original)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(translation)
    print(f"PASS: translated {len(original.splitlines())} source lines without omission")


if __name__ == "__main__":
    main()
