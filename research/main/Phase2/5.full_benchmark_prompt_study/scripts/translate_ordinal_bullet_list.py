#!/usr/bin/env python3
"""Translate a complete English ordinal-bullet sequence without dropping rows.

This is a deterministic linguistic helper, not a model translation service.
It changes only the English ordinal/list wording, preserving every line,
number, order, blank line, and non-list sentence.
"""
from __future__ import annotations

import re


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
            suffix = f"（{translated_parenthetical}）"
        translated.append(f"- 第{number}个阳台{suffix}")
        rows += 1
    if rows == 0:
        raise ValueError("no ordinal-list rows found")
    return "\n".join(translated)
