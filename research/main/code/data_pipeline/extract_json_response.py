"""Conservatively recover one JSON object from a model response without editing it."""
from __future__ import annotations

import json
import re


def extract_json_object(raw: str) -> dict | None:
    """Prefer a fenced JSON block, then the last balanced object that parses."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = fenced + _balanced_objects(raw)
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _balanced_objects(text: str) -> list[str]:
    objects, start, depth, quoted, escaped = [], None, 0, False, False
    for index, char in enumerate(text):
        if quoted:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == '"': quoted = False
            continue
        if char == '"': quoted = True
        elif char == "{":
            if depth == 0: start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None: objects.append(text[start:index + 1])
    return objects
