#!/usr/bin/env python3
"""Inspect only task-format edge cases; never load image bytes."""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(os.environ.get(
    "WORK5_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data",
)).resolve()
DATA = ROOT / "datasets"
OUT = ROOT / "source-audit" / "task-edge-cases.json"


def short(value, limit=240):
    value = "" if value is None else str(value)
    return value if len(value) <= limit else value[:limit] + "…"


def main() -> None:
    zoom_table = pq.read_table(DATA / "zoombench/data/test.parquet", columns=["id", "query", "response", "question_type"])
    by_type = defaultdict(Counter)
    zoom_examples = {}
    for row in zoom_table.to_pylist():
        qt = str(row.get("question_type") or "").strip().casefold() or "<blank>"
        answer = str(row.get("response") or "").strip()
        answer_kind = "letter" if re.fullmatch(r"[A-Z]", answer) else "numeric" if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer) else "other"
        by_type[qt][answer_kind] += 1
        zoom_examples.setdefault(qt, []).append({
            "source_id": row.get("id"),
            "query_excerpt": short(row.get("query")),
            "reference_answer": short(answer, 80),
        })
    for values in zoom_examples.values():
        del values[3:]

    mme_edges = {}
    for track, folder in [("mme-realworld", "mme-realworld"), ("mme-realworld-cn", "mme-realworld-cn")]:
        files = sorted((DATA / folder).rglob("*.parquet"))
        examples = []
        counts = Counter()
        for file in files:
            table = pq.read_table(file, columns=["index", "question", "multi-choice options", "answer"])
            for row in table.to_pylist():
                options = row.get("multi-choice options") or []
                counts[f"n={len(options)}"] += 1
                if len(options) == 6 and len(examples) < 40:
                    examples.append({
                        "index": row.get("index"),
                        "answer": row.get("answer"),
                        "question_excerpt": short(row.get("question")),
                        "options": [short(option, 300) for option in options],
                    })
        mme_edges[track] = {"option_count": dict(counts), "six_option_examples": examples}

    result = {
        "zoombench_question_type_by_reference_format": {
            key: dict(value) for key, value in by_type.items()
        },
        "zoombench_examples_by_type": zoom_examples,
        "mme_option_space_and_six_option_cases": mme_edges,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
