#!/usr/bin/env python3
"""Check stable-key uniqueness and answer/task fields without reading image payloads."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(os.environ.get(
    "WORK5_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data",
)).resolve()
DATA = ROOT / "datasets"
OUT = ROOT / "source-audit" / "sample-key-answer-audit.json"
SPECS = {
    "vstar": ("vstar", "question_id", ["label", "category"]),
    "zoombench": ("zoombench", "id", ["question_type", "response"]),
    "hrbench-4k": ("hrbench-4k", "index", ["answer", "category", "cycle_category"]),
    "hrbench-8k": ("hrbench-8k", "index", ["answer", "category", "cycle_category"]),
    "mme-realworld": ("mme-realworld", "index", ["answer", "category", "l2-category", "multi-choice options"]),
    "mme-realworld-cn": ("mme-realworld-cn", "index", ["answer", "category", "l2-category", "multi-choice options"]),
}


def main() -> None:
    result = {}
    for track, (folder, key, candidates) in SPECS.items():
        files = sorted((DATA / folder).rglob("*.parquet"))
        if track == "hrbench-4k":
            files = [p for p in files if p.name == "hr_bench_4k.parquet"]
        elif track == "hrbench-8k":
            files = [p for p in files if p.name == "hr_bench_8k.parquet"]
        counts = {name: Counter() for name in candidates}
        keys, missing_keys, rows = set(), 0, 0
        for file in files:
            schema = pq.read_schema(file)
            columns = [name for name in [key, *candidates] if name in schema.names]
            table = pq.read_table(file, columns=columns)
            pyrows = table.to_pylist()
            rows += len(pyrows)
            for row in pyrows:
                value = row.get(key)
                if value is None or str(value).strip() == "":
                    missing_keys += 1
                else:
                    keys.add(str(value))
                for name in candidates:
                    if name not in row:
                        continue
                    value = row[name]
                    if name == "multi-choice options":
                        value = f"n={len(value or [])}"
                    elif name == "question_type":
                        value = str(value or "").strip().casefold() or "<blank>"
                    elif isinstance(value, list):
                        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                    else:
                        value = "<NULL>" if value is None else str(value)
                    counts[name][value] += 1
        result[track] = {
            "rows": rows,
            "key_field": key,
            "unique_nonempty_keys": len(keys),
            "missing_keys": missing_keys,
            "duplicate_keys": rows - missing_keys - len(keys),
            "field_counts": {name: dict(sorted(counter.items())) for name, counter in counts.items()},
        }
        print(f"{track}: rows={rows} unique_keys={len(keys)} missing={missing_keys} duplicates={rows - missing_keys - len(keys)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
