#!/usr/bin/env python3
"""Audit pinned parquet schemas and task/category frequencies without copying media."""
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
OUT = ROOT / "source-audit" / "schema-and-category-audit.json"
TRACKS = {
    "vstar": "vstar",
    "zoombench": "zoombench",
    "hrbench-4k": "hrbench-4k",
    "hrbench-8k": "hrbench-8k",
    "mme-realworld": "mme-realworld",
    "mme-realworld-cn": "mme-realworld-cn",
}
MEDIA_NAMES = {"image", "images", "bytes", "crop_image", "crop_images"}


def main() -> None:
    result = {}
    for track, folder in TRACKS.items():
        directory = DATA / folder
        files = sorted(directory.rglob("*.parquet"))
        if track == "hrbench-4k":
            files = [p for p in files if p.name == "hr_bench_4k.parquet"]
        elif track == "hrbench-8k":
            files = [p for p in files if p.name == "hr_bench_8k.parquet"]
        schemas = []
        frequencies: dict[str, Counter] = {}
        rows = 0
        for file in files:
            pf = pq.ParquetFile(file)
            rows += pf.metadata.num_rows
            names = pf.schema_arrow.names
            if not schemas:
                schemas = [{"name": field.name, "type": str(field.type)} for field in pf.schema_arrow]
                selected = [
                    name for name in names
                    if any(tag in name.casefold() for tag in ("category", "type", "task", "subset"))
                    and name.casefold() not in MEDIA_NAMES
                ]
                frequencies = {name: Counter() for name in selected}
            if frequencies:
                table = pq.read_table(file, columns=list(frequencies))
                for name in frequencies:
                    frequencies[name].update(
                        "<NULL>" if value is None else str(value)
                        for value in table[name].to_pylist()
                    )
        result[track] = {
            "files": [str(p.relative_to(ROOT)) for p in files],
            "rows": rows,
            "schema": schemas,
            "category_or_task_frequencies": {
                name: dict(sorted(counter.items())) for name, counter in frequencies.items()
            },
        }
        print(f"{track}: {rows} rows; {len(files)} parquet files", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
