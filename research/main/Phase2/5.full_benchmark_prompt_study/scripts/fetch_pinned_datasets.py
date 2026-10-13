#!/usr/bin/env python3
"""Fetch the exact Vision-OPD dataset revisions into the designated data volume."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


DATA_ROOT = Path(os.environ.get(
    "WORK5_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data",
)).resolve()
HERE = Path(__file__).resolve().parents[1]
SOURCES = HERE / "dataset_sources.tsv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cache = DATA_ROOT / "hf-cache"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["TMPDIR"] = "/tmp"
    audit = DATA_ROOT / "source-audit"
    audit.mkdir(exist_ok=True)
    manifest = []
    with SOURCES.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            repo, revision = row["hub_repo"], row["revision"]
            target = DATA_ROOT / "datasets" / row["benchmark"]
            target.mkdir(parents=True, exist_ok=True)
            parquet_pattern = {
                "hrbench-4k": "**/hr_bench_4k.parquet",
                "hrbench-8k": "**/hr_bench_8k.parquet",
            }.get(row["benchmark"], "*.parquet")
            parquet_patterns = [parquet_pattern]
            if parquet_pattern.startswith("**/"):
                parquet_patterns.append(parquet_pattern.removeprefix("**/"))
            print(f"FETCH {repo}@{revision} -> {target}", flush=True)
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                local_dir=str(target),
                cache_dir=str(cache / "hub"),
                allow_patterns=parquet_patterns + ["README.md", "*.json", "**/*.json"],
            )
            if row["benchmark"].startswith("hrbench-"):
                matching = sorted(target.rglob(Path(parquet_pattern).name))
            else:
                matching = sorted(target.rglob("*.parquet"))
            for path in matching:
                metadata = pq.ParquetFile(path).metadata
                manifest.append({
                    "benchmark": row["benchmark"],
                    "repo": repo,
                    "revision": revision,
                    "file": str(path.relative_to(DATA_ROOT)),
                    "rows": metadata.num_rows,
                    "row_groups": metadata.num_row_groups,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
                print(f"  {path.name}: rows={metadata.num_rows} bytes={path.stat().st_size}", flush=True)
    out = audit / "pinned-parquet-manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    totals = {}
    for item in manifest:
        totals[item["benchmark"]] = totals.get(item["benchmark"], 0) + item["rows"]
    (audit / "row-counts.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
