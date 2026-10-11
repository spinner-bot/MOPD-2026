#!/usr/bin/env python3
"""Extract the frozen CV-Bench sample images into a persistent archive."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--two-d-parquet", required=True, type=Path)
    parser.add_argument("--three-d-parquet", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    records = sample.get("records", [])
    parquet_by_split = {
        "2D": args.two_d_parquet,
        "3D": args.three_d_parquet,
    }
    lookup = {}
    for split, parquet_path in parquet_by_split.items():
        table = pq.read_table(parquet_path, columns=["filename", "image"])
        for row in table.to_pylist():
            filename = row["filename"]
            image = row["image"]["bytes"]
            lookup[(split, filename)] = image

    args.output_root.mkdir(parents=True, exist_ok=True)
    resolved = []
    for record in records:
        key = (record["split"], record["filename"])
        image = lookup.get(key)
        if image is None:
            raise ValueError(f"missing image for {key}")
        suffix = Path(record["filename"]).suffix or ".jpg"
        rel = Path(record["split"]) / (record["sample_id"].replace("/", "_") + suffix)
        out = args.output_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(image)
        resolved.append({
            "sample_id": record["sample_id"],
            "split": record["split"],
            "source_filename": record["filename"],
            "archive_path": str(rel),
            "bytes": len(image),
            "sha256": hashlib.sha256(image).hexdigest(),
        })

    payload = {
        "schema_version": "cvbench-resolved-sample-v1",
        "sample_manifest_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "count": len(resolved),
        "records": resolved,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(resolved), "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

