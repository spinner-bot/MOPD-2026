#!/usr/bin/env python3
"""Build the immutable multimodal inference input from frozen manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample = read_json(args.sample)
    image_manifest = read_json(args.images)
    image_by_id = {row["sample_id"]: row for row in image_manifest["records"]}
    records = []
    for row in sample["records"]:
        sample_id = row["sample_id"]
        if sample_id not in image_by_id:
            raise SystemExit(f"missing resolved image for {sample_id}")
        image = image_by_id[sample_id]
        records.append(
            {
                **row,
                "image_archive_path": image["archive_path"],
                "image_sha256": image["sha256"],
            }
        )

    payload = {
        "schema": "short-output-inference-input-v1",
        "sample_manifest": str(args.sample),
        "image_manifest": str(args.images),
        "count": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
