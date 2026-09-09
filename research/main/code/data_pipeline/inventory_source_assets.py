"""Inventory pinned payload resources and materialized source assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def tree_stats(root: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for directory, _, files in os.walk(root):
        if Path(directory).parts[-1] == ".cache":
            continue
        for name in files:
            path = Path(directory) / name
            if ".cache" in path.parts:
                continue
            try:
                size += path.stat().st_size
                count += 1
            except FileNotFoundError:
                continue
    return count, size


def read_revisions(path: Path) -> list[dict]:
    resources = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name, revision, destination = line.split("\t")
        resources.append({"resource": name, "revision": revision, "destination": destination})
    return resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revisions", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    resources = read_revisions(args.revisions)
    for resource in resources:
        destination = Path(resource["destination"])
        resource["exists"] = destination.is_dir()
        resource["file_count_excluding_cache"], resource["bytes_excluding_cache"] = (
            tree_stats(destination) if destination.is_dir() else (0, 0)
        )

    materialized = {}
    for name, relative in {
        "vision_opd": "data/processed/vision-opd-6k",
        "m3vqa": "data/processed/m3vqa/images",
    }.items():
        path = args.payload_root / relative
        count, size = tree_stats(path) if path.is_dir() else (0, 0)
        materialized[name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "complete_marker": (path / ".complete").is_file()
            or (path / ".images.complete").is_file(),
            "file_count": count,
            "bytes": size,
        }

    result = {"resources": resources, "materialized_assets": materialized}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
