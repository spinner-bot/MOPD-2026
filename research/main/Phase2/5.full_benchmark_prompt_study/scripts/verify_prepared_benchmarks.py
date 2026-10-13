#!/usr/bin/env python3
"""Independent post-write gate for prepared records, media hashes, and paired IDs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "WORK5_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data",
)).resolve()
PREPARED = ROOT / "prepared"
SOURCE_COUNTS = json.loads((ROOT / "source-audit/row-counts.json").read_text(encoding="utf-8"))
PREP_AUDIT = json.loads((ROOT / "source-audit/prepared-input-audit.json").read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if set(SOURCE_COUNTS) != set(PREP_AUDIT):
        raise ValueError("prepared audit tracks do not match pinned source-count tracks")
    summary = {"tracks": {}, "total_rows": 0, "total_full_images": 0, "total_crop_images": 0}
    for track, source_rows in SOURCE_COUNTS.items():
        directory = PREPARED / track
        input_path = directory / "benchmark.json"
        media_path = directory / "media-manifest.json"
        report = PREP_AUDIT[track]
        if input_path.stat().st_size == 0 or file_sha256(input_path) != report["benchmark_json_sha256"]:
            raise ValueError(f"{track}: prepared JSON digest mismatch")
        records = json.loads(input_path.read_text(encoding="utf-8"))
        if len(records) != source_rows or report["rows"] != source_rows:
            raise ValueError(f"{track}: row count differs from fixed-revision source")
        uids = [row.get("sample_uid") for row in records]
        if None in uids or len(set(uids)) != source_rows:
            raise ValueError(f"{track}: missing or duplicate sample_uid")
        if any(not row.get("query") or not row.get("response") for row in records):
            raise ValueError(f"{track}: empty query/reference answer")

        media = json.loads(media_path.read_text(encoding="utf-8"))
        indexed = {}
        crops_by_uid = {}
        for item in media:
            key = (item["sample_uid"], item["kind"])
            if key in indexed:
                raise ValueError(f"{track}: duplicate media-manifest key {key}")
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise ValueError(f"{track}: missing/size-mismatched media {path}")
            if file_sha256(path) != item["sha256"]:
                raise ValueError(f"{track}: SHA-256 mismatch for {path}")
            indexed[key] = item
            if item["kind"] == "crop":
                crops_by_uid.setdefault(item["sample_uid"], []).append(item["path"])
        for row in records:
            uid = row["sample_uid"]
            full = indexed.get((uid, "full"))
            if full is None or row.get("images") != [full["path"]]:
                raise ValueError(f"{track}: full-image reference mismatch for {uid}")
            crops = crops_by_uid.get(uid, [])
            if row.get("crop_images", []) != crops:
                raise ValueError(f"{track}: crop-image reference mismatch for {uid}")

        full_count = sum(1 for item in media if item["kind"] == "full")
        crop_count = sum(1 for item in media if item["kind"] == "crop")
        if full_count != source_rows:
            raise ValueError(f"{track}: expected one full image per QA, found {full_count}")
        summary["tracks"][track] = {
            "rows": len(records),
            "unique_uids": len(uids),
            "full_images": full_count,
            "crop_images": crop_count,
            "benchmark_json_sha256": report["benchmark_json_sha256"],
            "media_files_hashed": len(media),
        }
        summary["total_rows"] += len(records)
        summary["total_full_images"] += full_count
        summary["total_crop_images"] += crop_count
        print(f"PASS {track}: rows={len(records)} full={full_count} crop={crop_count}", flush=True)

    if summary["total_rows"] != sum(SOURCE_COUNTS.values()):
        raise ValueError("overall prepared sample total mismatch")
    summary["five_condition_generation_requests"] = 5 * summary["total_rows"]
    output = ROOT / "source-audit/prepared-integrity-verification.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS ALL: {summary['total_rows']} rows, {summary['total_full_images']} full images, "
          f"{summary['total_crop_images']} crop images, {summary['five_condition_generation_requests']} generation requests", flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
