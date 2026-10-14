#!/usr/bin/env python3
"""Fetch pinned CV-Bench images for the twelve published cases and verify bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


REVISION = "eb5a19ace092c73ff4e61da6ceda16e190bf32dc"
BASE_URL = f"https://huggingface.co/datasets/nyu-visionx/CV-Bench/resolve/{REVISION}/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    args = parser.parse_args()

    cases = json.loads((args.work_root / "derived/run-60011513/case_manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads((args.source_root / "raw/resolved_image_manifest.json").read_text(encoding="utf-8"))
    by_id = {record["sample_id"]: record for record in resolved["records"]}
    assert len(cases) == 12 and len({case["sample_id"] for case in cases}) == 12
    output = args.work_root / "raw/case_images"
    output.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for case in cases:
        image = by_id[case["sample_id"]]
        source_path = image["source_filename"]
        url = BASE_URL + urllib.parse.quote(source_path, safe="/")
        destination = output / f"case-{case['case_index']:02d}{Path(source_path).suffix.lower()}"
        if destination.exists():
            data = destination.read_bytes()
        else:
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "MOPD-2026-case-image-audit/1.0"})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        data = response.read()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
        digest = hashlib.sha256(data).hexdigest()
        if digest != image["sha256"] or len(data) != image["bytes"]:
            raise ValueError(f"case {case['case_index']:02d}: source image does not match frozen image manifest")
        destination.write_bytes(data)
        downloaded.append({
            "case_index": case["case_index"],
            "sample_id": case["sample_id"],
            "file": destination.name,
            "source_filename": source_path,
            "bytes": len(data),
            "sha256": digest,
            "source_revision": REVISION,
        })
        print(f"verified case {case['case_index']:02d}: {destination.name} ({len(data)} bytes)")

    (output / "case_image_manifest.json").write_text(
        json.dumps(downloaded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
