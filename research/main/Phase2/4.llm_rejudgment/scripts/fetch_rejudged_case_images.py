#!/usr/bin/env python3
"""Fetch and hash-verify benchmark images for corrected paired-flip examples."""

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.repo_root
    w2 = root / "research/main/Phase2/2.short_output_prompt_study"
    w4 = root / "research/main/Phase2/4.llm_rejudgment"
    cases = json.loads((w4 / "derived/run-60011513/transitions/rejudged_case_manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads((w2 / "raw/resolved_image_manifest.json").read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in resolved["records"]}
    # A question may appear twice when it flips under both S1 and S3. Keep
    # those as separate analytical cases; uniqueness is required only for the
    # report's case indices, not for the underlying question/image IDs.
    if len(cases) != 12 or len({row["case_index"] for row in cases}) != 12:
        raise ValueError("expected twelve uniquely indexed corrected flip-case records")
    output = w4 / "raw/rejudged_case_images"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        image = by_id[case["sample_id"]]
        ext = Path(image["source_filename"]).suffix.lower()
        dest = output / f"case-{case['case_index']:02d}{ext}"
        if dest.exists():
            data = dest.read_bytes()
        else:
            url = BASE_URL + urllib.parse.quote(image["source_filename"], safe="/")
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "MOPD-2026-corrected-flip-report/1.0"})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        data = response.read()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
        digest = hashlib.sha256(data).hexdigest()
        if digest != image["sha256"] or len(data) != image["bytes"]:
            raise ValueError(f"image checksum/size mismatch for case {case['case_index']}")
        dest.write_bytes(data)
        records.append({
            **case,
            "file": dest.name,
            "source_filename": image["source_filename"],
            "bytes": len(data),
            "sha256": digest,
            "source_revision": REVISION,
        })
        print(f"verified case {case['case_index']:02d}: {dest.name} ({len(data)} bytes)")
    (output / "image_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
