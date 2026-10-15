#!/usr/bin/env python3
"""Read-only verification of every source hash recorded by a Work5 analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "phase2-work5-statistics-v4"
EXPECTED_ROWS = 160810


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_analysis_sources(analysis_dir: Path, workers: int = 8) -> dict[str, Any]:
    manifest_path = analysis_dir / "analysis-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"Expected {EXPECTED_SCHEMA} manifest")
    if manifest.get("joined_records") != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} joined records")
    source_files = manifest.get("audit", {}).get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("Manifest contains no audit.source_files entries")

    normalized: list[tuple[Path, str]] = []
    for item in source_files:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError("Every source entry must contain path and sha256")
        normalized.append((Path(item["path"]), str(item["sha256"]).lower()))

    paths = [str(path) for path, _ in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError("Manifest contains duplicate source paths")

    def check(entry: tuple[Path, str]) -> dict[str, str]:
        path, expected = entry
        if not path.is_file():
            return {"path": str(path), "status": "missing", "expected": expected}
        actual = sha256_file(path)
        if actual != expected:
            return {"path": str(path), "status": "mismatch", "expected": expected, "actual": actual}
        return {"path": str(path), "status": "ok"}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(check, normalized))
    failures = [item for item in results if item["status"] != "ok"]
    return {
        "schema": EXPECTED_SCHEMA,
        "joined_records": EXPECTED_ROWS,
        "manifest": str(manifest_path),
        "source_count": len(results),
        "verified_count": len(results) - len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = verify_analysis_sources(args.analysis_dir, args.workers)
    if result["failures"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(f"PASS: SHA-256 verified for all {result['verified_count']} recorded analysis inputs")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
