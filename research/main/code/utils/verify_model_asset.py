#!/usr/bin/env python3
"""Verify a relocated local model against the frozen revision and file inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FROZEN_REVISIONS_SHA256 = "ca5044669a0fa91c0f1aec1b90d03abd285ead153d38c02300fe49cb9f2c7b00"
FROZEN_CHECKSUMS_SHA256 = "18e81c36cb698633bc266a6f40f934b84ea2678dfd890f8b5aba72845b90b4ef"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_revision(revisions_tsv: Path, repo: str) -> dict[str, str]:
    with revisions_tsv.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source, delimiter="\t") if row.get("repo") == repo]
    if len(rows) != 1:
        raise ValueError(f"revision inventory must contain exactly one row for {repo}")
    return rows[0]


def load_expected_files(checksums_path: Path, model_basename: str) -> dict[str, str]:
    expected = {}
    with checksums_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            parts = line.rstrip("\n").split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"invalid checksum inventory line {line_number}")
            digest, archived_path = parts
            archived = Path(archived_path.strip())
            if archived.parent.name != model_basename:
                continue
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"invalid checksum on line {line_number}")
            if archived.name in expected:
                raise ValueError(f"duplicate expected model filename: {archived.name}")
            expected[archived.name] = digest
    if not expected:
        raise ValueError(f"no checksum records for model directory {model_basename}")
    return expected


def verify_model_asset(
    *,
    model_dir: Path,
    repo: str,
    revision: str,
    revisions_tsv: Path,
    checksums_path: Path,
    expected_revisions_sha256: str = FROZEN_REVISIONS_SHA256,
    expected_checksums_sha256: str = FROZEN_CHECKSUMS_SHA256,
) -> dict:
    errors = []
    resolved_model = model_dir.resolve(strict=True)
    revisions_hash = sha256_file(revisions_tsv)
    checksums_hash = sha256_file(checksums_path)
    if revisions_hash != expected_revisions_sha256:
        errors.append("revisions_inventory_sha256_mismatch")
    if checksums_hash != expected_checksums_sha256:
        errors.append("checksums_inventory_sha256_mismatch")

    revision_row = load_revision(revisions_tsv, repo)
    if revision_row.get("revision") != revision:
        errors.append("model_revision_mismatch")
    archived_basename = Path(str(revision_row.get("path", ""))).name
    if not archived_basename or resolved_model.name != archived_basename:
        errors.append("model_directory_basename_mismatch")

    expected = load_expected_files(checksums_path, archived_basename)
    observed_names = sorted(path.name for path in resolved_model.iterdir() if path.is_file())
    expected_names = sorted(expected)
    missing = sorted(set(expected_names) - set(observed_names))
    unexpected = sorted(set(observed_names) - set(expected_names))
    if missing:
        errors.append("missing_model_files")
    if unexpected:
        errors.append("unexpected_model_files")

    files = {}
    for name in sorted(set(expected_names) & set(observed_names)):
        path = resolved_model / name
        actual = sha256_file(path)
        matches = actual == expected[name]
        if not matches:
            errors.append(f"model_file_sha256_mismatch:{name}")
        files[name] = {
            "bytes": path.stat().st_size,
            "expected_sha256": expected[name],
            "actual_sha256": actual,
            "matches": matches,
        }

    return {
        "status": "passed" if not errors else "failed",
        "repo": repo,
        "revision": revision,
        "model_dir": str(resolved_model),
        "model_directory_basename": resolved_model.name,
        "revisions_inventory": str(revisions_tsv.resolve()),
        "revisions_inventory_sha256": revisions_hash,
        "checksums_inventory": str(checksums_path.resolve()),
        "checksums_inventory_sha256": checksums_hash,
        "expected_files": len(expected_names),
        "observed_files": len(observed_names),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "verified_bytes": sum(item["bytes"] for item in files.values()),
        "files": files,
        "validation_errors": sorted(set(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--revisions-tsv", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite model asset gate: {args.output}")
    try:
        summary = verify_model_asset(
            model_dir=args.model_dir,
            repo=args.repo,
            revision=args.revision,
            revisions_tsv=args.revisions_tsv,
            checksums_path=args.checksums,
        )
    except (OSError, ValueError) as error:
        summary = {"status": "failed", "validation_errors": [f"{type(error).__name__}: {error}"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        sink.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit("model asset gate failed")


if __name__ == "__main__":
    main()
