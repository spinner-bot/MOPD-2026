#!/usr/bin/env python3
"""Validate completed B2a blind human reviews without auto-filling labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_image_affordance_blind_calibration import (
    load_jsonl,
    validate_completed_review,
)


def require_new_outputs(paths: list[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("accepted, rejected, and summary outputs must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError("refusing to overwrite existing finalizer output: " + ",".join(existing))


def write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as sink:
        sink.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def finalize(original_rows: list[dict], reviewed_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    original_by_id = {}
    for row in original_rows:
        key = str(row.get("calibration_record_id", "")).strip()
        if not key or key in original_by_id:
            raise ValueError("missing or duplicate original calibration_record_id")
        original_by_id[key] = row

    reviewed_by_id = {}
    for row in reviewed_rows:
        key = str(row.get("calibration_record_id", "")).strip()
        if not key or key in reviewed_by_id:
            raise ValueError("missing or duplicate reviewed calibration_record_id")
        if key not in original_by_id:
            raise ValueError(f"unknown reviewed calibration_record_id: {key}")
        reviewed_by_id[key] = row
    if set(reviewed_by_id) != set(original_by_id):
        missing = sorted(set(original_by_id) - set(reviewed_by_id))
        raise ValueError(f"reviewed packet is incomplete: missing {','.join(missing)}")

    accepted, rejected = [], []
    reasons = Counter()
    decisions = Counter()
    for key in original_by_id:
        row = reviewed_by_id[key]
        errors = validate_completed_review(original_by_id[key], row)
        if errors:
            rejected.append({
                "calibration_record_id": key,
                "image_candidate_id": row.get("image_candidate_id"),
                "validation_errors": errors,
            })
            reasons.update(errors)
        else:
            accepted.append(row)
            decisions[str(row["human_affordance_decision"])] += 1
    summary = {
        "original_records": len(original_rows),
        "reviewed_records": len(reviewed_rows),
        "valid_signed_reviews": len(accepted),
        "invalid_reviews": len(rejected),
        "valid_decisions": dict(sorted(decisions.items())),
        "validation_errors": dict(sorted(reasons.items())),
        "machine_proposals_exposed": False,
        "source_questions_exposed": False,
        "automatically_filled_human_fields": 0,
    }
    return accepted, rejected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    try:
        require_new_outputs([args.accepted, args.rejected, args.summary])
        accepted, rejected, summary = finalize(
            load_jsonl(args.original), load_jsonl(args.reviewed)
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_jsonl_exclusive(args.accepted, accepted)
    write_jsonl_exclusive(args.rejected, rejected)
    write_json_exclusive(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
