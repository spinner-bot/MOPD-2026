#!/usr/bin/env python3
"""Validate a completed human-review export for the frozen 60 x 5 Work5 packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from full_campaign_layout import CONDITIONS, TRACKS


FIELDS = ("extraction_faithful", "semantic_correct", "judge_valid")
VALUES = {
    "Yes / 是": "yes",
    "No / 否": "no",
    "Unclear / 不确定": "unclear",
    "Pending / 待审": "pending",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_review_export(
    export: dict[str, Any], sample_manifest: dict[str, Any], reviewer: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if export.get("schema") != "work5-manual-review-decisions-v1":
        raise ValueError("Unsupported or missing human-review export schema")
    if not reviewer.strip():
        raise ValueError("Reviewer identifier must be nonempty")
    selection = sample_manifest.get("tracks_selection")
    if sample_manifest.get("schema") != "phase2-work5-manual-audit-sample-v2":
        raise ValueError("Expected the frozen manual-audit sample v2 manifest")
    if not isinstance(selection, dict) or set(selection) != set(TRACKS):
        raise ValueError("Sample manifest does not cover exactly the six Work5 tracks")

    case_lookup: dict[int, tuple[str, str]] = {}
    for track in TRACKS:
        selected = selection[track].get("selected", [])
        if len(selected) != 10:
            raise ValueError(f"{track}: expected ten selected cases")
        for item in selected:
            case_index = len(case_lookup) + 1
            case_lookup[case_index] = (track, item["sample_uid"])
    if len(case_lookup) != 60:
        raise ValueError("Expected exactly 60 selected cases")

    rows = export.get("rows")
    if not isinstance(rows, list) or len(rows) != 300:
        raise ValueError(f"Expected exactly 300 paired review rows, got {len(rows) if isinstance(rows, list) else 'invalid'}")
    case_pattern = re.compile(r"^\s*Case\s+(\d+)\b", re.IGNORECASE)
    condition_pattern = re.compile(r"^\s*(S0|SC|S1|S2|S3)\s*[·|]", re.IGNORECASE)
    normalized: dict[tuple[int, str], dict[str, Any]] = {}
    field_counts = {field: Counter() for field in FIELDS}

    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Review row {row_number} is not an object")
        case_match = case_pattern.search(str(row.get("case_title", "")))
        condition_match = condition_pattern.search(str(row.get("condition", "")))
        if not case_match or not condition_match:
            raise ValueError(f"Review row {row_number} has an unrecognized case/condition label")
        case_index = int(case_match.group(1))
        condition = condition_match.group(1).upper()
        if case_index not in case_lookup or condition not in CONDITIONS:
            raise ValueError(f"Review row {row_number} has an out-of-range case or condition")
        key = (case_index, condition)
        if key in normalized:
            raise ValueError(f"Duplicate review entry for case {case_index}/{condition}")
        expected_track, sample_uid = case_lookup[case_index]
        if expected_track not in str(row.get("condition", "")):
            raise ValueError(f"Review row {row_number} track disagrees with the frozen case index")
        decisions = row.get("decisions")
        if not isinstance(decisions, dict) or set(decisions) != {*FIELDS, "note"}:
            raise ValueError(f"Review row {row_number} has missing or unexpected decision fields")
        normalized_decisions = {}
        for field in FIELDS:
            raw_value = decisions[field]
            if raw_value not in VALUES:
                raise ValueError(f"Review row {row_number}/{field} has invalid value {raw_value!r}")
            value = VALUES[raw_value]
            if value == "pending":
                raise ValueError(f"Review row {row_number}/{field} is still pending")
            field_counts[field][value] += 1
            normalized_decisions[field] = value
        note = decisions["note"]
        if not isinstance(note, str):
            raise ValueError(f"Review row {row_number} note must be text")
        if any(value in {"no", "unclear"} for value in normalized_decisions.values()) and not note.strip():
            raise ValueError(f"Review row {row_number} has a No/Unclear decision without an evidence note")
        normalized[key] = {
            "case_index": case_index,
            "track": expected_track,
            "sample_uid": sample_uid,
            "condition": condition,
            **normalized_decisions,
            "note": note.strip(),
        }

    expected_keys = {(case_index, condition) for case_index in case_lookup for condition in CONDITIONS}
    if set(normalized) != expected_keys:
        missing = sorted(expected_keys - set(normalized))[:5]
        raise ValueError(f"Review rows do not cover all 60 x 5 pairs; missing examples: {missing}")
    completed_rows = [normalized[key] for key in sorted(expected_keys)]
    summary = {
        "schema": "phase2-work5-manual-review-validation-v1",
        "reviewer": reviewer.strip(),
        "completed_rows": len(completed_rows),
        "cases": len(case_lookup),
        "conditions": list(CONDITIONS),
        "decision_counts": {field: dict(counts) for field, counts in field_counts.items()},
        "rows_with_evidence_notes": sum(bool(row["note"]) for row in completed_rows),
        "not_an_automatic_score_replacement": True,
    }
    return completed_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True,
                        help="JSON downloaded from the manual review HTML packet")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True,
                        help="Exact HTML packet used for review")
    parser.add_argument("--reviewer", required=True,
                        help="Reviewer name/identifier to record in the validation manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing review validation: {args.output}")
    export = json.loads(args.decisions.read_text(encoding="utf-8"))
    manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    completed_rows, summary = validate_review_export(export, manifest, args.reviewer)
    summary.update({
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_export_sha256": sha256_file(args.decisions),
        "sample_manifest_sha256": sha256_file(args.sample_manifest),
        "review_packet_sha256": sha256_file(args.packet),
    })
    summary["review_rows"] = completed_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"PASS: validated {summary['completed_rows']} decisions by {summary['reviewer']}")
    print(f"output={args.output}; export_sha256={summary['decision_export_sha256']}")


if __name__ == "__main__":
    main()
