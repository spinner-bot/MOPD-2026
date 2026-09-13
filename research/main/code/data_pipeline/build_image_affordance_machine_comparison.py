#!/usr/bin/env python3
"""Reveal B1 machine proposals only after a complete B2a blind review is locked."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_image_affordance_blind_calibration import load_jsonl, stable_packet_sha256
from finalize_image_affordance_blind_review import (
    finalize,
    require_new_outputs,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from process_image_affordance_screen import load as load_machine_raw
from process_image_affordance_screen import parse_records


EXPECTED_PACKET_SHA256 = "b61839ddb4111f1be5d2d67f165c40dc05cd10daf8d5ca02a202b707a1057a58"
EXPECTED_RECORDS = 32
PHASE2_HUMAN_FIELDS = (
    "human_machine_description_faithful",
    "human_machine_path_match",
    "human_machine_false_two_step",
    "human_machine_comparison_notes",
    "human_machine_comparison_reviewer",
    "human_machine_comparison_reviewed_at",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_comparison_packet(
    original_rows: list[dict],
    reviewed_rows: list[dict],
    machine_raw_rows: list[dict],
    *,
    reviewed_packet_sha256: str,
    expected_packet_sha256: str = EXPECTED_PACKET_SHA256,
    expected_records: int = EXPECTED_RECORDS,
) -> tuple[list[dict], dict]:
    if stable_packet_sha256(original_rows) != expected_packet_sha256:
        raise ValueError("unexpected frozen B2a packet SHA-256")
    if len(original_rows) != expected_records:
        raise ValueError(f"expected {expected_records} frozen records")
    try:
        accepted, rejected, _ = finalize(original_rows, reviewed_rows)
    except ValueError as error:
        raise ValueError(f"B2a review must be complete before machine reveal: {error}") from error
    if rejected or len(accepted) != expected_records:
        raise ValueError("B2a review must be complete before machine reveal")

    original_ids = {row["image_candidate_id"] for row in original_rows}
    raw_ids = [str(row.get("image_candidate_id", "")).strip() for row in machine_raw_rows]
    if any(not key for key in raw_ids) or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("machine raw records have missing or duplicate image_candidate_id")
    unknown = sorted(set(raw_ids) - original_ids)
    if unknown:
        raise ValueError(f"machine raw record outside B2a packet: {unknown[0]}")

    parsed, machine_rejected, parse_summary = parse_records(machine_raw_rows)
    valid_by_id = {row["image_candidate_id"]: row for row in parsed}
    invalid_by_id = {row["image_candidate_id"]: row for row in machine_rejected}
    if set(valid_by_id) & set(invalid_by_id):
        raise ValueError("machine result is both valid and invalid")

    output = []
    status_counts = Counter()
    contingency = Counter()
    for review in accepted:
        key = review["image_candidate_id"]
        row = dict(review)
        if key in valid_by_id:
            machine = valid_by_id[key]
            proposal = machine["proposal"]
            status = "parsed_valid"
            decision = proposal["decision"]
            rejection_reasons: list[str] = []
            provenance = machine["raw_record_provenance"]
            human_positive = review["human_affordance_decision"] == "candidate"
            machine_positive = decision == "candidate"
            contingency[
                ("human_candidate" if human_positive else "human_noncandidate")
                + "__"
                + ("machine_candidate" if machine_positive else "machine_noncandidate")
            ] += 1
        elif key in invalid_by_id:
            machine = invalid_by_id[key]
            proposal = machine.get("proposal")
            status = "invalid"
            decision = None
            rejection_reasons = machine["rejection_reasons"]
            provenance = machine["raw_record_provenance"]
        else:
            proposal = None
            status = "unavailable"
            decision = None
            rejection_reasons = []
            provenance = None
        status_counts[status] += 1
        row.update({
            "comparison_phase": "phase_2_machine_reveal",
            "phase_1_review_locked": True,
            "phase_1_reviewed_packet_sha256": reviewed_packet_sha256,
            "machine_output_status": status,
            "machine_decision": decision,
            "machine_proposal": proposal,
            "machine_rejection_reasons": rejection_reasons,
            "machine_provenance": provenance,
            "human_machine_comparison_verified": False,
            "human_machine_comparison_status": "pending",
        })
        row.update({field: "" for field in PHASE2_HUMAN_FIELDS})
        output.append(row)

    machine_candidates = contingency["human_candidate__machine_candidate"] + contingency[
        "human_noncandidate__machine_candidate"
    ]
    human_candidates_assessed = contingency["human_candidate__machine_candidate"] + contingency[
        "human_candidate__machine_noncandidate"
    ]
    summary = {
        "records": len(output),
        "phase_1_valid_signed_reviews": len(accepted),
        "phase_1_reviewed_packet_sha256": reviewed_packet_sha256,
        "machine_output_status_counts": dict(sorted(status_counts.items())),
        "machine_parse_summary": parse_summary,
        "assessed_valid_machine_outputs": status_counts["parsed_valid"],
        "contingency_on_valid_machine_outputs": dict(sorted(contingency.items())),
        "machine_candidate_precision_on_valid_intersection": (
            contingency["human_candidate__machine_candidate"] / machine_candidates
            if machine_candidates else None
        ),
        "machine_candidate_recall_on_valid_intersection": (
            contingency["human_candidate__machine_candidate"] / human_candidates_assessed
            if human_candidates_assessed else None
        ),
        "machine_output_unavailable_is_not_machine_reject": True,
        "phase_2_human_verified_records": 0,
        "warning": (
            "Precision/recall use only parsed-valid machine outputs. Invalid and unavailable records "
            "are separate missingness and cannot be counted as machine negatives."
        ),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--machine-raw", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    try:
        require_new_outputs([args.output, args.summary])
        reviewed_sha256 = file_sha256(args.reviewed)
        output, summary = build_comparison_packet(
            load_jsonl(args.original),
            load_jsonl(args.reviewed),
            load_machine_raw(args.machine_raw),
            reviewed_packet_sha256=reviewed_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_jsonl_exclusive(args.output, output)
    summary["comparison_packet_sha256"] = stable_packet_sha256(output)
    write_json_exclusive(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
