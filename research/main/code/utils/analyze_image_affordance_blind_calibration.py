#!/usr/bin/env python3
"""Summarize a complete B2a blind review and apply frozen routing rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from build_image_affordance_blind_calibration import load_jsonl, stable_packet_sha256
from finalize_image_affordance_blind_review import finalize, require_new_outputs, write_json_exclusive


EXPECTED_PACKET_SHA256 = "b61839ddb4111f1be5d2d67f165c40dc05cd10daf8d5ca02a202b707a1057a58"
EXPECTED_RECORDS = 32


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def route_for(candidate_count: int) -> dict[str, str]:
    """Frozen B2a feasibility route, set before human labels are observed."""

    if candidate_count >= 8:
        return {
            "route": "proceed_to_b3_eight_image_canary",
            "meaning": "B2a supplies at least eight witnessed image-only R/PR targets for a B3 construction canary.",
        }
    if candidate_count >= 1:
        return {
            "route": "compare_equal_size_blind_source_samples_before_scaling",
            "meaning": "R/PR affordance exists, but this packet does not supply an eight-image B3 canary; keep M3VQA supplemental while comparing sources.",
        }
    return {
        "route": "stop_m3vqa_as_primary_affordance_source",
        "meaning": "No witnessed target was found in 32 images; this does not prove zero population affordance, but M3VQA is not scaled as the primary source.",
    }


def analyze(
    original_rows: list[dict],
    reviewed_rows: list[dict],
    *,
    expected_packet_sha256: str = EXPECTED_PACKET_SHA256,
    expected_records: int = EXPECTED_RECORDS,
) -> dict:
    if stable_packet_sha256(original_rows) != expected_packet_sha256:
        raise ValueError("unexpected frozen B2a packet SHA-256")
    if len(original_rows) != expected_records:
        raise ValueError(f"expected {expected_records} frozen records")

    accepted, rejected, completion = finalize(original_rows, reviewed_rows)
    if rejected or len(accepted) != expected_records:
        raise ValueError("B2a review is not complete and strictly valid")
    reviewers = sorted({str(row["human_reviewer"]).strip() for row in accepted})
    if len(reviewers) != 1:
        raise ValueError("one blind-review file must contain exactly one stable reviewer ID")

    decisions = Counter(row["human_affordance_decision"] for row in accepted)
    families = Counter(
        row["human_affordance_family"]
        for row in accepted
        if row["human_affordance_decision"] == "candidate"
    )
    usable = sum(row["human_image_usable"] is True for row in accepted)
    candidates = decisions["candidate"]
    candidate_rate = candidates / expected_records
    projected = math.ceil(136 / candidate_rate) if candidate_rate else None
    result = {
        "status": "complete_blind_calibration_summary",
        "records": expected_records,
        "reviewer_id": reviewers[0],
        "decision_counts": dict(sorted(decisions.items())),
        "usable_images": usable,
        "candidate_images": candidates,
        "candidate_family_counts": dict(sorted(families.items())),
        "candidate_rate_all_images": candidate_rate,
        "candidate_rate_all_images_wilson_95": wilson_interval(candidates, expected_records),
        "candidate_rate_usable_images": candidates / usable if usable else None,
        "projected_images_to_review_for_136_at_point_rate": projected,
        "routing": route_for(candidates),
        "human_verified_records": len(accepted),
        "machine_results_used": False,
        "source_questions_used": False,
        "interpretation_boundary": (
            "One-reviewer source-feasibility calibration. A candidate is a witnessed affordance; "
            "reject/ambiguous does not prove the image has no other valid target."
        ),
        "completion_gate": completion,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-original-sha256", default=EXPECTED_PACKET_SHA256)
    parser.add_argument("--expected-records", type=int, default=EXPECTED_RECORDS)
    args = parser.parse_args()
    try:
        require_new_outputs([args.output])
        result = analyze(
            load_jsonl(args.original),
            load_jsonl(args.reviewed),
            expected_packet_sha256=args.expected_original_sha256,
            expected_records=args.expected_records,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    result["reviewed_packet_sha256"] = hashlib.sha256(args.reviewed.read_bytes()).hexdigest()
    write_json_exclusive(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
