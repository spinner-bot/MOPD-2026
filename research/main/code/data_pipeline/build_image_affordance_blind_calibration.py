#!/usr/bin/env python3
"""Build phase-one B2 calibration records with machine proposals hidden.

The packet is a deterministic prefix of an already frozen, question-free
image sample. Reviewers first formulate a concrete closed-world target and
its minimal evidence path from the image alone. Machine proposals are joined
only after this phase has been saved, which prevents proposal anchoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from image_affordance_contract import validate_image_candidate


RULES_VERSION = "image-affordance-blind-calibration-v1"
HUMAN_FIELDS = (
    "human_image_usable",
    "human_affordance_decision",
    "human_target_question",
    "human_canonical_answer",
    "human_affordance_family",
    "human_evidence_descriptions",
    "human_reasoning_steps",
    "human_counterfactual_required_evidence",
    "human_difficult_visual_evidence",
    "human_external_knowledge_needed",
    "human_reviewer",
    "human_reviewed_at",
    "human_notes",
)
FORBIDDEN_KEYS = {
    "machine_output_status",
    "machine_decision",
    "machine_proposal",
    "machine_rejection_reasons",
    "machine_provenance",
    "final_question_source",
    "reference_answers",
    "reference_answer_alias_groups",
    "canonical_answer_text",
    "evidence",
    "evidence_urls",
    "question_hop",
    "reasoning_depth_candidate",
    "task_family_candidate",
}


def stable_packet_sha256(rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_blind_packet(source_candidates: list[dict], size: int) -> tuple[list[dict], dict]:
    if size < 1 or size > len(source_candidates):
        raise ValueError(f"calibration size {size} outside 1..{len(source_candidates)}")

    seen: set[str] = set()
    packet = []
    for source_index, source in enumerate(source_candidates[:size]):
        errors = validate_image_candidate(source)
        key = str(source.get("image_candidate_id", "")).strip()
        if errors:
            raise ValueError(f"invalid source candidate {key or '<missing>'}: {','.join(errors)}")
        if FORBIDDEN_KEYS & source.keys():
            raise ValueError(f"forbidden source field in {key}")
        if key in seen:
            raise ValueError(f"duplicate source image_candidate_id: {key}")
        seen.add(key)

        row = {
            "calibration_record_id": f"b2a-{source_index:04d}-{key}",
            "calibration_source_index": source_index,
            "image_candidate_id": key,
            "source_image_id": source["source_image_id"],
            "source_dataset": source.get("source_dataset"),
            "image_path": source["image_original"],
            "image_sha256": source["image_sha256"],
            "review_phase": "phase_1_blind_image_only",
            "machine_proposal_exposed": False,
            "source_question_exposed": False,
            "review_status": "pending_blind_human_affordance_review",
            "human_verified": False,
        }
        row.update({field: "" for field in HUMAN_FIELDS})
        packet.append(row)

    summary = {
        "rules_version": RULES_VERSION,
        "source_records": len(source_candidates),
        "calibration_records": len(packet),
        "selection": "ordered prefix of frozen question-free image sample",
        "machine_proposals_exposed": False,
        "source_questions_exposed": False,
        "automatically_human_verified": 0,
        "packet_sha256": stable_packet_sha256(packet),
    }
    return packet, summary


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    try:
        packet, summary = build_blind_packet(load_jsonl(args.source_input), args.size)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_jsonl(args.output, packet)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
