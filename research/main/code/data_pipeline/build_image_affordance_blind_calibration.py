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
from datetime import datetime
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
DECISIONS = {"candidate", "reject", "ambiguous"}
FAMILIES = {"R", "PR"}
EDITABLE_REVIEW_FIELDS = {*HUMAN_FIELDS, "human_verified", "review_status"}


def stable_packet_sha256(rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    return str(value if value is not None else "").strip()


def _nonempty_text_list(value: object, minimum: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= minimum
        and all(isinstance(item, str) and _text(item) for item in value)
        and len({_text(item).casefold() for item in value}) == len(value)
    )


def validate_completed_review(original: dict, reviewed: dict) -> list[str]:
    """Validate a human-signed blind judgment without filling any field."""

    errors = []
    immutable_keys = (set(original) | set(reviewed)) - EDITABLE_REVIEW_FIELDS
    for key in sorted(immutable_keys):
        if original.get(key) != reviewed.get(key):
            errors.append(f"immutable_field_changed:{key}")
    if FORBIDDEN_KEYS & reviewed.keys():
        errors.append("review_firewall_violation")
    if reviewed.get("machine_proposal_exposed") is not False:
        errors.append("machine_proposal_was_exposed")
    if reviewed.get("source_question_exposed") is not False:
        errors.append("source_question_was_exposed")
    if reviewed.get("human_verified") is not True:
        errors.append("human_verified_not_true")
    if reviewed.get("review_status") != "human_blind_affordance_review_complete":
        errors.append("review_status_not_complete")
    if type(reviewed.get("human_image_usable")) is not bool:
        errors.append("human_image_usable_not_bool")
    if type(reviewed.get("human_external_knowledge_needed")) is not bool:
        errors.append("human_external_knowledge_needed_not_bool")

    decision = reviewed.get("human_affordance_decision")
    if decision not in DECISIONS:
        errors.append("invalid_human_affordance_decision")
    if not _text(reviewed.get("human_reviewer")):
        errors.append("missing_human_reviewer")
    reviewed_at = _text(reviewed.get("human_reviewed_at"))
    try:
        parsed_time = datetime.fromisoformat(reviewed_at)
        if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
            errors.append("human_reviewed_at_missing_timezone")
    except ValueError:
        errors.append("invalid_human_reviewed_at")

    if decision == "candidate":
        if reviewed.get("human_image_usable") is not True:
            errors.append("candidate_image_not_usable")
        if reviewed.get("human_external_knowledge_needed") is not False:
            errors.append("candidate_needs_external_knowledge")
        if not _text(reviewed.get("human_target_question")):
            errors.append("candidate_missing_target_question")
        answer = reviewed.get("human_canonical_answer")
        if not (
            isinstance(answer, str) and _text(answer)
            or _nonempty_text_list(answer, 1)
        ):
            errors.append("candidate_missing_canonical_answer")
        if reviewed.get("human_affordance_family") not in FAMILIES:
            errors.append("candidate_invalid_affordance_family")
        evidence = reviewed.get("human_evidence_descriptions")
        evidence_valid = _nonempty_text_list(evidence, 2)
        if not evidence_valid:
            errors.append("candidate_requires_two_visual_evidence_items")
        if not _nonempty_text_list(reviewed.get("human_reasoning_steps"), 2):
            errors.append("candidate_requires_two_reasoning_steps")
        counterfactual = reviewed.get("human_counterfactual_required_evidence")
        counterfactual_valid = _nonempty_text_list(counterfactual, 2)
        if not counterfactual_valid:
            errors.append("candidate_requires_two_counterfactual_evidence_items")
        elif evidence_valid and {
            _text(item).casefold() for item in counterfactual
        } != {_text(item).casefold() for item in evidence}:
            errors.append("counterfactual_evidence_must_match_visual_evidence")
        difficult = reviewed.get("human_difficult_visual_evidence")
        if not _nonempty_text_list(difficult, 0):
            errors.append("invalid_difficult_visual_evidence")
        elif evidence_valid and any(
            _text(item).casefold() not in {_text(value).casefold() for value in evidence}
            for item in difficult
        ):
            errors.append("difficult_visual_evidence_not_in_required_evidence")
        elif reviewed.get("human_affordance_family") == "PR" and not difficult:
            errors.append("pr_requires_difficult_visual_evidence")
        elif reviewed.get("human_affordance_family") == "R" and difficult:
            errors.append("r_must_not_have_difficult_visual_evidence")
    elif decision in {"reject", "ambiguous"} and not _text(reviewed.get("human_notes")):
        errors.append("noncandidate_requires_notes")

    return sorted(set(errors))


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
