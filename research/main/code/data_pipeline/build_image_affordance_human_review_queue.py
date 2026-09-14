#!/usr/bin/env python3
"""Build the B2 image-only human review queue without importing source questions.

Every source image enters the queue exactly once, including images whose B1
machine output was invalid or missing. This lets B2 estimate both proposal
precision and missed affordances. Machine proposals remain suggestions; all
human judgment fields are emitted blank.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from image_affordance_contract import validate_affordance_proposal, validate_image_candidate


HUMAN_AFFORDANCE_FIELDS = (
    "human_image_usable",
    "human_machine_description_faithful",
    "human_closed_world_reasoning_affordance",
    "human_affordance_family",
    "human_valid_entity_ids",
    "human_valid_path_indices",
    "human_difficult_entity_ids",
    "human_external_identification_needed",
    "human_review_decision",
    "human_reviewer",
    "human_reviewed_at",
    "human_notes",
)

MACHINE_FIELDS = (
    "image_candidate_id",
    "source_image_id",
    "source_dataset",
    "image_path",
    "image_sha256",
    "source_record_count",
    "machine_output_status",
    "machine_decision",
    "machine_proposal",
    "machine_rejection_reasons",
    "machine_provenance",
    "source_questions_exposed_to_screen",
    "review_status",
    "human_verified",
)
CSV_COLUMNS = (*MACHINE_FIELDS, *HUMAN_AFFORDANCE_FIELDS)
IDENTITY_FIELDS = ("image_candidate_id", "source_image_id", "image_original", "image_sha256")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _identity_errors(source: dict, output: dict) -> list[str]:
    errors = []
    if output.get("stage") != "image_affordance":
        errors.append("stage_mismatch")
    embedded = output.get("candidate")
    if not isinstance(embedded, dict):
        return errors + ["machine_candidate_not_object"]
    for field in IDENTITY_FIELDS:
        if embedded.get(field) != source.get(field):
            errors.append(f"candidate_{field}_mismatch")
    return errors


def _machine_index(rows: list[dict], known_ids: set[str], label: str) -> tuple[dict[str, dict], list[dict]]:
    index: dict[str, dict] = {}
    rejects = []
    for row_number, row in enumerate(rows, start=1):
        key = str(row.get("image_candidate_id", "")).strip()
        reasons = []
        if not key:
            reasons.append("missing_image_candidate_id")
        elif key not in known_ids:
            reasons.append("unknown_image_candidate_id")
        elif key in index:
            reasons.append("duplicate_machine_output")
        if reasons:
            rejects.append({
                "input_kind": label,
                "row_number": row_number,
                "image_candidate_id": key,
                "rejection_reasons": reasons,
            })
            if key in index:
                index[key] = {"_duplicate": True}
            continue
        index[key] = row
    return index, rejects


def build_queue(
    source_candidates: list[dict],
    parsed_rows: list[dict],
    rejected_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    source_by_id: dict[str, dict] = {}
    for row in source_candidates:
        errors = validate_image_candidate(row)
        key = str(row.get("image_candidate_id", "")).strip()
        if errors:
            raise ValueError(f"invalid source candidate {key or '<missing>'}: {','.join(errors)}")
        if key in source_by_id:
            raise ValueError(f"duplicate source image_candidate_id: {key}")
        source_by_id[key] = row

    known = set(source_by_id)
    parsed_by_id, rejects = _machine_index(parsed_rows, known, "parsed")
    rejected_by_id, more_rejects = _machine_index(rejected_rows, known, "rejected")
    rejects.extend(more_rejects)
    overlap = set(parsed_by_id) & set(rejected_by_id)
    for key in sorted(overlap):
        rejects.append({
            "input_kind": "parsed_and_rejected",
            "image_candidate_id": key,
            "rejection_reasons": ["duplicate_machine_output_across_files"],
        })
        parsed_by_id[key] = {"_duplicate": True}
        rejected_by_id[key] = {"_duplicate": True}

    queue = []
    status_counts = Counter()
    decision_counts = Counter()
    for key in sorted(source_by_id):
        source = source_by_id[key]
        parsed = parsed_by_id.get(key)
        rejected = rejected_by_id.get(key)
        row = parsed or rejected
        output_status = "missing_output"
        proposal = None
        rejection_reasons: list[str] = []
        provenance: dict[str, Any] = {}

        if isinstance(row, dict) and row.get("_duplicate") is True:
            output_status = "invalid_duplicate_output"
            rejection_reasons = ["duplicate_machine_output"]
        elif isinstance(row, dict):
            identity_errors = _identity_errors(source, row)
            if identity_errors:
                output_status = "invalid_identity_output"
                rejection_reasons = identity_errors
                rejects.append({
                    "input_kind": "joined_output",
                    "image_candidate_id": key,
                    "rejection_reasons": identity_errors,
                })
            else:
                proposal = row.get("proposal") if isinstance(row.get("proposal"), dict) else None
                provenance = row.get("raw_record_provenance") if isinstance(row.get("raw_record_provenance"), dict) else {}
                if parsed is row:
                    proposal_errors = validate_affordance_proposal(proposal)
                    if proposal_errors:
                        output_status = "invalid_parsed_output"
                        rejection_reasons = proposal_errors
                        rejects.append({
                            "input_kind": "parsed",
                            "image_candidate_id": key,
                            "rejection_reasons": proposal_errors,
                        })
                    else:
                        output_status = "parsed_valid"
                else:
                    output_status = "rejected_output"
                    rejection_reasons = sorted(set(str(value) for value in row.get("rejection_reasons", [])))

        decision = proposal.get("decision") if isinstance(proposal, dict) else None
        record = {
            "image_candidate_id": key,
            "source_image_id": source["source_image_id"],
            "source_dataset": source.get("source_dataset"),
            "image_path": source["image_original"],
            "image_sha256": source["image_sha256"],
            "source_record_count": source.get("source_record_count"),
            "machine_output_status": output_status,
            "machine_decision": decision,
            "machine_proposal": proposal,
            "machine_rejection_reasons": rejection_reasons,
            "machine_provenance": provenance,
            "source_questions_exposed_to_screen": False,
            "review_status": "pending_independent_image_affordance_review",
            "human_verified": False,
        }
        record.update({field: "" for field in HUMAN_AFFORDANCE_FIELDS})
        queue.append(record)
        status_counts[output_status] += 1
        decision_counts[str(decision) if decision is not None else "none"] += 1

    return queue, rejects, {
        "source_images": len(source_candidates),
        "queued_for_human_review": len(queue),
        "machine_output_statuses": dict(sorted(status_counts.items())),
        "machine_decisions": dict(sorted(decision_counts.items())),
        "machine_output_reject_records": len(rejects),
        "source_questions_exposed": False,
        "automatically_human_verified": 0,
        "review_scope": "all pilot images, including invalid or missing machine outputs",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered = {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        writer.writerow(rendered)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\ufeff" + buffer.getvalue(), encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--join-rejects", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    try:
        queue, rejects, summary = build_queue(
            load_jsonl(args.source_input), load_jsonl(args.parsed), load_jsonl(args.rejected)
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_jsonl(args.queue_jsonl, queue)
    write_csv(args.queue_csv, queue)
    write_jsonl(args.join_rejects, rejects)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
