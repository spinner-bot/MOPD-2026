#!/usr/bin/env python3
"""Build a complete, evidence-firewalled human review queue from W2B output.

The input is one or more parsed ``construct.jsonl`` files produced by
``process_teacher_gap_candidate_stage.py``. Only structurally valid
``accept_candidate`` proposals enter the queue. Every human field is emitted
blank; model proposals remain explicitly labelled as machine suggestions.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_m3vqa_teacher_gap_review_packet import HUMAN_FIELDS
from teacher_gap_candidate_contract import (
    validate_audit,
    validate_blind,
    validate_candidate,
    validate_construct,
)


MACHINE_FIELDS = (
    "candidate_id",
    "source_image_id",
    "image_path",
    "question",
    "reference_answers",
    "reference_answer_alias_groups",
    "canonical_answer_text",
    "blind_proposal",
    "audit_proposal",
    "construction_proposal",
    "machine_provenance",
    "review_status",
)
CSV_COLUMNS = (*MACHINE_FIELDS, *HUMAN_FIELDS)


def load_jsonl(patterns: list[str]) -> list[dict]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(value) for value in (matches or [pattern]))
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def validate_constructed_row(row: dict) -> list[str]:
    errors = []
    if row.get("stage") != "construct":
        errors.append("stage_not_construct")
    candidate = row.get("candidate")
    blind = row.get("blind_proposal")
    audit = row.get("audit_proposal")
    proposal = row.get("proposal")
    if not isinstance(candidate, dict):
        errors.append("candidate_not_object")
        candidate = {}
    else:
        errors.extend(validate_candidate(candidate))
        if str(row.get("candidate_id", "")).strip() != str(candidate.get("candidate_id", "")).strip():
            errors.append("candidate_id_mismatch")
        if str(row.get("source_image_id", "")).strip() != str(candidate.get("source_image_id", "")).strip():
            errors.append("source_image_id_mismatch")
    if not isinstance(blind, dict):
        errors.append("blind_proposal_not_object")
    else:
        errors.extend(f"blind:{error}" for error in validate_blind(blind))
    if not isinstance(audit, dict):
        errors.append("audit_proposal_not_object")
    else:
        errors.extend(f"audit:{error}" for error in validate_audit(audit, candidate))
    if not isinstance(proposal, dict):
        errors.append("construction_proposal_not_object")
    else:
        errors.extend(
            f"construct:{error}"
            for error in validate_construct(proposal, candidate, audit)
        )
        if proposal.get("decision") != "accept_candidate":
            errors.append("construction_decision_not_accept")
    return sorted(set(errors))


def queue_record(row: dict) -> dict:
    candidate = row["candidate"]
    record = {
        "candidate_id": row["candidate_id"],
        "source_image_id": row["source_image_id"],
        "image_path": candidate["image_original"],
        "question": candidate["final_question_source"],
        "reference_answers": candidate["reference_answers"],
        "reference_answer_alias_groups": candidate["reference_answer_alias_groups"],
        "canonical_answer_text": candidate["canonical_answer_text"],
        "blind_proposal": row["blind_proposal"],
        "audit_proposal": row["audit_proposal"],
        "construction_proposal": row["proposal"],
        "machine_provenance": {
            "construct": row.get("raw_record_provenance"),
            "screening": row.get("screening_provenance"),
        },
        "review_status": "machine_candidate_pending_independent_human_review",
    }
    record.update({field: "" for field in HUMAN_FIELDS})
    return record


def build_queue(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    accepted, rejected = [], []
    seen = set()
    reason_counts = Counter()
    for row in sorted(rows, key=lambda item: str(item.get("candidate_id", ""))):
        candidate_id = str(row.get("candidate_id", "")).strip()
        reasons = validate_constructed_row(row)
        if not candidate_id:
            reasons.append("missing_candidate_id")
        elif candidate_id in seen:
            reasons.append("duplicate_candidate_id")
        seen.add(candidate_id)
        if reasons:
            unique = sorted(set(reasons))
            reason_counts.update(unique)
            rejected.append({"candidate_id": candidate_id or None, "rejection_reasons": unique})
        else:
            accepted.append(queue_record(row))
    summary = {
        "input_records": len(rows),
        "queued_for_human_review": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "human_fields_blank": list(HUMAN_FIELDS),
        "official_evidence_included": False,
        "automatically_human_verified": 0,
    }
    return accepted, rejected, summary


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered = {
            key: dumps(value) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        writer.writerow(rendered)
    path.write_bytes(("\ufeff" + buffer.getvalue()).encode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constructed", nargs="+", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        queued, rejected, summary = build_queue(load_jsonl(args.constructed))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_jsonl(args.output_jsonl, queued)
    write_csv(args.output_csv, queued)
    write_jsonl(args.rejects, rejected)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(dumps(summary))


if __name__ == "__main__":
    main()
