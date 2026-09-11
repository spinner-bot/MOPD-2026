"""Deterministically parse, validate, and join raw 1016 candidate-stage output."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from extract_json_response import extract_json_object
from teacher_gap_candidate_contract import (
    STAGES,
    is_construction_candidate,
    proposal_errors,
)


def load_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    expanded = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        expanded.extend(matches or [pattern])
    for value in expanded:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_records(rows: list[dict], stage: str) -> tuple[list[dict], list[dict], dict]:
    parsed_rows, rejects = [], []
    reasons = Counter()
    seen = set()
    for raw in rows:
        key = (raw.get("candidate_id"), raw.get("stage"))
        row_reasons = []
        if key in seen:
            row_reasons.append("duplicate_candidate_stage")
        seen.add(key)
        if raw.get("stage") != stage:
            row_reasons.append("stage_mismatch")
        if raw.get("error"):
            row_reasons.append("runner_error")
        if raw.get("truncated") is not False or raw.get("eos_reached") is not True:
            row_reasons.append("truncated_output")
        proposal = extract_json_object(str(raw.get("raw_model_output", "")))
        parse_method = "deterministic_last_json_object" if proposal is not None else None
        if proposal is None:
            row_reasons.append("invalid_json")
        else:
            row_reasons.extend(proposal_errors(stage, proposal, raw.get("candidate", {})))
        output = {
            "candidate_id": raw.get("candidate_id"),
            "source_image_id": raw.get("source_image_id"),
            "stage": stage,
            "candidate": raw.get("candidate"),
            "proposal": proposal,
            "parse_method": parse_method,
            "raw_record_provenance": {
                "rank": raw.get("rank"),
                "model_revision": raw.get("model_revision"),
                "code_commit": raw.get("code_commit"),
                "image_checksum": raw.get("image_checksum"),
                "prompt_hash": raw.get("prompt_hash"),
                "chat_template_hash": raw.get("chat_template_hash"),
                "thinking_mode": raw.get("thinking_mode"),
            },
        }
        if stage == "construct":
            output["blind_proposal"] = raw.get("blind_proposal")
            output["audit_proposal"] = raw.get("audit_proposal")
            output["screening_provenance"] = raw.get("screening_provenance")
        if row_reasons:
            output["rejection_reasons"] = sorted(set(row_reasons))
            rejects.append(output)
            reasons.update(output["rejection_reasons"])
        else:
            parsed_rows.append(output)
    summary = {
        "stage": stage,
        "observed": len(rows),
        "parsed_valid": len(parsed_rows),
        "rejected": len(rejects),
        "parse_rate": len(parsed_rows) / len(rows) if rows else 0.0,
        "rejection_reasons": dict(sorted(reasons.items())),
    }
    if stage == "audit":
        summary["knowledge_labels"] = dict(sorted(Counter(
            row["proposal"]["knowledge_dependency_label"] for row in parsed_rows
        ).items()))
        summary["task_families"] = dict(sorted(Counter(
            str(row["proposal"].get("task_family")) for row in parsed_rows
        ).items()))
    if stage == "construct":
        summary["decisions"] = dict(sorted(Counter(
            row["proposal"]["decision"] for row in parsed_rows
        ).items()))
    return parsed_rows, rejects, summary


def screening_reasons(blind: dict | None, audit: dict | None) -> list[str]:
    reasons = []
    if not isinstance(blind, dict):
        return ["missing_blind_proposal"]
    if not isinstance(audit, dict):
        return ["missing_audit_proposal"]
    if blind.get("requires_unprovided_fact") is not False:
        reasons.append("blind_external_fact_risk")
    if blind.get("ambiguous") is not False:
        reasons.append("blind_ambiguous")
    if blind.get("image_is_necessary") is not True:
        reasons.append("blind_image_not_necessary")
    if audit.get("knowledge_dependency_label") != "image_only":
        reasons.append(f"audit_{audit.get('knowledge_dependency_label', 'invalid')}")
    for key in ("reference_entailed", "unique_answer", "image_is_necessary", "full_image_reference_valid"):
        if audit.get(key) is not True:
            reasons.append(f"audit_{key}_false")
    if audit.get("task_family") not in {"R", "PR"}:
        reasons.append("audit_not_R_or_PR")
    return reasons


def select_construction(blind_rows: list[dict], audit_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    blind_by_id = {row["candidate_id"]: row for row in blind_rows}
    audit_by_id = {row["candidate_id"]: row for row in audit_rows}
    candidate_ids = sorted(set(blind_by_id) | set(audit_by_id))
    selected, rejected = [], []
    reason_counts = Counter()
    for candidate_id in candidate_ids:
        blind_row, audit_row = blind_by_id.get(candidate_id), audit_by_id.get(candidate_id)
        blind = blind_row.get("proposal") if blind_row else None
        audit = audit_row.get("proposal") if audit_row else None
        reasons = screening_reasons(blind, audit)
        if not reasons and is_construction_candidate(blind, audit):
            selected.append({
                "candidate_id": candidate_id,
                "source_image_id": audit_row["source_image_id"],
                "candidate": audit_row["candidate"],
                "blind_proposal": blind,
                "audit_proposal": audit,
                "screening_provenance": {
                    "blind": blind_row.get("raw_record_provenance"),
                    "audit": audit_row.get("raw_record_provenance"),
                },
                "status": "model_candidate_pending_construction_and_human_review",
            })
        else:
            reasons = reasons or ["screening_contract_failed"]
            reason_counts.update(reasons)
            rejected.append({
                "candidate_id": candidate_id,
                "source_image_id": (audit_row or blind_row or {}).get("source_image_id"),
                "rejection_reasons": sorted(set(reasons)),
                "blind_proposal": blind,
                "audit_proposal": audit,
            })
    return selected, rejected, {
        "observed_candidate_ids": len(candidate_ids),
        "selected_for_construction": len(selected),
        "rejected": len(rejected),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "warning": "selected records are model candidates, not human-verified image_only/R truth",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parse_parser = subparsers.add_parser("parse")
    parse_parser.add_argument("--stage", choices=STAGES, required=True)
    parse_parser.add_argument("--inputs", nargs="+", required=True)
    parse_parser.add_argument("--parsed-output", type=Path, required=True)
    parse_parser.add_argument("--rejected-output", type=Path, required=True)
    parse_parser.add_argument("--summary", type=Path, required=True)
    select_parser = subparsers.add_parser("select-construction")
    select_parser.add_argument("--blind", nargs="+", required=True)
    select_parser.add_argument("--audit", nargs="+", required=True)
    select_parser.add_argument("--selected-output", type=Path, required=True)
    select_parser.add_argument("--rejected-output", type=Path, required=True)
    select_parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "parse":
        parsed_rows, rejects, summary = parse_records(load_jsonl(args.inputs), args.stage)
        write_jsonl(args.parsed_output, parsed_rows)
        write_jsonl(args.rejected_output, rejects)
    else:
        selected, rejects, summary = select_construction(load_jsonl(args.blind), load_jsonl(args.audit))
        write_jsonl(args.selected_output, selected)
        write_jsonl(args.rejected_output, rejects)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
