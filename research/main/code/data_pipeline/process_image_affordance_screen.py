#!/usr/bin/env python3
"""Parse raw image-affordance records without promoting machine proposals."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from extract_json_response import extract_json_object
from image_affordance_contract import affordance_lock_errors, validate_affordance_proposal


def load(patterns: list[str]) -> list[dict]:
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with Path(path).open(encoding="utf-8") as source:
                rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def parse_records(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    parsed, rejected, seen = [], [], set()
    counts = Counter()
    for raw in rows:
        key = str(raw.get("image_candidate_id", "")).strip()
        errors = []
        if key in seen:
            errors.append("duplicate_image_candidate_id")
        seen.add(key)
        if raw.get("stage") != "image_affordance":
            errors.append("stage_mismatch")
        if raw.get("error"):
            errors.append("runner_error")
        if raw.get("truncated") is not False or raw.get("eos_reached") is not True:
            errors.append("truncated_output")
        proposal = extract_json_object(str(raw.get("raw_model_output", "")))
        if proposal is None:
            errors.append("invalid_json")
        else:
            errors.extend(validate_affordance_proposal(proposal))
            attempts = raw.get("attempts")
            chosen = attempts[-1] if isinstance(attempts, list) and attempts else {}
            errors.extend(affordance_lock_errors(
                proposal,
                chosen.get("preserved_decision_fields", {}) if isinstance(chosen, dict) else {},
            ))
        output = {
            "image_candidate_id": key,
            "source_image_id": raw.get("source_image_id"),
            "stage": "image_affordance",
            "candidate": raw.get("candidate"),
            "proposal": proposal,
            "parse_method": "deterministic_last_json_object" if proposal is not None else None,
            "raw_record_provenance": {field: raw.get(field) for field in (
                "rank", "input_index", "world_size", "model_revision", "code_commit",
                "model_dtype", "image_checksum", "prompt_hash", "chat_template_hash",
                "thinking_mode", "chat_template", "generation_config", "attempt_count",
                "chosen_attempt", "fallback_used", "fallback_recovered",
                "primary_failure_reasons",
            )},
            "review_status": "machine_affordance_proposal_pending_independent_human_review",
        }
        if errors:
            output["rejection_reasons"] = sorted(set(errors))
            rejected.append(output)
            counts.update(output["rejection_reasons"])
        else:
            parsed.append(output)
    decisions = Counter(row["proposal"]["decision"] for row in parsed)
    return parsed, rejected, {
        "observed": len(rows), "parsed_valid": len(parsed), "rejected": len(rejected),
        "parse_rate": len(parsed) / len(rows) if rows else 0.0,
        "decisions": dict(sorted(decisions.items())),
        "machine_candidates": decisions.get("candidate", 0),
        "human_verified": 0,
        "rejection_reasons": dict(sorted(counts.items())),
    }


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--parsed-output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    parsed, rejected, summary = parse_records(load(args.inputs))
    write(args.parsed_output, parsed)
    write(args.rejected_output, rejected)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
