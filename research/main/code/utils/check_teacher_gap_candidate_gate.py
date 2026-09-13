#!/usr/bin/env python3
"""Fail-closed configuration/completeness gate for 1016 candidate stages."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))

from process_teacher_gap_candidate_stage import parse_records
from teacher_gap_candidate_contract import STAGES


PROMPT_FIELDS = {
    "blind": ["image_original", "final_question_source"],
    "audit": ["image_original", "final_question_source", "reference_answers"],
    "construct": [
        "image_original",
        "final_question_source",
        "reference_answers",
        "audit_proposal.required_evidence",
        "audit_proposal.minimal_reasoning_graph",
    ],
}


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_value(value) -> str:
    return "" if value is None else str(value)


def summarize_and_validate(
    rows: list[dict],
    stage: str,
    *,
    min_total: int = 1,
    expected_total: int | None = None,
    min_parse_rate: float = 0.95,
    expected_world_size: int = 4,
    expected_thinking: bool | None = None,
    expected_max_new_tokens: int | None = None,
) -> tuple[dict, list[str]]:
    parsed, rejected, parse_summary = parse_records(rows, stage)
    ids = [row.get("candidate_id") for row in rows]
    errors = sum(bool(row.get("error")) for row in rows)
    truncated = sum(
        row.get("truncated") is not False or row.get("eos_reached") is not True
        for row in rows
    )
    revisions = sorted({text_value(row.get("model_revision")) for row in rows})
    commits = sorted({text_value(row.get("code_commit")) for row in rows})
    dtypes = sorted({text_value(row.get("model_dtype")) for row in rows})
    world_sizes = sorted({
        row.get("world_size")
        for row in rows
        if isinstance(row.get("world_size"), int) and not isinstance(row.get("world_size"), bool)
    })
    thinking_values = [row.get("thinking_mode") for row in rows]
    thinking_modes = {value for value in thinking_values if type(value) is bool}
    generation_configs = {canonical_json(row.get("generation_config")) for row in rows}
    input_indices = [row.get("input_index") for row in rows]

    summary = {
        "stage": stage,
        "total": len(rows),
        "valid_parsed": len(parsed),
        "rejected": len(rejected),
        "parse_rate": len(parsed) / len(rows) if rows else 0.0,
        "runner_errors": errors,
        "truncated_or_missing_eos": truncated,
        "unique_candidate_ids": len(set(ids)),
        "model_revisions": revisions,
        "code_commits": commits,
        "model_dtypes": dtypes,
        "world_sizes": world_sizes,
        "thinking_modes": sorted(str(value) for value in thinking_modes),
        "generation_configs": sorted(generation_configs),
        "parse_summary": parse_summary,
    }
    failures = []
    if len(rows) < min_total:
        failures.append(f"total={len(rows)} < min_total={min_total}")
    if expected_total is not None and len(rows) != expected_total:
        failures.append(f"total={len(rows)} != expected_total={expected_total}")
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        failures.append("candidate ids are missing or duplicated")
    if summary["parse_rate"] < min_parse_rate:
        failures.append(f"parse_rate={summary['parse_rate']:.6f} < {min_parse_rate:.6f}")
    if errors:
        failures.append(f"runner_errors={errors}")
    if truncated:
        failures.append(f"truncated_or_missing_eos={truncated}")
    if len(revisions) != 1 or not revisions[0]:
        failures.append("model revision must be one non-empty value")
    if len(commits) != 1 or not commits[0] or commits[0] == "unknown":
        failures.append("code commit must be one known value")
    if len(dtypes) != 1 or not dtypes[0]:
        failures.append("model dtype must be one non-empty value")
    if world_sizes != [expected_world_size]:
        failures.append(f"world_size must be {expected_world_size}")
    if any(type(value) is not bool for value in thinking_values) or len(thinking_modes) != 1:
        failures.append("thinking_mode must be one explicit boolean value")
    elif expected_thinking is not None and thinking_modes != {expected_thinking}:
        failures.append(f"thinking_mode must be {expected_thinking}")
    if len(generation_configs) != 1:
        failures.append("generation_config must be identical across all records")

    for row in rows:
        key = str(row.get("candidate_id", "<missing>"))
        if row.get("prompt_fields_used") != PROMPT_FIELDS[stage]:
            failures.append(f"{key}: prompt_fields_used violates {stage} allow-list")
        if row.get("chat_template") != {
            "enable_thinking": row.get("thinking_mode"),
            "add_generation_prompt": True,
        }:
            failures.append(f"{key}: chat_template provenance mismatch")
        config = row.get("generation_config")
        if (
            not isinstance(config, dict)
            or config.get("do_sample") is not False
            or config.get("structured_output_repetition_constraints") != "disabled"
            or not isinstance(config.get("max_new_tokens"), int)
            or isinstance(config.get("max_new_tokens"), bool)
            or config.get("max_new_tokens") < 1
        ):
            failures.append(f"{key}: invalid generation_config")
        elif expected_max_new_tokens is not None and config["max_new_tokens"] != expected_max_new_tokens:
            failures.append(f"{key}: max_new_tokens differs from submitted value")
        for field in ("image_checksum", "prompt_hash", "chat_template_hash"):
            if not str(row.get(field, "")).strip():
                failures.append(f"{key}: missing {field}")
        input_index = row.get("input_index")
        rank = row.get("rank")
        if not isinstance(input_index, int) or isinstance(input_index, bool) or input_index < 0:
            failures.append(f"{key}: invalid input_index")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < expected_world_size:
            failures.append(f"{key}: invalid rank")
        elif isinstance(input_index, int) and not isinstance(input_index, bool) and rank != input_index % expected_world_size:
            failures.append(f"{key}: rank disagrees with input sharding")

    if expected_total is not None and sorted(
        value for value in input_indices if isinstance(value, int) and not isinstance(value, bool)
    ) != list(range(expected_total)):
        failures.append("input_index values must cover expected input exactly once")
    return summary, sorted(set(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--min-total", type=int, default=1)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--min-parse-rate", type=float, default=0.95)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--expected-thinking", choices=("true", "false"))
    parser.add_argument("--expected-max-new-tokens", type=int)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    expected_thinking = None if args.expected_thinking is None else args.expected_thinking == "true"
    summary, failures = summarize_and_validate(
        rows,
        args.stage,
        min_total=args.min_total,
        expected_total=args.expected_total,
        min_parse_rate=args.min_parse_rate,
        expected_world_size=args.expected_world_size,
        expected_thinking=expected_thinking,
        expected_max_new_tokens=args.expected_max_new_tokens,
    )
    summary["files"] = paths
    summary["failures"] = failures
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit("teacher-gap candidate gate FAILED: " + "; ".join(failures))
    print("teacher-gap candidate gate PASSED")


if __name__ == "__main__":
    main()
