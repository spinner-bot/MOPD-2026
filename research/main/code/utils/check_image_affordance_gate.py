#!/usr/bin/env python3
"""Fail-closed completeness and provenance gate for image-affordance B1."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))

from extract_json_response import extract_json_object
from image_affordance_contract import (
    affordance_fallback_lock,
    affordance_lock_errors,
    render_affordance_prompt,
    render_compact_affordance_prompt,
    stable_hash,
    validate_affordance_proposal,
    validate_image_candidate,
)
from process_image_affordance_screen import parse_records


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_rows(
    rows: list[dict], *, expected_total: int, min_parse_rate: float,
    expected_world_size: int = 4, max_truncated: int = 0,
    expected_commit: str | None = None,
    expected_primary_max_new_tokens: int | None = None,
    expected_fallback_max_new_tokens: int | None = None,
) -> tuple[dict, list[str]]:
    parsed, rejected, parse_summary = parse_records(rows)
    failures = []
    ids = [str(row.get("image_candidate_id", "")).strip() for row in rows]
    errors = sum(bool(row.get("error")) for row in rows)
    truncated = sum(row.get("truncated") is not False or row.get("eos_reached") is not True for row in rows)
    revisions = {str(row.get("model_revision", "")) for row in rows}
    commits = {str(row.get("code_commit", "")) for row in rows}
    configs = {canonical(row.get("generation_config")) for row in rows}
    summary = {
        "total": len(rows), "valid_parsed": len(parsed), "rejected": len(rejected),
        "parse_rate": len(parsed) / len(rows) if rows else 0.0,
        "runner_errors": errors, "truncated_or_missing_eos": truncated,
        "fallback_used": sum(row.get("fallback_used") is True for row in rows),
        "fallback_recovered": sum(row.get("fallback_recovered") is True for row in rows),
        "parse_summary": parse_summary, "code_commits": sorted(commits),
        "model_revisions": sorted(revisions), "generation_configs": sorted(configs),
    }
    if len(rows) != expected_total:
        failures.append(f"total={len(rows)} != expected_total={expected_total}")
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        failures.append("image_candidate_ids missing or duplicated")
    input_indices = [row.get("input_index") for row in rows]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in input_indices) or sorted(input_indices) != list(range(expected_total)):
        failures.append("input indices must cover the expected sample exactly")
    if summary["parse_rate"] < min_parse_rate:
        failures.append(f"parse_rate={summary['parse_rate']:.6f} < {min_parse_rate:.6f}")
    if errors:
        failures.append(f"runner_errors={errors}")
    if truncated > max_truncated:
        failures.append(f"truncated_or_missing_eos={truncated} > {max_truncated}")
    if len(revisions) != 1 or "" in revisions:
        failures.append("model revision must be one non-empty value")
    if len(commits) != 1 or "" in commits or "unknown" in commits:
        failures.append("code commit must be one known value")
    elif expected_commit is not None and commits != {expected_commit}:
        failures.append("code commit differs from submitted value")
    if len(configs) != 1:
        failures.append("generation_config must be identical")

    for row in rows:
        key = str(row.get("image_candidate_id", "<missing>"))
        candidate = row.get("candidate")
        candidate_valid = isinstance(candidate, dict) and not validate_image_candidate(candidate)
        if not candidate_valid:
            failures.append(f"{key}: invalid image candidate")
            candidate = candidate if isinstance(candidate, dict) else {}
        if row.get("source_image_id") != candidate.get("source_image_id"):
            failures.append(f"{key}: source image identity mismatch")
        if key != str(candidate.get("image_candidate_id", "")).strip():
            failures.append(f"{key}: image candidate identity mismatch")
        if row.get("image_checksum") != candidate.get("image_sha256"):
            failures.append(f"{key}: image checksum mismatch")
        if row.get("prompt_fields_used") != ["image_original"]:
            failures.append(f"{key}: prompt allow-list violation")
        if row.get("world_size") != expected_world_size:
            failures.append(f"{key}: world_size mismatch")
        input_index = row.get("input_index")
        rank = row.get("rank")
        if (
            not isinstance(input_index, int) or isinstance(input_index, bool)
            or not isinstance(rank, int) or isinstance(rank, bool)
            or rank != input_index % expected_world_size
        ):
            failures.append(f"{key}: rank disagrees with deterministic sharding")
        if row.get("thinking_mode") is not False:
            failures.append(f"{key}: thinking must be false")
        if row.get("chat_template") != {"enable_thinking": False, "add_generation_prompt": True}:
            failures.append(f"{key}: chat template mismatch")
        config = row.get("generation_config")
        if (
            not isinstance(config, dict)
            or config.get("do_sample") is not False
            or config.get("max_attempts") != 2
            or config.get("structured_output_repetition_constraints") != "disabled"
            or not isinstance(config.get("primary_max_new_tokens"), int)
            or isinstance(config.get("primary_max_new_tokens"), bool)
            or config.get("primary_max_new_tokens") < 1
            or not isinstance(config.get("fallback_max_new_tokens"), int)
            or isinstance(config.get("fallback_max_new_tokens"), bool)
            or config.get("fallback_max_new_tokens") < 1
        ):
            failures.append(f"{key}: invalid generation config")
        elif (
            expected_primary_max_new_tokens is not None
            and config.get("primary_max_new_tokens") != expected_primary_max_new_tokens
        ) or (
            expected_fallback_max_new_tokens is not None
            and config.get("fallback_max_new_tokens") != expected_fallback_max_new_tokens
        ):
            failures.append(f"{key}: generation token budget mismatch")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
            failures.append(f"{key}: invalid attempts")
            continue
        if any(not isinstance(attempt, dict) for attempt in attempts):
            failures.append(f"{key}: attempt must be an object")
            continue
        if row.get("attempt_count") != len(attempts) or row.get("chosen_attempt") != len(attempts) - 1:
            failures.append(f"{key}: attempt provenance mismatch")
        primary_expected = stable_hash(render_affordance_prompt(candidate)) if candidate_valid else None
        if attempts[0].get("attempt_kind") != "primary" or (
            candidate_valid and attempts[0].get("prompt_hash") != primary_expected
        ):
            failures.append(f"{key}: primary prompt mismatch")
        if len(attempts) == 2:
            primary = extract_json_object(str(attempts[0].get("raw_model_output", "")))
            fallback_expected = stable_hash(render_compact_affordance_prompt(primary))
            if attempts[1].get("attempt_kind") != "compact_fallback" or attempts[1].get("prompt_hash") != fallback_expected:
                failures.append(f"{key}: fallback prompt mismatch")
            if attempts[1].get("preserved_decision_fields") != affordance_fallback_lock(primary):
                failures.append(f"{key}: fallback lock mismatch")
        for attempt in attempts:
            lock = attempt.get("preserved_decision_fields")
            if not isinstance(lock, dict):
                failures.append(f"{key}: attempt decision lock must be an object")
                lock = {}
            proposal = extract_json_object(str(attempt.get("raw_model_output", "")))
            recomputed = []
            if attempt.get("truncated") is True:
                recomputed.append("truncated_output")
            if proposal is None:
                recomputed.append("invalid_json")
            else:
                recomputed.extend(validate_affordance_proposal(proposal))
                recomputed.extend(affordance_lock_errors(proposal, lock))
            if set(attempt.get("validation_errors", [])) != set(recomputed):
                failures.append(f"{key}: attempt validation errors mismatch")
            if attempt.get("schema_valid") != (not recomputed):
                failures.append(f"{key}: attempt schema_valid mismatch")
        chosen = attempts[-1]
        for field in ("prompt_hash", "chat_template_hash", "input_tokens", "generated_tokens", "eos_reached", "truncated", "raw_model_output"):
            if row.get(field) != chosen.get(field):
                failures.append(f"{key}: selected attempt {field} mismatch")
    summary["failures"] = sorted(set(failures))
    return summary, summary["failures"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--min-parse-rate", type=float, default=0.9)
    parser.add_argument("--max-truncated", type=int, default=0)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-primary-max-new-tokens", type=int)
    parser.add_argument("--expected-fallback-max-new-tokens", type=int)
    args = parser.parse_args()
    rows = []
    paths = sorted(glob.glob(args.input_glob))
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    summary, failures = validate_rows(
        rows, expected_total=args.expected_total, min_parse_rate=args.min_parse_rate,
        max_truncated=args.max_truncated, expected_commit=args.expected_commit,
        expected_primary_max_new_tokens=args.expected_primary_max_new_tokens,
        expected_fallback_max_new_tokens=args.expected_fallback_max_new_tokens,
    )
    summary["files"] = paths
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit("image-affordance gate FAILED: " + "; ".join(failures))
    print("image-affordance gate PASSED")


if __name__ == "__main__":
    main()
