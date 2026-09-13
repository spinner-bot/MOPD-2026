#!/usr/bin/env python3
"""Audit W2 teacher-gap candidate screen/construct run archives.

This tool reads the real archive layout used by the W2 sbatch scripts and
delegates parsing and candidate selection to the same authoritative functions
as the gates. It never promotes model output to human truth and never reports
candidate-screening counts as a measured teacher gap.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))

from process_teacher_gap_candidate_stage import parse_records, select_construction  # noqa: E402


def load_jsonl_paths(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def stage_paths(run_dir: Path, stage: str, scope: str = "raw") -> list[Path]:
    pattern = str(run_dir / scope / stage / "rank-*.jsonl")
    return [Path(value) for value in sorted(glob.glob(pattern))]


def analyze_raw_stage(run_dir: Path, stage: str, scope: str = "raw") -> tuple[list[dict], dict]:
    paths = stage_paths(run_dir, stage, scope)
    if not paths:
        return [], {
            "stage": stage,
            "scope": scope,
            "status": "not_available",
            "files": [],
            "observed": 0,
        }
    raw = load_jsonl_paths(paths)
    parsed, rejected, summary = parse_records(raw, stage)
    truncated = sum(
        row.get("truncated") is not False or row.get("eos_reached") is not True
        for row in raw
    )
    runner_errors = sum(bool(row.get("error")) for row in raw)
    return parsed, {
        **summary,
        "scope": scope,
        "status": "complete_raw_set" if raw else "empty_raw_set",
        "files": [str(path) for path in paths],
        "truncated_or_missing_eos": truncated,
        "runner_errors": runner_errors,
        "valid_candidate_ids": [row["candidate_id"] for row in parsed],
        "rejected_candidate_ids": [row["candidate_id"] for row in rejected],
    }


def analyze_screen_output(run_dir: Path) -> dict:
    blind_rows, blind_summary = analyze_raw_stage(run_dir, "blind")
    audit_rows, audit_summary = analyze_raw_stage(run_dir, "audit")
    result = {
        "run_dir": str(run_dir),
        "run_type": "screen",
        "stages": {"blind": blind_summary, "audit": audit_summary},
        "teacher_gap_measured": False,
        "scientific_status": "candidate_screening_only",
        "selection": {"status": "not_available_until_both_full_stages_exist"},
        "human_review_queue": [],
    }
    if blind_rows and audit_rows:
        selected, rejected, summary = select_construction(blind_rows, audit_rows)
        result["selection"] = {
            **summary,
            "status": "model_selection_complete",
            "rejected_candidate_ids": [row["candidate_id"] for row in rejected],
        }
        result["human_review_queue"] = [{
            "candidate_id": row["candidate_id"],
            "source_image_id": row["source_image_id"],
            "status": "model_candidate_pending_construction_and_human_review",
        } for row in selected]
    return result


def analyze_construct_output(run_dir: Path) -> dict:
    parsed, summary = analyze_raw_stage(run_dir, "construct")
    decisions = {"accept_candidate": 0, "reject": 0}
    queue = []
    for row in parsed:
        decision = row["proposal"].get("decision")
        if decision in decisions:
            decisions[decision] += 1
        if decision == "accept_candidate":
            queue.append({
                "candidate_id": row["candidate_id"],
                "source_image_id": row["source_image_id"],
                "status": "constructed_model_candidate_pending_human_review",
            })
    return {
        "run_dir": str(run_dir),
        "run_type": "construct",
        "stage": summary,
        "decisions": decisions,
        "human_review_queue": queue,
        "automatically_human_verified": 0,
        "teacher_gap_measured": False,
        "scientific_status": "candidate_construction_only",
    }


def analyze_run(run_dir: Path) -> dict:
    name = run_dir.name.casefold()
    if "screen" in name:
        return analyze_screen_output(run_dir)
    if "construct" in name:
        return analyze_construct_output(run_dir)
    raise ValueError("run directory name must contain 'screen' or 'construct'")


def compact_comparison(first: dict, second: dict) -> dict:
    """Compare matching raw stages without inventing missing-stage zeros."""

    comparison = {}
    if first["run_type"] != second["run_type"]:
        raise ValueError("comparison requires two runs of the same type")
    if first["run_type"] == "screen":
        stages = ("blind", "audit")
        first_stages, second_stages = first["stages"], second["stages"]
    else:
        stages = ("construct",)
        first_stages = {"construct": first["stage"]}
        second_stages = {"construct": second["stage"]}
    for stage in stages:
        before, after = first_stages[stage], second_stages[stage]
        if before.get("status") == "not_available" or after.get("status") == "not_available":
            comparison[stage] = {"status": "not_comparable"}
            continue
        comparison[stage] = {
            "status": "comparable",
            "before": {
                "observed": before["observed"],
                "parse_rate": before["parse_rate"],
                "truncated_or_missing_eos": before["truncated_or_missing_eos"],
            },
            "after": {
                "observed": after["observed"],
                "parse_rate": after["parse_rate"],
                "truncated_or_missing_eos": after["truncated_or_missing_eos"],
            },
            "parse_rate_delta": after["parse_rate"] - before["parse_rate"],
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="earlier archive of the same type")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        raise SystemExit(f"missing run directory: {args.run_dir}")
    try:
        result = analyze_run(args.run_dir)
        if args.compare:
            result["comparison"] = compact_comparison(analyze_run(args.compare), result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
