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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--min-total", type=int, default=1)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--min-parse-rate", type=float, default=0.95)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    parsed, rejected, parse_summary = parse_records(rows, args.stage)
    ids = [row.get("candidate_id") for row in rows]
    errors = sum(bool(row.get("error")) for row in rows)
    truncated = sum(row.get("truncated") is not False or row.get("eos_reached") is not True for row in rows)
    revisions = sorted({str(row.get("model_revision", "")) for row in rows})
    commits = sorted({str(row.get("code_commit", "")) for row in rows})
    summary = {
        "stage": args.stage,
        "files": paths,
        "total": len(rows),
        "valid_parsed": len(parsed),
        "rejected": len(rejected),
        "parse_rate": len(parsed) / len(rows) if rows else 0.0,
        "runner_errors": errors,
        "truncated_or_missing_eos": truncated,
        "unique_candidate_ids": len(set(ids)),
        "model_revisions": revisions,
        "code_commits": commits,
        "parse_summary": parse_summary,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))

    failures = []
    if len(rows) < args.min_total:
        failures.append(f"total={len(rows)} < min_total={args.min_total}")
    if args.expected_total is not None and len(rows) != args.expected_total:
        failures.append(f"total={len(rows)} != expected_total={args.expected_total}")
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        failures.append("candidate ids are missing or duplicated")
    if summary["parse_rate"] < args.min_parse_rate:
        failures.append(f"parse_rate={summary['parse_rate']:.6f} < {args.min_parse_rate:.6f}")
    if errors:
        failures.append(f"runner_errors={errors}")
    if truncated:
        failures.append(f"truncated_or_missing_eos={truncated}")
    if len(revisions) != 1 or not revisions[0]:
        failures.append("model revision must be one non-empty value")
    if len(commits) != 1 or not commits[0] or commits[0] == "unknown":
        failures.append("code commit must be one known value")
    if failures:
        raise SystemExit("teacher-gap candidate gate FAILED: " + "; ".join(failures))
    print("teacher-gap candidate gate PASSED")


if __name__ == "__main__":
    main()

