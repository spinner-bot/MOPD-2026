"""Parse raw 1017 pilot inference, apply the gate, and make the frozen decision."""

from __future__ import annotations

import argparse
import glob
import json
import hashlib
import math
from pathlib import Path

from exploratory_teacher_gap import (
    CHOICES,
    decide_teacher_gap,
    validate_manifest_bundle,
    validate_result_records,
)


def load_jsonl(patterns: list[str]) -> list[dict]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    rows = []
    for value in paths:
        with open(value, encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as sink:
        sink.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def parse_constrained_answer(raw: dict) -> tuple[str | None, str | None]:
    sequences = raw.get("choice_token_ids")
    token_ids = raw.get("generated_token_ids")
    if not isinstance(sequences, dict) or set(sequences) != set(CHOICES) or not isinstance(token_ids, list):
        return None, None
    matches = []
    for choice in CHOICES:
        sequence = sequences[choice]
        if isinstance(sequence, list) and token_ids[: len(sequence)] == sequence:
            matches.append(choice)
    if len(matches) == 1:
        return matches[0], "constrained_token_trie"
    text = str(raw.get("raw_generation", "")).strip()
    if text in CHOICES:
        return text, "exact_text_fallback"
    return None, None


def enrich_raw_record(raw: dict) -> dict:
    row = dict(raw)
    parsed, method = parse_constrained_answer(raw)
    row["parsed_answer"] = parsed
    row["parse_method"] = method
    scores = raw.get("choice_log_likelihoods")
    if not isinstance(scores, dict) or set(scores) != set(CHOICES):
        return row
    values = [float(scores[choice]) for choice in CHOICES]
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    probabilities = {choice: weights[index] / total for index, choice in enumerate(CHOICES)}
    correct_choice = raw.get("correct_choice")
    row["choice_probabilities"] = probabilities
    if correct_choice in CHOICES:
        row["correct"] = parsed == correct_choice
        row["correct_log_prob"] = float(scores[correct_choice])
        incorrect = max(float(scores[choice]) for choice in CHOICES if choice != correct_choice)
        row["margin"] = float(scores[correct_choice]) - incorrect
    row["entropy"] = -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--inputs", nargs="+", required=True)
    validate_parser.add_argument("--expected-samples", type=int, required=True)
    validate_parser.add_argument("--parsed-output", type=Path, required=True)
    validate_parser.add_argument("--summary", type=Path, required=True)
    decide_parser = subparsers.add_parser("decide")
    decide_parser.add_argument("--input", type=Path, required=True)
    decide_parser.add_argument("--expected-samples", type=int, default=96)
    decide_parser.add_argument("--leakage-free", action="store_true", required=True)
    decide_parser.add_argument("--output", type=Path, required=True)
    bundle_parser = subparsers.add_parser("validate-manifests")
    bundle_parser.add_argument("--canary", type=Path, required=True)
    bundle_parser.add_argument("--dev", type=Path, required=True)
    bundle_parser.add_argument("--locked", type=Path, required=True)
    bundle_parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate-manifests":
        if args.summary.exists():
            raise SystemExit(f"refusing to overwrite manifest gate summary: {args.summary}")
        paths = {"canary": args.canary, "dev": args.dev, "locked": args.locked}
        partitions = {name: load_jsonl([str(path)]) for name, path in paths.items()}
        errors = validate_manifest_bundle(partitions)
        summary = {
            "status": "passed" if not errors else "failed",
            "partition_records": {name: len(rows) for name, rows in partitions.items()},
            "partition_sha256": {name: file_sha256(path) for name, path in paths.items()},
            "validation_errors": errors,
            "performance_results_read": False,
        }
        write_json_new(args.summary, summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if errors:
            raise SystemExit("teacher-gap manifest bundle FAILED: " + "; ".join(errors))
    elif args.command == "validate":
        if args.parsed_output.exists() or args.summary.exists():
            raise SystemExit("refusing to overwrite parsed output or validation summary")
        rows = [enrich_raw_record(row) for row in load_jsonl(args.inputs)]
        errors = validate_result_records(rows, expected_samples=args.expected_samples)
        summary = {
            "expected_samples": args.expected_samples,
            "expected_records": args.expected_samples * 4,
            "observed_records": len(rows),
            "unique_sample_condition": len({(row.get("sample_id"), row.get("condition")) for row in rows}),
            "validation_errors": errors,
            "status": "passed" if not errors else "failed",
            "note": "No condition-level performance is emitted before the completeness gate passes.",
        }
        write_json_new(args.summary, summary)
        if errors:
            raise SystemExit("teacher-gap result gate FAILED: " + "; ".join(errors))
        write_jsonl(args.parsed_output, rows)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite decision output: {args.output}")
        rows = load_jsonl([str(args.input)])
        decision = decide_teacher_gap(
            rows,
            leakage_free=args.leakage_free,
            expected_samples=args.expected_samples,
        )
        write_json_new(args.output, decision)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        if decision["decision"] == "INVALID":
            raise SystemExit("teacher-gap decision refused invalid records")


if __name__ == "__main__":
    main()
