#!/usr/bin/env python3
"""Join preserved rule labels with LLM judgments and quantify their differences."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    if not a:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    p_a = sum(a) / len(a)
    p_b = sum(b) / len(b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def compare(a: list[bool], b: list[bool]) -> dict:
    rule_only = sum(x and not y for x, y in zip(a, b))
    llm_only = sum(not x and y for x, y in zip(a, b))
    discordant = rule_only + llm_only
    if discordant == 0:
        p = 1.0
    else:
        lower_tail = min(llm_only, rule_only)
        p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(lower_tail + 1)) / (2 ** discordant))
    return {
        "rule_only_correct": rule_only,
        "llm_only_correct": llm_only,
        "discordant_correctness": discordant,
        "mcnemar_exact_p": p,
        "cohens_kappa_correctness": cohens_kappa(a, b),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--rule-per-sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requests = {}
    for _, row in read_jsonl(args.input_freeze):
        key = (row["sample_id"], row["condition"])
        if key in requests:
            raise ValueError(f"duplicate frozen request: {key}")
        requests[key] = row
    if len(requests) != 10000:
        raise ValueError(f"expected 10,000 frozen requests, got {len(requests)}")

    rules = {}
    with args.rule_per_sample.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["sample_id"], row["condition"])
            if key in rules:
                raise ValueError(f"duplicate old rule label: {key}")
            rules[key] = row
    if set(rules) != set(requests):
        raise ValueError(f"rule/frozen keys differ: rule={len(rules)}, input={len(requests)}")

    latest = {}
    technical_errors = []
    invalid_outputs = []
    for path in sorted(args.judge_dir.glob("judge-shard-*.jsonl")):
        for line_no, row in read_jsonl(path):
            key = (row.get("sample_id"), row.get("condition"))
            if key not in requests:
                raise ValueError(f"unknown output key at {path}:{line_no}: {key}")
            latest[key] = row
            if row.get("request_status") == "TECHNICAL_ERROR":
                technical_errors.append({"path": str(path), "line": line_no, "key": key, "error": row.get("judge_error")})
            elif row.get("request_status") != "OK":
                invalid_outputs.append({"path": str(path), "line": line_no, "key": key, "error": row.get("validation_error")})

    missing = sorted(set(requests) - set(latest))
    bad = [key for key in requests if key not in latest or latest[key].get("request_status") != "OK"]
    if missing or bad:
        raise ValueError(
            f"judge completeness gate failed: missing={len(missing)}, non-OK latest={len(bad)}, "
            f"technical_errors={len(technical_errors)}, invalid={len(invalid_outputs)}"
        )
    if len(latest) != 10000:
        raise ValueError(f"expected 10,000 unique judged keys, got {len(latest)}")

    output_rows = []
    old_status_cross = Counter()
    old_llm_answer_matches = 0
    both_answered = 0
    delta_records = []
    per_condition = []
    all_rule_correct, all_llm_correct = [], []

    for key, req in requests.items():
        old = rules[key]
        judge = latest[key]
        if judge.get("response_sha256") != req["response_sha256"]:
            raise ValueError(f"source response hash mismatch: {key}")
        old_answer = (old.get("parsed_answer") or "").strip().upper() or None
        old_correct = as_bool(old.get("correct"))
        llm_status = judge["status"]
        llm_answer = judge["selected_option"]
        llm_correct = bool(llm_status == "ANSWERED" and llm_answer == req["gold_option"])
        old_answered = bool(old.get("answer_valid") and as_bool(old.get("answer_valid")))
        old_status_cross[("ANSWERED" if old_answered else "NO_PARSED_ANSWER", llm_status)] += 1
        if old_answered and llm_status == "ANSWERED":
            both_answered += 1
            old_llm_answer_matches += old_answer == llm_answer

        row = {
            "sample_id": req["sample_id"],
            "condition": req["condition"],
            "task": old.get("task", ""),
            "source": old.get("source", ""),
            "split": old.get("split", ""),
            "gold_answer": req["gold_option"],
            "rule_parsed_answer": old_answer or "",
            "rule_answer_valid": old_answered,
            "rule_correct": old_correct,
            "llm_status": llm_status,
            "llm_parsed_answer": llm_answer or "",
            "llm_correct": llm_correct,
            "llm_evidence": judge.get("evidence", ""),
            "llm_reason_code": judge.get("reason_code", ""),
            "llm_raw_text": judge.get("judge_raw_text", ""),
            "response_chars": req["response_chars"],
            "response_sha256": req["response_sha256"],
            "api_error": req.get("api_error") or "",
            "completion_tokens": (req.get("usage") or {}).get("completion_tokens", ""),
            "prompt_tokens": (req.get("usage") or {}).get("prompt_tokens", ""),
            "response": req["response"],
        }
        output_rows.append(row)
        all_rule_correct.append(old_correct)
        all_llm_correct.append(llm_correct)
        if old_answer != llm_answer or old_correct != llm_correct or llm_status != "ANSWERED":
            delta_records.append({**row, "model_judgment": judge})

    fields = list(output_rows[0])
    with (args.output_dir / "per_sample_rejudged.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    with (args.output_dir / "judgment_differences.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in delta_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    for condition in CONDITIONS:
        subset = [row for row in output_rows if row["condition"] == condition]
        old_flags = [bool(row["rule_correct"]) for row in subset]
        llm_flags = [bool(row["llm_correct"]) for row in subset]
        exact_answered = sum(bool(row["rule_parsed_answer"]) and bool(row["llm_parsed_answer"]) for row in subset)
        exact_n = sum(
            bool(row["rule_parsed_answer"])
            and bool(row["llm_parsed_answer"])
            and row["rule_parsed_answer"] == row["llm_parsed_answer"]
            for row in subset
        )
        comparison = compare(old_flags, llm_flags)
        per_condition.append({
            "condition": condition,
            "n": len(subset),
            "rule_accuracy": sum(old_flags) / len(subset),
            "llm_accuracy": sum(llm_flags) / len(subset),
            "llm_answered_n": sum(row["llm_status"] == "ANSWERED" for row in subset),
            "llm_no_answer_n": sum(row["llm_status"] == "NO_ANSWER" for row in subset),
            "llm_ambiguous_n": sum(row["llm_status"] == "AMBIGUOUS" for row in subset),
            "old_new_answer_letter_match_n": exact_n,
            "old_new_answered_both_n": exact_answered,
            "old_new_answer_letter_match_rate_among_both_answered": exact_n / exact_answered if exact_answered else None,
            **comparison,
        })

    overall = compare(all_rule_correct, all_llm_correct)
    summary = {
        "schema": "phase2-rule-vs-llm-rejudgment-v1",
        "n": len(output_rows),
        "llm_status_counts": dict(Counter(row["llm_status"] for row in output_rows)),
        "rule_accuracy": sum(all_rule_correct) / len(all_rule_correct),
        "llm_accuracy": sum(all_llm_correct) / len(all_llm_correct),
        "rule_llm_correctness_comparison": overall,
        "old_new_answered_n": both_answered,
        "old_new_same_answer_n": old_llm_answer_matches,
        "old_new_same_answer_rate_among_both_answered": old_llm_answer_matches / both_answered if both_answered else None,
        "old_parse_status_by_llm_status": {
            f"{old_status}|{llm_status}": count for (old_status, llm_status), count in sorted(old_status_cross.items())
        },
        "per_condition": per_condition,
        "difference_records_n": len(delta_records),
        "technical_errors_encountered": technical_errors,
        "invalid_outputs_encountered": invalid_outputs,
        "interpretation_note": "The LLM label is an adjudication result, not an independently verified gold standard; report disagreement and manual-audit evidence separately.",
    }
    (args.output_dir / "detector_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "detector_by_condition.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_condition[0]))
        writer.writeheader()
        writer.writerows(per_condition)
    print(json.dumps({"n": len(output_rows), "statuses": summary["llm_status_counts"], "differences": len(delta_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
