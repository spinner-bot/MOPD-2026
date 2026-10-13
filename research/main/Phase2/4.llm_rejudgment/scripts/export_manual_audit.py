#!/usr/bin/env python3
"""Draw a reproducible stratified human audit sample from final joined labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--per-condition-agreement", type=int, default=20)
    parser.add_argument("--per-condition-disagreement", type=int, default=20)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if len(all_rows) != 10000:
        raise ValueError(f"expected 10,000 joined rows, got {len(all_rows)}")

    selected = []
    strata_counts = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in all_rows if row["condition"] == condition]
        if len(condition_rows) != 2000:
            raise ValueError(f"expected 2,000 rows in {condition}, got {len(condition_rows)}")
        answered = [row for row in condition_rows if row["llm_status"] == "ANSWERED"]
        agreement = [
            row for row in answered
            if truth(row["rule_answer_valid"]) and row["rule_parsed_answer"] == row["llm_parsed_answer"]
        ]
        disagreement = [
            row for row in answered
            if truth(row["rule_answer_valid"]) and row["rule_parsed_answer"] != row["llm_parsed_answer"]
        ]
        strata = [
            ("both_answered_same_option", agreement, args.per_condition_agreement),
            ("both_answered_different_option", disagreement, args.per_condition_disagreement),
        ]
        for stratum, population, requested_n in strata:
            n = min(requested_n, len(population))
            draws = rng.sample(population, n)
            strata_counts[f"{condition}|{stratum}"] = {"population_n": len(population), "sample_n": n}
            for row in draws:
                selected.append({**row, "audit_stratum": stratum, "audit_population_n": len(population), "audit_sample_n": n, "audit_weight": len(population) / n if n else ""})

        special = [row for row in condition_rows if row["llm_status"] != "ANSWERED"]
        strata_counts[f"{condition}|llm_nonanswered_all"] = {"population_n": len(special), "sample_n": len(special)}
        for row in special:
            selected.append({**row, "audit_stratum": "llm_nonanswered_all", "audit_population_n": len(special), "audit_sample_n": len(special), "audit_weight": 1.0})

    selected.sort(key=lambda row: (CONDITIONS.index(row["condition"]), row["audit_stratum"], row["sample_id"]))
    output_fields = [
        "audit_id", "audit_seed", "audit_stratum", "audit_population_n", "audit_sample_n", "audit_weight",
        "sample_id", "condition", "task", "source", "question", "choices_json", "response", "gold_answer",
        "rule_parsed_answer", "rule_answer_valid", "rule_correct", "llm_status", "llm_parsed_answer", "llm_correct",
        "llm_evidence", "llm_evidence_validation", "llm_raw_text", "response_sha256",
        "reviewer_status", "reviewer_option", "manual_extraction_correct", "manual_correctness", "reviewer_notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for audit_id, row in enumerate(selected, 1):
            writer.writerow({
                "audit_id": audit_id,
                "audit_seed": args.seed,
                "choices_json": row.get("choices", ""),
                "gold_answer": row.get("gold_answer", ""),
                **{field: row.get(field, "") for field in output_fields if field not in {"audit_id", "audit_seed", "choices_json", "gold_answer"}},
            })

    manifest = {
        "schema": "phase2-llm-judge-manual-audit-sample-v1",
        "seed": args.seed,
        "population_n": len(all_rows),
        "sample_n": len(selected),
        "design": "Per condition, simple random sample without replacement from both-answer-same-option and both-answer-different-option strata; include every LLM NO_ANSWER/AMBIGUOUS row.",
        "strata": strata_counts,
        "output_csv": str(args.output),
        "review_fields": ["reviewer_status", "reviewer_option", "manual_extraction_correct", "manual_correctness", "reviewer_notes"],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_n": len(selected), "strata": strata_counts, "output": str(args.output), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
