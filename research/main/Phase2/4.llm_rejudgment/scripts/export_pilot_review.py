#!/usr/bin/env python3
"""Export a human-review worksheet for a frozen pilot judgment run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--rule-per-sample", type=Path, required=True)
    parser.add_argument("--judge-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requests = {(r["sample_id"], r["condition"]): r for r in jsonl(args.input_freeze)}
    judges = {(r["sample_id"], r["condition"]): r for r in jsonl(args.judge_jsonl)}
    rules = {}
    with args.rule_per_sample.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rules[(row["sample_id"], row["condition"])] = row
    if set(judges) - set(requests) or set(judges) - set(rules):
        raise ValueError("pilot rows do not join exactly to frozen inputs and preserved rule labels")
    if len(judges) != 50:
        raise ValueError(f"expected exactly 50 pilot judgments; found {len(judges)}")

    fields = [
        "sample_id", "condition", "question", "choices_json", "response",
        "llm_status", "llm_option", "llm_evidence", "llm_raw_text",
        "gold_option", "rule_option", "rule_correct",
        "manual_status", "manual_option", "manual_extraction_correct",
        "manual_correctness", "reviewer_notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(judges, key=lambda k: (k[1], k[0])):
            req, judge, rule = requests[key], judges[key], rules[key]
            writer.writerow({
                "sample_id": req["sample_id"],
                "condition": req["condition"],
                "question": req["question"],
                "choices_json": json.dumps(req["choices"], ensure_ascii=False),
                "response": req["response"],
                "llm_status": judge.get("status", ""),
                "llm_option": judge.get("selected_option") or "",
                "llm_evidence": judge.get("evidence", ""),
                "llm_raw_text": judge.get("judge_raw_text", ""),
                "gold_option": req["gold_option"],
                "rule_option": rule.get("parsed_answer", ""),
                "rule_correct": rule.get("correct", ""),
                "manual_status": "",
                "manual_option": "",
                "manual_extraction_correct": "",
                "manual_correctness": "",
                "reviewer_notes": "",
            })

    print(json.dumps({"rows": len(judges), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
