#!/usr/bin/env python3
"""Derive paired metrics and statistical comparisons from five raw JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]
REASONING_RE = re.compile(
    r"\b(?:because|therefore|thus|so|since|reason|looking at|the image shows|i can see|first|second|hence)\b",
    re.I,
)
ANSWER_RE = re.compile(r"(?<![A-Za-z])(?:\(?\s*([ABCD])\s*\)?)(?![A-Za-z])", re.I)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_answer(text: str):
    text = (text or "").strip()
    if not text:
        return None
    exact = re.fullmatch(r"\(?\s*([ABCD])\s*\)?[.!:]?", text, re.I)
    if exact:
        return exact.group(1).upper()
    marked = re.search(r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?\s*([ABCD])\b", text, re.I)
    if marked:
        return marked.group(1).upper()
    first = ANSWER_RE.search(text)
    return first.group(1).upper() if first else None


def metric_row(record, labels):
    text = record.get("response") or ""
    answer = parse_answer(text)
    gold = str(record.get("gold_answer", record.get("response_key", ""))).upper()
    if not gold:
        gold = str(labels.get(record["sample_id"], "")).upper()
    valid_labels = set("ABCD")
    return {
        "sample_id": record["sample_id"],
        "condition": record["condition"],
        "split": record.get("split"),
        "task": record.get("task"),
        "source": record.get("source"),
        "gold_answer": gold,
        "parsed_answer": answer,
        "answer_valid": answer in valid_labels,
        "correct": answer == gold if gold else None,
        "strict_format": bool(re.fullmatch(r"[ABCD]", text.strip())),
        "reasoning_marker": bool(REASONING_RE.search(text)),
        "response_chars": len(text),
        "response_words": len(text.split()),
        "response": text,
        "api_error": record.get("api_error"),
        "image_sha256": record.get("image_sha256"),
    }


def bootstrap_ci(values, seed=261011, iterations=5000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    for i in range(iterations):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def mcnemar(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    discordant = int(np.sum(a != b))
    if discordant == 0:
        p = 1.0
    else:
        p = float(binomtest(int(np.sum((~a) & b)), discordant, 0.5).pvalue)
    return {"a_only": int(np.sum(a & ~b)), "b_only": int(np.sum(~a & b)), "discordant": discordant, "p_value": p}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sample = load_json(args.sample)
    labels = {r["sample_id"]: str(r.get("response", "")).upper() for r in sample["records"]}
    rows = []
    counts = {}
    by_condition = {}
    for condition in CONDITIONS:
        path = args.run_root / "outputs" / f"{condition}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing output: {path}")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_condition[condition] = {r["sample_id"]: r for r in records}
        counts[condition] = len(records)
        rows.extend(metric_row(r, labels) for r in records)

    expected = set(labels)
    for condition in CONDITIONS:
        actual = set(by_condition[condition])
        if actual != expected:
            raise SystemExit(f"sample mismatch in {condition}: expected {len(expected)}, got {len(actual)}")

    with (args.output / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        correct = np.array([bool(r["correct"]) for r in subset])
        valid = np.array([bool(r["answer_valid"]) for r in subset])
        strict = np.array([bool(r["strict_format"]) for r in subset])
        lengths = np.array([r["response_chars"] for r in subset], dtype=float)
        reasoning = np.array([bool(r["reasoning_marker"]) for r in subset])
        summary.append(
            {
                "condition": condition,
                "n": len(subset),
                "accuracy": float(correct.mean()),
                "valid_answer_rate": float(valid.mean()),
                "strict_format_rate": float(strict.mean()),
                "reasoning_marker_rate": float(reasoning.mean()),
                "response_chars_mean": float(lengths.mean()),
                "response_chars_median": float(np.median(lengths)),
                "response_chars_ci95": bootstrap_ci(lengths),
                "api_error_count": sum(bool(r["api_error"]) for r in subset),
            }
        )

    paired = []
    base = {r["sample_id"]: r for r in rows if r["condition"] == "S0"}
    for condition in CONDITIONS[1:]:
        current = {r["sample_id"]: r for r in rows if r["condition"] == condition}
        a = np.array([bool(base[k]["correct"]) for k in sorted(expected)])
        b = np.array([bool(current[k]["correct"]) for k in sorted(expected)])
        diff = b.astype(float) - a.astype(float)
        paired.append(
            {
                "comparison": f"{condition} vs S0",
                "accuracy_difference": float(diff.mean()),
                "accuracy_difference_ci95": bootstrap_ci(diff),
                "mcnemar": mcnemar(a, b),
            }
        )

    payload = {
        "schema": "short-output-derived-analysis-v1",
        "sample_size": len(expected),
        "counts": counts,
        "summary": summary,
        "paired_comparisons": paired,
        "notes": [
            "gold_answer is taken from the frozen sample manifest response field.",
            "reasoning_marker_rate is a lexical screening metric, not a semantic proof of reasoning.",
            "raw model responses are preserved separately; this file contains derived metrics only.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_size": len(expected), "counts": counts, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
