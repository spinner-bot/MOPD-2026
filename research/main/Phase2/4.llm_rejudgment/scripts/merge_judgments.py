#!/usr/bin/env python3
"""Join preserved rule labels with LLM judgments and quantify their differences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def bootstrap_ci(values: list[float], seed: int, iterations: int = 5000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * (iterations - 1))], means[int(0.975 * (iterations - 1))]]


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

    shard_paths = sorted(args.judge_dir.glob("judge-shard-*.jsonl"))
    if not shard_paths:
        raise ValueError("no judge shard JSONL files found")
    shard_manifests = {}
    for path in shard_paths:
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise ValueError(f"missing shard manifest: {manifest_path}")
        shard_manifests[path] = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_shards_values = {manifest.get("num_shards") for manifest in shard_manifests.values()}
    shard_indices = {manifest.get("shard_index") for manifest in shard_manifests.values()}
    if len(num_shards_values) != 1:
        raise ValueError(f"inconsistent num_shards in manifests: {num_shards_values}")
    num_shards = num_shards_values.pop()
    if not isinstance(num_shards, int) or num_shards < 1:
        raise ValueError(f"invalid num_shards in manifests: {num_shards}")
    if shard_indices != set(range(num_shards)) or len(shard_paths) != num_shards:
        raise ValueError(f"shard coverage failed: files={len(shard_paths)}, indices={shard_indices}, expected={num_shards}")
    expected_keys_by_shard = {index: set() for index in range(num_shards)}
    for index, key in enumerate(requests):
        expected_keys_by_shard[index % num_shards].add(key)
    expected_input_hash = sha256(args.input_freeze)
    prompt_hashes = set()
    model_config_hashes = set()
    for path, manifest in shard_manifests.items():
        index = manifest["shard_index"]
        if path.name != f"judge-shard-{index}.jsonl":
            raise ValueError(f"output filename/index mismatch: {path.name} vs {index}")
        expected_conditions = Counter(requests[key]["condition"] for key in expected_keys_by_shard[index])
        if manifest.get("input_sha256") != expected_input_hash:
            raise ValueError(f"frozen input hash mismatch in {path}")
        if manifest.get("selected_records") != len(expected_keys_by_shard[index]):
            raise ValueError(f"manifest selected_records mismatch in {path}")
        if manifest.get("successful_records") != len(expected_keys_by_shard[index]):
            raise ValueError(f"manifest successful_records mismatch in {path}")
        if manifest.get("selected_condition_counts") != dict(expected_conditions):
            raise ValueError(f"manifest condition balance mismatch in {path}")
        if manifest.get("limit_per_condition") is not None:
            raise ValueError(f"full run manifest has a pilot limit: {path}")
        if not manifest.get("finished_at_utc"):
            raise ValueError(f"shard has no finished timestamp: {path}")
        prompt_hashes.add(manifest.get("prompt_sha256"))
        model_config_hashes.add(manifest.get("model_config_sha256"))
    if len(prompt_hashes) != 1 or None in prompt_hashes:
        raise ValueError(f"prompt hashes differ across shards: {prompt_hashes}")
    if len(model_config_hashes) != 1 or None in model_config_hashes:
        raise ValueError(f"model config hashes differ across shards: {model_config_hashes}")

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
    first_shard_for_key = {}
    latest_by_shard = {manifest["shard_index"]: set() for manifest in shard_manifests.values()}
    for path in shard_paths:
        shard_index = shard_manifests[path]["shard_index"]
        for line_no, row in read_jsonl(path):
            key = (row.get("sample_id"), row.get("condition"))
            if key not in requests:
                raise ValueError(f"unknown output key at {path}:{line_no}: {key}")
            expected_shard = next((i for i, keys in expected_keys_by_shard.items() if key in keys), None)
            if expected_shard != shard_index:
                raise ValueError(f"row routed to wrong shard at {path}:{line_no}: key={key}, expected={expected_shard}")
            if key in first_shard_for_key and first_shard_for_key[key] != path:
                raise ValueError(f"duplicate key across shard files: {key} in {first_shard_for_key[key]} and {path}")
            first_shard_for_key[key] = path
            latest_by_shard[shard_index].add(key)
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
    for index, expected_keys in expected_keys_by_shard.items():
        if latest_by_shard[index] != expected_keys:
            raise ValueError(f"shard {index} key coverage mismatch: {len(latest_by_shard[index])}/{len(expected_keys)}")

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
        evidence = judge.get("evidence", "")
        if llm_status == "ANSWERED" and (not evidence or evidence not in req["response"]):
            raise ValueError(f"ANSWERED evidence is not an exact source substring: {key}")
        if evidence and evidence not in req["response"]:
            raise ValueError(f"non-verbatim evidence remains after validation: {key}")
        evidence_validation = judge.get("evidence_validation") or (
            "exact_substring" if evidence else "empty_optional_evidence"
        )
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
            "correct": llm_correct,
            "parsed_answer": llm_answer or "",
            "answer_valid": llm_status == "ANSWERED",
            "strict_format": as_bool(old.get("strict_format")),
            "reasoning_marker": as_bool(old.get("reasoning_marker")),
            "llm_evidence": evidence,
            "llm_evidence_validation": evidence_validation,
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

    summary_rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        subset = [row for row in output_rows if row["condition"] == condition]
        correct = [float(bool(row["llm_correct"])) for row in subset]
        chars = [float(row["response_chars"]) for row in subset]
        summary_rows.append({
            "condition": condition,
            "n": len(subset),
            "accuracy": sum(correct) / len(correct),
            "valid_answer_rate": sum(row["llm_status"] == "ANSWERED" for row in subset) / len(subset),
            "strict_format_rate": sum(bool(row["strict_format"]) for row in subset) / len(subset),
            "reasoning_marker_rate": sum(bool(row["reasoning_marker"]) for row in subset) / len(subset),
            "response_chars_mean": sum(chars) / len(chars),
            "response_chars_median": sorted(chars)[len(chars) // 2] if len(chars) % 2 else (sorted(chars)[len(chars)//2 - 1] + sorted(chars)[len(chars)//2]) / 2,
            "response_chars_ci95": bootstrap_ci(chars, seed=261013 + condition_index),
            "api_error_count": sum(bool(row["api_error"]) for row in subset),
        })
    by_condition_id = {
        condition: {row["sample_id"]: bool(row["llm_correct"]) for row in output_rows if row["condition"] == condition}
        for condition in CONDITIONS
    }
    paired_comparisons = []
    for condition_index, condition in enumerate(CONDITIONS[1:], start=1):
        ids = sorted(by_condition_id["S0"])
        a = [by_condition_id["S0"][sample_id] for sample_id in ids]
        b = [by_condition_id[condition][sample_id] for sample_id in ids]
        diff = [float(y) - float(x) for x, y in zip(a, b)]
        paired_comparisons.append({
            "comparison": f"{condition} vs S0",
            "accuracy_difference": sum(diff) / len(diff),
            "accuracy_difference_ci95": bootstrap_ci(diff, seed=261023 + condition_index),
            "mcnemar": compare(a, b),
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
        "llm_evidence_validation_counts": dict(Counter(row["llm_evidence_validation"] for row in output_rows)),
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
    (args.output_dir / "summary.json").write_text(json.dumps({
        "schema": "short-output-rejudged-summary-v1",
        "sample_size": 2000,
        "counts": dict(Counter(row["condition"] for row in output_rows)),
        "summary": summary_rows,
        "paired_comparisons": paired_comparisons,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(output_rows), "statuses": summary["llm_status_counts"], "differences": len(delta_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
