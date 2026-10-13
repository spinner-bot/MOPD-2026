#!/usr/bin/env python3
"""Reproducible paired analysis of verified Work5 outputs and two score sources.

This script is intentionally read-only with respect to raw inputs. It writes
versioned aggregate CSV/JSON tables beneath DATA_ROOT/derived only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

CONDITIONS = ("S0", "SC", "S1", "S2", "S3")
TRACKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")
CAMPAIGN = "full/runs/prompt-registry-v1"
N_BOOTSTRAP = 2000


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keyed_unique(rows: list[dict], path: Path, key="sample_uid") -> dict[str, dict]:
    result = {}
    for row in rows:
        uid = row.get(key)
        if not isinstance(uid, str) or not uid or uid in result:
            raise ValueError(f"{path}: missing or duplicate {key}: {uid!r}")
        result[uid] = row
    return result


def normalize_number(value: str) -> Decimal | None:
    text = str(value).strip().replace(",", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[。．.]$", "", text)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def answer_matches(extracted: dict, gold: str, answer_format: str) -> bool:
    if extracted.get("status") != "ANSWERED":
        return False
    if answer_format == "MCQ_OPTION":
        chosen = extracted.get("selected_option")
        gold_match = re.fullmatch(r"\s*(?:\(([A-F])\)|([A-F])(?:[.)])?)\s*", str(gold), re.I)
        return bool(gold_match and chosen and chosen.upper() == (gold_match.group(1) or gold_match.group(2)).upper())
    predicted = normalize_number(extracted.get("answer_text", ""))
    target = normalize_number(gold)
    return predicted is not None and target is not None and predicted == target


def exact_mcnemar_p(better: int, worse: int) -> float:
    """Two-sided exact binomial test on discordant paired outcomes."""
    n = better + worse
    if n == 0:
        return 1.0
    if better == worse:
        return 1.0
    cutoff = min(better, worse)
    log_pmf = -n * math.log(2)
    log_tail = log_pmf
    for k in range(cutoff):
        log_pmf += math.log(n - k) - math.log(k + 1)
        peak = max(log_tail, log_pmf)
        log_tail = peak + math.log1p(math.exp(min(log_tail, log_pmf) - peak))
    return min(1.0, 2.0 * math.exp(log_tail))


def holm_adjust(p_values: list[float]) -> list[float]:
    ordered = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(ordered):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = rank
        start = end
    return ranks


def spearman_rho(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x, y = average_ranks(x_values), average_ranks(y_values)
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def cluster_bootstrap_delta(rows: list[dict], a: str, b: str, source: str, seed: int, iterations=N_BOOTSTRAP):
    import numpy as np

    clusters: dict[str, dict[tuple[str, str], dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        clusters[row["cluster_uid"]][(row["track"], row["sample_uid"])][row["condition"]] = row
    values = []
    for samples in clusters.values():
        paired = [conditions for conditions in samples.values() if a in conditions and b in conditions]
        ca = sum(item[f"{source}_correct"] for item in (pair[a] for pair in paired))
        cb = sum(item[f"{source}_correct"] for item in (pair[b] for pair in paired))
        if paired:
            values.append((len(paired), ca, cb))
    if not values:
        return (None, None)
    matrix = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n_clusters = len(values)
    deltas = np.empty(iterations, dtype=float)
    for i in range(iterations):
        selected = rng.integers(0, n_clusters, size=n_clusters)
        draw = matrix[selected].sum(axis=0)
        deltas[i] = (draw[2] - draw[1]) / draw[0]
    return tuple(float(x) for x in np.percentile(deltas, [2.5, 97.5]))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(data_root: Path) -> tuple[list[dict], dict]:
    campaign = data_root / CAMPAIGN
    if not (campaign / "output-verification.json").is_file():
        raise FileNotFoundError("Successful full-output verification marker is required")
    verification = load_json(campaign / "output-verification.json")
    audits = {"duplicate_extraction_attempts": 0, "native_rows": 0, "extraction_rows": 0,
              "source_files": [], "prompt_registry_sha256": verification.get("prompt_registry_sha256")}
    if verification.get("total_generation_outputs") != 160810:
        raise ValueError("Verified output manifest does not describe 160,810 full requests")
    expected_conditions = set(CONDITIONS)
    if set(verification.get("conditions", {})) != expected_conditions:
        raise ValueError("Full output manifest is missing one or more conditions")

    rows = []
    audits["source_files"].append({"role": "full_output_verification", "path": str(campaign / "output-verification.json"),
                                   "sha256": sha256_file(campaign / "output-verification.json")})
    for track in TRACKS:
        prepared_path = data_root / "prepared" / track / "benchmark.json"
        audits["source_files"].append({"role": "benchmark_input", "path": str(prepared_path), "sha256": sha256_file(prepared_path)})
        source = keyed_unique(load_json(prepared_path), prepared_path)
        if len(source) != {"vstar": 191, "zoombench": 845, "hrbench-4k": 800,
                           "hrbench-8k": 800, "mme-realworld": 23609,
                           "mme-realworld-cn": 5917}[track]:
            raise ValueError(f"{track}: prepared sample count differs from frozen dataset audit")

        for condition in CONDITIONS:
            tag = f"OPD-V-Qwen3.5-4B-{condition}"
            output_path = campaign / "outputs" / condition / track / f"{tag}_answer.jsonl"
            audits["source_files"].append({"role": "model_output", "condition": condition, "track": track,
                                            "path": str(output_path), "sha256": sha256_file(output_path)})
            outputs = keyed_unique(load_jsonl(output_path), output_path)
            if set(outputs) != set(source):
                raise ValueError(f"{condition}/{track}: inference UID coverage mismatch")

            native_path = (data_root / "official-scoring/vision-opd-native-v1" / condition / track
                           / "judge" / track / f"{tag}_answer.jsonl")
            audits["source_files"].append({"role": "native_score", "condition": condition, "track": track,
                                            "path": str(native_path), "sha256": sha256_file(native_path)})
            native_raw = load_json(native_path)  # Official scorer writes a JSON array despite .jsonl suffix.
            native = keyed_unique(native_raw, native_path)
            if set(native) != set(source):
                raise ValueError(f"{condition}/{track}: native-score UID coverage mismatch")
            audits["native_rows"] += len(native)

            extraction_path = data_root / "judgment/runs/answer-extraction-v1" / track / f"{condition}.jsonl"
            audits["source_files"].append({"role": "answer_extraction", "condition": condition, "track": track,
                                            "path": str(extraction_path), "sha256": sha256_file(extraction_path)})
            extraction_attempts = load_jsonl(extraction_path)
            audits["extraction_rows"] += len(extraction_attempts)
            extracted = {}
            for attempt in extraction_attempts:
                key = attempt.get("sample_uid")
                if key in extracted:
                    audits["duplicate_extraction_attempts"] += 1
                extracted[key] = attempt  # Last attempt is the final retry result.
            if set(extracted) != set(source):
                raise ValueError(f"{condition}/{track}: answer-extraction UID coverage mismatch")

            for uid, item in source.items():
                output = outputs[uid]
                score = native[uid]
                extraction = extracted[uid]
                if output.get("prompt_condition") != condition:
                    raise ValueError(f"{condition}/{track}/{uid}: prompt condition mismatch")
                judge = str(score.get("judge", "")).strip().casefold()
                if extraction.get("request_status") != "OK":
                    raise ValueError(f"{condition}/{track}/{uid}: answer-extraction status {extraction.get('request_status')}")

                images = item.get("images") or []
                usage = output.get("api_usage") or {}
                question_type = item.get("question_type")
                answer_format = "NUMERIC_SHORT_ANSWER" if track == "zoombench" and question_type == "blank" else "MCQ_OPTION"
                extracted_correct = answer_matches(extraction, str(item.get("response") or ""), answer_format)
                rows.append({
                    "track": track, "condition": condition, "sample_uid": uid,
                    "cluster_uid": f"{track}:{images[0]}" if images else f"{track}:{uid}",
                    "category": item.get("category") or "(missing)",
                    "l2_category": item.get("l2-category") or "(missing)",
                    "cycle_category": item.get("cycle_category") if item.get("cycle_category") is not None else "(missing)",
                    "question_type": question_type or "(missing)",
                    "language": item.get("language") or "(missing)",
                    "answer_format": answer_format,
                    "native_correct": int(judge == "yes"),
                    "native_judge_source": score.get("judge_source"),
                    "native_judge_raw": score.get("judge"),
                    "native_judge_exact_binary": int(judge in {"yes", "no"}),
                    "extraction_correct": int(extracted_correct),
                    "extraction_status": extraction.get("status"),
                    "extraction_reason": extraction.get("reason_code"),
                    "native_extraction_agree": int(int(judge == "yes") == int(extracted_correct)),
                    "response_chars": len(str(output.get("model_answer") or "")),
                    "response_words": len(str(output.get("model_answer") or "").split()),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_api_tokens": usage.get("total_tokens"),
                })
    if len(rows) != 160810:
        raise ValueError(f"Expected 160,810 joined records, got {len(rows)}")
    return rows, audits


def summarize(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = defaultdict(list)
    by_condition_track: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_condition_track[(row["condition"], row["track"])].append(row)

    for (condition, track), group in by_condition_track.items():
        record = {"scope": "track", "stratum": track, "condition": condition, "n": len(group)}
        for source in ("native", "extraction"):
            record[f"{source}_correct"] = sum(row[f"{source}_correct"] for row in group)
            record[f"{source}_accuracy"] = record[f"{source}_correct"] / len(group)
        record["judge_extraction_agreement"] = statistics.mean(row["native_extraction_agree"] for row in group)
        record["native_judge_exact_binary_rate"] = statistics.mean(row["native_judge_exact_binary"] for row in group)
        record["mean_response_chars"] = statistics.mean(row["response_chars"] for row in group)
        record["median_response_chars"] = statistics.median(row["response_chars"] for row in group)
        for field in ("prompt_tokens", "completion_tokens", "total_api_tokens"):
            vals = [row[field] for row in group if isinstance(row[field], (int, float))]
            record[f"{field}_coverage"] = len(vals) / len(group)
            record[f"{field}_mean"] = statistics.mean(vals) if vals else None
            record[f"{field}_median"] = statistics.median(vals) if vals else None
        tables["condition_summary"].append(record)

    strata_fields = ("category", "l2_category", "cycle_category", "question_type", "language", "answer_format")
    for field in strata_fields:
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            groups[(row["track"], str(row[field]), row["condition"])].append(row)
        for (track, value, condition), group in groups.items():
            rec = {"stratum_type": field, "track": track, "stratum": value, "condition": condition, "n": len(group)}
            for source in ("native", "extraction"):
                rec[f"{source}_correct"] = sum(row[f"{source}_correct"] for row in group)
                rec[f"{source}_accuracy"] = rec[f"{source}_correct"] / len(group)
            tables["strata_summary"].append(rec)

    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        rec = {"condition": condition, "n": len(group)}
        rec["micro_native_accuracy"] = statistics.mean(row["native_correct"] for row in group)
        rec["micro_extraction_accuracy"] = statistics.mean(row["extraction_correct"] for row in group)
        per_track = [x for x in tables["condition_summary"] if x["condition"] == condition]
        rec["macro_track_native_accuracy"] = statistics.mean(x["native_accuracy"] for x in per_track)
        rec["macro_track_extraction_accuracy"] = statistics.mean(x["extraction_accuracy"] for x in per_track)
        rec["native_extraction_agreement"] = statistics.mean(row["native_extraction_agree"] for row in group)
        rec["native_judge_exact_binary_rate"] = statistics.mean(row["native_judge_exact_binary"] for row in group)
        rec["mean_response_chars"] = statistics.mean(row["response_chars"] for row in group)
        rec["median_response_chars"] = statistics.median(row["response_chars"] for row in group)
        for field in ("prompt_tokens", "completion_tokens", "total_api_tokens"):
            vals = [row[field] for row in group if isinstance(row[field], (int, float))]
            rec[f"{field}_coverage"] = len(vals) / len(group)
            rec[f"{field}_mean"] = statistics.mean(vals) if vals else None
            rec[f"{field}_median"] = statistics.median(vals) if vals else None
        tables["condition_summary"].append(rec)

    score_sources = ("native", "extraction")
    scopes: list[tuple[str, str | None, list[dict]]] = [("micro_all", None, rows)]
    for track in TRACKS:
        scopes.append(("track", track, [row for row in rows if row["track"] == track]))
    comparisons = list(itertools.combinations(CONDITIONS, 2))
    for source in score_sources:
        for scope_name, scope_value, subset in scopes:
            pvals, pending_records = [], []
            for comp_idx, (a, b) in enumerate(comparisons):
                # Rows are in long format; build condition-aligned pairs by stable sample key.
                condition_maps = {cond: {(r["track"], r["sample_uid"]): r for r in subset if r["condition"] == cond}
                                  for cond in (a, b)}
                common = sorted(set(condition_maps[a]) & set(condition_maps[b]))
                better = sum(condition_maps[b][key][f"{source}_correct"] > condition_maps[a][key][f"{source}_correct"] for key in common)
                worse = sum(condition_maps[b][key][f"{source}_correct"] < condition_maps[a][key][f"{source}_correct"] for key in common)
                diff = statistics.mean(condition_maps[b][key][f"{source}_correct"] - condition_maps[a][key][f"{source}_correct"] for key in common) if common else 0.0
                ci_low, ci_high = cluster_bootstrap_delta(subset, a, b, source, seed + comp_idx + (0 if scope_name == "micro_all" else TRACKS.index(scope_value) + 1) * 100)
                pval = exact_mcnemar_p(better, worse)
                pending_records.append({
                    "score_source": source, "scope": scope_name, "stratum": scope_value or "all_tracks",
                    "condition_a": a, "condition_b": b, "n_paired": len(common),
                    "b_correct_a_wrong": better, "a_correct_b_wrong": worse,
                    "paired_delta_b_minus_a": diff, "paired_delta_percentage_points": 100 * diff,
                    "cluster_bootstrap_95ci_low_pp": 100 * ci_low, "cluster_bootstrap_95ci_high_pp": 100 * ci_high,
                    "mcnemar_exact_p": pval,
                })
                pvals.append(pval)
            for rec, adj in zip(pending_records, holm_adjust(pvals)):
                rec["mcnemar_holm_p"] = adj
                tables["paired_comparisons"].append(rec)

    by_source: dict[tuple[str, str, int, int], int] = Counter()
    for row in rows:
        by_source[(row["condition"], row["track"], row["native_correct"], row["extraction_correct"])] += 1
        tables["judge_source"].append({"condition": row["condition"], "track": row["track"], "judge_source": row["native_judge_source"]})
    tables["score_disagreement"] = [
        {"condition": c, "track": t, "native_correct": n, "extraction_correct": e, "n": count}
        for (c, t, n, e), count in sorted(by_source.items())
    ]
    tables["judge_source"] = [
        {"condition": c, "track": t, "judge_source": source, "n": count}
        for (c, t, source), count in sorted(Counter((r["condition"], r["track"], r["native_judge_source"]) for r in rows).items())
    ]
    tables["native_judge_value_audit"] = [
        {"condition": c, "track": t, "judge_value": value, "n": count}
        for (c, t, value), count in sorted(Counter(
            (r["condition"], r["track"], str(r["native_judge_raw"])) for r in rows
        ).items())
    ]
    tables["extraction_status"] = [
        {"condition": c, "track": t, "status": status, "reason_code": reason, "n": count}
        for (c, t, status, reason), count in sorted(Counter(
            (r["condition"], r["track"], str(r["extraction_status"]), str(r["extraction_reason"])) for r in rows
        ).items())
    ]
    association_scopes = [("micro_all", None, rows)]
    association_scopes.extend(("track", track, [row for row in rows if row["track"] == track]) for track in TRACKS)
    for scope_name, scope_value, scope_rows in association_scopes:
        for condition in CONDITIONS:
            group = [row for row in scope_rows if row["condition"] == condition]
            for metric in ("response_chars", "response_words", "completion_tokens", "total_api_tokens"):
                for source in ("native", "extraction"):
                    selected = [row for row in group if isinstance(row[metric], (int, float))]
                    tables["length_accuracy_association"].append({
                        "scope": scope_name, "stratum": scope_value or "all_tracks", "condition": condition,
                        "length_metric": metric, "score_source": source, "n": len(selected),
                        "spearman_rho": spearman_rho(
                            [row[metric] for row in selected],
                            [row[f"{source}_correct"] for row in selected],
                        ),
                    })
    for source in ("native", "extraction"):
        for metric in ("response_chars", "response_words", "completion_tokens"):
            groups = defaultdict(list)
            for row in rows:
                if row[metric] is not None:
                    groups[(row["condition"], row["track"], row[f"{source}_correct"])].append(row[metric])
            for (condition, track, correct), vals in sorted(groups.items()):
                tables["response_length_by_score"].append({
                    "score_source": source, "metric": metric, "condition": condition,
                    "track": track, "correct": correct, "n": len(vals),
                    "mean": statistics.mean(vals), "median": statistics.median(vals),
                    "p90": sorted(vals)[int(0.9 * (len(vals) - 1))],
                })
    return tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-version", default="full-statistics-v1")
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    output = args.data_root / "derived" / args.output-version
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite derived analysis: {output}")
    staging = output.parent / f".building-{args.output_version}-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete prior analysis staging area: {staging}")
    rows, audit = collect(args.data_root)
    tables = summarize(rows, args.seed)
    staging.mkdir(parents=True, exist_ok=False)
    write_csv(staging / "sample_level_merged.csv", rows)
    for name, values in tables.items():
        write_csv(staging / f"{name}.csv", values)
    manifest = {
        "schema": "phase2-work5-statistics-v1", "input_verification": CAMPAIGN,
        "tracks": list(TRACKS), "conditions": list(CONDITIONS),
        "prompt_registry_sha256": audit["prompt_registry_sha256"],
        "joined_records": len(rows), "seed": args.seed, "bootstrap_iterations": N_BOOTSTRAP,
        "paired_comparison_method": "two-sided exact McNemar; Holm-adjusted within each score source and scope",
        "confidence_intervals": "paired cluster bootstrap by image; 95 percent percentile interval",
        "score_sources": ["unmodified Vision-OPD native judge", "Qwen structured text-only answer extraction"],
        "audit": audit,
        "interpretation": "No pooled score should replace benchmark-stratified results; report micro and equal-track macro summaries separately.",
    }
    (staging / "analysis-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.rename(output)
    print(f"PASS: analyzed {len(rows)} joined condition-sample records; wrote {len(tables)} aggregate tables to {output}")


if __name__ == "__main__":
    main()
