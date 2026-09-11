"""Fail-closed validation and decision logic for protocol 1016.

This module intentionally contains no model inference.  It freezes the CPU-side
contracts that must pass before a locked pilot can be interpreted.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


CONDITIONS = ("S", "T_Q", "T_QA", "T_generic")
CHOICES = ("A", "B", "C", "D")


def validate_locked_manifest(samples: list[dict], expected_n: int = 96) -> list[str]:
    """Validate content-independent invariants of the locked pilot manifest."""

    errors: list[str] = []
    if len(samples) != expected_n:
        errors.append(f"expected {expected_n} samples, found {len(samples)}")
    sample_ids = [str(row.get("sample_id", "")).strip() for row in samples]
    source_ids = [str(row.get("source_image_id", "")).strip() for row in samples]
    if any(not value for value in sample_ids):
        errors.append("missing sample_id")
    if any(not value for value in source_ids):
        errors.append("missing source_image_id")
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("sample_id values must be unique")
    if len(set(source_ids)) != len(source_ids):
        errors.append("source_image_id values must be unique")

    labels = Counter(row.get("final_answer") for row in samples)
    if expected_n % len(CHOICES) == 0:
        target = expected_n // len(CHOICES)
        if labels != Counter({choice: target for choice in CHOICES}):
            errors.append(f"answer labels must be balanced at {target} per choice")
    elif set(labels) - set(CHOICES):
        errors.append("final_answer must be A, B, C, or D")

    required_text = (
        "source_dataset",
        "source_record_id",
        "image_original",
        "image_checksum",
        "final_question",
        "option_A",
        "option_B",
        "option_C",
        "option_D",
        "generic_context",
    )
    for index, row in enumerate(samples):
        row_id = sample_ids[index] or f"row-{index}"
        missing = [key for key in required_text if not str(row.get(key, "")).strip()]
        if missing:
            errors.append(f"{row_id}: missing fields {','.join(missing)}")
        if row.get("knowledge_dependency_label") != "image_only":
            errors.append(f"{row_id}: knowledge_dependency_label must be image_only")
        if row.get("task_family") not in {"R", "PR"}:
            errors.append(f"{row_id}: task_family must be R or PR")
        if row.get("full_image_reference_valid") is not True:
            errors.append(f"{row_id}: full_image_reference_valid must be true")
        if row.get("human_verified") is not True:
            errors.append(f"{row_id}: human_verified must be true")
        if row.get("answer_leakage") not in {0, False}:
            errors.append(f"{row_id}: answer_leakage must be zero")
        evidence = row.get("required_evidence")
        if not isinstance(evidence, list) or len(evidence) < 2 or any(not str(v).strip() for v in evidence):
            errors.append(f"{row_id}: required_evidence must contain at least two entries")
        graph = row.get("minimal_reasoning_graph")
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not graph["nodes"]:
            errors.append(f"{row_id}: minimal_reasoning_graph.nodes is required")
        guides = row.get("guides")
        if not isinstance(guides, list) or not 1 <= len(guides) <= 4:
            errors.append(f"{row_id}: guides must contain 1..4 items")
        elif any(
            not isinstance(guide, dict)
            or not str(guide.get("question", "")).strip()
            or not str(guide.get("answer", "")).strip()
            for guide in guides
        ):
            errors.append(f"{row_id}: every guide requires question and answer")
        options = [str(row.get(f"option_{choice}", "")).strip().casefold() for choice in CHOICES]
        if len(set(options)) != len(options):
            errors.append(f"{row_id}: options must be distinct")
    return errors


def validate_partition_isolation(partitions: dict[str, list[dict]]) -> list[str]:
    """Reject source-image or known near-duplicate-cluster partition overlap."""

    errors = []
    owners: dict[str, str] = {}
    cluster_owners: dict[str, str] = {}
    for name in sorted(partitions):
        for row in partitions[name]:
            source_id = str(row.get("source_image_id", "")).strip()
            if not source_id:
                errors.append(f"{name}: missing source_image_id")
            elif source_id in owners and owners[source_id] != name:
                errors.append(f"source_image_id {source_id} overlaps {owners[source_id]} and {name}")
            else:
                owners[source_id] = name
            cluster_id = str(row.get("source_cluster_id", "")).strip()
            if cluster_id:
                if cluster_id in cluster_owners and cluster_owners[cluster_id] != name:
                    errors.append(
                        f"source_cluster_id {cluster_id} overlaps {cluster_owners[cluster_id]} and {name}"
                    )
                else:
                    cluster_owners[cluster_id] = name
    return errors


def validate_result_records(
    records: list[dict],
    expected_samples: int = 96,
    min_parse_rate: float = 0.99,
) -> list[str]:
    """Apply the protocol's completeness and provenance gate."""

    errors: list[str] = []
    expected_total = expected_samples * len(CONDITIONS)
    if len(records) != expected_total:
        errors.append(f"expected {expected_total} result records, found {len(records)}")
    keys = [(row.get("sample_id"), row.get("condition")) for row in records]
    if len(set(keys)) != len(keys):
        errors.append("sample_id × condition keys must be unique")
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_sample[str(row.get("sample_id", ""))].append(row)
    if len(by_sample) != expected_samples:
        errors.append(f"expected {expected_samples} unique sample ids, found {len(by_sample)}")

    parsed = 0
    global_fields = (
        "manifest_hash",
        "model_revision",
        "processor_revision",
        "code_commit",
        "shell_hash",
    )
    global_values = {field: set() for field in global_fields}
    for index, row in enumerate(records):
        key = f"{row.get('sample_id', '<missing>')}/{row.get('condition', '<missing>')}"
        if row.get("condition") not in CONDITIONS:
            errors.append(f"{key}: unsupported condition")
        if row.get("error") not in {None, ""}:
            errors.append(f"{key}: error is present")
        if row.get("truncated") is not False or row.get("eos_reached") is not True:
            errors.append(f"{key}: output is truncated or missing EOS")
        if row.get("parsed_answer") in CHOICES:
            parsed += 1
        if row.get("correct_choice") not in CHOICES:
            errors.append(f"{key}: correct_choice must be A, B, C, or D")
        if not isinstance(row.get("correct"), bool):
            errors.append(f"{key}: correct must be boolean")
        elif row.get("parsed_answer") in CHOICES and row["correct"] != (row["parsed_answer"] == row.get("correct_choice")):
            errors.append(f"{key}: correct disagrees with parsed_answer and correct_choice")
        if not isinstance(row.get("correct_log_prob"), (int, float)) or not math.isfinite(row["correct_log_prob"]):
            errors.append(f"{key}: correct_log_prob must be finite")
        scores = row.get("choice_log_likelihoods")
        probabilities = row.get("choice_probabilities")
        if not isinstance(scores, dict) or set(scores) != set(CHOICES):
            errors.append(f"{key}: four choice log-likelihoods are required")
        elif not all(isinstance(v, (int, float)) and math.isfinite(v) for v in scores.values()):
            errors.append(f"{key}: choice log-likelihoods must be finite")
        elif row.get("correct_choice") in CHOICES and isinstance(row.get("correct_log_prob"), (int, float)) and not math.isclose(
            float(row["correct_log_prob"]), float(scores[row["correct_choice"]]), rel_tol=0, abs_tol=1e-8
        ):
            errors.append(f"{key}: correct_log_prob disagrees with choice score")
        if not isinstance(probabilities, dict) or set(probabilities) != set(CHOICES):
            errors.append(f"{key}: four choice probabilities are required")
        elif not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in probabilities.values()):
            errors.append(f"{key}: choice probabilities must be finite and nonnegative")
        elif not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-6):
            errors.append(f"{key}: choice probabilities must sum to one")
        for field in global_fields:
            value = str(row.get(field, "")).strip()
            if not value:
                errors.append(f"{key}: missing {field}")
            global_values[field].add(value)

    if records and parsed / len(records) < min_parse_rate:
        errors.append(f"parse rate {parsed / len(records):.6f} is below {min_parse_rate:.6f}")
    for field, values in global_values.items():
        if len(values) != 1:
            errors.append(f"{field} must be identical across all records")

    invariant_fields = ("image_checksum", "question_hash", "options_hash")
    for sample_id, rows in by_sample.items():
        if {row.get("condition") for row in rows} != set(CONDITIONS):
            errors.append(f"{sample_id}: all four conditions are required")
        for field in invariant_fields:
            values = {str(row.get(field, "")).strip() for row in rows}
            if "" in values or len(values) != 1:
                errors.append(f"{sample_id}: {field} must be present and invariant")
        prompt_hashes = [str(row.get("prompt_hash", "")).strip() for row in rows]
        if any(not value for value in prompt_hashes):
            errors.append(f"{sample_id}: prompt_hash is required")
    return errors


def exact_mcnemar_one_sided(baseline_correct: list[bool], treatment_correct: list[bool]) -> dict:
    """Exact one-sided McNemar test for treatment gains over baseline."""

    if len(baseline_correct) != len(treatment_correct) or not baseline_correct:
        raise ValueError("paired non-empty arrays of equal length are required")
    regressions = sum(bool(s) and not bool(t) for s, t in zip(baseline_correct, treatment_correct))
    gains = sum(not bool(s) and bool(t) for s, t in zip(baseline_correct, treatment_correct))
    discordant = regressions + gains
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = sum(math.comb(discordant, k) for k in range(gains, discordant + 1)) / (2**discordant)
    return {
        "gains": gains,
        "regressions": regressions,
        "discordant": discordant,
        "p_value": p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values keyed like the input."""

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p_value) in enumerate(ordered):
        running = max(running, (count - rank) * p_value)
        adjusted[key] = min(1.0, running)
    return adjusted


def paired_cluster_bootstrap(
    baseline: list[float],
    treatment: list[float],
    cluster_ids: list[str],
    n_bootstrap: int = 10_000,
    seed: int = 20260911,
) -> dict:
    """Bootstrap paired continuous differences by source-image cluster."""

    if not baseline or len(baseline) != len(treatment) or len(baseline) != len(cluster_ids):
        raise ValueError("paired non-empty values and cluster ids of equal length are required")
    grouped: dict[str, list[float]] = defaultdict(list)
    for base, treated, cluster in zip(baseline, treatment, cluster_ids):
        difference = float(treated) - float(base)
        if not math.isfinite(difference):
            raise ValueError("bootstrap differences must be finite")
        grouped[str(cluster)].append(difference)
    clusters = sorted(grouped)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [value for cluster in sampled for value in grouped[cluster]]
        draws.append(sum(values) / len(values))
    draws.sort()

    def percentile(probability: float) -> float:
        position = (len(draws) - 1) * probability
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return draws[lower]
        weight = position - lower
        return draws[lower] * (1 - weight) + draws[upper] * weight

    observed = sum(float(t) - float(b) for b, t in zip(baseline, treatment)) / len(baseline)
    return {
        "mean_diff": observed,
        "ci_lower": percentile(0.025),
        "ci_upper": percentile(0.975),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def _condition_vectors(records: list[dict]) -> tuple[list[str], dict[str, dict[str, dict]]]:
    by_condition: dict[str, dict[str, dict]] = {condition: {} for condition in CONDITIONS}
    for row in records:
        condition = row.get("condition")
        if condition in by_condition:
            by_condition[condition][str(row.get("sample_id"))] = row
    ids = sorted(by_condition["S"])
    if not ids or any(sorted(by_condition[condition]) != ids for condition in CONDITIONS):
        raise ValueError("all conditions must contain the same non-empty sample ids")
    return ids, by_condition


def decide_teacher_gap(
    records: list[dict],
    *,
    leakage_free: bool,
    expected_samples: int = 96,
    n_bootstrap: int = 10_000,
    seed: int = 20260911,
) -> dict:
    """Apply the pre-registered 1016 Go/Revise/Stop decision tree."""

    validation_errors = validate_result_records(records, expected_samples=expected_samples)
    if validation_errors:
        return {"decision": "INVALID", "validation_errors": validation_errors}
    ids, rows = _condition_vectors(records)
    source_clusters = [str(rows["S"][sample_id]["source_image_id"]) for sample_id in ids]
    baseline_correct = [bool(rows["S"][sample_id]["correct"]) for sample_id in ids]
    baseline_logp = [float(rows["S"][sample_id]["correct_log_prob"]) for sample_id in ids]
    generic_correct = [bool(rows["T_generic"][sample_id]["correct"]) for sample_id in ids]
    generic_logp = [float(rows["T_generic"][sample_id]["correct_log_prob"]) for sample_id in ids]

    comparisons = {}
    raw_p = {}
    for condition in ("T_Q", "T_QA"):
        treatment_correct = [bool(rows[condition][sample_id]["correct"]) for sample_id in ids]
        treatment_logp = [float(rows[condition][sample_id]["correct_log_prob"]) for sample_id in ids]
        mcnemar = exact_mcnemar_one_sided(baseline_correct, treatment_correct)
        bootstrap = paired_cluster_bootstrap(
            baseline_logp,
            treatment_logp,
            source_clusters,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        raw_p[condition] = mcnemar["p_value"]
        comparisons[condition] = {
            "accuracy_delta_vs_S": sum(treatment_correct) / len(ids) - sum(baseline_correct) / len(ids),
            "correct_log_prob_vs_S": bootstrap,
            "mcnemar_vs_S": mcnemar,
            "accuracy_delta_vs_generic": sum(treatment_correct) / len(ids) - sum(generic_correct) / len(ids),
            "correct_log_prob_delta_vs_generic": sum(treatment_logp[i] - generic_logp[i] for i in range(len(ids))) / len(ids),
        }
    adjusted = holm_adjust(raw_p)
    for condition in comparisons:
        comparisons[condition]["mcnemar_vs_S"]["holm_p_value"] = adjusted[condition]

    def strong_go(condition: str) -> bool:
        result = comparisons[condition]
        structured_advantage = (
            result["accuracy_delta_vs_generic"] >= 0.04
            or result["correct_log_prob_delta_vs_generic"] >= 0.10
        )
        return (
            leakage_free
            and result["accuracy_delta_vs_S"] >= 0.08
            and result["mcnemar_vs_S"]["holm_p_value"] < 0.05
            and result["correct_log_prob_vs_S"]["mean_diff"] >= 0.15
            and result["correct_log_prob_vs_S"]["ci_lower"] > 0
            and structured_advantage
            and result["mcnemar_vs_S"]["gains"] > result["mcnemar_vs_S"]["regressions"]
        )

    if strong_go("T_Q"):
        decision = "GO_T_Q"
    elif strong_go("T_QA"):
        decision = "GO_T_QA"
    else:
        oracle = comparisons["T_QA"]
        stop = (
            oracle["accuracy_delta_vs_S"] < 0.03
            and (
                oracle["correct_log_prob_vs_S"]["mean_diff"] < 0.05
                or oracle["correct_log_prob_vs_S"]["ci_lower"] <= 0
            )
            and oracle["mcnemar_vs_S"]["gains"] <= oracle["mcnemar_vs_S"]["regressions"]
        )
        decision = "STOP_CURRENT_RECIPE" if stop else "REVISE"
    return {
        "decision": decision,
        "leakage_free": leakage_free,
        "n_samples": len(ids),
        "comparisons": comparisons,
    }
