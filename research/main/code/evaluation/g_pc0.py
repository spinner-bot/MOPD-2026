"""Contracts and frozen decision rule for the G-PC0 mechanism control.

G-PC0 is deliberately separate from the natural-image 8/32/96 manifests.  It
uses the same four-condition runner and result integrity checks, but its gold
labels come from a deterministic renderer rather than a human semantic review.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATA_PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "data_pipeline"
sys.path.insert(0, str(DATA_PIPELINE_ROOT))

from reasoning_graph_contract import validate_guide_graph_alignment, validate_reasoning_graph


CHOICES = ("A", "B", "C", "D")
FAMILIES = ("directed_path", "sequential_transform", "grid_trace", "set_relation")
PROFILE = "g_pc0_v1"
SOURCE_DATASET = "MOPD-G-PC0-v1"
GENERATOR_ID = "mopd-g-pc0-v1"


def _text(row: dict, key: str) -> str:
    return str(row.get(key, "")).strip()


def validate_g_pc0_manifest(samples: list[dict], expected_n: int = 16) -> list[str]:
    """Validate the frozen procedural manifest without claiming human review."""

    errors: list[str] = []
    if expected_n != 16:
        errors.append("G-PC0 v1 requires exactly 16 samples")
    if len(samples) != expected_n:
        errors.append(f"expected {expected_n} samples, found {len(samples)}")

    sample_ids = [_text(row, "sample_id") for row in samples]
    image_ids = [_text(row, "source_image_id") for row in samples]
    cluster_ids = [_text(row, "source_cluster_id") for row in samples]
    record_ids = [_text(row, "source_record_id") for row in samples]
    for name, values in (
        ("sample_id", sample_ids),
        ("source_image_id", image_ids),
        ("source_cluster_id", cluster_ids),
        ("source_record_id", record_ids),
    ):
        if any(not value for value in values):
            errors.append(f"missing {name}")
        if len(set(values)) != len(values):
            errors.append(f"{name} values must be unique")

    labels = Counter(row.get("final_answer") for row in samples)
    if labels != Counter({choice: 4 for choice in CHOICES}):
        errors.append("answer labels must be balanced at 4 per choice")
    families = Counter(row.get("positive_control_family") for row in samples)
    if families != Counter({family: 4 for family in FAMILIES}):
        errors.append("positive-control families must contain exactly four samples each")
    spec_hashes = {_text(row, "generation_spec_hash") for row in samples}
    if len(spec_hashes) != 1 or any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in spec_hashes
    ):
        errors.append("generation_spec_hash must be one shared lowercase SHA-256")

    required_text = (
        "source_dataset",
        "source_record_id",
        "source_image_id",
        "source_cluster_id",
        "image_original",
        "image_checksum",
        "final_question",
        "option_A",
        "option_B",
        "option_C",
        "option_D",
        "canonical_answer_text",
        "generic_context",
        "generator_id",
        "generation_spec_hash",
        "positive_control_family",
    )
    for index, row in enumerate(samples):
        row_id = sample_ids[index] or f"row-{index}"
        expected_id = f"gpc0-{index + 1:02d}"
        expected_family = FAMILIES[index // 4]
        expected_label = CHOICES[index % 4]
        if row_id != expected_id:
            errors.append(f"row-{index}: sample_id must be {expected_id}")
        if row.get("source_record_id") != expected_id:
            errors.append(f"{row_id}: source_record_id must equal sample_id")
        if row.get("positive_control_family") != expected_family:
            errors.append(f"{row_id}: positive_control_family must be {expected_family}")
        if row.get("final_answer") != expected_label:
            errors.append(f"{row_id}: final_answer must follow frozen A/B/C/D schedule")
        missing = [key for key in required_text if not _text(row, key)]
        if missing:
            errors.append(f"{row_id}: missing fields {','.join(missing)}")
        if row.get("manifest_profile") != PROFILE:
            errors.append(f"{row_id}: manifest_profile must be {PROFILE}")
        if row.get("experiment_role") != "mechanism_positive_control":
            errors.append(f"{row_id}: experiment_role must be mechanism_positive_control")
        if row.get("source_dataset") != SOURCE_DATASET:
            errors.append(f"{row_id}: source_dataset must be {SOURCE_DATASET}")
        if row.get("generator_id") != GENERATOR_ID:
            errors.append(f"{row_id}: generator_id must be {GENERATOR_ID}")
        if row.get("generator_seed") != 20260914:
            errors.append(f"{row_id}: generator_seed must be 20260914")
        if row.get("procedural_gold_verified") is not True:
            errors.append(f"{row_id}: procedural_gold_verified must be true")
        if row.get("human_verified") is not False:
            errors.append(f"{row_id}: human_verified must be false")
        if _text(row, "human_reviewer") or _text(row, "human_reviewed_at"):
            errors.append(f"{row_id}: human provenance must remain empty")
        if row.get("knowledge_dependency_label") != "image_only":
            errors.append(f"{row_id}: knowledge_dependency_label must be image_only")
        if row.get("task_family") != "R":
            errors.append(f"{row_id}: task_family must be R")
        if row.get("full_image_reference_valid") is not True:
            errors.append(f"{row_id}: full_image_reference_valid must be true")
        if row.get("unique_answer") is not True:
            errors.append(f"{row_id}: unique_answer must be true")
        if row.get("answer_leakage") not in {0, False}:
            errors.append(f"{row_id}: answer_leakage must be zero")
        if row.get("external_fact_injected") not in {0, False}:
            errors.append(f"{row_id}: external_fact_injected must be zero")

        depth = row.get("reasoning_depth")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 3:
            errors.append(f"{row_id}: reasoning_depth must be an integer >= 3")
        evidence = row.get("required_evidence")
        evidence_valid = (
            isinstance(evidence, list)
            and len(evidence) >= 3
            and all(isinstance(value, str) and value.strip() for value in evidence)
            and len({value.strip().casefold() for value in evidence}) == len(evidence)
        )
        if not evidence_valid:
            errors.append(f"{row_id}: required_evidence must contain at least three distinct entries")
        if row.get("difficult_visual_evidence") != []:
            errors.append(f"{row_id}: difficult_visual_evidence must be empty for synthetic R")

        graph = row.get("minimal_reasoning_graph")
        graph_errors, _ = validate_reasoning_graph(
            graph, declared_depth=depth, require_two_evidence=True
        )
        errors.extend(f"{row_id}: minimal_reasoning_graph {error}" for error in graph_errors)
        if evidence_valid and isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
            graph_evidence = {
                _text(node, "claim").casefold()
                for node in graph["nodes"]
                if isinstance(node, dict) and node.get("type") == "evidence"
            }
            if graph_evidence != {value.strip().casefold() for value in evidence}:
                errors.append(f"{row_id}: required_evidence must match graph evidence")

        guides = row.get("guides")
        if not isinstance(guides, list) or not 1 <= len(guides) <= 4:
            errors.append(f"{row_id}: guides must contain 1..4 items")
        else:
            canonical = _text(row, "canonical_answer_text").casefold()
            for guide_index, guide in enumerate(guides):
                if not isinstance(guide, dict):
                    errors.append(f"{row_id}: guide {guide_index} must be an object")
                    continue
                if not _text(guide, "question") or not _text(guide, "answer"):
                    errors.append(f"{row_id}: guide {guide_index} requires question and answer")
                if canonical and _text(guide, "answer").casefold() == canonical:
                    errors.append(f"{row_id}: guide {guide_index} reveals canonical answer")
            errors.extend(
                f"{row_id}: {error}" for error in validate_guide_graph_alignment(guides, graph)
            )

        options = [_text(row, f"option_{choice}") for choice in CHOICES]
        if any(not option for option in options) or len({option.casefold() for option in options}) != 4:
            errors.append(f"{row_id}: options must be four distinct non-empty strings")
        correct = row.get("final_answer")
        if correct in CHOICES and _text(row, f"option_{correct}") != _text(row, "canonical_answer_text"):
            errors.append(f"{row_id}: correct option must equal canonical_answer_text")
        trace = row.get("solution_trace")
        if not isinstance(trace, list) or len(trace) < 4 or any(
            not isinstance(value, str) or not value.strip() for value in trace
        ):
            errors.append(f"{row_id}: solution_trace must contain at least four states")
        elif trace[-1].strip() != _text(row, "canonical_answer_text"):
            errors.append(f"{row_id}: solution_trace must end at canonical_answer_text")
    return sorted(set(errors))


def validate_g_pc0_assets(samples: list[dict], image_root: Path) -> list[str]:
    """Bind every procedural record to an in-root, checksum-matched image."""

    errors: list[str] = []
    root = image_root.resolve(strict=True)
    for row in samples:
        row_id = _text(row, "sample_id") or "<missing>"
        try:
            path = (root / _text(row, "image_original")).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("path escapes image root")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != _text(row, "image_checksum"):
                errors.append(f"{row_id}: image checksum differs from manifest")
        except (OSError, ValueError) as error:
            errors.append(f"{row_id}: image asset invalid: {error}")
    return sorted(set(errors))


def decide_g_pc0(records: list[dict], validation_errors: list[str] | None = None) -> dict[str, Any]:
    """Apply the result-blind 1035 G-PC0 decision tree to validated records."""

    if validation_errors:
        return {"decision": "INVALID", "validation_errors": sorted(set(validation_errors))}
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in records:
        by_sample[str(row.get("sample_id", ""))][str(row.get("condition", ""))] = row
    if len(by_sample) != 16 or any(set(rows) != {"S", "T_Q", "T_QA", "T_generic"} for rows in by_sample.values()):
        return {"decision": "INVALID", "validation_errors": ["requires 16 complete four-condition samples"]}

    ordered = sorted(by_sample)
    baseline = [bool(by_sample[sample_id]["S"].get("correct")) for sample_id in ordered]
    tqa = [bool(by_sample[sample_id]["T_QA"].get("correct")) for sample_id in ordered]
    gains = sum(not before and after for before, after in zip(baseline, tqa))
    regressions = sum(before and not after for before, after in zip(baseline, tqa))
    baseline_correct = sum(baseline)
    if baseline_correct >= 14:
        decision = "CEILING_INCONCLUSIVE"
    elif gains >= 4 and gains - regressions >= 3:
        decision = "MECHANISM_PASS"
    else:
        decision = "INGESTION_OR_RECIPE_FAILURE"

    condition_correct = {
        condition: sum(bool(by_sample[sample_id][condition].get("correct")) for sample_id in ordered)
        for condition in ("S", "T_Q", "T_QA", "T_generic")
    }
    per_family: dict[str, dict[str, int]] = {}
    for family in FAMILIES:
        family_ids = [
            sample_id
            for sample_id in ordered
            if by_sample[sample_id]["S"].get("positive_control_family") == family
        ]
        per_family[family] = {
            condition: sum(
                bool(by_sample[sample_id][condition].get("correct")) for sample_id in family_ids
            )
            for condition in ("S", "T_Q", "T_QA", "T_generic")
        }
        per_family[family]["n"] = len(family_ids)
    return {
        "decision": decision,
        "n_samples": 16,
        "condition_correct": condition_correct,
        "tqa_vs_s": {"gains": gains, "regressions": regressions, "net_flips": gains - regressions},
        "per_family_descriptive_only": per_family,
        "inferential_statistics_used": False,
        "natural_image_claim_authorized": False,
        "training_data_authorized": False,
    }


def diagnose_g_pc0(records: list[dict], decision: str) -> dict[str, Any]:
    """Describe a gated G-PC0 run without replacing its frozen decision rule."""

    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in records:
        by_sample[str(row["sample_id"])][str(row["condition"])] = row
    conditions = ("S", "T_Q", "T_QA", "T_generic")
    ordered = sorted(by_sample)
    metrics: dict[str, dict[str, Any]] = {}
    argmax_mismatches: dict[str, list[str]] = {}
    for condition in conditions:
        condition_rows = [by_sample[sample_id][condition] for sample_id in ordered]
        prediction_counts = Counter(str(row["parsed_answer"]) for row in condition_rows)
        gold_counts = Counter(str(row["correct_choice"]) for row in condition_rows)
        correct_by_gold = {
            choice: sum(
                bool(row["correct"]) for row in condition_rows if row["correct_choice"] == choice
            )
            for choice in CHOICES
        }
        mismatches = []
        for row in condition_rows:
            scores = row["choice_log_likelihoods"]
            score_argmax = max(CHOICES, key=lambda choice: (scores[choice], -CHOICES.index(choice)))
            if score_argmax != row["parsed_answer"]:
                mismatches.append(str(row["sample_id"]))
        argmax_mismatches[condition] = mismatches
        metrics[condition] = {
            "correct": sum(bool(row["correct"]) for row in condition_rows),
            "prediction_label_counts": dict(sorted(prediction_counts.items())),
            "gold_label_counts": dict(sorted(gold_counts.items())),
            "correct_by_gold_label": correct_by_gold,
            "max_prediction_label_share": max(prediction_counts.values()) / len(condition_rows),
            "mean_correct_full_vocab_log_prob": sum(float(row["correct_log_prob"]) for row in condition_rows)
            / len(condition_rows),
            "mean_correct_four_choice_probability": sum(
                float(row["choice_probabilities"][row["correct_choice"]]) for row in condition_rows
            )
            / len(condition_rows),
            "mean_margin": sum(float(row["margin"]) for row in condition_rows) / len(condition_rows),
            "mean_entropy": sum(float(row["entropy"]) for row in condition_rows) / len(condition_rows),
            "mean_choice_probability": {
                choice: sum(float(row["choice_probabilities"][choice]) for row in condition_rows)
                / len(condition_rows)
                for choice in CHOICES
            },
            "input_tokens": {
                "min": min(int(row["input_tokens"]) for row in condition_rows),
                "mean": sum(int(row["input_tokens"]) for row in condition_rows) / len(condition_rows),
                "max": max(int(row["input_tokens"]) for row in condition_rows),
            },
        }

    paired: dict[str, dict[str, Any]] = {}
    for condition in ("T_Q", "T_QA", "T_generic"):
        gains = [sample_id for sample_id in ordered if not by_sample[sample_id]["S"]["correct"] and by_sample[sample_id][condition]["correct"]]
        regressions = [sample_id for sample_id in ordered if by_sample[sample_id]["S"]["correct"] and not by_sample[sample_id][condition]["correct"]]
        paired[condition] = {
            "gains": gains,
            "regressions": regressions,
            "net_flips": len(gains) - len(regressions),
            "mean_correct_full_vocab_log_prob_delta": sum(
                float(by_sample[sample_id][condition]["correct_log_prob"])
                - float(by_sample[sample_id]["S"]["correct_log_prob"])
                for sample_id in ordered
            )
            / len(ordered),
            "mean_correct_four_choice_probability_delta": sum(
                float(by_sample[sample_id][condition]["choice_probabilities"][by_sample[sample_id][condition]["correct_choice"]])
                - float(by_sample[sample_id]["S"]["choice_probabilities"][by_sample[sample_id]["S"]["correct_choice"]])
                for sample_id in ordered
            )
            / len(ordered),
        }
    collapse = {
        condition: metrics[condition]["max_prediction_label_share"] >= 0.75
        for condition in conditions
    }
    return {
        "analysis_status": "posthoc_descriptive",
        "frozen_decision": decision,
        "decision_changed": False,
        "n_samples": len(ordered),
        "condition_metrics": metrics,
        "paired_vs_S": paired,
        "generation_vs_score_argmax_mismatches": argmax_mismatches,
        "output_label_collapse_threshold": 0.75,
        "output_label_collapse": collapse,
        "natural_image_claim_authorized": False,
        "training_data_authorized": False,
    }


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as sink:
        sink.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
