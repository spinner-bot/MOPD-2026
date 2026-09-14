"""Contracts and frozen decision rule for the G-PC0 mechanism control.

G-PC0 is deliberately separate from the natural-image 8/32/96 manifests.  It
uses the same four-condition runner and result integrity checks, but its gold
labels come from a deterministic renderer rather than a human semantic review.
"""

from __future__ import annotations

import hashlib
import json
import math
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
ROTATION_PROFILE = "g_pc0_label_rotation_diagnostic_v1"
ROTATION_ROLE = "posthoc_label_rotation_diagnostic"
BASE_MANIFEST_SHA256 = "10b09d122a5781df564a63cac8449d368a2d785c3fb06ba15c3887db2433bb1c"


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


def label_rotation_mapping(rotation: int) -> dict[str, str]:
    """Map displayed labels to frozen semantic option IDs for one cyclic rotation."""

    if rotation not in range(4):
        raise ValueError("label rotation must be in 0..3")
    return {
        displayed: CHOICES[(index + rotation) % len(CHOICES)]
        for index, displayed in enumerate(CHOICES)
    }


def validate_g_pc0_rotation_manifest(
    samples: list[dict],
    base_samples: list[dict] | None,
    base_manifest_hash: str | None,
    expected_n: int = 64,
) -> list[str]:
    """Bind the 4x label-rotation diagnostic to the exact frozen G-PC0 rows."""

    errors: list[str] = []
    if expected_n != 64 or len(samples) != expected_n:
        errors.append(f"G-PC0 label rotation requires exactly 64 rows, found {len(samples)}")
    if base_manifest_hash != BASE_MANIFEST_SHA256:
        errors.append("label rotation requires the frozen G-PC0 v1 base manifest hash")
    if base_samples is None:
        return sorted(set(errors + ["label rotation requires explicit base manifest rows"]))
    base_errors = validate_g_pc0_manifest(base_samples, expected_n=16)
    errors.extend(f"base manifest: {error}" for error in base_errors)
    base_by_id = {_text(row, "sample_id"): row for row in base_samples}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        groups[_text(row, "base_sample_id")].append(row)
    if set(groups) != set(base_by_id):
        errors.append("expanded base_sample_id set differs from frozen base manifest")
    if len({_text(row, "sample_id") for row in samples}) != len(samples):
        errors.append("expanded sample_id values must be unique")
    labels = Counter(row.get("final_answer") for row in samples)
    if labels != Counter({choice: 16 for choice in CHOICES}):
        errors.append("expanded answer labels must be balanced at 16 per choice")
    rotation_hashes = {_text(row, "rotation_spec_hash") for row in samples}
    if len(rotation_hashes) != 1 or any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in rotation_hashes
    ):
        errors.append("rotation_spec_hash must be one shared lowercase SHA-256")

    immutable_fields = (
        "source_dataset",
        "source_image_id",
        "source_cluster_id",
        "image_original",
        "image_checksum",
        "final_question",
        "canonical_answer_text",
        "generic_context",
        "knowledge_dependency_label",
        "task_family",
        "full_image_reference_valid",
        "unique_answer",
        "human_verified",
        "human_reviewer",
        "human_reviewed_at",
        "procedural_gold_verified",
        "answer_leakage",
        "external_fact_injected",
        "required_evidence",
        "difficult_visual_evidence",
        "reasoning_depth",
        "minimal_reasoning_graph",
        "guides",
        "solution_trace",
        "positive_control_family",
        "generator_id",
        "generator_seed",
        "generation_spec_hash",
        "eligible_for_natural_g0",
        "eligible_for_training",
    )
    for base_index, base in enumerate(base_samples):
        base_id = _text(base, "sample_id")
        rows = groups.get(base_id, [])
        if len(rows) != 4 or {row.get("label_rotation") for row in rows} != set(range(4)):
            errors.append(f"{base_id}: requires exactly rotations 0..3")
            continue
        for row in sorted(rows, key=lambda value: value["label_rotation"]):
            rotation = row["label_rotation"]
            row_id = f"{base_id}-lr{rotation}"
            if row.get("sample_id") != row_id or row.get("source_record_id") != row_id:
                errors.append(f"{row_id}: diagnostic sample/source record ID mismatch")
            if row.get("manifest_profile") != ROTATION_PROFILE:
                errors.append(f"{row_id}: manifest_profile must be {ROTATION_PROFILE}")
            if row.get("experiment_role") != ROTATION_ROLE:
                errors.append(f"{row_id}: experiment_role must be {ROTATION_ROLE}")
            if row.get("base_manifest_sha256") != BASE_MANIFEST_SHA256:
                errors.append(f"{row_id}: base manifest hash mismatch")
            for field in immutable_fields:
                if row.get(field) != base.get(field):
                    errors.append(f"{row_id}: immutable field differs from base: {field}")
            semantic_options = {choice: base[f"option_{choice}"] for choice in CHOICES}
            if row.get("semantic_options") != semantic_options:
                errors.append(f"{row_id}: semantic_options differ from base")
            if row.get("correct_semantic_option_id") != base.get("final_answer"):
                errors.append(f"{row_id}: correct semantic option differs from base")
            mapping = label_rotation_mapping(rotation)
            if row.get("displayed_label_to_semantic_option") != mapping:
                errors.append(f"{row_id}: displayed-label mapping is not the frozen cyclic rotation")
            for displayed, semantic in mapping.items():
                if row.get(f"option_{displayed}") != base.get(f"option_{semantic}"):
                    errors.append(f"{row_id}: displayed option {displayed} has wrong semantic content")
            correct_labels = [
                displayed
                for displayed, semantic in mapping.items()
                if semantic == base.get("final_answer")
            ]
            if len(correct_labels) != 1 or row.get("final_answer") != correct_labels[0]:
                errors.append(f"{row_id}: rotated correct label mismatch")
            if row.get(f"option_{row.get('final_answer')}") != base.get("canonical_answer_text"):
                errors.append(f"{row_id}: rotated correct text differs from canonical answer")
    expected_order = [f"gpc0-{index + 1:02d}-lr{rotation}" for index in range(16) for rotation in range(4)]
    if [_text(row, "sample_id") for row in samples] != expected_order:
        errors.append("expanded rows must follow frozen base-major then rotation-major order")
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


def decide_g_pc0_label_rotation(
    records: list[dict], validation_errors: list[str] | None = None
) -> dict[str, Any]:
    """Aggregate four cyclic labels back to semantic options and apply 1046."""

    if validation_errors:
        return {"decision": "INVALID", "validation_errors": sorted(set(validation_errors))}
    grouped: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    errors: list[str] = []
    for row in records:
        base_id = str(row.get("base_sample_id", ""))
        condition = str(row.get("condition", ""))
        rotation = row.get("label_rotation")
        if not base_id or condition not in {"S", "T_Q", "T_QA", "T_generic"} or rotation not in range(4):
            errors.append("every row requires valid base_sample_id, condition and label_rotation")
            continue
        if rotation in grouped[base_id][condition]:
            errors.append(f"{base_id}/{condition}/lr{rotation}: duplicate")
        grouped[base_id][condition][rotation] = row
    if len(grouped) != 16:
        errors.append(f"requires 16 base samples, found {len(grouped)}")
    for base_id, conditions in grouped.items():
        if set(conditions) != {"S", "T_Q", "T_QA", "T_generic"}:
            errors.append(f"{base_id}: requires four conditions")
        for condition, rotations in conditions.items():
            if set(rotations) != set(range(4)):
                errors.append(f"{base_id}/{condition}: requires rotations 0..3")
    if errors:
        return {"decision": "INVALID", "validation_errors": sorted(set(errors))}

    aggregated: list[dict[str, Any]] = []
    for base_id in sorted(grouped):
        for condition in ("S", "T_Q", "T_QA", "T_generic"):
            rows = grouped[base_id][condition]
            semantic_scores = {choice: [] for choice in CHOICES}
            mapped_generation_votes = Counter()
            correct_semantic = None
            family = None
            for rotation in range(4):
                row = rows[rotation]
                mapping = row.get("displayed_label_to_semantic_option")
                scores = row.get("choice_log_likelihoods")
                if not isinstance(mapping, dict) or set(mapping) != set(CHOICES):
                    errors.append(f"{base_id}/{condition}/lr{rotation}: invalid label mapping")
                    continue
                if not isinstance(scores, dict) or set(scores) != set(CHOICES):
                    errors.append(f"{base_id}/{condition}/lr{rotation}: invalid choice scores")
                    continue
                for displayed in CHOICES:
                    semantic_scores[mapping[displayed]].append(float(scores[displayed]))
                parsed = row.get("parsed_answer")
                if parsed in CHOICES:
                    mapped_generation_votes[mapping[parsed]] += 1
                correct_semantic = row.get("correct_semantic_option_id")
                family = row.get("positive_control_family")
            means = {
                semantic: sum(values) / len(values) if len(values) == 4 else float("nan")
                for semantic, values in semantic_scores.items()
            }
            prediction = max(CHOICES, key=lambda choice: (means[choice], -CHOICES.index(choice)))
            vote_prediction = max(
                CHOICES, key=lambda choice: (mapped_generation_votes[choice], -CHOICES.index(choice))
            )
            aggregated.append({
                "base_sample_id": base_id,
                "condition": condition,
                "positive_control_family": family,
                "correct_semantic_option_id": correct_semantic,
                "semantic_mean_log_likelihoods": means,
                "semantic_prediction": prediction,
                "correct": prediction == correct_semantic,
                "mapped_generation_votes": {choice: mapped_generation_votes[choice] for choice in CHOICES},
                "mapped_generation_vote_prediction": vote_prediction,
                "mapped_generation_vote_correct": vote_prediction == correct_semantic,
            })
    if errors or any(
        not math.isfinite(value)
        for row in aggregated
        for value in row["semantic_mean_log_likelihoods"].values()
    ):
        if not errors:
            errors.append("every semantic score requires four finite rotation values")
        return {"decision": "INVALID", "validation_errors": sorted(set(errors))}

    table = {(row["base_sample_id"], row["condition"]): row for row in aggregated}
    base_ids = sorted(grouped)
    baseline = [table[(base_id, "S")]["correct"] for base_id in base_ids]
    tqa = [table[(base_id, "T_QA")]["correct"] for base_id in base_ids]
    gains = sum(not before and after for before, after in zip(baseline, tqa))
    regressions = sum(before and not after for before, after in zip(baseline, tqa))
    baseline_correct = sum(baseline)
    if baseline_correct >= 14:
        decision = "ROTATION_CEILING_INCONCLUSIVE"
    elif gains >= 4 and gains - regressions >= 3:
        decision = "LABEL_INTERFACE_MASKED_G_SIGNAL"
    else:
        decision = "GUIDE_OR_MODEL_INGESTION_FAILURE_AFTER_ROTATION"
    condition_correct = {
        condition: sum(table[(base_id, condition)]["correct"] for base_id in base_ids)
        for condition in ("S", "T_Q", "T_QA", "T_generic")
    }
    return {
        "decision": decision,
        "diagnostic_only": True,
        "gpc0_v1_frozen_decision": "INGESTION_OR_RECIPE_FAILURE",
        "gpc0_v1_decision_changed": False,
        "n_base_samples": 16,
        "n_rotation_samples": 64,
        "condition_correct_after_semantic_aggregation": condition_correct,
        "tqa_vs_s": {"gains": gains, "regressions": regressions, "net_flips": gains - regressions},
        "aggregation": "mean raw single-token label log-likelihood by semantic option across four rotations",
        "tie_break": list(CHOICES),
        "aggregated_records": aggregated,
        "inferential_statistics_used": False,
        "natural_image_claim_authorized": False,
        "training_data_authorized": False,
    }


def diagnose_g_pc0_label_rotation(records: list[dict], frozen_decision: str) -> dict[str, Any]:
    """Emit posthoc descriptive views without changing the 1046 decision."""

    recomputed = decide_g_pc0_label_rotation(records)
    if recomputed.get("decision") == "INVALID":
        raise ValueError("label-rotation diagnosis requires complete valid record structure")
    aggregated = recomputed["aggregated_records"]
    conditions = ("S", "T_Q", "T_QA", "T_generic")
    table = {(row["base_sample_id"], row["condition"]): row for row in aggregated}
    base_ids = sorted({row["base_sample_id"] for row in aggregated})
    flips = []
    for base_id in base_ids:
        baseline = table[(base_id, "S")]
        treatment = table[(base_id, "T_QA")]
        if baseline["correct"] != treatment["correct"]:
            flips.append({
                "base_sample_id": base_id,
                "positive_control_family": baseline["positive_control_family"],
                "correct_semantic_option_id": baseline["correct_semantic_option_id"],
                "s_prediction": baseline["semantic_prediction"],
                "tqa_prediction": treatment["semantic_prediction"],
                "direction": "gain" if treatment["correct"] else "regression",
            })
    family_correct = {}
    for family in FAMILIES:
        family_correct[family] = {
            condition: sum(
                row["correct"]
                for row in aggregated
                if row["condition"] == condition and row["positive_control_family"] == family
            )
            for condition in conditions
        }
        family_correct[family]["n"] = sum(
            row["condition"] == "S" and row["positive_control_family"] == family
            for row in aggregated
        )
    metrics = {}
    for condition in conditions:
        condition_rows = [row for row in aggregated if row["condition"] == condition]
        margins = []
        for row in condition_rows:
            scores = row["semantic_mean_log_likelihoods"]
            correct = row["correct_semantic_option_id"]
            margins.append(scores[correct] - max(scores[choice] for choice in CHOICES if choice != correct))
        raw_rows = [row for row in records if row.get("condition") == condition]
        metrics[condition] = {
            "semantic_correct": sum(row["correct"] for row in condition_rows),
            "semantic_prediction_counts": dict(sorted(Counter(row["semantic_prediction"] for row in condition_rows).items())),
            "mapped_generation_vote_correct": sum(row["mapped_generation_vote_correct"] for row in condition_rows),
            "raw_displayed_prediction_counts": dict(sorted(Counter(str(row["parsed_answer"]) for row in raw_rows).items())),
            "correct_semantic_margin": {
                "mean": sum(margins) / len(margins),
                "median": (sorted(margins)[7] + sorted(margins)[8]) / 2,
                "min": min(margins),
                "max": max(margins),
            },
        }
    return {
        "analysis_status": "posthoc_descriptive",
        "frozen_decision": frozen_decision,
        "recomputed_decision": recomputed["decision"],
        "decision_changed": False,
        "paired_s_to_tqa_flips": flips,
        "per_family_descriptive_only": family_correct,
        "condition_metrics": metrics,
        "natural_image_claim_authorized": False,
        "training_data_authorized": False,
    }


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as sink:
        sink.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
