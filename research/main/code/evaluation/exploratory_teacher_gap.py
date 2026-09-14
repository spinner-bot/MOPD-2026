"""Fail-closed validation and decision logic for protocol 1017.

This module intentionally contains no model inference.  It freezes the CPU-side
contracts that must pass before a locked pilot can be interpreted.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


DATA_PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "data_pipeline"
sys.path.insert(0, str(DATA_PIPELINE_ROOT))

from reasoning_graph_contract import validate_guide_graph_alignment, validate_reasoning_graph
from g_pc0 import validate_g_pc0_manifest, validate_g_pc0_rotation_manifest
from teacher_gap_inputs import NATURAL_ROTATION_CONDITIONS


CONDITIONS = ("S", "T_Q", "T_QA", "T_generic")
CHOICES = ("A", "B", "C", "D")
NATURAL_ROTATION_PROFILE = "natural_label_rotation_v2_dual_generic"
NATURAL_ROTATION_ROLE = "natural_teacher_gap_label_robust_measurement"
NATURAL_ROTATION_SPEC = {
    "version": NATURAL_ROTATION_PROFILE,
    "rotation_order": [0, 1, 2, 3],
    "displayed_to_semantic_rule": "semantic_index=(displayed_index+rotation) mod 4",
    "score_aggregation": "mean raw label-sequence log-likelihood by semantic option",
    "continuous_metric": "log_softmax over four semantic mean scores",
    "conditions": list(NATURAL_ROTATION_CONDITIONS),
    "matched_generic_controls": {"T_Q": "T_generic_Q", "T_QA": "T_generic_QA"},
    "tie_break": list(CHOICES),
}
NATURAL_ROTATION_SPEC_HASH = hashlib.sha256(
    json.dumps(NATURAL_ROTATION_SPEC, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def validate_manifest_profile(
    samples: list[dict],
    expected_n: int = 96,
    manifest_profile: str = "natural",
    base_samples: list[dict] | None = None,
    base_manifest_hash: str | None = None,
) -> list[str]:
    """Dispatch the shared runner to an explicitly declared manifest contract."""

    if manifest_profile == "natural":
        return validate_locked_manifest(samples, expected_n=expected_n)
    if manifest_profile == "g_pc0":
        return validate_g_pc0_manifest(samples, expected_n=expected_n)
    if manifest_profile == "g_pc0_rotation":
        return validate_g_pc0_rotation_manifest(
            samples,
            base_samples=base_samples,
            base_manifest_hash=base_manifest_hash,
            expected_n=expected_n,
        )
    if manifest_profile == "natural_rotation":
        return validate_natural_rotation_manifest(
            samples,
            base_samples=base_samples,
            base_manifest_hash=base_manifest_hash,
            expected_n=expected_n,
        )
    return [f"unsupported manifest profile: {manifest_profile}"]


def conditions_for_profile(manifest_profile: str) -> tuple[str, ...]:
    """Keep completed four-condition profiles separate from natural v0.4."""

    return NATURAL_ROTATION_CONDITIONS if manifest_profile == "natural_rotation" else CONDITIONS


def frozen_schedule(
    samples: list[dict], manifest_hash: str, conditions: tuple[str, ...] = CONDITIONS
) -> list[tuple[dict, str]]:
    """Return the deterministic sample-condition order shared by runner and gate."""

    pairs = [(sample, condition) for sample in samples for condition in conditions]
    return sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{manifest_hash}\0{pair[0]['sample_id']}\0{pair[1]}".encode("utf-8")
        ).hexdigest(),
    )


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
    cluster_ids = [str(row.get("source_cluster_id", "")).strip() for row in samples]
    if any(not value for value in cluster_ids):
        errors.append("missing source_cluster_id")
    if len(set(cluster_ids)) != len(cluster_ids):
        errors.append("source_cluster_id values must be unique")

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
        "generic_context_t_q",
        "generic_context_t_qa",
        "human_reviewer",
        "human_reviewed_at",
    )
    for index, row in enumerate(samples):
        row_id = sample_ids[index] or f"row-{index}"
        missing = [key for key in required_text if not str(row.get(key, "")).strip()]
        if missing:
            errors.append(f"{row_id}: missing fields {','.join(missing)}")
        length_control = row.get("generic_length_control")
        if not isinstance(length_control, dict):
            errors.append(f"{row_id}: generic_length_control must be an object")
        else:
            if length_control.get("algorithm") != "neutral-prefix-nearest-per-sample-v1":
                errors.append(f"{row_id}: invalid generic length algorithm")
            word_bank_hash = str(length_control.get("neutral_word_bank_sha256", ""))
            if len(word_bank_hash) != 64 or any(char not in "0123456789abcdef" for char in word_bank_hash):
                errors.append(f"{row_id}: invalid neutral word bank sha256")
            tokenizer_source = str(length_control.get("tokenizer_source", "")).strip()
            tokenizer_hashes = length_control.get("tokenizer_artifact_sha256")
            if not tokenizer_source:
                errors.append(f"{row_id}: missing generic tokenizer source")
            tokenizer_revision = str(length_control.get("tokenizer_revision", ""))
            if len(tokenizer_revision) != 40 or any(
                char not in "0123456789abcdef" for char in tokenizer_revision
            ):
                errors.append(f"{row_id}: invalid generic tokenizer revision")
            if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
                errors.append(f"{row_id}: missing generic tokenizer artifact hashes")
            elif any(
                len(str(value)) != 64 or any(char not in "0123456789abcdef" for char in str(value))
                for value in tokenizer_hashes.values()
            ):
                errors.append(f"{row_id}: invalid generic tokenizer artifact sha256")
            tolerance = length_control.get("tolerance")
            if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance != 0.10:
                errors.append(f"{row_id}: generic length tolerance must equal 0.10")
            for pair_name in ("T_Q_vs_T_generic_Q", "T_QA_vs_T_generic_QA"):
                check = length_control.get(pair_name)
                if not isinstance(check, dict):
                    errors.append(f"{row_id}: missing generic length check {pair_name}")
                    continue
                if check.get("sample_id") != row_id:
                    errors.append(f"{row_id}: generic length check sample_id mismatch for {pair_name}")
                treatment_tokens = check.get("treatment_tokens")
                control_tokens = check.get("control_tokens")
                relative_difference = check.get("relative_difference")
                valid_numbers = (
                    isinstance(treatment_tokens, int) and not isinstance(treatment_tokens, bool)
                    and isinstance(control_tokens, int) and not isinstance(control_tokens, bool)
                    and treatment_tokens > 0 and control_tokens > 0
                    and isinstance(relative_difference, (int, float))
                    and not isinstance(relative_difference, bool)
                )
                if not valid_numbers:
                    errors.append(f"{row_id}: invalid generic length values for {pair_name}")
                    continue
                expected_difference = abs(control_tokens - treatment_tokens) / treatment_tokens
                if abs(relative_difference - expected_difference) > 1e-12:
                    errors.append(f"{row_id}: inconsistent generic length difference for {pair_name}")
                if relative_difference > 0.10 or check.get("passes") is not True:
                    errors.append(f"{row_id}: per-sample generic length contract failed for {pair_name}")
        finalization = row.get("finalization_provenance")
        if not isinstance(finalization, dict):
            errors.append(f"{row_id}: finalization_provenance must be an object")
        else:
            code_commit = str(finalization.get("code_commit", ""))
            if len(code_commit) != 40 or any(char not in "0123456789abcdef" for char in code_commit):
                errors.append(f"{row_id}: invalid finalizer code commit")
            for field in ("runtime_code_sha256", "input_sha256"):
                hashes = finalization.get(field)
                if not isinstance(hashes, dict) or not hashes:
                    errors.append(f"{row_id}: missing finalizer {field}")
                elif any(
                    len(str(value)) != 64
                    or any(char not in "0123456789abcdef" for char in str(value))
                    for value in hashes.values()
                ):
                    errors.append(f"{row_id}: invalid finalizer {field}")
            if not isinstance(finalization.get("seed"), int) or isinstance(finalization.get("seed"), bool):
                errors.append(f"{row_id}: invalid finalizer seed")
        if row.get("knowledge_dependency_label") != "image_only":
            errors.append(f"{row_id}: knowledge_dependency_label must be image_only")
        if row.get("task_family") not in {"R", "PR"}:
            errors.append(f"{row_id}: task_family must be R or PR")
        if row.get("full_image_reference_valid") is not True:
            errors.append(f"{row_id}: full_image_reference_valid must be true")
        if row.get("unique_answer") is not True:
            errors.append(f"{row_id}: unique_answer must be true")
        if row.get("human_verified") is not True:
            errors.append(f"{row_id}: human_verified must be true")
        reviewed_at = str(row.get("human_reviewed_at", "")).strip()
        try:
            parsed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
                errors.append(f"{row_id}: human_reviewed_at requires timezone")
        except ValueError:
            errors.append(f"{row_id}: human_reviewed_at must be valid ISO 8601")
        if row.get("answer_leakage") not in {0, False}:
            errors.append(f"{row_id}: answer_leakage must be zero")
        if row.get("external_fact_injected") not in {0, False}:
            errors.append(f"{row_id}: external_fact_injected must be zero")
        depth = row.get("reasoning_depth")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 2:
            errors.append(f"{row_id}: reasoning_depth must be an integer >= 2")
        evidence = row.get("required_evidence")
        evidence_valid = (
            isinstance(evidence, list)
            and len(evidence) >= 2
            and all(isinstance(value, str) and value.strip() for value in evidence)
            and len({value.strip().casefold() for value in evidence}) == len(evidence)
        )
        if not evidence_valid:
            errors.append(f"{row_id}: required_evidence must contain at least two entries")
        graph = row.get("minimal_reasoning_graph")
        graph_errors, _ = validate_reasoning_graph(
            graph,
            declared_depth=depth,
            require_two_evidence=True,
        )
        errors.extend(f"{row_id}: minimal_reasoning_graph {error}" for error in graph_errors)
        if evidence_valid and isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
            graph_evidence = {
                str(node.get("claim", "")).strip().casefold()
                for node in graph["nodes"]
                if isinstance(node, dict) and node.get("type") == "evidence"
            }
            if graph_evidence != {value.strip().casefold() for value in evidence}:
                errors.append(f"{row_id}: required_evidence must match graph evidence")

        difficult = row.get("difficult_visual_evidence")
        difficult_valid = (
            isinstance(difficult, list)
            and all(isinstance(value, str) and value.strip() for value in difficult)
            and len({value.strip().casefold() for value in difficult}) == len(difficult)
        )
        if not difficult_valid:
            errors.append(f"{row_id}: difficult_visual_evidence must be a distinct string list")
        elif evidence_valid and any(
            value.strip().casefold() not in {item.strip().casefold() for item in evidence}
            for value in difficult
        ):
            errors.append(f"{row_id}: difficult_visual_evidence must be required evidence")
        elif row.get("task_family") == "PR" and not difficult:
            errors.append(f"{row_id}: PR requires difficult_visual_evidence")
        elif row.get("task_family") == "R" and difficult:
            errors.append(f"{row_id}: R must not have difficult_visual_evidence")

        guides = row.get("guides")
        if not isinstance(guides, list) or not 1 <= len(guides) <= 4:
            errors.append(f"{row_id}: guides must contain 1..4 items")
        elif any(
            not isinstance(guide, dict)
            or not str(guide.get("question", "")).strip()
            or not str(guide.get("answer", "")).strip()
            or not str(guide.get("supports_next", "")).strip()
            for guide in guides
        ):
            errors.append(f"{row_id}: every guide requires question, answer, and supports_next")
        errors.extend(
            f"{row_id}: {error}"
            for error in validate_guide_graph_alignment(guides, graph)
        )
        options = [str(row.get(f"option_{choice}", "")).strip().casefold() for choice in CHOICES]
        if len(set(options)) != len(options):
            errors.append(f"{row_id}: options must be distinct")
        correct_choice = row.get("final_answer")
        if correct_choice in CHOICES and str(row.get(f"option_{correct_choice}", "")).strip() != str(
            row.get("canonical_answer_text", "")
        ).strip():
            errors.append(f"{row_id}: correct option must equal canonical_answer_text")
    return errors


def natural_label_rotation_mapping(rotation: int) -> dict[str, str]:
    if rotation not in range(4):
        raise ValueError("label rotation must be in 0..3")
    return {
        displayed: CHOICES[(index + rotation) % len(CHOICES)]
        for index, displayed in enumerate(CHOICES)
    }


def expand_natural_rotation_manifest(
    base_samples: list[dict], base_manifest_hash: str
) -> list[dict]:
    """Create all four non-selective displayed-label views for natural samples."""

    base_errors = validate_locked_manifest(base_samples, expected_n=len(base_samples))
    if base_errors:
        raise ValueError("base natural manifest invalid: " + "; ".join(base_errors))
    if len(base_manifest_hash) != 64 or any(c not in "0123456789abcdef" for c in base_manifest_hash):
        raise ValueError("base manifest hash must be lowercase SHA-256")
    rows = []
    for base in base_samples:
        semantic_options = {choice: base[f"option_{choice}"] for choice in CHOICES}
        for rotation in range(4):
            mapping = natural_label_rotation_mapping(rotation)
            row = deepcopy(base)
            row.update({
                "sample_id": f"{base['sample_id']}-lr{rotation}",
                "manifest_profile": NATURAL_ROTATION_PROFILE,
                "experiment_role": NATURAL_ROTATION_ROLE,
                "base_sample_id": base["sample_id"],
                "base_manifest_sha256": base_manifest_hash,
                "label_rotation": rotation,
                "rotation_spec_hash": NATURAL_ROTATION_SPEC_HASH,
                "semantic_options": semantic_options,
                "correct_semantic_option_id": base["final_answer"],
                "displayed_label_to_semantic_option": mapping,
            })
            for displayed, semantic in mapping.items():
                row[f"option_{displayed}"] = semantic_options[semantic]
            row["final_answer"] = next(
                displayed for displayed, semantic in mapping.items() if semantic == base["final_answer"]
            )
            rows.append(row)
    return rows


def validate_natural_rotation_manifest(
    samples: list[dict],
    base_samples: list[dict] | None,
    base_manifest_hash: str | None,
    expected_n: int,
) -> list[str]:
    """Bind natural rotation views to an exact human-reviewed base manifest."""

    errors = []
    if expected_n % 4 or len(samples) != expected_n:
        errors.append(f"natural label rotation requires exactly {expected_n} rows, found {len(samples)}")
    if base_samples is None or base_manifest_hash is None:
        return sorted(set(errors + ["natural label rotation requires explicit base manifest and hash"]))
    if expected_n != 4 * len(base_samples):
        errors.append("expanded size must equal four times the base manifest size")
    errors.extend(
        f"base manifest: {error}"
        for error in validate_locked_manifest(base_samples, expected_n=len(base_samples))
    )
    if len(base_manifest_hash) != 64 or any(c not in "0123456789abcdef" for c in base_manifest_hash):
        errors.append("base manifest hash must be lowercase SHA-256")
    base_by_id = {str(row.get("sample_id", "")): row for row in base_samples}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        groups[str(row.get("base_sample_id", ""))].append(row)
    if set(groups) != set(base_by_id):
        errors.append("expanded base_sample_id set differs from base manifest")
    controlled = {
        "sample_id", "manifest_profile", "experiment_role", "base_sample_id",
        "base_manifest_sha256", "label_rotation", "rotation_spec_hash",
        "semantic_options", "correct_semantic_option_id", "displayed_label_to_semantic_option",
        "option_A", "option_B", "option_C", "option_D", "final_answer",
    }
    for base in base_samples:
        base_id = str(base.get("sample_id", ""))
        rows = groups.get(base_id, [])
        if len(rows) != 4 or {row.get("label_rotation") for row in rows} != set(range(4)):
            errors.append(f"{base_id}: requires exactly rotations 0..3")
            continue
        semantic_options = {choice: base[f"option_{choice}"] for choice in CHOICES}
        for row in rows:
            rotation = row["label_rotation"]
            row_id = f"{base_id}-lr{rotation}"
            if row.get("sample_id") != row_id:
                errors.append(f"{row_id}: expanded sample ID mismatch")
            if row.get("manifest_profile") != NATURAL_ROTATION_PROFILE or row.get("experiment_role") != NATURAL_ROTATION_ROLE:
                errors.append(f"{row_id}: natural rotation profile/role mismatch")
            if row.get("base_manifest_sha256") != base_manifest_hash:
                errors.append(f"{row_id}: base manifest hash mismatch")
            if row.get("rotation_spec_hash") != NATURAL_ROTATION_SPEC_HASH:
                errors.append(f"{row_id}: rotation spec hash mismatch")
            for field in set(base) - controlled:
                if row.get(field) != base.get(field):
                    errors.append(f"{row_id}: base field differs: {field}")
            if set(row) - set(base) - controlled:
                errors.append(f"{row_id}: unexpected expanded fields")
            if row.get("semantic_options") != semantic_options:
                errors.append(f"{row_id}: semantic options mismatch")
            if row.get("correct_semantic_option_id") != base.get("final_answer"):
                errors.append(f"{row_id}: correct semantic option mismatch")
            mapping = natural_label_rotation_mapping(rotation)
            if row.get("displayed_label_to_semantic_option") != mapping:
                errors.append(f"{row_id}: cyclic mapping mismatch")
            for displayed, semantic in mapping.items():
                if row.get(f"option_{displayed}") != semantic_options[semantic]:
                    errors.append(f"{row_id}: displayed option {displayed} mismatch")
            correct_label = next(
                displayed for displayed, semantic in mapping.items() if semantic == base.get("final_answer")
            )
            if row.get("final_answer") != correct_label:
                errors.append(f"{row_id}: displayed correct label mismatch")
    expected_order = [
        f"{base['sample_id']}-lr{rotation}" for base in base_samples for rotation in range(4)
    ]
    if [str(row.get("sample_id", "")) for row in samples] != expected_order:
        errors.append("natural rotations must follow base-major then rotation-major order")
    return sorted(set(errors))


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


def validate_manifest_bundle(partitions: dict[str, list[dict]]) -> list[str]:
    """Validate the frozen 8/32/96 manifests together before GPU inference."""

    expected = {"canary": 8, "dev": 32, "locked": 96}
    errors = []
    if set(partitions) != set(expected):
        return ["manifest bundle must contain canary, dev, and locked partitions"]
    for name in ("canary", "dev", "locked"):
        errors.extend(
            f"{name}: {error}"
            for error in validate_locked_manifest(partitions[name], expected_n=expected[name])
        )
    errors.extend(validate_partition_isolation(partitions))
    return errors


def validate_results_against_manifest(
    records: list[dict],
    samples: list[dict],
    manifest_hash: str,
    conditions: tuple[str, ...] = CONDITIONS,
) -> list[str]:
    """Bind raw records to the exact frozen sample metadata they claim to use."""

    errors = []
    expected = {str(row.get("sample_id", "")).strip(): row for row in samples}
    if "" in expected or len(expected) != len(samples):
        return ["manifest sample_id values must be non-empty and unique"]
    observed_ids = {str(row.get("sample_id", "")).strip() for row in records}
    if observed_ids != set(expected):
        missing = sorted(set(expected) - observed_ids)
        extra = sorted(observed_ids - set(expected))
        errors.append(f"raw sample ids differ from manifest: missing={missing}, extra={extra}")

    expected_schedule = {
        (str(sample["sample_id"]), condition): index
        for index, (sample, condition) in enumerate(frozen_schedule(samples, manifest_hash, conditions))
    }

    for index, row in enumerate(records):
        sample_id = str(row.get("sample_id", "")).strip()
        sample = expected.get(sample_id)
        if sample is None:
            continue
        schedule_key = (sample_id, str(row.get("condition", "")))
        if row.get("schedule_index") != expected_schedule.get(schedule_key):
            errors.append(
                f"record-{index}/{sample_id}: schedule_index differs from frozen schedule"
            )
        expected_values = {
            "manifest_hash": manifest_hash,
            "source_dataset": str(sample.get("source_dataset", "")),
            "source_record_id": str(sample.get("source_record_id", "")),
            "source_image_id": str(sample.get("source_image_id", "")),
            "source_cluster_id": str(sample.get("source_cluster_id", "")),
            "task_family": str(sample.get("task_family", "")),
            "reasoning_depth": str(sample.get("reasoning_depth", "")),
            "image_checksum": str(sample.get("image_checksum", "")),
            "correct_choice": str(sample.get("final_answer", "")),
            "question_hash": hashlib.sha256(
                str(sample.get("final_question", "")).encode("utf-8")
            ).hexdigest(),
            "options_hash": hashlib.sha256(
                json.dumps(
                    [sample.get(f"option_{choice}") for choice in CHOICES],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        for field in (
            "experiment_role",
            "positive_control_family",
            "generator_id",
            "base_sample_id",
            "base_manifest_sha256",
            "rotation_spec_hash",
            "correct_semantic_option_id",
        ):
            if str(sample.get(field, "")).strip():
                expected_values[field] = str(sample[field])
        for field in ("label_rotation",):
            if field in sample:
                expected_values[field] = sample[field]
        for field, value in expected_values.items():
            if row.get(field) != value and str(row.get(field, "")) != value:
                errors.append(f"record-{index}/{sample_id}: {field} differs from frozen manifest")
        for field in ("semantic_options", "displayed_label_to_semantic_option"):
            if field in sample and row.get(field) != sample[field]:
                errors.append(f"record-{index}/{sample_id}: {field} differs from frozen manifest")
    return errors


def validate_result_records(
    records: list[dict],
    expected_samples: int = 96,
    min_parse_rate: float = 0.99,
    conditions: tuple[str, ...] = CONDITIONS,
) -> list[str]:
    """Apply the protocol's completeness and provenance gate."""

    errors: list[str] = []
    expected_total = expected_samples * len(conditions)
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
        "model_dtype",
    )
    global_values = {field: set() for field in global_fields}
    global_json_fields = (
        "chat_template",
        "generation_config",
        "choice_token_ids",
        "runtime_code_hashes",
    )
    global_json_values = {field: set() for field in global_json_fields}
    schedule_indices = []
    for index, row in enumerate(records):
        key = f"{row.get('sample_id', '<missing>')}/{row.get('condition', '<missing>')}"
        if row.get("condition") not in conditions:
            errors.append(f"{key}: unsupported condition")
        schedule_index = row.get("schedule_index")
        rank = row.get("rank")
        world_size = row.get("world_size")
        if not isinstance(schedule_index, int) or isinstance(schedule_index, bool):
            errors.append(f"{key}: schedule_index must be an integer")
        else:
            schedule_indices.append(schedule_index)
        if world_size != 4:
            errors.append(f"{key}: world_size must be 4")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < 4:
            errors.append(f"{key}: rank must be an integer in 0..3")
        elif (
            isinstance(schedule_index, int)
            and not isinstance(schedule_index, bool)
            and rank != schedule_index % 4
        ):
            errors.append(f"{key}: rank disagrees with frozen schedule sharding")
        if row.get("error") not in {None, ""}:
            errors.append(f"{key}: error is present")
        if row.get("truncated") is not False or row.get("eos_reached") is not True:
            errors.append(f"{key}: output is truncated or missing EOS")
        if row.get("parsed_answer") in CHOICES:
            parsed += 1
        if row.get("correct_choice") not in CHOICES:
            errors.append(f"{key}: correct_choice must be A, B, C, or D")
        if row.get("task_family") not in {"R", "PR"}:
            errors.append(f"{key}: task_family must be R or PR")
        reasoning_depth = row.get("reasoning_depth")
        if not isinstance(reasoning_depth, int) or isinstance(reasoning_depth, bool) or reasoning_depth < 2:
            errors.append(f"{key}: reasoning_depth must be an integer >= 2")
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
        for field in global_json_fields:
            value = row.get(field)
            if value is None:
                errors.append(f"{key}: missing {field}")
                continue
            try:
                global_json_values[field].add(
                    json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
            except (TypeError, ValueError):
                errors.append(f"{key}: {field} must be JSON serializable")

        if row.get("chat_template") != {"enable_thinking": False, "add_generation_prompt": True}:
            errors.append(f"{key}: chat_template must disable thinking and add generation prompt")
        if row.get("choice_continuation_verified") is not True:
            errors.append(f"{key}: choice continuation must be verified against the chat-template boundary")
        generation = row.get("generation_config")
        if (
            not isinstance(generation, dict)
            or generation.get("do_sample") is not False
            or generation.get("constrained_to") != list(CHOICES)
        ):
            errors.append(f"{key}: generation_config violates frozen deterministic choice contract")
        choice_tokens = row.get("choice_token_ids")
        if (
            not isinstance(choice_tokens, dict)
            or set(choice_tokens) != set(CHOICES)
            or any(
                not isinstance(value, list)
                or not value
                or any(not isinstance(token, int) or isinstance(token, bool) for token in value)
                for value in choice_tokens.values()
            )
            or len({tuple(value) for value in choice_tokens.values()}) != len(CHOICES)
        ):
            errors.append(f"{key}: choice_token_ids must contain four distinct non-empty integer sequences")
        elif len({len(value) for value in choice_tokens.values()}) != 1:
            errors.append(f"{key}: choice_token_ids must have equal lengths")

    if records and parsed / len(records) < min_parse_rate:
        errors.append(f"parse rate {parsed / len(records):.6f} is below {min_parse_rate:.6f}")
    for field, values in global_values.items():
        if len(values) != 1:
            errors.append(f"{field} must be identical across all records")
    for field, values in global_json_values.items():
        if len(values) != 1:
            errors.append(f"{field} must be identical across all records")
    if sorted(schedule_indices) != list(range(expected_total)):
        errors.append("schedule_index values must cover the frozen schedule exactly once")

    invariant_fields = (
        "source_dataset",
        "source_record_id",
        "source_image_id",
        "source_cluster_id",
        "task_family",
        "reasoning_depth",
        "image_checksum",
        "question_hash",
        "options_hash",
    )
    for sample_id, rows in by_sample.items():
        if {row.get("condition") for row in rows} != set(conditions):
            errors.append(f"{sample_id}: exact profile condition set is required")
        for field in invariant_fields:
            values = {str(row.get(field, "")).strip() for row in rows}
            if "" in values or len(values) != 1:
                errors.append(f"{sample_id}: {field} must be present and invariant")
        for field in (
            "experiment_role",
            "positive_control_family",
            "generator_id",
            "base_sample_id",
            "base_manifest_sha256",
            "rotation_spec_hash",
            "correct_semantic_option_id",
            "label_rotation",
        ):
            values = {str(row.get(field, "")).strip() for row in rows}
            if any(values) and ("" in values or len(values) != 1):
                errors.append(f"{sample_id}: {field} must be present and invariant when used")
        for field in ("semantic_options", "displayed_label_to_semantic_option"):
            values = {
                json.dumps(row.get(field), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for row in rows
                if field in row
            }
            if values and (len(values) != 1 or any(field not in row for row in rows)):
                errors.append(f"{sample_id}: {field} must be present and invariant when used")
        prompt_hashes = [str(row.get("prompt_hash", "")).strip() for row in rows]
        if any(not value for value in prompt_hashes):
            errors.append(f"{sample_id}: prompt_hash is required")
        elif len(set(prompt_hashes)) != len(conditions):
            errors.append(f"{sample_id}: each condition must have a distinct prompt_hash")
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


def _condition_vectors(
    records: list[dict], conditions: tuple[str, ...] = CONDITIONS
) -> tuple[list[str], dict[str, dict[str, dict]]]:
    by_condition: dict[str, dict[str, dict]] = {condition: {} for condition in conditions}
    for row in records:
        condition = row.get("condition")
        if condition in by_condition:
            by_condition[condition][str(row.get("sample_id"))] = row
    ids = sorted(by_condition["S"])
    if not ids or any(sorted(by_condition[condition]) != ids for condition in conditions):
        raise ValueError("all conditions must contain the same non-empty sample ids")
    return ids, by_condition


def _decide_teacher_gap_validated(
    records: list[dict],
    *,
    leakage_free: bool,
    expected_samples: int = 96,
    n_bootstrap: int = 10_000,
    seed: int = 20260911,
    generic_controls: dict[str, str] | None = None,
) -> dict:
    """Apply 1017 statistics after the caller has validated the observation space."""

    generic_controls = generic_controls or {"T_Q": "T_generic", "T_QA": "T_generic"}
    required_conditions = tuple(dict.fromkeys(("S", "T_Q", "T_QA", *generic_controls.values())))
    ids, rows = _condition_vectors(records, required_conditions)
    source_clusters = [str(rows["S"][sample_id]["source_cluster_id"]) for sample_id in ids]
    baseline_correct = [bool(rows["S"][sample_id]["correct"]) for sample_id in ids]
    baseline_logp = [float(rows["S"][sample_id]["correct_log_prob"]) for sample_id in ids]
    comparisons = {}
    raw_p = {}
    for condition in ("T_Q", "T_QA"):
        generic_condition = generic_controls[condition]
        generic_correct = [bool(rows[generic_condition][sample_id]["correct"]) for sample_id in ids]
        generic_logp = [float(rows[generic_condition][sample_id]["correct_log_prob"]) for sample_id in ids]
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
            "matched_generic_condition": generic_condition,
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
        # A contaminated locked set cannot support either a positive Go or a
        # negative Stop claim, even when its observed effect is near zero.
        decision = "STOP_CURRENT_RECIPE" if stop and leakage_free else "REVISE"
    return {
        "decision": decision,
        "leakage_free": leakage_free,
        "n_samples": len(ids),
        "matched_generic_controls": generic_controls,
        "comparisons": comparisons,
    }


def decide_teacher_gap(
    records: list[dict],
    *,
    leakage_free: bool,
    expected_samples: int = 96,
    n_bootstrap: int = 10_000,
    seed: int = 20260911,
) -> dict:
    """Apply the original fixed-label 1017 decision for preserved legacy runs."""

    validation_errors = validate_result_records(records, expected_samples=expected_samples)
    if validation_errors:
        return {"decision": "INVALID", "validation_errors": validation_errors}
    return _decide_teacher_gap_validated(
        records,
        leakage_free=leakage_free,
        expected_samples=expected_samples,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def aggregate_natural_rotation_results(
    records: list[dict], expected_samples: int = 96
) -> tuple[list[dict], list[str]]:
    """Reduce four displayed-label views to one semantic observation per condition."""

    errors: list[str] = []
    grouped: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in records:
        base_id = str(row.get("base_sample_id", ""))
        condition = str(row.get("condition", ""))
        rotation = row.get("label_rotation")
        if not base_id or condition not in NATURAL_ROTATION_CONDITIONS or rotation not in range(4):
            errors.append("every rotation result requires base_sample_id, condition and rotation 0..3")
            continue
        if rotation in grouped[base_id][condition]:
            errors.append(f"{base_id}/{condition}/lr{rotation}: duplicate")
        grouped[base_id][condition][rotation] = row
    if len(grouped) != expected_samples:
        errors.append(f"expected {expected_samples} base samples, found {len(grouped)}")
    observations = []
    for base_id in sorted(grouped):
        conditions = grouped[base_id]
        if set(conditions) != set(NATURAL_ROTATION_CONDITIONS):
            errors.append(f"{base_id}: all five natural conditions are required")
            continue
        for condition in NATURAL_ROTATION_CONDITIONS:
            rotations = conditions[condition]
            if set(rotations) != set(range(4)):
                errors.append(f"{base_id}/{condition}: rotations 0..3 are required")
                continue
            semantic_values = {choice: [] for choice in CHOICES}
            mapped_votes = Counter()
            correct_values = set()
            cluster_values = set()
            for rotation in range(4):
                row = rotations[rotation]
                mapping = row.get("displayed_label_to_semantic_option")
                scores = row.get("choice_log_likelihoods")
                if mapping != natural_label_rotation_mapping(rotation):
                    errors.append(f"{base_id}/{condition}/lr{rotation}: mapping mismatch")
                    continue
                if not isinstance(scores, dict) or set(scores) != set(CHOICES) or not all(
                    isinstance(value, (int, float)) and math.isfinite(value) for value in scores.values()
                ):
                    errors.append(f"{base_id}/{condition}/lr{rotation}: invalid choice scores")
                    continue
                for displayed, semantic in mapping.items():
                    semantic_values[semantic].append(float(scores[displayed]))
                if row.get("parsed_answer") in CHOICES:
                    mapped_votes[mapping[row["parsed_answer"]]] += 1
                correct_values.add(str(row.get("correct_semantic_option_id", "")))
                cluster_values.add(str(row.get("source_cluster_id", "")))
            if len(correct_values) != 1 or next(iter(correct_values), "") not in CHOICES:
                errors.append(f"{base_id}/{condition}: correct semantic option must be invariant")
                continue
            if len(cluster_values) != 1 or not next(iter(cluster_values), ""):
                errors.append(f"{base_id}/{condition}: source cluster must be invariant")
                continue
            if any(len(values) != 4 for values in semantic_values.values()):
                errors.append(f"{base_id}/{condition}: every semantic option needs four scores")
                continue
            means = {choice: sum(semantic_values[choice]) / 4 for choice in CHOICES}
            maximum = max(means.values())
            weights = {choice: math.exp(means[choice] - maximum) for choice in CHOICES}
            total = sum(weights.values())
            probabilities = {choice: weights[choice] / total for choice in CHOICES}
            log_probabilities = {choice: math.log(probabilities[choice]) for choice in CHOICES}
            prediction = max(CHOICES, key=lambda choice: (means[choice], -CHOICES.index(choice)))
            correct = next(iter(correct_values))
            max_votes = max(mapped_votes.values(), default=0)
            vote_winners = [choice for choice in CHOICES if mapped_votes[choice] == max_votes]
            observations.append({
                "sample_id": base_id,
                "condition": condition,
                "source_cluster_id": next(iter(cluster_values)),
                "correct": prediction == correct,
                "correct_choice": correct,
                "parsed_answer": prediction,
                "correct_log_prob": log_probabilities[correct],
                "semantic_correct_log_prob": log_probabilities[correct],
                "semantic_margin": means[correct] - max(means[choice] for choice in CHOICES if choice != correct),
                "semantic_prediction": prediction,
                "semantic_mean_log_likelihoods": means,
                "semantic_probabilities": probabilities,
                "mapped_generation_votes": {choice: mapped_votes[choice] for choice in CHOICES},
                "mapped_generation_vote_prediction": vote_winners[0] if len(vote_winners) == 1 else None,
                "mapped_generation_vote_tie": len(vote_winners) != 1,
            })
    return observations, sorted(set(errors))


def decide_teacher_gap_label_rotation(
    records: list[dict],
    *,
    leakage_free: bool,
    expected_samples: int = 96,
    validation_errors: list[str] | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 20260911,
) -> dict:
    """Apply 1017 v0.3 to base-sample semantic observations from four rotations."""

    observations, aggregation_errors = aggregate_natural_rotation_results(records, expected_samples)
    errors = sorted(set((validation_errors or []) + aggregation_errors))
    if errors:
        return {"decision": "INVALID", "validation_errors": errors}
    result = _decide_teacher_gap_validated(
        observations,
        leakage_free=leakage_free,
        expected_samples=expected_samples,
        n_bootstrap=n_bootstrap,
        seed=seed,
        generic_controls={"T_Q": "T_generic_Q", "T_QA": "T_generic_QA"},
    )
    result.update({
        "measurement_profile": NATURAL_ROTATION_PROFILE,
        "field_semantics": {
            "accuracy_delta_vs_S": "semantic_accuracy_delta_vs_S",
            "accuracy_delta_vs_generic": "semantic_accuracy_delta_vs_generic",
            "correct_log_prob_vs_S": "semantic_correct_log_prob_vs_S",
            "correct_log_prob_delta_vs_generic": "semantic_correct_log_prob_delta_vs_generic",
        },
        "statistical_unit": "base_sample",
        "n_rotation_views": expected_samples * 4,
        "n_condition_records": expected_samples * 4 * len(NATURAL_ROTATION_CONDITIONS),
        "continuous_metric": "semantic_correct_log_prob",
        "generation_vote_used_for_decision": False,
        "semantic_observations": observations,
    })
    return result
