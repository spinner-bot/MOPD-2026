"""Tests for the isolated G-PC0 procedural mechanism control."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "evaluation"))

from exploratory_teacher_gap import frozen_schedule, validate_manifest_profile
from build_g_pc0_label_rotation import build_bundle as build_rotation_bundle
from g_pc0 import (
    BASE_MANIFEST_SHA256,
    decide_g_pc0,
    decide_g_pc0_label_rotation,
    diagnose_g_pc0,
    label_rotation_mapping,
    validate_g_pc0_assets,
    validate_g_pc0_manifest,
    validate_g_pc0_rotation_manifest,
)
from generate_g_pc0 import generate_bundle
from process_exploratory_teacher_gap import main as process_teacher_gap


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def result_rows(
    baseline_correct: set[int], tqa_correct: set[int], tq_correct: set[int] | None = None
) -> list[dict]:
    rows = []
    tq_correct = tq_correct or baseline_correct
    for index in range(16):
        family = ("directed_path", "sequential_transform", "grid_trace", "set_relation")[index // 4]
        for condition in ("S", "T_Q", "T_QA", "T_generic"):
            correct_set = {
                "S": baseline_correct,
                "T_Q": tq_correct,
                "T_QA": tqa_correct,
                "T_generic": baseline_correct,
            }[condition]
            rows.append({
                "sample_id": f"gpc0-{index + 1:02d}",
                "condition": condition,
                "correct": index in correct_set,
                "positive_control_family": family,
            })
    return rows


def complete_result_rows(samples: list[dict], manifest_hash: str) -> list[dict]:
    rows = []
    schedule = frozen_schedule(samples, manifest_hash)
    for schedule_index, (sample, condition) in enumerate(schedule):
        sample_index = int(sample["sample_id"].split("-")[-1]) - 1
        correct = sample_index < 6 or (condition == "T_QA" and sample_index < 10)
        correct_choice = sample["final_answer"]
        parsed = correct_choice if correct else "ABCD"[("ABCD".index(correct_choice) + 1) % 4]
        scores = {choice: -2.0 for choice in "ABCD"}
        scores[correct_choice] = -0.5 if correct else -1.5
        scores[parsed] = -0.5
        weights = {choice: math.exp(score) for choice, score in scores.items()}
        total = sum(weights.values())
        probabilities = {choice: value / total for choice, value in weights.items()}
        rows.append({
            "sample_id": sample["sample_id"],
            "source_dataset": sample["source_dataset"],
            "source_record_id": sample["source_record_id"],
            "source_image_id": sample["source_image_id"],
            "source_cluster_id": sample["source_cluster_id"],
            "task_family": sample["task_family"],
            "reasoning_depth": sample["reasoning_depth"],
            "experiment_role": sample["experiment_role"],
            "positive_control_family": sample["positive_control_family"],
            "generator_id": sample["generator_id"],
            "condition": condition,
            "schedule_index": schedule_index,
            "rank": schedule_index % 4,
            "world_size": 4,
            "correct": correct,
            "correct_choice": correct_choice,
            "correct_log_prob": scores[correct_choice],
            "parsed_answer": parsed,
            "choice_log_likelihoods": scores,
            "choice_probabilities": probabilities,
            "margin": scores[correct_choice] - max(scores[choice] for choice in "ABCD" if choice != correct_choice),
            "entropy": -sum(value * math.log(value) for value in probabilities.values()),
            "input_tokens": 100,
            "truncated": False,
            "eos_reached": True,
            "error": None,
            "manifest_hash": manifest_hash,
            "model_revision": "frozen-model",
            "processor_revision": "frozen-model",
            "code_commit": "frozen-commit",
            "shell_hash": "frozen-shell",
            "model_dtype": "torch.bfloat16",
            "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
            "generation_config": {"do_sample": False, "constrained_to": list("ABCD")},
            "choice_token_ids": {choice: [index + 1] for index, choice in enumerate("ABCD")},
            "choice_continuation_verified": True,
            "runtime_code_hashes": {"runner": "hash"},
            "image_checksum": sample["image_checksum"],
            "question_hash": hashlib.sha256(sample["final_question"].encode()).hexdigest(),
            "options_hash": hashlib.sha256(json.dumps(
                [sample[f"option_{choice}"] for choice in "ABCD"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()).hexdigest(),
            "prompt_hash": f"prompt-{sample['sample_id']}-{condition}",
        })
    return rows


class GPC0Tests(unittest.TestCase):
    @staticmethod
    def frozen_base_manifest() -> Path:
        return ROOT / "outputs" / "runs" / "20260914T0916-teacher-gap-g-pc0-v1" / "outputs" / "manifest.jsonl"

    def test_generator_builds_balanced_isolated_valid_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = generate_bundle(root / "bundle", "bundle/images", asset_root=root)
            samples = load_jsonl(root / "bundle" / "manifest.jsonl")
            self.assertEqual(validate_g_pc0_manifest(samples), [])
            self.assertEqual(validate_g_pc0_assets(samples, root), [])
            self.assertEqual(validate_manifest_profile(samples, 16, "g_pc0"), [])
            self.assertTrue(validate_manifest_profile(samples, 16, "natural"))
            self.assertEqual(Counter(row["final_answer"] for row in samples), Counter({choice: 4 for choice in "ABCD"}))
            self.assertEqual(Counter(row["positive_control_family"] for row in samples), Counter({
                "directed_path": 4,
                "sequential_transform": 4,
                "grid_trace": 4,
                "set_relation": 4,
            }))
            self.assertEqual(summary["human_verified_records"], 0)
            self.assertEqual(summary["natural_g0_records"], 0)
            self.assertEqual(summary["training_records"], 0)
            for row in samples[12:]:
                self.assertNotEqual(row["solution_trace"][1], "empty set")
                self.assertNotEqual(row["solution_trace"][1], row["solution_trace"][2])

    def test_generator_is_byte_deterministic_for_images_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = generate_bundle(root / "first", "first/images", asset_root=root)
            second = generate_bundle(root / "second", "second/images", asset_root=root)
            self.assertEqual(first["image_sha256"], second["image_sha256"])
            first_rows = load_jsonl(root / "first" / "manifest.jsonl")
            second_rows = load_jsonl(root / "second" / "manifest.jsonl")
            for row in first_rows + second_rows:
                row["image_original"] = Path(row["image_original"]).name
            self.assertEqual(first_rows, second_rows)

    def test_contract_rejects_human_claim_and_guide_answer_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_bundle(root / "bundle", "bundle/images", asset_root=root)
            samples = load_jsonl(root / "bundle" / "manifest.jsonl")
            human_claim = copy.deepcopy(samples)
            human_claim[0]["human_verified"] = True
            self.assertTrue(any("human_verified must be false" in error for error in validate_g_pc0_manifest(human_claim)))
            leaked = copy.deepcopy(samples)
            leaked[0]["guides"][0]["answer"] = leaked[0]["canonical_answer_text"]
            self.assertTrue(any("reveals canonical answer" in error for error in validate_g_pc0_manifest(leaked)))

    def test_asset_gate_rejects_mutated_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_bundle(root / "bundle", "bundle/images", asset_root=root)
            samples = load_jsonl(root / "bundle" / "manifest.jsonl")
            image = root / samples[0]["image_original"]
            image.write_bytes(image.read_bytes() + b"mutation")
            self.assertTrue(any("checksum differs" in error for error in validate_g_pc0_assets(samples, root)))

    def test_preflight_cli_checks_profile_assets_and_token_lengths(self):
        class Tokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                if text in "ABCD" and len(text) == 1:
                    return [ord(text)]
                return list(range(10))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_bundle(root / "bundle", "bundle/images", asset_root=root)
            tokenizer_root = root / "tokenizer"
            tokenizer_root.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                (tokenizer_root / name).write_text(name, encoding="utf-8")
            summary = root / "preflight.json"
            argv = [
                "process_exploratory_teacher_gap.py",
                "validate-positive-control-manifest",
                "--manifest", str(root / "bundle" / "manifest.jsonl"),
                "--image-root", str(root),
                "--tokenizer", str(tokenizer_root),
                "--summary", str(summary),
            ]
            with patch("sys.argv", argv), patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ), patch("builtins.print"):
                process_teacher_gap()
            result = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["generic_length_check"]["passes"])
            self.assertFalse(result["performance_results_read"])

    def test_frozen_decision_tree(self):
        ceiling = decide_g_pc0(result_rows(set(range(14)), set(range(14))))
        self.assertEqual(ceiling["decision"], "CEILING_INCONCLUSIVE")

        mechanism = decide_g_pc0(result_rows(set(range(6)), set(range(10))))
        self.assertEqual(mechanism["decision"], "MECHANISM_PASS")
        self.assertEqual(mechanism["tqa_vs_s"], {"gains": 4, "regressions": 0, "net_flips": 4})
        self.assertFalse(mechanism["natural_image_claim_authorized"])

        weak = decide_g_pc0(result_rows(set(range(6)), set(range(8))))
        self.assertEqual(weak["decision"], "INGESTION_OR_RECIPE_FAILURE")

        regression = decide_g_pc0(result_rows(set(range(6)), {3, 4, 5, 6, 7, 8, 9}))
        self.assertEqual(regression["tqa_vs_s"]["gains"], 4)
        self.assertEqual(regression["tqa_vs_s"]["regressions"], 3)
        self.assertEqual(regression["decision"], "INGESTION_OR_RECIPE_FAILURE")

    def test_decision_cli_revalidates_manifest_and_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_bundle(root / "bundle", "bundle/images", asset_root=root)
            manifest = root / "bundle" / "manifest.jsonl"
            samples = load_jsonl(manifest)
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            parsed = root / "results.jsonl"
            parsed.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in complete_result_rows(samples, manifest_hash)),
                encoding="utf-8",
            )
            output = root / "decision.json"
            argv = [
                "process_exploratory_teacher_gap.py",
                "decide-positive-control",
                "--input", str(parsed),
                "--manifest", str(manifest),
                "--image-root", str(root),
                "--output", str(output),
            ]
            with patch("sys.argv", argv), patch("builtins.print"):
                process_teacher_gap()
            decision = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(decision["decision"], "MECHANISM_PASS")
            self.assertEqual(decision["manifest_sha256"], manifest_hash)
            self.assertFalse(decision["natural_image_claim_authorized"])
            diagnostic = diagnose_g_pc0(load_jsonl(parsed), decision["decision"])
            self.assertTrue(all(not values for values in diagnostic["generation_vs_score_argmax_mismatches"].values()))
            self.assertFalse(diagnostic["decision_changed"])

    def test_label_rotation_bundle_is_exact_and_mutations_fail(self):
        base_manifest = self.frozen_base_manifest()
        self.assertEqual(hashlib.sha256(base_manifest.read_bytes()).hexdigest(), BASE_MANIFEST_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rotation"
            summary = build_rotation_bundle(base_manifest, output)
            base = load_jsonl(base_manifest)
            rows = load_jsonl(output / "manifest.jsonl")
            self.assertEqual(summary["records"], 64)
            self.assertEqual(validate_g_pc0_rotation_manifest(rows, base, BASE_MANIFEST_SHA256), [])
            self.assertEqual(Counter(row["final_answer"] for row in rows), Counter({choice: 16 for choice in "ABCD"}))
            for row in rows:
                self.assertEqual(row["displayed_label_to_semantic_option"], label_rotation_mapping(row["label_rotation"]))
                self.assertFalse(row["eligible_for_natural_g0"])
                self.assertFalse(row["eligible_for_training"])
            mutated = copy.deepcopy(rows)
            mutated[0]["option_A"] = "mutated"
            self.assertTrue(validate_g_pc0_rotation_manifest(mutated, base, BASE_MANIFEST_SHA256))
            reordered = rows[::-1]
            self.assertTrue(any("order" in error for error in validate_g_pc0_rotation_manifest(reordered, base, BASE_MANIFEST_SHA256)))
            self.assertTrue(validate_g_pc0_rotation_manifest(rows, base, "0" * 64))

    def test_rotation_aggregation_cancels_static_label_prior_and_applies_rule(self):
        records = []
        label_bias = {"A": 20.0, "B": 10.0, "C": 0.0, "D": -10.0}
        conditions = ("S", "T_Q", "T_QA", "T_generic")
        for index in range(16):
            correct_semantic = "ABCD"[index % 4]
            wrong_semantic = "ABCD"[(index % 4 + 1) % 4]
            for condition in conditions:
                target = correct_semantic if (
                    index < 6 or (condition == "T_QA" and index < 10)
                ) else wrong_semantic
                for rotation in range(4):
                    mapping = label_rotation_mapping(rotation)
                    scores = {
                        displayed: label_bias[displayed] + (40.0 if semantic == target else 0.0)
                        for displayed, semantic in mapping.items()
                    }
                    records.append({
                        "base_sample_id": f"gpc0-{index + 1:02d}",
                        "condition": condition,
                        "label_rotation": rotation,
                        "displayed_label_to_semantic_option": mapping,
                        "choice_log_likelihoods": scores,
                        "parsed_answer": max("ABCD", key=lambda choice: scores[choice]),
                        "correct_semantic_option_id": correct_semantic,
                        "positive_control_family": ("directed_path", "sequential_transform", "grid_trace", "set_relation")[index // 4],
                    })
        decision = decide_g_pc0_label_rotation(records)
        self.assertEqual(decision["decision"], "LABEL_INTERFACE_MASKED_G_SIGNAL")
        self.assertEqual(decision["condition_correct_after_semantic_aggregation"]["S"], 6)
        self.assertEqual(decision["condition_correct_after_semantic_aggregation"]["T_QA"], 10)
        self.assertEqual(decision["tqa_vs_s"], {"gains": 4, "regressions": 0, "net_flips": 4})
        self.assertEqual(decision["gpc0_v1_frozen_decision"], "INGESTION_OR_RECIPE_FAILURE")
        self.assertFalse(decision["gpc0_v1_decision_changed"])
        self.assertFalse(decision["natural_image_claim_authorized"])
        self.assertFalse(decision["training_data_authorized"])


if __name__ == "__main__":
    unittest.main()
