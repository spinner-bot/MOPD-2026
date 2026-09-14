"""Contract tests for the frozen 1017 exploratory teacher-gap protocol."""

import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))

from exploratory_teacher_gap import (
    CONDITIONS,
    aggregate_natural_rotation_results,
    decide_teacher_gap,
    decide_teacher_gap_label_rotation,
    expand_natural_rotation_manifest,
    exact_mcnemar_one_sided,
    frozen_schedule,
    validate_manifest_bundle,
    validate_manifest_profile,
    validate_locked_manifest,
    validate_natural_rotation_manifest,
    validate_partition_isolation,
    validate_result_records,
    validate_results_against_manifest,
)
from teacher_gap_inputs import COMMON_SHELL, render_teacher_gap_input, validate_generic_length
from process_exploratory_teacher_gap import enrich_raw_record, main as process_teacher_gap
from run_exploratory_teacher_gap import (
    allowed_token_function,
    choice_token_sequences,
    conditional_choice_scores,
    main as run_teacher_gap,
    runtime_code_hashes,
    validate_choice_continuations,
)


def manifest_sample(index: int) -> dict:
    label = "ABCD"[index % 4]
    sample = {
        "sample_id": f"sample-{index:03d}",
        "source_image_id": f"image-{index:03d}",
        "source_cluster_id": f"cluster-{index:03d}",
        "source_dataset": "M3VQA",
        "source_record_id": f"m3vqa-{index:03d}",
        "image_original": f"images/{index:03d}.jpg",
        "image_checksum": f"sha256-{index:03d}",
        "final_question": "Which item completes the visible relation?",
        "option_A": f"alpha-{index}",
        "option_B": f"beta-{index}",
        "option_C": f"gamma-{index}",
        "option_D": f"delta-{index}",
        "final_answer": label,
        "generic_context": "Inspect the relevant visible elements, compare the choices, and verify the conclusion.",
        "knowledge_dependency_label": "image_only",
        "task_family": "R",
        "full_image_reference_valid": True,
        "unique_answer": True,
        "human_verified": True,
        "human_reviewer": "reviewer-1",
        "human_reviewed_at": "2026-09-11T18:00:00+08:00",
        "answer_leakage": 0,
        "external_fact_injected": 0,
        "required_evidence": ["left relation", "right relation"],
        "difficult_visual_evidence": [],
        "reasoning_depth": 2,
        "minimal_reasoning_graph": {
            "nodes": [
                {"id": "e1", "type": "evidence", "claim": "left relation"},
                {"id": "e2", "type": "evidence", "claim": "right relation"},
                {"id": "h1", "type": "intermediate", "claim": "combined relation"},
                {"id": "a", "type": "answer", "claim": "selected item"},
            ],
            "edges": [["e1", "h1"], ["e2", "h1"], ["h1", "a"]],
        },
        "guides": [
            {"question": "What relation is visible on the left?", "answer": "The first relation.", "supports_next": "h1"},
            {"question": "How does it apply on the right?", "answer": "In the same order.", "supports_next": "a"},
        ],
    }
    sample["canonical_answer_text"] = sample[f"option_{label}"]
    return sample


def result_records(tq_gains: int, tqa_gains: int, generic_gains: int = 1, logp_tq: float = 0.30, logp_tqa: float = 0.35) -> list[dict]:
    rows = []
    schedule_index = 0
    for index in range(96):
        baseline = index < 48
        correct_by_condition = {
            "S": baseline,
            "T_Q": baseline or 48 <= index < 48 + tq_gains,
            "T_QA": baseline or 48 <= index < 48 + tqa_gains,
            "T_generic": baseline or 48 <= index < 48 + generic_gains,
        }
        deltas = {"S": 0.0, "T_Q": logp_tq, "T_QA": logp_tqa, "T_generic": 0.02}
        for condition in CONDITIONS:
            rows.append({
                "sample_id": f"sample-{index:03d}",
                "source_dataset": "M3VQA",
                "source_record_id": f"m3vqa-{index:03d}",
                "source_image_id": f"image-{index:03d}",
                "source_cluster_id": f"cluster-{index:03d}",
                "task_family": "R",
                "reasoning_depth": 2,
                "condition": condition,
                "schedule_index": schedule_index,
                "rank": schedule_index % 4,
                "world_size": 4,
                "correct": correct_by_condition[condition],
                "correct_choice": "A",
                "correct_log_prob": -1.0 + deltas[condition],
                "parsed_answer": "A" if correct_by_condition[condition] else "B",
                "choice_log_likelihoods": {"A": -1.0 + deltas[condition], "B": -2.0, "C": -3.0, "D": -4.0},
                "choice_probabilities": {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
                "truncated": False,
                "eos_reached": True,
                "error": None,
                "manifest_hash": "manifest-1",
                "model_revision": "model-1",
                "processor_revision": "processor-1",
                "code_commit": "commit-1",
                "shell_hash": "shell-1",
                "model_dtype": "torch.bfloat16",
                "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
                "generation_config": {"do_sample": False, "constrained_to": ["A", "B", "C", "D"]},
                "choice_token_ids": {"A": [1], "B": [2], "C": [3], "D": [4]},
                "choice_continuation_verified": True,
                "runtime_code_hashes": {
                    "exploratory_teacher_gap.py": "hash-1",
                    "reasoning_graph_contract.py": "hash-2",
                    "run_exploratory_teacher_gap.py": "hash-3",
                    "teacher_gap_inputs.py": "hash-4",
                },
                "image_checksum": f"image-sha-{index}",
                "question_hash": f"question-sha-{index}",
                "options_hash": f"options-sha-{index}",
                "prompt_hash": f"prompt-{index}-{condition}",
            })
            schedule_index += 1
    return rows


class ExploratoryTeacherGapTests(unittest.TestCase):
    def test_natural_rotation_validate_and_decide_cli_end_to_end(self):
        samples = [manifest_sample(index) for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            base.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in samples), encoding="utf-8")
            base_hash = hashlib.sha256(base.read_bytes()).hexdigest()
            expanded = expand_natural_rotation_manifest(samples, base_hash)
            manifest = root / "rotation.jsonl"
            manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in expanded), encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            schedule = frozen_schedule(expanded, manifest_hash)
            choice_tokens = {choice: [index + 1] for index, choice in enumerate("ABCD")}
            label_bias = {"A": 20.0, "B": 10.0, "C": 0.0, "D": -10.0}
            raw_rows = []
            for schedule_index, (sample, condition) in enumerate(schedule):
                mapping = sample["displayed_label_to_semantic_option"]
                correct_semantic = sample["correct_semantic_option_id"]
                target = correct_semantic if condition == "T_QA" else "ABCD"[("ABCD".index(correct_semantic) + 1) % 4]
                scores = {
                    displayed: label_bias[displayed] + (40.0 if semantic == target else 0.0)
                    for displayed, semantic in mapping.items()
                }
                parsed = max("ABCD", key=lambda choice: scores[choice])
                row = {
                    "sample_id": sample["sample_id"],
                    "source_dataset": sample["source_dataset"],
                    "source_record_id": sample["source_record_id"],
                    "source_image_id": sample["source_image_id"],
                    "source_cluster_id": sample["source_cluster_id"],
                    "task_family": sample["task_family"],
                    "reasoning_depth": sample["reasoning_depth"],
                    "condition": condition,
                    "schedule_index": schedule_index,
                    "rank": schedule_index % 4,
                    "world_size": 4,
                    "manifest_hash": manifest_hash,
                    "model_revision": "model-1",
                    "processor_revision": "model-1",
                    "code_commit": "commit-1",
                    "runtime_code_hashes": {"runner": "hash"},
                    "prompt_hash": f"prompt-{sample['sample_id']}-{condition}",
                    "shell_hash": "shell-1",
                    "question_hash": hashlib.sha256(sample["final_question"].encode()).hexdigest(),
                    "options_hash": hashlib.sha256(json.dumps(
                        [sample[f"option_{choice}"] for choice in "ABCD"],
                        ensure_ascii=False, separators=(",", ":"),
                    ).encode()).hexdigest(),
                    "correct_choice": sample["final_answer"],
                    "choice_token_ids": choice_tokens,
                    "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
                    "generation_config": {"do_sample": False, "constrained_to": list("ABCD")},
                    "image_checksum": sample["image_checksum"],
                    "choice_log_likelihoods": scores,
                    "choice_continuation_verified": True,
                    "generated_token_ids": choice_tokens[parsed] + [99],
                    "generated_tokens": 2,
                    "eos_reached": True,
                    "truncated": False,
                    "raw_generation": parsed,
                    "model_dtype": "torch.bfloat16",
                    "error": None,
                }
                for field in (
                    "experiment_role", "base_sample_id", "base_manifest_sha256", "label_rotation",
                    "rotation_spec_hash", "semantic_options", "correct_semantic_option_id",
                    "displayed_label_to_semantic_option",
                ):
                    row[field] = sample[field]
                raw_rows.append(row)
            raw = root / "raw.jsonl"
            raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows), encoding="utf-8")
            parsed = root / "parsed.jsonl"
            gate = root / "gate.json"
            argv = [
                "process_exploratory_teacher_gap.py", "validate", "--inputs", str(raw),
                "--manifest", str(manifest), "--base-manifest", str(base),
                "--manifest-profile", "natural_rotation", "--expected-samples", "32",
                "--parsed-output", str(parsed), "--summary", str(gate),
            ]
            with patch("sys.argv", argv), patch("builtins.print"):
                process_teacher_gap()
            self.assertEqual(json.loads(gate.read_text())["status"], "passed")
            decision_path = root / "decision.json"
            argv = [
                "process_exploratory_teacher_gap.py", "decide-natural-label-rotation",
                "--input", str(parsed), "--manifest", str(manifest),
                "--base-manifest", str(base), "--expected-samples", "8",
                "--leakage-free", "--output", str(decision_path),
            ]
            with patch("sys.argv", argv), patch("builtins.print"):
                process_teacher_gap()
            decision = json.loads(decision_path.read_text())
            self.assertEqual(decision["decision"], "GO_T_QA")
            self.assertEqual(decision["n_samples"], 8)
            self.assertEqual(decision["n_condition_records"], 128)

    def test_natural_rotation_preflight_cli_binds_base_assets_and_tokenizer(self):
        class Tokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [ord(text)] if text in "ABCD" and len(text) == 1 else list(range(20))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            samples = [manifest_sample(index) for index in range(8)]
            for index, sample in enumerate(samples):
                content = f"image-{index}".encode()
                (image_dir / f"{index:03d}.jpg").write_bytes(content)
                sample["image_checksum"] = hashlib.sha256(content).hexdigest()
            base = root / "base.jsonl"
            base.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in samples), encoding="utf-8")
            base_hash = hashlib.sha256(base.read_bytes()).hexdigest()
            expanded = expand_natural_rotation_manifest(samples, base_hash)
            manifest = root / "rotation.jsonl"
            manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in expanded), encoding="utf-8")
            tokenizer_root = root / "tokenizer"
            tokenizer_root.mkdir()
            summary = root / "preflight.json"
            argv = [
                "process_exploratory_teacher_gap.py", "validate-natural-label-rotation-manifest",
                "--manifest", str(manifest), "--base-manifest", str(base),
                "--expected-samples", "8", "--image-root", str(root),
                "--tokenizer", str(tokenizer_root), "--summary", str(summary),
            ]
            with patch("sys.argv", argv), patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ), patch("builtins.print"):
                process_teacher_gap()
            result = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["manifest_records"], 32)
            self.assertFalse(result["performance_results_read"])
            self.assertFalse(result["gpu_used"])

    def test_natural_rotation_manifest_and_semantic_aggregation(self):
        samples = [manifest_sample(index) for index in range(8)]
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples).encode()
        base_hash = hashlib.sha256(payload).hexdigest()
        expanded = expand_natural_rotation_manifest(samples, base_hash)
        self.assertEqual(validate_natural_rotation_manifest(expanded, samples, base_hash, 32), [])
        self.assertEqual(validate_manifest_profile(
            expanded, 32, "natural_rotation", samples, base_hash
        ), [])
        self.assertEqual(len(expanded), 32)
        self.assertEqual({row["source_record_id"] for row in expanded}, {row["source_record_id"] for row in samples})
        mutated = json.loads(json.dumps(expanded))
        mutated[0]["displayed_label_to_semantic_option"]["A"] = "D"
        self.assertTrue(validate_natural_rotation_manifest(mutated, samples, base_hash, 32))

        label_bias = {"A": 20.0, "B": 10.0, "C": 0.0, "D": -10.0}
        records = []
        for row in expanded:
            mapping = row["displayed_label_to_semantic_option"]
            correct = row["correct_semantic_option_id"]
            wrong = "ABCD"[("ABCD".index(correct) + 1) % 4]
            for condition in CONDITIONS:
                target = correct if condition == "T_QA" else wrong
                scores = {
                    displayed: label_bias[displayed] + (40.0 if semantic == target else 0.0)
                    for displayed, semantic in mapping.items()
                }
                records.append({
                    "base_sample_id": row["base_sample_id"],
                    "condition": condition,
                    "label_rotation": row["label_rotation"],
                    "displayed_label_to_semantic_option": mapping,
                    "choice_log_likelihoods": scores,
                    "parsed_answer": max("ABCD", key=lambda choice: scores[choice]),
                    "correct_semantic_option_id": correct,
                    "source_cluster_id": row["source_cluster_id"],
                })
        observations, errors = aggregate_natural_rotation_results(records, expected_samples=8)
        self.assertEqual(errors, [])
        self.assertEqual(len(observations), 32)
        self.assertEqual(sum(row["correct"] for row in observations if row["condition"] == "S"), 0)
        self.assertEqual(sum(row["correct"] for row in observations if row["condition"] == "T_QA"), 8)
        self.assertTrue(all("mapped_generation_vote_tie" in row for row in observations))
        decision = decide_teacher_gap_label_rotation(
            records, leakage_free=True, expected_samples=8, n_bootstrap=100
        )
        self.assertEqual(decision["decision"], "GO_T_QA")
        self.assertEqual(decision["n_samples"], 8)
        self.assertEqual(decision["n_rotation_views"], 32)
        self.assertEqual(decision["n_condition_records"], 128)
        self.assertFalse(decision["generation_vote_used_for_decision"])

    def test_validate_cli_binds_complete_raw_to_manifest(self):
        samples = [manifest_sample(index) for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "canary.jsonl"
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples),
                encoding="utf-8",
            )
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            schedule = frozen_schedule(samples, manifest_hash)
            raw_rows = []
            choice_tokens = {"A": [1], "B": [2], "C": [3], "D": [4]}
            for schedule_index, (sample, condition) in enumerate(schedule):
                answer = sample["final_answer"]
                raw_rows.append({
                    "sample_id": sample["sample_id"],
                    "source_dataset": sample["source_dataset"],
                    "source_record_id": sample["source_record_id"],
                    "source_image_id": sample["source_image_id"],
                    "source_cluster_id": sample["source_cluster_id"],
                    "task_family": sample["task_family"],
                    "reasoning_depth": sample["reasoning_depth"],
                    "condition": condition,
                    "schedule_index": schedule_index,
                    "rank": schedule_index % 4,
                    "world_size": 4,
                    "manifest_hash": manifest_hash,
                    "model_revision": "model-1",
                    "processor_revision": "model-1",
                    "code_commit": "commit-1",
                    "prompt_hash": f"prompt-{sample['sample_id']}-{condition}",
                    "shell_hash": "shell-1",
                    "question_hash": hashlib.sha256(sample["final_question"].encode()).hexdigest(),
                    "options_hash": hashlib.sha256(json.dumps(
                        [sample[f"option_{choice}"] for choice in "ABCD"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()).hexdigest(),
                    "correct_choice": answer,
                    "choice_token_ids": choice_tokens,
                    "choice_continuation_verified": True,
                    "runtime_code_hashes": {"runner": "hash"},
                    "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
                    "generation_config": {"do_sample": False, "constrained_to": list("ABCD")},
                    "image_checksum": sample["image_checksum"],
                    "choice_log_likelihoods": {
                        choice: (0.0 if choice == answer else -1.0) for choice in "ABCD"
                    },
                    "generated_token_ids": choice_tokens[answer] + [99],
                    "generated_tokens": 2,
                    "eos_reached": True,
                    "truncated": False,
                    "raw_generation": answer,
                    "model_dtype": "torch.bfloat16",
                    "error": None,
                })
            raw = root / "raw.jsonl"
            raw.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            parsed = root / "parsed.jsonl"
            summary = root / "summary.json"
            argv = [
                "process_exploratory_teacher_gap.py",
                "validate",
                "--inputs", str(raw),
                "--manifest", str(manifest),
                "--expected-samples", "8",
                "--parsed-output", str(parsed),
                "--summary", str(summary),
            ]
            with patch("sys.argv", argv), patch("builtins.print"):
                process_teacher_gap()
            gate = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(gate["status"], "passed")
            self.assertEqual(gate["manifest_sha256"], manifest_hash)
            self.assertEqual(len(parsed.read_text(encoding="utf-8").splitlines()), 32)

    def test_runner_refuses_a_commit_changed_after_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            manifest = root / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            argv = [
                "run_exploratory_teacher_gap.py",
                "--manifest", str(manifest),
                "--output", str(root / "output.jsonl"),
                "--model", str(model),
                "--model-revision", "model-revision",
                "--code-commit", "submitted-commit",
                "--image-root", str(root),
            ]
            with patch("sys.argv", argv), patch(
                "run_exploratory_teacher_gap.git_commit", return_value="newer-commit"
            ):
                with self.assertRaisesRegex(SystemExit, "code commit changed after submission"):
                    run_teacher_gap()

    def test_runner_refuses_to_overwrite_raw_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            manifest = root / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            output = root / "raw.jsonl"
            output.write_text("immutable\n", encoding="utf-8")
            argv = [
                "run_exploratory_teacher_gap.py",
                "--manifest", str(manifest),
                "--output", str(output),
                "--model", str(model),
                "--model-revision", "model-revision",
                "--code-commit", "same-commit",
                "--image-root", str(root),
            ]
            with patch("sys.argv", argv), patch(
                "run_exploratory_teacher_gap.git_commit", return_value="same-commit"
            ):
                with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                    run_teacher_gap()
            self.assertEqual(output.read_text(encoding="utf-8"), "immutable\n")

    def test_runner_records_all_runtime_code_hashes(self):
        hashes = runtime_code_hashes()
        self.assertEqual(
            set(hashes),
            {
                "exploratory_teacher_gap.py",
                "g_pc0.py",
                "reasoning_graph_contract.py",
                "run_exploratory_teacher_gap.py",
                "teacher_gap_inputs.py",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_choice_continuation_must_preserve_template_boundary(self):
        class CharacterTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [ord(value) for value in text]

        sequences = {choice: [ord(choice)] for choice in "ABCD"}
        validate_choice_continuations(CharacterTokenizer(), "assistant\n", sequences)
        with self.assertRaisesRegex(ValueError, "differs from frozen"):
            validate_choice_continuations(
                CharacterTokenizer(), "assistant\n", {**sequences, "A": [999]}
            )

        class RetokenizingTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [1] if text == "prefix" else [2]

        with self.assertRaisesRegex(ValueError, "retokenizes"):
            validate_choice_continuations(
                RetokenizingTokenizer(), "prefix", {choice: [index] for index, choice in enumerate("ABCD")}
            )

    def test_choice_scoring_extends_multimodal_token_types(self):
        import torch

        class Model:
            def __call__(self, **kwargs):
                self.assert_shapes(kwargs)
                return type("Output", (), {"logits": torch.zeros(1, 3, 128)})()

            @staticmethod
            def assert_shapes(kwargs):
                assert tuple(kwargs["input_ids"].shape) == (1, 3)
                assert tuple(kwargs["attention_mask"].shape) == (1, 3)
                assert tuple(kwargs["mm_token_type_ids"].shape) == (1, 3)
                assert kwargs["mm_token_type_ids"][0, -1].item() == 0

        inputs = {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "mm_token_type_ids": torch.tensor([[1, 0]]),
            "pixel_values": torch.zeros(5, 7),
        }
        scores = conditional_choice_scores(
            Model(), inputs, {choice: [32 + index] for index, choice in enumerate("ABCD")}, torch
        )
        self.assertEqual(set(scores), set("ABCD"))
        self.assertEqual(tuple(inputs["mm_token_type_ids"].shape), (1, 2))

    def test_locked_manifest_fails_closed_on_overlap_and_balance(self):
        samples = [manifest_sample(index) for index in range(96)]
        self.assertEqual(validate_locked_manifest(samples), [])
        samples[95]["source_image_id"] = samples[0]["source_image_id"]
        samples[95]["final_answer"] = "A"
        errors = validate_locked_manifest(samples)
        self.assertTrue(any("source_image_id" in error for error in errors))
        self.assertTrue(any("balanced" in error for error in errors))
        self.assertTrue(validate_partition_isolation({"dev": samples[:1], "locked": samples[:1]}))

    def test_locked_manifest_rejects_duplicate_near_duplicate_cluster(self):
        samples = [manifest_sample(index) for index in range(96)]
        samples[95]["source_cluster_id"] = samples[0]["source_cluster_id"]
        self.assertTrue(any(
            "source_cluster_id values must be unique" in error
            for error in validate_locked_manifest(samples)
        ))

    def test_manifest_bundle_requires_isolated_8_32_96_partitions(self):
        partitions = {
            "canary": [manifest_sample(index) for index in range(8)],
            "dev": [manifest_sample(index) for index in range(8, 40)],
            "locked": [manifest_sample(index) for index in range(40, 136)],
        }
        self.assertEqual(validate_manifest_bundle(partitions), [])
        partitions["locked"][0]["source_cluster_id"] = partitions["canary"][0]["source_cluster_id"]
        errors = validate_manifest_bundle(partitions)
        self.assertTrue(any("overlaps canary and locked" in error for error in errors))

    def test_locked_manifest_requires_human_provenance_and_graph_contract(self):
        samples = [manifest_sample(index) for index in range(96)]
        samples[0]["human_reviewer"] = ""
        samples[0]["human_reviewed_at"] = "2026-09-11T18:00:00"
        samples[1]["minimal_reasoning_graph"]["edges"] = [["missing", "a"]]
        errors = validate_locked_manifest(samples)
        self.assertTrue(any("human_reviewer" in error for error in errors))
        self.assertTrue(any("requires timezone" in error for error in errors))
        self.assertTrue(any("unknown_endpoint" in error for error in errors))

    def test_locked_manifest_preserves_answer_and_r_pr_evidence_semantics(self):
        samples = [manifest_sample(index) for index in range(96)]
        samples[0]["canonical_answer_text"] = "not the correct option"
        samples[1]["required_evidence"] = ["other evidence", "right relation"]
        samples[2]["task_family"] = "PR"
        errors = validate_locked_manifest(samples)
        self.assertTrue(any("correct option must equal canonical" in error for error in errors))
        self.assertTrue(any("required_evidence must match graph" in error for error in errors))
        self.assertTrue(any("PR requires difficult" in error for error in errors))

    def test_common_shell_changes_only_privileged_block(self):
        sample = manifest_sample(0)
        rendered = {condition: render_teacher_gap_input(sample, condition) for condition in CONDITIONS}
        self.assertEqual({item.shell_hash for item in rendered.values()}, {rendered["S"].shell_hash})
        self.assertNotEqual(rendered["S"].prompt_hash, rendered["T_Q"].prompt_hash)
        self.assertIn("Guide 1:", rendered["T_Q"].privileged_context)
        self.assertNotIn("Intermediate answer", rendered["T_Q"].privileged_context)
        self.assertIn("Intermediate answer 1:", rendered["T_QA"].privileged_context)
        self.assertIn("{privileged_context}", COMMON_SHELL)

    def test_generic_length_uses_experiment_tokenizer_contract(self):
        samples = [manifest_sample(0), manifest_sample(1)]
        for sample in samples:
            sample["generic_context"] = "one two three four five six seven eight nine ten"
            sample["guides"] = [{"question": "one two three four five six seven eight?", "answer": "unused"}]
        result = validate_generic_length(samples, lambda text: len(text.split()), tolerance=0.20)
        self.assertTrue(result["passes"])

    def test_exact_mcnemar_is_directional(self):
        baseline = [False] * 16
        treatment = [True] * 16
        forward = exact_mcnemar_one_sided(baseline, treatment)
        reverse = exact_mcnemar_one_sided(treatment, baseline)
        self.assertAlmostEqual(forward["p_value"], 1 / 65536)
        self.assertGreater(reverse["p_value"], 0.99)

    def test_incomplete_results_are_invalid(self):
        rows = result_records(tq_gains=16, tqa_gains=18)
        rows.pop()
        self.assertTrue(validate_result_records(rows))
        self.assertEqual(
            decide_teacher_gap(rows, leakage_free=True, n_bootstrap=200)["decision"],
            "INVALID",
        )

    def test_result_gate_rejects_condition_and_schedule_drift(self):
        rows = result_records(tq_gains=16, tqa_gains=18)
        rows[0]["chat_template"]["enable_thinking"] = True
        rows[1]["source_cluster_id"] = "wrong-cluster"
        rows[2]["schedule_index"] = rows[1]["schedule_index"]
        rows[3]["prompt_hash"] = rows[2]["prompt_hash"]
        rows[4]["task_family"] = "P"
        rows[5]["reasoning_depth"] = "2"
        errors = validate_result_records(rows)
        self.assertTrue(any("chat_template" in error for error in errors))
        self.assertTrue(any("source_cluster_id" in error for error in errors))
        self.assertTrue(any("schedule_index" in error for error in errors))
        self.assertTrue(any("distinct prompt_hash" in error for error in errors))
        self.assertTrue(any("task_family must be R or PR" in error for error in errors))
        self.assertTrue(any("reasoning_depth must be an integer" in error for error in errors))

    def test_result_records_are_bound_to_exact_manifest_metadata(self):
        samples = [manifest_sample(index) for index in range(96)]
        manifest_hash = "frozen-manifest-hash"
        rows = result_records(tq_gains=16, tqa_gains=18)
        expected_by_id = {sample["sample_id"]: sample for sample in samples}
        for row in rows:
            sample = expected_by_id[row["sample_id"]]
            row.update({
                "manifest_hash": manifest_hash,
                "source_dataset": sample["source_dataset"],
                "source_record_id": sample["source_record_id"],
                "source_image_id": sample["source_image_id"],
                "source_cluster_id": sample["source_cluster_id"],
                "task_family": sample["task_family"],
                "reasoning_depth": sample["reasoning_depth"],
                "image_checksum": sample["image_checksum"],
                "correct_choice": sample["final_answer"],
                "question_hash": hashlib.sha256(sample["final_question"].encode()).hexdigest(),
                "options_hash": hashlib.sha256(json.dumps(
                    [sample[f"option_{choice}"] for choice in "ABCD"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()).hexdigest(),
            })
        schedule = {
            (sample["sample_id"], condition): index
            for index, (sample, condition) in enumerate(frozen_schedule(samples, manifest_hash))
        }
        for row in rows:
            row["schedule_index"] = schedule[(row["sample_id"], row["condition"])]
            row["rank"] = row["schedule_index"] % 4
        self.assertEqual(validate_results_against_manifest(rows, samples, manifest_hash), [])
        rows[0]["options_hash"] = "tampered"
        rows[1]["manifest_hash"] = "other-manifest"
        rows[2]["schedule_index"], rows[3]["schedule_index"] = (
            rows[3]["schedule_index"], rows[2]["schedule_index"]
        )
        errors = validate_results_against_manifest(rows, samples, manifest_hash)
        self.assertTrue(any("options_hash differs" in error for error in errors))
        self.assertTrue(any("manifest_hash differs" in error for error in errors))
        self.assertTrue(any("schedule_index differs" in error for error in errors))

    def test_strong_tq_go(self):
        decision = decide_teacher_gap(
            result_records(tq_gains=16, tqa_gains=18),
            leakage_free=True,
            n_bootstrap=500,
        )
        self.assertEqual(decision["decision"], "GO_T_Q")

    def test_tqa_pivot_when_questions_alone_are_weak(self):
        decision = decide_teacher_gap(
            result_records(tq_gains=4, tqa_gains=16, logp_tq=0.05, logp_tqa=0.30),
            leakage_free=True,
            n_bootstrap=500,
        )
        self.assertEqual(decision["decision"], "GO_T_QA")

    def test_stop_current_recipe_when_oracle_is_null(self):
        decision = decide_teacher_gap(
            result_records(tq_gains=0, tqa_gains=0, generic_gains=0, logp_tq=0.0, logp_tqa=0.0),
            leakage_free=True,
            n_bootstrap=500,
        )
        self.assertEqual(decision["decision"], "STOP_CURRENT_RECIPE")

    def test_leakage_forces_revise(self):
        decision = decide_teacher_gap(
            result_records(tq_gains=16, tqa_gains=18),
            leakage_free=False,
            n_bootstrap=500,
        )
        self.assertEqual(decision["decision"], "REVISE")

    def test_frozen_schedule_is_deterministic_and_complete(self):
        samples = [manifest_sample(0), manifest_sample(1)]
        first = [(row["sample_id"], condition) for row, condition in frozen_schedule(samples, "hash")]
        second = [(row["sample_id"], condition) for row, condition in frozen_schedule(samples, "hash")]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)

    def test_choice_trie_allows_only_complete_labels_then_eos(self):
        allowed = allowed_token_function(3, {"A": [10], "B": [11], "C": [12], "D": [13]}, {99})

        class FakeIds:
            def __init__(self, values): self.values = values
            def __getitem__(self, item): return FakeIds(self.values[item])
            def tolist(self): return self.values

        self.assertEqual(allowed(0, FakeIds([1, 2, 3])), [10, 11, 12, 13])
        self.assertEqual(allowed(0, FakeIds([1, 2, 3, 10])), [99])

    def test_choice_scores_require_equal_length_label_tokens(self):
        class UnequalTokenizer:
            def encode(self, value, add_special_tokens=False):
                return [ord(value)] if value != "D" else [68, 69]

        with self.assertRaisesRegex(ValueError, "equal lengths"):
            choice_token_sequences(UnequalTokenizer())

        rows = result_records(tq_gains=1, tqa_gains=1)
        rows[0]["choice_token_ids"]["D"] = [4, 5]
        self.assertTrue(any(
            "choice_token_ids must have equal lengths" in error
            for error in validate_result_records(rows)
        ))

    def test_raw_result_enrichment_is_deterministic(self):
        row = result_records(tq_gains=1, tqa_gains=1)[0]
        row.update({
            "correct_choice": "A",
            "choice_token_ids": {"A": [10], "B": [11], "C": [12], "D": [13]},
            "generated_token_ids": [10, 99],
            "raw_generation": "A",
        })
        enriched = enrich_raw_record(row)
        self.assertEqual(enriched["parsed_answer"], "A")
        self.assertTrue(enriched["correct"])
        self.assertAlmostEqual(sum(enriched["choice_probabilities"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
