"""Contract tests for the frozen 1016 exploratory teacher-gap protocol."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))

from exploratory_teacher_gap import (
    CONDITIONS,
    decide_teacher_gap,
    exact_mcnemar_one_sided,
    validate_locked_manifest,
    validate_partition_isolation,
    validate_result_records,
)
from teacher_gap_inputs import COMMON_SHELL, render_teacher_gap_input, validate_generic_length
from process_exploratory_teacher_gap import enrich_raw_record
from run_exploratory_teacher_gap import allowed_token_function, frozen_schedule


def manifest_sample(index: int) -> dict:
    label = "ABCD"[index % 4]
    return {
        "sample_id": f"sample-{index:03d}",
        "source_image_id": f"image-{index:03d}",
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
        "human_verified": True,
        "answer_leakage": 0,
        "required_evidence": ["left relation", "right relation"],
        "minimal_reasoning_graph": {"nodes": [{"id": "e1"}, {"id": "e2"}, {"id": "a"}]},
        "guides": [
            {"question": "What relation is visible on the left?", "answer": "The first relation."},
            {"question": "How does it apply on the right?", "answer": "In the same order."},
        ],
    }


def result_records(tq_gains: int, tqa_gains: int, generic_gains: int = 1, logp_tq: float = 0.30, logp_tqa: float = 0.35) -> list[dict]:
    rows = []
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
                "source_image_id": f"image-{index:03d}",
                "condition": condition,
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
                "image_checksum": f"image-sha-{index}",
                "question_hash": f"question-sha-{index}",
                "options_hash": f"options-sha-{index}",
                "prompt_hash": f"prompt-{index}-{condition}",
            })
    return rows


class ExploratoryTeacherGapTests(unittest.TestCase):
    def test_locked_manifest_fails_closed_on_overlap_and_balance(self):
        samples = [manifest_sample(index) for index in range(96)]
        self.assertEqual(validate_locked_manifest(samples), [])
        samples[95]["source_image_id"] = samples[0]["source_image_id"]
        samples[95]["final_answer"] = "A"
        errors = validate_locked_manifest(samples)
        self.assertTrue(any("source_image_id" in error for error in errors))
        self.assertTrue(any("balanced" in error for error in errors))
        self.assertTrue(validate_partition_isolation({"dev": samples[:1], "locked": samples[:1]}))

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
