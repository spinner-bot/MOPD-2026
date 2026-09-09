"""Regression checks for Stage 0 selection, depth, gatekeeping, and run archives."""

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "code" / "data_pipeline", ROOT / "code" / "evaluation", ROOT / "code" / "utils"):
    sys.path.insert(0, str(folder))

from metrics import check_stage0_go_criteria
from freeze_text_controls import freeze_controls
from guide_contract import validate_t03_proposal
from quality_contract import eligible, validate_quality_judgement
from reasoning_depth import ReasoningGraph, ReasoningNode, compute_depth_from_graph
from run_archive import initialize_run
from split_dataset import SplitConfig, stratified_split, validate_split
from stage0_selection import depth_category, validate_stage0_coverage


def metric_row(acc_gap, logp_gap, depth=3):
    return {
        "accuracy_gap": acc_gap,
        "log_prob_gap": logp_gap,
        "reasoning_depth": depth,
    }


class Stage0ReadinessTests(unittest.TestCase):
    def test_depth_graph_direct_and_three_hops(self):
        direct = ReasoningGraph(
            nodes=[
                ReasoningNode("E", "evidence", "evidence"),
                ReasoningNode("A", "answer", "answer", ["E"]),
            ],
            final_answer_node_id="A",
        )
        chained = ReasoningGraph(
            nodes=[
                ReasoningNode("E", "evidence", "evidence"),
                ReasoningNode("H1", "hypothesis", "h1", ["E"]),
                ReasoningNode("H2", "hypothesis", "h2", ["H1"]),
                ReasoningNode("A", "answer", "answer", ["H2"]),
            ],
            final_answer_node_id="A",
        )
        self.assertEqual(compute_depth_from_graph(direct), 1)
        self.assertEqual(compute_depth_from_graph(chained), 3)

    def test_d4_is_categorized_as_d4_plus(self):
        samples = [{"reasoning_depth": 4} for _ in range(2)]
        coverage = validate_stage0_coverage(samples)
        self.assertEqual(depth_category(4), "D4+")
        self.assertEqual(coverage["depth_coverage"], {"D4+": 2})

    def test_stage0_images_do_not_overlap_other_splits(self):
        samples = []
        for family in ("P", "R", "PR"):
            for index in range(12):
                samples.append({"source_image_id": f"{family}-{index}", "task_family": family})
        splits = stratified_split(
            samples, SplitConfig(train_size=18, val_size=6, test_size=6, stage0_size=6)
        )
        self.assertTrue(splits["stage0"])
        self.assertEqual(validate_split(splits), [])

    def test_empty_or_unpaired_gate_cannot_pass(self):
        self.assertEqual(check_stage0_go_criteria({})["overall_go"], False)
        metrics = {
            "T_Q": [metric_row(.1, .2)],
            "T_QA": [metric_row(.1, .2)],
            "T_random": [metric_row(.0, .0)],
            "T_generic": [metric_row(.0, .0)],
        }
        self.assertEqual(check_stage0_go_criteria(metrics)["overall_go"], False)

    def test_gate_requires_all_evidence(self):
        metrics = {
            "T_Q": [metric_row(.08, .25), metric_row(.08, .25)],
            "T_QA": [metric_row(.05, .15), metric_row(.05, .15)],
            "T_random": [metric_row(.01, .01), metric_row(.01, .01)],
            "T_generic": [metric_row(.01, .01), metric_row(.01, .01)],
        }
        tests = {
            "T_Q": {"significant": True},
            "T_Q_vs_T_random": {"significant": True},
            "T_Q_vs_T_generic": {"significant": True},
        }
        decision = check_stage0_go_criteria(metrics, paired_tests=tests)
        self.assertTrue(decision["overall_go"])
        self.assertEqual(decision["selected_text_method"], "T_Q")

    def test_run_archive_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = initialize_run(root, "run-001", "P0", "test archive")
            self.assertTrue((run_dir / "manifest.json").exists())
            with self.assertRaises(FileExistsError):
                initialize_run(root, "run-001", "P0", "test archive")

    def test_t03_contract_rejects_answer_leakage(self):
        proposal = {
            "decision": "accept_candidate",
            "task_family": "R",
            "guidance_strategy": "logical_bridge",
            "reasoning_depth": 3,
            "required_evidence": ["the label on the left"],
            "intermediate_states": ["identify label", "compare alternatives"],
            "guides": [{"question": "What label is visible?", "answer": "The label says blue."}],
            "final_answer_leaked_by_guide": False,
        }
        self.assertIn("guide 1 leaks final answer", validate_t03_proposal(proposal, "blue"))

    def test_t04_controls_are_deterministic_and_exclude_guides(self):
        sample = {
            "sample_id": "sample-17",
            "guide_1_question": "Which animal is at the gate?",
            "guide_1_answer": "A dog is at the gate.",
            "guide_2_question": "What is beside the gate?",
            "guide_2_answer": "A bicycle is beside it.",
        }
        pool = ["Which animal is at the gate?", "How many clouds are visible?", "What color is the sky?"]
        first, second = freeze_controls(sample, pool), freeze_controls(sample, pool)
        self.assertEqual(first, second)
        self.assertNotIn(sample["guide_1_question"], first["random_questions"])
        self.assertTrue(first["random_questions_frozen"])

    def test_t05_acceptance_gate_requires_every_high_quality_field(self):
        accepted = {
            "answerability": "high", "logical_dependency": "medium", "relevance": "high",
            "ambiguity": "low", "visual_grounding": "high", "answer_leakage": 0,
            "decision": "accept",
        }
        self.assertEqual(validate_quality_judgement(accepted), [])
        self.assertTrue(eligible(accepted))
        accepted["ambiguity"] = "medium"
        self.assertFalse(eligible(accepted))


if __name__ == "__main__":
    unittest.main()
