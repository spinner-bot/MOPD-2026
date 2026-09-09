"""Regression checks for Stage 0 selection, depth, gatekeeping, and run archives."""

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "code" / "data_pipeline", ROOT / "code" / "evaluation", ROOT / "code" / "utils"):
    sys.path.insert(0, str(folder))

from metrics import check_stage0_go_criteria
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


if __name__ == "__main__":
    unittest.main()
