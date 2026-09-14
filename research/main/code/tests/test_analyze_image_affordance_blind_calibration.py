"""Tests for B2a blind calibration routing and statistics."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "utils"))

from analyze_image_affordance_blind_calibration import analyze, route_for, wilson_interval
from build_image_affordance_blind_calibration import stable_packet_sha256


def original(index: int) -> dict:
    return {
        "calibration_record_id": f"b2a-{index:04d}-image-{index}",
        "calibration_source_index": index,
        "image_candidate_id": f"image-{index}",
        "source_image_id": f"source-{index}",
        "source_dataset": "synthetic",
        "image_path": f"/tmp/image-{index}.jpg",
        "image_sha256": f"sha-{index}",
        "review_phase": "phase_1_blind_image_only",
        "machine_proposal_exposed": False,
        "source_question_exposed": False,
        "review_status": "pending_blind_human_affordance_review",
        "human_verified": False,
        "human_image_usable": "",
        "human_affordance_decision": "",
        "human_target_question": "",
        "human_canonical_answer": "",
        "human_affordance_family": "",
        "human_evidence_descriptions": "",
        "human_reasoning_steps": "",
        "human_counterfactual_required_evidence": "",
        "human_difficult_visual_evidence": "",
        "human_external_knowledge_needed": "",
        "human_reviewer": "",
        "human_reviewed_at": "",
        "human_notes": "",
    }


def reviewed(row: dict, decision: str, reviewer: str = "reviewer-01") -> dict:
    result = deepcopy(row)
    result.update({
        "human_verified": True,
        "review_status": "human_blind_affordance_review_complete",
        "human_image_usable": True,
        "human_affordance_decision": decision,
        "human_external_knowledge_needed": False,
        "human_reviewer": reviewer,
        "human_reviewed_at": "2026-09-14T04:00:00+08:00",
        "human_notes": "No valid target found." if decision != "candidate" else "",
    })
    if decision == "candidate":
        result.update({
            "human_target_question": "Which route remains connected to the exit?",
            "human_canonical_answer": "right route",
            "human_affordance_family": "R",
            "human_evidence_descriptions": ["left route is blocked", "right route reaches exit"],
            "human_reasoning_steps": ["compare route continuity", "select the connected route"],
            "human_counterfactual_required_evidence": [
                "left route is blocked", "right route reaches exit",
            ],
            "human_difficult_visual_evidence": [],
        })
    return result


class AnalyzeImageAffordanceBlindCalibrationTests(unittest.TestCase):
    def test_frozen_routes_are_count_based(self):
        self.assertEqual(route_for(8)["route"], "proceed_to_b3_eight_image_canary")
        self.assertEqual(
            route_for(7)["route"],
            "compare_equal_size_blind_source_samples_before_scaling",
        )
        self.assertEqual(route_for(0)["route"], "stop_m3vqa_as_primary_affordance_source")

    def test_complete_packet_reports_witness_rate_without_machine_results(self):
        originals = [original(i) for i in range(8)]
        reviews = [reviewed(row, "candidate" if i < 2 else "reject") for i, row in enumerate(originals)]
        result = analyze(
            originals,
            reviews,
            expected_packet_sha256=stable_packet_sha256(originals),
            expected_records=8,
        )
        self.assertEqual(result["candidate_images"], 2)
        self.assertEqual(result["candidate_rate_all_images"], 0.25)
        self.assertEqual(result["projected_images_to_review_for_136_at_point_rate"], 544)
        self.assertFalse(result["machine_results_used"])
        low, high = result["candidate_rate_all_images_wilson_95"]
        self.assertLess(low, 0.25)
        self.assertGreater(high, 0.25)

    def test_incomplete_or_mixed_reviewer_packet_fails_closed(self):
        originals = [original(i) for i in range(2)]
        packet_hash = stable_packet_sha256(originals)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze(
                originals,
                [reviewed(originals[0], "reject")],
                expected_packet_sha256=packet_hash,
                expected_records=2,
            )
        with self.assertRaisesRegex(ValueError, "one stable reviewer"):
            analyze(
                originals,
                [reviewed(originals[0], "reject", "a"), reviewed(originals[1], "reject", "b")],
                expected_packet_sha256=packet_hash,
                expected_records=2,
            )

    def test_original_packet_hash_and_firewall_are_enforced(self):
        originals = [original(0)]
        reviews = [reviewed(originals[0], "reject")]
        with self.assertRaisesRegex(ValueError, "unexpected frozen"):
            analyze(originals, reviews, expected_records=1)
        reviews[0]["machine_proposal"] = {"decision": "candidate"}
        with self.assertRaisesRegex(ValueError, "not complete"):
            analyze(
                originals,
                reviews,
                expected_packet_sha256=stable_packet_sha256(originals),
                expected_records=1,
            )

    def test_invalid_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)


if __name__ == "__main__":
    unittest.main()
