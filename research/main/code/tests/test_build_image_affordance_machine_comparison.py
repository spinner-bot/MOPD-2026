"""Tests for post-lock B2b machine-proposal reveal."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "tests"))

from build_image_affordance_blind_calibration import stable_packet_sha256
from build_image_affordance_machine_comparison import build_comparison_packet
from test_analyze_image_affordance_blind_calibration import original, reviewed


def proposal(decision: str) -> dict:
    common = {
        "decision": decision,
        "observable_entities": [],
        "observable_relations": [],
        "candidate_reasoning_paths": [],
        "p_score": 0,
        "r_score": 0,
        "pr_score": 0,
        "closed_world_question_feasible": False,
        "external_identification_needed": False,
        "uncertainty": "none found",
    }
    if decision == "candidate":
        common.update({
            "observable_entities": [
                {"id": "e1", "description": "left route", "region": "left", "fine_detail_difficult": False},
                {"id": "e2", "description": "right route", "region": "right", "fine_detail_difficult": False},
            ],
            "observable_relations": [
                {"source_id": "e1", "relation": "separate from", "target_id": "e2"},
            ],
            "candidate_reasoning_paths": [{
                "family": "R",
                "evidence_entity_ids": ["e1", "e2"],
                "transformations": ["trace both routes", "compare connectivity"],
                "natural_question_type": "connection",
            }],
            "r_score": 2,
            "closed_world_question_feasible": True,
        })
    return common


def raw(row: dict, decision: str, *, malformed: bool = False) -> dict:
    return {
        "image_candidate_id": row["image_candidate_id"],
        "source_image_id": row["source_image_id"],
        "stage": "image_affordance",
        "candidate": {
            "image_candidate_id": row["image_candidate_id"],
            "source_image_id": row["source_image_id"],
        },
        "error": None,
        "truncated": False,
        "eos_reached": True,
        "raw_model_output": "not json" if malformed else json.dumps(proposal(decision)),
        "attempts": [],
    }


class BuildImageAffordanceMachineComparisonTests(unittest.TestCase):
    def test_missing_machine_outputs_remain_unavailable(self):
        originals = [original(i) for i in range(4)]
        reviews = [
            reviewed(originals[0], "candidate"),
            reviewed(originals[1], "reject"),
            reviewed(originals[2], "candidate"),
            reviewed(originals[3], "reject"),
        ]
        packet, summary = build_comparison_packet(
            originals,
            reviews,
            [raw(originals[0], "candidate"), raw(originals[1], "candidate"), raw(originals[2], "reject")],
            reviewed_packet_sha256="locked-review-sha",
            expected_packet_sha256=stable_packet_sha256(originals),
            expected_records=4,
        )
        self.assertEqual(
            summary["machine_output_status_counts"],
            {"parsed_valid": 3, "unavailable": 1},
        )
        self.assertEqual(summary["machine_candidate_precision_on_valid_intersection"], 0.5)
        self.assertEqual(summary["machine_candidate_recall_on_valid_intersection"], 0.5)
        self.assertEqual(packet[3]["machine_output_status"], "unavailable")
        self.assertIsNone(packet[3]["machine_decision"])
        self.assertFalse(packet[3]["human_machine_comparison_verified"])

    def test_invalid_machine_output_is_not_a_negative(self):
        originals = [original(i) for i in range(2)]
        reviews = [reviewed(row, "candidate") for row in originals]
        _, summary = build_comparison_packet(
            originals,
            reviews,
            [raw(originals[0], "candidate"), raw(originals[1], "reject", malformed=True)],
            reviewed_packet_sha256="locked-review-sha",
            expected_packet_sha256=stable_packet_sha256(originals),
            expected_records=2,
        )
        self.assertEqual(summary["machine_output_status_counts"], {"invalid": 1, "parsed_valid": 1})
        self.assertEqual(summary["machine_candidate_recall_on_valid_intersection"], 1.0)

    def test_machine_reveal_refuses_incomplete_review_and_unknown_image(self):
        originals = [original(i) for i in range(2)]
        packet_hash = stable_packet_sha256(originals)
        with self.assertRaisesRegex(ValueError, "complete before machine reveal"):
            build_comparison_packet(
                originals,
                [reviewed(originals[0], "reject")],
                [],
                reviewed_packet_sha256="sha",
                expected_packet_sha256=packet_hash,
                expected_records=2,
            )
        outside = deepcopy(originals[0])
        outside["image_candidate_id"] = "outside"
        with self.assertRaisesRegex(ValueError, "outside B2a"):
            build_comparison_packet(
                originals,
                [reviewed(row, "reject") for row in originals],
                [raw(outside, "reject")],
                reviewed_packet_sha256="sha",
                expected_packet_sha256=packet_hash,
                expected_records=2,
            )

    def test_duplicate_machine_raw_is_rejected(self):
        originals = [original(0)]
        machine = raw(originals[0], "reject")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_comparison_packet(
                originals,
                [reviewed(originals[0], "reject")],
                [machine, deepcopy(machine)],
                reviewed_packet_sha256="sha",
                expected_packet_sha256=stable_packet_sha256(originals),
                expected_records=1,
            )


if __name__ == "__main__":
    unittest.main()
