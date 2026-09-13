from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from build_image_affordance_human_review_queue import (
    HUMAN_AFFORDANCE_FIELDS,
    build_queue,
)


def source(key: str) -> dict:
    return {
        "image_candidate_id": key,
        "source_image_id": f"{key}.jpg",
        "source_dataset": "M3VQA",
        "image_original": f"/images/{key}.jpg",
        "image_sha256": key * 64,
        "source_record_count": 1,
        "source_questions_exposed_to_screen": False,
        "question_origin": "none_image_screening",
    }


def proposal(decision: str = "ambiguous") -> dict:
    return {
        "decision": decision,
        "observable_entities": [],
        "observable_relations": [],
        "candidate_reasoning_paths": [],
        "p_score": 0,
        "r_score": 0,
        "pr_score": 0,
        "closed_world_question_feasible": False,
        "external_identification_needed": False,
        "uncertainty": "low resolution",
    }


def output(candidate: dict, rejected: bool = False) -> dict:
    row = {
        "image_candidate_id": candidate["image_candidate_id"],
        "source_image_id": candidate["source_image_id"],
        "stage": "image_affordance",
        "candidate": dict(candidate),
        "proposal": proposal(),
        "raw_record_provenance": {"model_revision": "rev"},
    }
    if rejected:
        row["rejection_reasons"] = ["invalid_json"]
        row["proposal"] = None
    return row


class ImageAffordanceHumanReviewQueueTests(unittest.TestCase):
    def test_all_source_images_are_queued_with_blank_human_fields(self):
        sources = [source("a"), source("b"), source("c")]
        queue, rejects, summary = build_queue(
            sources, [output(sources[0])], [output(sources[1], rejected=True)]
        )
        self.assertEqual(rejects, [])
        self.assertEqual(len(queue), 3)
        self.assertEqual(
            [row["machine_output_status"] for row in queue],
            ["parsed_valid", "rejected_output", "missing_output"],
        )
        self.assertEqual(summary["queued_for_human_review"], 3)
        self.assertEqual(summary["automatically_human_verified"], 0)
        for row in queue:
            self.assertFalse(row["human_verified"])
            for field in HUMAN_AFFORDANCE_FIELDS:
                self.assertEqual(row[field], "")

    def test_identity_mismatch_is_exposed_and_never_treated_as_valid(self):
        candidate = source("a")
        machine = output(candidate)
        machine["candidate"]["image_sha256"] = "wrong"
        queue, rejects, _ = build_queue([candidate], [machine], [])
        self.assertEqual(queue[0]["machine_output_status"], "invalid_identity_output")
        self.assertIn("candidate_image_sha256_mismatch", queue[0]["machine_rejection_reasons"])
        self.assertEqual(len(rejects), 1)

    def test_duplicate_source_id_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            build_queue([source("a"), source("a")], [], [])

    def test_source_question_firewall_fails_closed(self):
        candidate = source("a")
        candidate["final_question_source"] = "secret question"
        with self.assertRaisesRegex(ValueError, "question_firewall_violation"):
            build_queue([candidate], [], [])


if __name__ == "__main__":
    unittest.main()
