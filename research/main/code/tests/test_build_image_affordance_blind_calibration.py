from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from build_image_affordance_blind_calibration import (
    FORBIDDEN_KEYS,
    HUMAN_FIELDS,
    build_blind_packet,
)


def source(key: str) -> dict:
    return {
        "image_candidate_id": key,
        "source_image_id": f"{key}.jpg",
        "source_dataset": "M3VQA",
        "image_original": f"/images/{key}.jpg",
        "image_sha256": key * 64,
        "source_questions_exposed_to_screen": False,
        "question_origin": "none_image_screening",
    }


class ImageAffordanceBlindCalibrationTests(unittest.TestCase):
    def test_prefix_is_preserved_and_all_human_fields_are_blank(self):
        sources = [source("a"), source("b"), source("c")]
        packet, summary = build_blind_packet(sources, 2)
        self.assertEqual([row["image_candidate_id"] for row in packet], ["a", "b"])
        self.assertEqual(summary["calibration_records"], 2)
        self.assertFalse(summary["machine_proposals_exposed"])
        self.assertFalse(summary["source_questions_exposed"])
        self.assertEqual(summary["automatically_human_verified"], 0)
        for row in packet:
            self.assertFalse(row["human_verified"])
            self.assertFalse(row["machine_proposal_exposed"])
            self.assertFalse(FORBIDDEN_KEYS & row.keys())
            for field in HUMAN_FIELDS:
                self.assertEqual(row[field], "")

    def test_output_and_hash_are_deterministic(self):
        sources = [source("a"), source("b")]
        self.assertEqual(build_blind_packet(sources, 2), build_blind_packet(sources, 2))

    def test_invalid_size_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            build_blind_packet([source("a")], 2)

    def test_duplicate_id_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            build_blind_packet([source("a"), source("a")], 2)

    def test_machine_or_question_content_fails_closed(self):
        machine_source = source("a")
        machine_source["machine_proposal"] = {"decision": "candidate"}
        with self.assertRaisesRegex(ValueError, "forbidden source field"):
            build_blind_packet([machine_source], 1)
        question_source = source("b")
        question_source["final_question_source"] = "secret"
        with self.assertRaisesRegex(ValueError, "question_firewall_violation"):
            build_blind_packet([question_source], 1)


if __name__ == "__main__":
    unittest.main()
