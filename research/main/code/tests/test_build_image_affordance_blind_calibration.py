from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from build_image_affordance_blind_calibration import (
    FORBIDDEN_KEYS,
    HUMAN_FIELDS,
    build_blind_packet,
    validate_completed_review,
)
from finalize_image_affordance_blind_review import finalize, require_new_outputs


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

    def completed_candidate(self) -> tuple[dict, dict]:
        original = build_blind_packet([source("a")], 1)[0][0]
        reviewed = dict(original)
        reviewed.update({
            "review_status": "human_blind_affordance_review_complete",
            "human_verified": True,
            "human_image_usable": True,
            "human_affordance_decision": "candidate",
            "human_target_question": "Which route reaches the marked exit?",
            "human_canonical_answer": "the left route",
            "human_affordance_family": "R",
            "human_evidence_descriptions": ["left route is open", "right route is blocked"],
            "human_reasoning_steps": ["compare both routes", "select the open route"],
            "human_counterfactual_required_evidence": [
                "left route is open", "right route is blocked"
            ],
            "human_difficult_visual_evidence": [],
            "human_external_knowledge_needed": False,
            "human_reviewer": "reviewer-1",
            "human_reviewed_at": "2026-09-14T04:00:00+08:00",
            "human_notes": "closed-world target",
        })
        return original, reviewed

    def test_complete_concrete_candidate_passes(self):
        original, reviewed = self.completed_candidate()
        self.assertEqual(validate_completed_review(original, reviewed), [])
        accepted, rejected, summary = finalize([original], [reviewed])
        self.assertEqual(accepted, [reviewed])
        self.assertEqual(rejected, [])
        self.assertEqual(summary["valid_decisions"], {"candidate": 1})

    def test_candidate_requires_counterfactual_evidence_and_human_signature(self):
        original, reviewed = self.completed_candidate()
        reviewed["human_counterfactual_required_evidence"] = []
        reviewed["human_verified"] = False
        errors = validate_completed_review(original, reviewed)
        self.assertIn("candidate_requires_two_counterfactual_evidence_items", errors)
        self.assertIn("human_verified_not_true", errors)

    def test_pr_requires_difficult_visual_evidence(self):
        original, reviewed = self.completed_candidate()
        reviewed["human_affordance_family"] = "PR"
        self.assertIn(
            "pr_requires_difficult_visual_evidence",
            validate_completed_review(original, reviewed),
        )

    def test_r_cannot_claim_a_difficult_visual_node(self):
        original, reviewed = self.completed_candidate()
        reviewed["human_difficult_visual_evidence"] = ["left route is open"]
        self.assertIn(
            "r_must_not_have_difficult_visual_evidence",
            validate_completed_review(original, reviewed),
        )

    def test_evidence_must_be_distinct_and_counterfactual_list_must_match(self):
        original, reviewed = self.completed_candidate()
        reviewed["human_evidence_descriptions"] = ["same evidence", "same evidence"]
        self.assertIn(
            "candidate_requires_two_visual_evidence_items",
            validate_completed_review(original, reviewed),
        )
        reviewed["human_evidence_descriptions"] = ["left route is open", "right route is blocked"]
        reviewed["human_counterfactual_required_evidence"] = ["left route is open", "unrelated item"]
        self.assertIn(
            "counterfactual_evidence_must_match_visual_evidence",
            validate_completed_review(original, reviewed),
        )

    def test_difficult_evidence_must_be_on_required_path(self):
        original, reviewed = self.completed_candidate()
        reviewed["human_affordance_family"] = "PR"
        reviewed["human_difficult_visual_evidence"] = ["tiny unrelated logo"]
        self.assertIn(
            "difficult_visual_evidence_not_in_required_evidence",
            validate_completed_review(original, reviewed),
        )

    def test_immutable_or_firewall_mutation_fails(self):
        original, reviewed = self.completed_candidate()
        reviewed["image_path"] = "/different.jpg"
        reviewed["machine_proposal"] = {"decision": "candidate"}
        errors = validate_completed_review(original, reviewed)
        self.assertIn("immutable_field_changed:image_path", errors)
        self.assertIn("review_firewall_violation", errors)

    def test_finalizer_requires_the_complete_original_set(self):
        originals = build_blind_packet([source("a"), source("b")], 2)[0]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            finalize(originals, [originals[0]])

    def test_finalizer_outputs_must_be_new_and_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            require_new_outputs([root / "accepted.jsonl", root / "rejected.jsonl", root / "summary.json"])
            existing = root / "existing.jsonl"
            existing.write_text("immutable\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                require_new_outputs([existing, root / "rejected.jsonl", root / "summary.json"])
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                require_new_outputs([root / "same", root / "same", root / "summary.json"])


if __name__ == "__main__":
    unittest.main()
