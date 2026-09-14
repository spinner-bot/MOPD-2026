"""Contract tests for the evidence-firewalled W2B human review queue."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from build_m3vqa_teacher_gap_review_packet import HUMAN_FIELDS
from build_teacher_gap_human_review_queue import build_queue


def constructed() -> dict:
    return {
        "candidate_id": "candidate-1",
        "source_image_id": "image-1",
        "stage": "construct",
        "candidate": {
            "candidate_id": "candidate-1",
            "source_image_id": "image-1",
            "image_original": "/images/image-1.jpg",
            "final_question_source": "Which shape follows both visible relations?",
            "reference_answers": ["triangle"],
            "reference_answer_alias_groups": [["triangle"]],
            "canonical_answer_text": "triangle",
            "evidence": [["OFFICIAL EVIDENCE MUST NOT LEAK"]],
            "evidence_urls": ["https://example.invalid/secret"],
        },
        "blind_proposal": {
            "candidate_answer": "triangle",
            "visible_evidence": ["left relation", "right relation"],
            "image_is_necessary": True,
            "requires_unprovided_fact": False,
            "ambiguous": False,
            "uncertainty": "low",
        },
        "audit_proposal": {
            "knowledge_dependency_label": "image_only",
            "reason_codes": ["visible_relations"],
            "reference_entailed": True,
            "unique_answer": True,
            "image_is_necessary": True,
            "full_image_reference_valid": True,
            "task_family": "R",
            "reasoning_depth": 2,
            "required_evidence": ["left relation", "right relation"],
            "minimal_reasoning_graph": {
                "nodes": [
                    {"id": "e1", "type": "evidence", "claim": "left relation"},
                    {"id": "e2", "type": "evidence", "claim": "right relation"},
                    {"id": "h1", "type": "intermediate", "claim": "joint relation"},
                    {"id": "a", "type": "answer", "claim": "triangle"},
                ],
                "edges": [["e1", "h1"], ["e2", "h1"], ["h1", "a"]],
            },
            "explanation": "Both visible relations determine the answer.",
        },
        "proposal": {
            "decision": "accept_candidate",
            "guides": [
                {"question": "What is the left relation?", "answer": "It reaches the top.", "supports_next": "h1"},
                {"question": "What is the right relation?", "answer": "It reaches the bottom.", "supports_next": "a"},
            ],
            "correct_answer_text": "triangle",
            "distractors": ["circle", "square", "pentagon"],
            "generic_context_seed": "inspect-compare-verify",
            "answer_leakage": False,
            "external_fact_injected": False,
            "rejection_reason": "",
        },
        "raw_record_provenance": {"model_revision": "revision"},
        "screening_provenance": {"blind": {"prompt_hash": "blind"}, "audit": {"prompt_hash": "audit"}},
    }


class HumanReviewQueueTests(unittest.TestCase):
    def test_queue_contains_machine_context_and_only_blank_human_fields(self):
        queued, rejected, summary = build_queue([constructed()])
        self.assertEqual(rejected, [])
        self.assertEqual(summary["queued_for_human_review"], 1)
        row = queued[0]
        self.assertEqual(row["construction_proposal"]["correct_answer_text"], "triangle")
        self.assertEqual(row["review_status"], "machine_candidate_pending_independent_human_review")
        for field in HUMAN_FIELDS:
            self.assertIn(field, row)
            self.assertEqual(row[field], "")
        rendered = str(row)
        self.assertNotIn("OFFICIAL EVIDENCE MUST NOT LEAK", rendered)
        self.assertNotIn("example.invalid", rendered)

    def test_invalid_or_rejected_construction_never_enters_queue(self):
        row = constructed()
        row["proposal"]["decision"] = "reject"
        queued, rejected, summary = build_queue([row])
        self.assertEqual(queued, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(summary["automatically_human_verified"], 0)

    def test_dangling_guide_target_never_enters_queue(self):
        row = constructed()
        row["proposal"]["guides"][0]["supports_next"] = "missing-node"
        queued, rejected, _ = build_queue([row])
        self.assertEqual(queued, [])
        self.assertTrue(any("unknown_supports_next" in error for error in rejected[0]["rejection_reasons"]))

    def test_guide_may_elicit_an_evidence_node(self):
        row = constructed()
        row["proposal"]["guides"][0]["supports_next"] = "e1"
        queued, rejected, _ = build_queue([row])
        self.assertEqual(len(queued), 1)
        self.assertEqual(rejected, [])

    def test_duplicate_candidate_ids_fail_closed(self):
        queued, rejected, _ = build_queue([constructed(), constructed()])
        self.assertEqual(len(queued), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("duplicate_candidate_id", rejected[0]["rejection_reasons"])

    def test_outer_and_inner_identity_must_match(self):
        row = constructed()
        row["candidate"]["source_image_id"] = "different-image"
        queued, rejected, _ = build_queue([row])
        self.assertEqual(queued, [])
        self.assertIn("source_image_id_mismatch", rejected[0]["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
