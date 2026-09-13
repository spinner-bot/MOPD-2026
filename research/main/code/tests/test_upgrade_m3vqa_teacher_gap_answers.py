"""Tests for faithful M3VQA answer-cardinality conversion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from upgrade_m3vqa_teacher_gap_answers import upgrade


def candidate(data_id: str = "data_1") -> dict:
    return {
        "candidate_id": f"m3vqa-{data_id}",
        "source_record_id": data_id,
        "source_image_id": "image.jpg",
        "image_original": "/images/image.jpg",
        "final_question_source": "Which breeds are shown?",
    }


class UpgradeM3VQAAnswersTests(unittest.TestCase):
    def test_preserves_multiple_required_answer_groups(self):
        raw = [{
            "data_id": "data_1",
            "image_id": "image.jpg",
            "question": "Which breeds are shown?",
            "question_hop": 2,
            "answers": ["Golden Retriever", "Akita"],
            "answer_evals": [["Golden Retriever", "Goldie"], ["Akita", "Akita Inu"]],
            "evidence": [["SECRET"]],
            "evidence_urls": ["https://example.invalid"],
        }]
        rows, rejects, summary = upgrade([candidate()], raw)
        self.assertEqual(rejects, [])
        self.assertEqual(rows[0]["reference_answers"], ["Golden Retriever", "Akita"])
        self.assertEqual(rows[0]["canonical_answer_text"], '["Golden Retriever","Akita"]')
        self.assertEqual(summary["target_cardinality_distribution"], {"2": 1})
        self.assertNotIn("SECRET", str(rows))
        self.assertNotIn("example.invalid", str(rows))

    def test_collapses_legacy_answers_when_official_eval_has_one_group(self):
        raw = [{
            "data_id": "data_1",
            "image_id": "image.jpg",
            "question": "Which breeds are shown?",
            "question_hop": 2,
            "answers": ["Portugal", "Spain"],
            "answer_evals": [["Portugal", "Spain", "PT", "ES"]],
        }]
        rows, rejects, summary = upgrade([candidate()], raw)
        self.assertEqual(rejects, [])
        self.assertEqual(rows[0]["reference_answers"], ["Portugal"])
        self.assertEqual(rows[0]["canonical_answer_text"], "Portugal")
        self.assertTrue(rows[0]["answer_contract"]["legacy_alias_list_collapsed"])
        self.assertEqual(summary["legacy_alias_lists_collapsed"], 1)

    def test_join_and_identity_mismatches_fail_closed(self):
        rows, rejects, _ = upgrade([candidate()], [])
        self.assertEqual(rows, [])
        self.assertEqual(rejects[0]["rejection_reasons"], ["raw_join_miss"])

        raw = [{
            "data_id": "data_1", "image_id": "other.jpg",
            "question": "Which breeds are shown?", "question_hop": 2,
            "answers": ["Akita"], "answer_evals": [["Akita"]],
        }]
        rows, rejects, _ = upgrade([candidate()], raw)
        self.assertEqual(rows, [])
        self.assertIn("source_image_id_mismatch", rejects[0]["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
