from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from build_m3vqa_image_affordance_pool import FORBIDDEN_OUTPUT_KEYS, build_pool, select_sample


class ImageAffordancePoolTests(unittest.TestCase):
    def test_deduplicates_images_and_removes_question_information(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "img.jpg").write_bytes(b"same-image")
            rows = [
                {"source_dataset": "M3VQA", "source_image_id": "img.jpg", "source_record_id": "q2", "final_question_source": "SECRET Q", "reference_answers": ["SECRET A"], "evidence": ["SECRET E"]},
                {"source_dataset": "M3VQA", "source_image_id": "img.jpg", "source_record_id": "q1", "evidence_urls": ["SECRET URL"]},
            ]
            output, rejects, summary = build_pool(rows, root)
        self.assertEqual(rejects, [])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["source_record_ids"], ["q1", "q2"])
        self.assertFalse(FORBIDDEN_OUTPUT_KEYS & output[0].keys())
        self.assertNotIn("SECRET", str(output[0]))
        self.assertFalse(summary["question_fields_exposed"])

    def test_output_is_deterministic_under_input_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jpg").write_bytes(b"a")
            (root / "b.jpg").write_bytes(b"b")
            rows = [
                {"source_dataset": "M3VQA", "source_image_id": "b.jpg", "source_record_id": "q2"},
                {"source_dataset": "M3VQA", "source_image_id": "a.jpg", "source_record_id": "q1"},
            ]
            first = build_pool(rows, root)
            second = build_pool(list(reversed(rows)), root)
        self.assertEqual(first, second)

    def test_invalid_identity_and_missing_image_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"source_dataset": "other", "source_image_id": "../x.jpg", "source_record_id": ""},
                {"source_dataset": "M3VQA", "source_image_id": "missing.jpg", "source_record_id": "q"},
            ]
            output, rejects, _ = build_pool(rows, Path(directory))
        self.assertEqual(output, [])
        self.assertEqual(len(rejects), 2)

    def test_sample_selection_is_deterministic_and_question_independent(self):
        rows = [{"image_candidate_id": f"i{index}", "opaque": index} for index in range(8)]
        self.assertEqual(select_sample(rows, 3, 7), select_sample(list(reversed(rows)), 3, 7))
        with self.assertRaises(ValueError):
            select_sample(rows, 9, 7)


if __name__ == "__main__":
    unittest.main()
