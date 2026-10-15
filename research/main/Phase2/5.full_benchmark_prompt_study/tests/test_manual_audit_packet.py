import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_manual_audit_packet import render_document
from full_campaign_layout import CONDITIONS


def sample_case():
    conditions = []
    for condition in CONDITIONS:
        conditions.append({
            "condition": condition, "track": "vstar", "native_correct": 1,
            "terminal_correct": 1, "extraction_correct": 1,
            "terminal_decision": "yes", "native_judge_format": "yes",
            "native_judge_source": "llm", "native_judge_raw": "Reasoning\nYes",
            "extraction": {"status": "ANSWERED", "selected_option": "A", "answer_text": "",
                           "evidence": "The answer is A."},
            "extraction_attempts": [{"judge_raw_text": "{}"}],
            "response_chars": 16, "completion_tokens": 8,
            "model_answer": "The answer is A.",
        })
    return {
        "case_index": 1, "track": "vstar", "sample_uid": "vstar:sample:1",
        "primary_stratum": "three_score_agreement", "all_flags": ["three_score_agreement"],
        "query": "Choose A. <unsafe>", "gold_answer": "A",
        "image_data_uri": "data:image/png;base64,dGVzdA==", "conditions": conditions,
    }


class ManualAuditPacketTests(unittest.TestCase):
    def test_document_is_offline_bilingual_and_contains_all_paired_reviews(self):
        document = render_document(
            [sample_case()],
            {"schema": "phase2-work5-manual-audit-sample-v2", "seed": 9, "source": {"sha256": "abc"}},
            [{"path": "/data/image.jpg", "sha256": "def"}],
            {"verified_count": 127},
        )
        self.assertIn("评分人工审阅样本", document)
        self.assertIn("The answer is A.", document)
        self.assertIn("&lt;unsafe&gt;", document)
        self.assertIn('data:image/png;base64,dGVzdA==', document)
        self.assertEqual(document.count("data:image/png;base64,"), 1)
        self.assertEqual(document.count('class="condition"'), 5)
        self.assertIn("Download review decisions / 下载复核记录", document)
        self.assertIn("localStorage", document)
        self.assertIn('schema:"work5-manual-review-decisions-v1"', document)
        self.assertNotIn('src="http', document)


if __name__ == "__main__":
    unittest.main()
