import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_full_results.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("work5_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class AnalysisContractTests(unittest.TestCase):
    def test_native_score_uses_exact_terminal_decision_only(self):
        self.assertEqual(analysis.native_score_from_output(
            "The answer may be Yes.\n</think>\n\nNo\n"
        ), ("no", 0, 1))
        self.assertEqual(analysis.native_score_from_output(
            "Reasoning says no, but the final choice is Yes\n\nYes\n"
        ), ("yes", 1, 1))
        self.assertEqual(analysis.native_score_from_output("The answer is correct."), ("other", 0, 0))

    def test_numeric_answers_compare_by_decimal_value(self):
        self.assertEqual(analysis.normalize_number("1,200.00."), analysis.normalize_number("1200"))
        self.assertIsNone(analysis.normalize_number("about 1200"))

    def test_mcq_scoring_uses_selected_literal_label(self):
        self.assertTrue(analysis.answer_matches(
            {"status": "ANSWERED", "selected_option": "E"}, "E", "MCQ_OPTION"
        ))
        self.assertFalse(analysis.answer_matches(
            {"status": "ANSWERED", "selected_option": "E"}, "D", "MCQ_OPTION"
        ))
        self.assertFalse(analysis.answer_matches(
            {"status": "AMBIGUOUS", "selected_option": None}, "E", "MCQ_OPTION"
        ))

    def test_extraction_retry_audit_preserves_error_class_and_latest_attempt(self):
        attempts = [
            {"sample_uid": "u1", "condition": "S0", "track": "vstar",
             "request_status": "INVALID_JUDGE_JSON",
             "judge_prompt_sha256": "prompt-v1",
             "validation_error": "ValueError: evidence is not a nonempty exact response substring"},
            {"sample_uid": "u1", "condition": "S0", "track": "vstar",
             "request_status": "OK", "judge_prompt_sha256": "prompt-v2"},
            {"sample_uid": "u2", "condition": "S0", "track": "vstar",
             "request_status": "INVALID_JUDGE_JSON",
             "validation_error": "ValueError: MCQ answer must be one listed literal label and have empty answer_text"},
            {"sample_uid": "u3", "condition": "S0", "track": "vstar", "request_status": "TECHNICAL_ERROR"},
            {"sample_uid": "u4", "condition": "S0", "track": "vstar",
             "request_status": "INVALID_JUDGE_JSON",
             "validation_error": "JSONDecodeError: Unterminated string"},
        ]
        latest, duplicate_count, audit = analysis.summarize_extraction_attempts(attempts)
        self.assertEqual(duplicate_count, 1)
        self.assertEqual(latest["u1"]["request_status"], "OK")
        classes = {row["error_class"]: row["attempts"] for row in audit}
        self.assertEqual(classes["evidence_not_exact"], 1)
        self.assertEqual(classes["mcq_option_contract"], 1)
        self.assertEqual(classes["technical_error"], 1)
        self.assertEqual(classes["malformed_judge_json"], 1)
        self.assertEqual(classes["none"], 1)
        self.assertIn("prompt-v1", {row["judge_prompt_sha256"] for row in audit})
        self.assertIn("prompt-v2", {row["judge_prompt_sha256"] for row in audit})

    def test_exact_mcnemar_and_holm(self):
        self.assertEqual(analysis.exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(analysis.exact_mcnemar_p(3, 0), 0.25)
        adjusted = analysis.holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_spearman_handles_tied_binary_scores(self):
        self.assertAlmostEqual(analysis.spearman_rho([1, 2, 3], [0, 1, 1]), 0.8660254037844387)
        self.assertIsNone(analysis.spearman_rho([1, 1], [0, 1]))

    def test_small_complete_paired_summary(self):
        try:
            import numpy  # noqa: F401
        except Exception as exc:
            self.skipTest(f"NumPy binary unavailable in this local test environment: {exc}")
        rows = []
        for track in analysis.TRACKS:
            for i, condition in enumerate(analysis.CONDITIONS):
                correct = int(i % 2 == 0)
                rows.append({
                    "track": track, "condition": condition, "sample_uid": f"{track}:u1",
                    "cluster_uid": f"{track}:image1", "native_correct": correct,
                    "extraction_correct": 1 - correct, "native_extraction_agree": 0,
                    "response_chars": 5, "response_words": 1, "prompt_tokens": 10,
                    "completion_tokens": 2, "total_api_tokens": 12,
                    "category": "cat", "l2_category": "l2", "cycle_category": 0,
                    "question_type": "mcq", "language": "en", "answer_format": "MCQ_OPTION",
                    "native_judge_decision": "yes" if correct else "no",
                    "native_judge_source": "llm", "native_judge_raw": "Yes",
                    "native_judge_exact_binary": 1, "extraction_status": "ANSWERED",
                    "extraction_reason": "explicit_final",
                })
        tables = analysis.summarize(rows, 42)
        self.assertEqual(len(tables["condition_summary"]), 35)
        self.assertEqual(len(tables["paired_comparisons"]), 140)
        self.assertEqual(len(tables["response_length_by_score"]), 180)
        self.assertEqual(len(tables["native_judge_decision_audit"]), 30)


if __name__ == "__main__":
    unittest.main()
