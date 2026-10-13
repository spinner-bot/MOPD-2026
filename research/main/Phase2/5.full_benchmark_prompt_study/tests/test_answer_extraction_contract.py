from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from prepare_answer_extraction import OPTION_LINE  # noqa: E402
from run_answer_extraction import validate  # noqa: E402


class AnswerExtractionContractTests(unittest.TestCase):
    def test_option_parser_handles_letter_and_parenthesized_labels(self):
        query = "Question?\nA. one\n(B) two\n(D) duplicated\n(D) another\n(E) five"
        labels = [a or b for a, b in OPTION_LINE.findall(query)]
        self.assertEqual(labels, ["A", "B", "D", "D", "E"])

    def test_mcq_allows_repeated_label_but_not_unlisted_label(self):
        row = {
            "answer_format": "MCQ_OPTION",
            "available_option_labels_in_order": ["A", "B", "D", "D", "E"],
            "model_response": "I considered D, but my final answer is E.",
        }
        answer = {
            "status": "ANSWERED", "selected_option": "E", "answer_text": "",
            "evidence": "my final answer is E", "reason_code": "explicit_final",
        }
        result = validate(__import__("json").dumps(answer), row)
        self.assertEqual(result["selected_option"], "E")
        answer["selected_option"] = "F"
        with self.assertRaises(ValueError):
            validate(__import__("json").dumps(answer), row)

    def test_numeric_short_answer_requires_exact_response_span(self):
        row = {
            "answer_format": "NUMERIC_SHORT_ANSWER",
            "available_option_labels_in_order": [],
            "model_response": "There are 4 visible windows.",
        }
        answer = {
            "status": "ANSWERED", "selected_option": None, "answer_text": "4",
            "evidence": "4 visible windows", "reason_code": "semantic_final",
        }
        self.assertEqual(validate(__import__("json").dumps(answer), row)["answer_text"], "4")
        answer["answer_text"] = "four"
        with self.assertRaises(ValueError):
            validate(__import__("json").dumps(answer), row)

    def test_no_answer_is_kept_as_a_nonanswer(self):
        row = {
            "answer_format": "MCQ_OPTION",
            "available_option_labels_in_order": ["A", "B", "C", "D"],
            "model_response": "I cannot determine this.",
        }
        answer = {
            "status": "NO_ANSWER", "selected_option": None, "answer_text": "",
            "evidence": "", "reason_code": "no_answer",
        }
        self.assertEqual(validate(__import__("json").dumps(answer), row)["status"], "NO_ANSWER")


if __name__ == "__main__":
    unittest.main()
