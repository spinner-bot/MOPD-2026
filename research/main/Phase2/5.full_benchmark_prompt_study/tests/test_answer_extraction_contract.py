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

    def test_option_parser_maps_cjk_ordinal_first_choice_to_ascii_label(self):
        query = "Question?\n（一） first\n(B) second\n(C) third\n(D) fourth"
        from prepare_answer_extraction import extract_option_labels
        self.assertEqual(extract_option_labels(query), ["A", "B", "C", "D"])

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

    def test_markdown_only_evidence_difference_recovers_exact_source_span(self):
        row = {
            "answer_format": "MCQ_OPTION",
            "available_option_labels_in_order": ["A", "B", "C", "D"],
            "model_response": "This matches option **(C)**. Final answer: **C**",
        }
        answer = {
            "status": "ANSWERED", "selected_option": "C", "answer_text": "",
            "evidence": "This matches option (C).", "reason_code": "explicit_final",
        }
        result = validate(__import__("json").dumps(answer), row)
        self.assertEqual(result["evidence"], "This matches option **(C)**.")
        self.assertEqual(result["evidence_validation"], "exact_substring_markdown_normalized")

    def test_whitespace_and_markdown_only_difference_recovers_verbatim_span(self):
        row = {
            "answer_format": "MCQ_OPTION",
            "available_option_labels_in_order": ["A", "B", "C", "D"],
            "model_response": "Explanation.\n\nFinal Answer:  \n**D**",
        }
        answer = {
            "status": "ANSWERED", "selected_option": "D", "answer_text": "",
            "evidence": "Final Answer: \n**D**", "reason_code": "explicit_final",
        }
        result = validate(__import__("json").dumps(answer), row)
        self.assertEqual(result["evidence"], "Final Answer:  \n**D")
        self.assertEqual(
            result["evidence_validation"],
            "exact_substring_whitespace_markdown_normalized",
        )

    def test_semantically_similar_but_nonmatching_evidence_is_still_rejected(self):
        row = {
            "answer_format": "MCQ_OPTION",
            "available_option_labels_in_order": ["A", "B", "C", "D"],
            "model_response": "The answer is **C**.",
        }
        answer = {
            "status": "ANSWERED", "selected_option": "C", "answer_text": "",
            "evidence": "This matches option C.", "reason_code": "explicit_final",
        }
        with self.assertRaisesRegex(ValueError, "exact response substring"):
            validate(__import__("json").dumps(answer), row)

    def test_cjk_first_option_is_validated_as_canonical_a(self):
        row = {
            "answer_format": "MCQ_OPTION",
            "available_option_labels_in_order": ["B", "C", "D", "E"],
            "query": "（一）右中心\n(B)左上角\n(C)左下角\n(D)右上角\n(E)没有",
            "model_response": "最佳答案为：A",
        }
        answer = {
            "status": "ANSWERED", "selected_option": "A", "answer_text": "",
            "evidence": "最佳答案为：A", "reason_code": "explicit_final",
        }
        self.assertEqual(validate(__import__("json").dumps(answer), row)["selected_option"], "A")

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
