import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_full_results.py"
SPEC = importlib.util.spec_from_file_location("work5_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class AnalysisContractTests(unittest.TestCase):
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

    def test_exact_mcnemar_and_holm(self):
        self.assertEqual(analysis.exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(analysis.exact_mcnemar_p(3, 0), 0.25)
        adjusted = analysis.holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
