"""Regression tests for the guide-effect selection firewall (1082)."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "code" / "data_pipeline", ROOT / "code" / "evaluation"):
    sys.path.insert(0, str(folder))

from dependency_test import (
    analyze_dependency_pattern,
    run_dependency_test,
    run_partial_chain_analysis,
)
from effect_selection_firewall import validate_effect_selection_firewall


class EffectSelectionFirewallTests(unittest.TestCase):
    def test_confirmatory_role_requires_explicit_result_blind_selection(self):
        record = {
            "dataset_role": "G1-locked-mechanism",
            "selection_uses_model_effect": False,
        }
        self.assertEqual(validate_effect_selection_firewall(record), [])

        missing_flag = {"dataset_role": "Stage1-internal-test"}
        self.assertIn(
            "confirmatory_role_requires_selection_uses_model_effect_false",
            validate_effect_selection_firewall(missing_flag),
        )

    def test_confirmatory_role_rejects_attached_effect_outputs(self):
        record = {
            "dataset_role": "G0-locked-method-selection",
            "selection_uses_model_effect": False,
            "teacher_gap_S_T": 0.2,
            "model_utilization_contribution_g1": 0.1,
        }
        errors = validate_effect_selection_firewall(record)
        self.assertEqual(len(errors), 1)
        self.assertIn("model_utilization_contribution_g1", errors[0])
        self.assertIn("teacher_gap_S_T", errors[0])

    def test_effect_filtered_training_requires_parent_and_rule_hashes(self):
        invalid = {
            "dataset_role": "Stage1-train",
            "selection_uses_model_effect": True,
        }
        errors = validate_effect_selection_firewall(invalid)
        self.assertIn("missing_or_invalid_parent_unfiltered_manifest_sha256", errors)
        self.assertIn("missing_or_invalid_effect_selection_rule_sha256", errors)

        valid = {
            **invalid,
            "parent_unfiltered_manifest_sha256": "a" * 64,
            "effect_selection_rule_sha256": "b" * 64,
        }
        self.assertEqual(validate_effect_selection_firewall(valid), [])

    def test_dependency_outputs_are_analysis_only_even_when_positive(self):
        guides = [{"question": "q1"}, {"question": "q2"}]
        scores = {2: 1.0, 1: 0.7}
        results = run_dependency_test(guides, lambda active: scores[len(active)])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.analysis_only for result in results))
        self.assertTrue(all(not result.selection_safe for result in results))

        summary = analyze_dependency_pattern(results)
        self.assertTrue(summary["analysis_only"])
        self.assertFalse(summary["selection_safe"])

        partial = run_partial_chain_analysis(guides, lambda active: float(len(active)))
        self.assertTrue(partial["analysis_only"])
        self.assertFalse(partial["selection_safe"])

        empty = analyze_dependency_pattern([])
        self.assertTrue(empty["analysis_only"])
        self.assertFalse(empty["selection_safe"])


if __name__ == "__main__":
    unittest.main()
