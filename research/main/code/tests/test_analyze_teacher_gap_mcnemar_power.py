"""Tests for exact McNemar pilot power calculations."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "utils"))

from analyze_teacher_gap_mcnemar_power import (
    exact_power,
    rejection_thresholds,
    upper_tail,
)


class AnalyzeTeacherGapMcnemarPowerTests(unittest.TestCase):
    def test_exact_null_tail_and_holm_threshold(self):
        self.assertAlmostEqual(upper_tail(8, 0.5, 8), 0.00390625)
        self.assertEqual(rejection_thresholds(12, 0.025)[12], 10)

    def test_reference_power_values(self):
        self.assertAlmostEqual(exact_power(96, 0.10, 0.08, 0.025), 0.643, delta=0.001)
        self.assertAlmostEqual(exact_power(96, 0.30, 0.08, 0.025), 0.231, delta=0.001)

    def test_more_samples_increase_power_for_fixed_alternative(self):
        self.assertGreater(
            exact_power(198, 0.15, 0.08, 0.025),
            exact_power(96, 0.15, 0.08, 0.025),
        )

    def test_impossible_delta_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_power(96, 0.05, 0.08, 0.025)


if __name__ == "__main__":
    unittest.main()
