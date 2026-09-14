import sys
import unittest
from pathlib import Path

import numpy as np


EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))

from statistical_tests import factorial_regression, holm_correction  # noqa: E402


class StatisticalTestsContractTest(unittest.TestCase):
    def setUp(self):
        # Eight image clusters, each observed in all four factorial cells.
        self.cluster = np.repeat(np.arange(8), 4)
        self.v = np.tile([0, 0, 1, 1], 8)
        self.t = np.tile([0, 1, 0, 1], 8)
        self.correct = np.array(
            [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
             0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
        )

    def test_cluster_ids_select_cluster_sandwich_covariance(self):
        result = factorial_regression(
            self.correct, self.v, self.t, cluster_ids=self.cluster
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["covariance_type"], "cluster_sandwich")
        self.assertEqual(result["n_clusters"], 8)
        self.assertEqual(result["analysis_role"], "descriptive_smoke_only")
        self.assertFalse(result["g1_confirmatory_model"])

    def test_degenerate_clusters_fail_closed(self):
        result = factorial_regression(
            self.correct, self.v, self.t, cluster_ids=np.zeros(len(self.correct))
        )
        self.assertIn("at least two clusters", result["error"])

    def test_holm_is_monotone_in_sorted_p_values(self):
        result = holm_correction(np.array([0.03, 0.001, 0.04]))
        np.testing.assert_allclose(result["pvals_corrected"], [0.06, 0.003, 0.06])

    def test_unsupported_multiple_test_method_fails(self):
        with self.assertRaisesRegex(ValueError, "only the auditable Holm"):
            holm_correction(np.array([0.01]), method="fdr_bh")


if __name__ == "__main__":
    unittest.main()
