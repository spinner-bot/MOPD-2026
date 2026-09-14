import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "code" / "evaluation"
sys.path.insert(0, str(EVALUATION))

from metrics import check_stage0_go_criteria  # noqa: E402


class EvaluationEntrypointRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((EVALUATION / "evaluation_entrypoints.json").read_text())
        cls.entries = cls.registry["entries"]

    def test_natural_g0_has_exactly_one_authoritative_entry(self):
        authorized = [name for name, row in self.entries.items() if row.get("can_authorize_natural_g0")]
        self.assertEqual(authorized, ["natural_g0_label_rotation"])
        self.assertEqual(
            self.entries[authorized[0]]["required_profile"],
            "natural_label_rotation_v2_dual_generic",
        )

    def test_current_cli_exposes_rotation_but_not_legacy_decide(self):
        source = (EVALUATION / "process_exploratory_teacher_gap.py").read_text()
        self.assertIn('add_parser("decide-natural-label-rotation")', source)
        self.assertNotIn('add_parser("decide")', source)

    def test_retired_metrics_shim_cannot_authorize(self):
        result = check_stage0_go_criteria({"T_Q": [{"accuracy_gap": 1.0}]})
        self.assertFalse(result["overall_go"])
        self.assertEqual(result["status"], "retired_unsafe")
        self.assertFalse(self.entries["legacy_metrics_stage0_go"]["can_authorize_natural_g0"])

    def test_future_confirmatory_entries_are_unimplemented(self):
        self.assertEqual(self.entries["g1_confirmatory"]["status"], "not_implemented_not_authorized")
        self.assertEqual(self.entries["stage1_transfer"]["status"], "not_implemented_not_authorized")
        self.assertFalse(self.entries["generic_factorial_regression"]["can_authorize_g1_claim"])


if __name__ == "__main__":
    unittest.main()
