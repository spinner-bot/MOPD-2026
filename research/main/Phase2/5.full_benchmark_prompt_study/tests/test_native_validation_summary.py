from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from validate_native_long_context import terminal_decision  # noqa: E402


class NativeValidationSummaryTests(unittest.TestCase):
    def test_decision_uses_exact_final_line_not_reasoning_mentions(self):
        text = "The prompt asks whether output should be Yes.\n</think>\n\nNo\n"
        self.assertEqual(terminal_decision(text), "No")

    def test_accepts_exact_yes_or_no(self):
        self.assertEqual(terminal_decision("\nYes\n"), "Yes")
        self.assertEqual(terminal_decision("No\n\n"), "No")

    def test_other_terminal_text_is_not_forced_to_a_binary_label(self):
        self.assertEqual(terminal_decision("It seems correct."), "other")


if __name__ == "__main__":
    unittest.main()
