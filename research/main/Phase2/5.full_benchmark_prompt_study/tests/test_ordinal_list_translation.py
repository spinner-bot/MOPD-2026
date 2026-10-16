import html
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from translate_ordinal_bullet_list import english_number_value, translate_ordinal_bullet_list


class OrdinalListTranslationTests(unittest.TestCase):
    def test_cardinal_and_ordinal_number_words(self):
        examples = {
            "first": 1,
            "twenty-first": 21,
            "one hundred and first": 101,
            "twenty-seven thousand and twenty": 27020,
        }
        for phrase, expected in examples.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(english_number_value(phrase), expected)

    def test_every_source_list_line_is_preserved_as_one_translated_line(self):
        path = (Path(__file__).resolve().parents[1] / "report" /
                "manual-score-audit-sample-v1.html")
        source = path.read_text(encoding="utf-8")
        case = re.search(r'<section class="case" id="case-14">(.*?)</section>', source, re.S)
        self.assertIsNotNone(case)
        article = re.findall(r'<article class="condition">(.*?)</article>', case.group(1), re.S)[1]
        raw = re.search(
            r'<details><summary>Model response[^<]*</summary><pre>(.*?)</pre></details>',
            article, re.S,
        ).group(1)
        original = html.unescape(raw)
        translated = translate_ordinal_bullet_list(original)
        original_lines = original.splitlines()
        translated_lines = translated.splitlines()
        self.assertEqual(len(original_lines), len(translated_lines))
        self.assertEqual(sum("balcony down" in line for line in original_lines) +
                         sum("Top balcony" in line for line in original_lines), 3620)
        self.assertEqual(sum("个阳台" in line for line in translated_lines), 3620)
        self.assertTrue(translated_lines[-1].startswith("- 第27020个阳台"))
        self.assertIn("Looking at the image", original_lines[0])
        self.assertIn("Let’s count them:", original_lines[2])
        self.assertEqual(translated_lines[0], original_lines[0])
        self.assertEqual(translated_lines[2], original_lines[2])


if __name__ == "__main__":
    unittest.main()
