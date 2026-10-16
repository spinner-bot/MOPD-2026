import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from add_manual_review_bilingual_layer import render_bilingual, validate_manifest


def complete_manifest():
    return {
        "schema": "manual-review-bilingual-translations-v1",
        "page": {key: f"page {key}" for key in ("intro", "notice", "summary")},
        "cases": [
            {
                "case_index": index,
                "question": f"题目译文{index}",
                "stratum": f"分层译文{index}",
                "flags": f"标记译文{index}",
                "reference_answer": f"参考答案{index}",
                "conditions": [
                    {
                        "condition": condition,
                        "scores": "评分译文",
                        "extraction_evidence": "抽取和证据译文",
                        "model_response": "模型回答译文",
                        "judge_output": "裁判输出译文",
                    }
                    for condition in ("S0", "SC", "S1", "S2", "S3")
                ],
            }
            for index in range(1, 61)
        ],
    }


def source_packet():
    parts = [
        "<html><head><style>body{}</style></head><body>",
        "<h1>Title</h1><p>Intro source</p><div class=\"notice\">Notice source</div>",
        "<p>Cases: 60 · paired condition views: 300 · tracks: six · seed: 20261016</p>",
    ]
    for index in range(1, 61):
        parts.append(
            f'<section class="case"><h3>Case {index} <span>stratum</span></h3>'
            '<p class="flags">flags</p><h4>Original benchmark input / source</h4>'
            f'<pre>Question source {index}</pre><p><b>Reference answer / source</b> A</p>'
        )
        for condition in ("S0", "SC", "S1", "S2", "S3"):
            parts.append(
                '<article class="condition"><div class="scores">scores source</div>'
                '<p><b>Extracted answer / source</b> A<br><b>Evidence / source</b> evidence source</p>'
                f'<details><summary>Model response / source ({condition})</summary><pre>Model source {index} {condition}</pre></details>'
                f'<details><summary>Full native judge output / source ({condition})</summary><pre>Judge source {index} {condition}</pre></details>'
                '</article>'
            )
        parts.append("</section>")
    parts.append("</body></html>")
    return "".join(parts)


class ManualReviewBilingualLayerTests(unittest.TestCase):
    def test_frozen_source_has_all_uniform_translation_anchors(self):
        path = Path(__file__).resolve().parents[1] / "report" / "manual-score-audit-sample-v1.html"
        source = path.read_text(encoding="utf-8")
        self.assertEqual(source.count('<section class="case"'), 60)
        self.assertEqual(source.count("<details><summary>Model response"), 300)
        self.assertEqual(source.count("<details><summary>Full native judge output"), 300)
        self.assertEqual(source.count("<select data-review="), 900)
        self.assertEqual(source.count("<textarea data-review="), 300)
        self.assertEqual(source.count('src="data:image/'), 60)

    def test_manifest_rejects_any_missing_translation(self):
        manifest = complete_manifest()
        manifest["cases"][17]["conditions"][3]["judge_output"] = ""
        with self.assertRaisesRegex(ValueError, "cases\\[17\\].conditions\\[3\\].judge_output"):
            validate_manifest(manifest)

    def test_one_uniform_panel_is_added_for_every_required_translation(self):
        source = source_packet()
        result = render_bilingual(source, complete_manifest())
        self.assertEqual(result.count('<details class="zh-translation">'), 1083)
        self.assertEqual(result.count("Question source "), 60)
        self.assertEqual(result.count("Model source "), 300)
        self.assertEqual(result.count("Judge source "), 300)
        self.assertIn("<pre>Model source 1 S0</pre>", result)
        self.assertIn("题目译文60", result)
        self.assertEqual(result.count("<select"), source.count("<select"))
        self.assertEqual(result.count("<textarea"), source.count("<textarea"))


if __name__ == "__main__":
    unittest.main()
