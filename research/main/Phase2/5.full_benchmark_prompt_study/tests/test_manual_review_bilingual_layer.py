import sys
import re
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from add_manual_review_bilingual_layer import (
    render_bilingual,
    resolve_translation_files,
    validate_manifest,
)


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
                f'<details><summary>Model response / source (16 chars, 8 tokens)</summary><pre>Model source {index} {condition}</pre></details>'
                f'<details><summary>Full native judge output / source (yes)</summary><pre>Judge source {index} {condition}</pre></details>'
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
        self.assertEqual(len(re.findall(
            r'<summary>Model response[^<]*\((\d+) chars, (\d+) tokens\)</summary>', source)), 300)
        formats = re.findall(
            r'<summary>Full native judge output[^<]*\(([^()]*)\)</summary>', source)
        self.assertEqual(len(formats), 300)
        self.assertEqual(set(formats), {"yes", "non_binary"})

    def test_manifest_rejects_any_missing_translation(self):
        manifest = complete_manifest()
        manifest["cases"][17]["conditions"][3]["judge_output"] = ""
        with self.assertRaisesRegex(ValueError, "cases\\[17\\].conditions\\[3\\].judge_output"):
            validate_manifest(manifest)

    def test_long_translation_file_is_resolved_verbatim_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (root / "long-zh.txt").open("w", encoding="utf-8", newline="") as handle:
                handle.write("第1行\n第2行")
            with (root / "part-1.txt").open("w", encoding="utf-8", newline="") as handle:
                handle.write("第一段\n")
            with (root / "part-2.txt").open("w", encoding="utf-8", newline="") as handle:
                handle.write("第二段")
            manifest = complete_manifest()
            row = manifest["cases"][0]["conditions"][0]
            row.pop("model_response")
            row["model_response_file"] = "long-zh.txt"
            expanded = resolve_translation_files(manifest, root)
            self.assertEqual(expanded["cases"][0]["conditions"][0]["model_response"], "第1行\n第2行")

            manifest = complete_manifest()
            row = manifest["cases"][0]["conditions"][0]
            row.pop("model_response")
            row["model_response_files"] = ["part-1.txt", "part-2.txt"]
            expanded = resolve_translation_files(manifest, root)
            self.assertEqual(expanded["cases"][0]["conditions"][0]["model_response"], "第一段\n\n第二段")

            manifest = complete_manifest()
            row = manifest["cases"][0]["conditions"][0]
            row["model_response_file"] = "../outside.txt"
            with self.assertRaisesRegex(ValueError, "escapes manifest directory"):
                resolve_translation_files(manifest, root)

    def test_repetition_recipe_expands_every_unit_and_records_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = {
                "schema": "verbatim-repetition-translation-v1",
                "source_prefix": "前文：",
                "source_unit": "repeat",
                "source_separator": ", ",
                "repetitions": 3,
                "source_suffix": " tail",
                "translated_prefix": "译文：",
                "translated_unit": "重复",
                "translated_separator": "，",
                "translated_suffix": " 末尾",
            }
            (root / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False), encoding="utf-8")
            manifest = complete_manifest()
            row = manifest["cases"][0]["conditions"][0]
            row.pop("model_response")
            row["model_response_repetition_file"] = "recipe.json"
            expanded = resolve_translation_files(manifest, root)
            resolved = expanded["cases"][0]["conditions"][0]
            self.assertEqual(resolved["model_response"], "译文：重复，重复，重复 末尾")
            expected_source = "前文：repeat, repeat, repeat tail"
            self.assertEqual(
                resolved["model_response_source_sha256"],
                hashlib.sha256(expected_source.encode("utf-8")).hexdigest(),
            )

    def test_repetition_recipe_must_match_frozen_source_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = {
                "schema": "verbatim-repetition-translation-v1",
                "source_prefix": "Model ",
                "source_unit": "source 1 S0",
                "source_separator": "",
                "repetitions": 1,
                "source_suffix": "",
                "translated_prefix": "",
                "translated_unit": "模型原文译文",
                "translated_separator": "",
                "translated_suffix": "",
            }
            recipe_path = root / "recipe.json"
            recipe_path.write_text(json.dumps(recipe, ensure_ascii=False), encoding="utf-8")
            manifest = complete_manifest()
            row = manifest["cases"][0]["conditions"][0]
            row.pop("model_response")
            row["model_response_repetition_file"] = "recipe.json"
            resolved = resolve_translation_files(manifest, root)
            result = render_bilingual(source_packet(), resolved)
            self.assertIn("\n\n模型原文译文</pre>", result)

            recipe["source_prefix"] = "different frozen source "
            recipe_path.write_text(json.dumps(recipe, ensure_ascii=False), encoding="utf-8")
            manifest = complete_manifest()
            row = manifest["cases"][0]["conditions"][0]
            row.pop("model_response")
            row["model_response_repetition_file"] = "recipe.json"
            resolved = resolve_translation_files(manifest, root)
            with self.assertRaisesRegex(ValueError, "does not match repetition recipe"):
                render_bilingual(source_packet(), resolved)

    def test_one_uniform_panel_is_added_for_every_required_translation(self):
        source = source_packet()
        result = render_bilingual(source, complete_manifest())
        self.assertEqual(result.count('<details class="zh-translation">'), 1083)
        self.assertEqual(result.count("Question source "), 60)
        self.assertEqual(result.count("Model source "), 300)
        self.assertEqual(result.count("Judge source "), 300)
        self.assertIn("<pre>Model source 1 S0</pre>", result)
        self.assertEqual(result.count("输出长度：16个字符，8个词元"), 300)
        self.assertEqual(result.count("原生输出类型：单一肯定标签（是）"), 300)
        self.assertIn("题目译文60", result)
        self.assertEqual(result.count("<select"), source.count("<select"))
        self.assertEqual(result.count("<textarea"), source.count("<textarea"))


if __name__ == "__main__":
    unittest.main()
