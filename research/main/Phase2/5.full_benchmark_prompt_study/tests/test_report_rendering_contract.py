import json
import tempfile
import unittest
from pathlib import Path

import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_full_report.py"
SPEC = importlib.util.spec_from_file_location("work5_report", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ReportRenderingContractTests(unittest.TestCase):
    def test_refuses_unverified_or_partial_analysis_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "analysis-manifest.json").write_text(
                json.dumps({"schema": "phase2-work5-statistics-v1", "joined_records": 100}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "160,810"):
                report.load_analysis(root)

    def test_document_is_bilingual_and_embeds_figures_without_external_assets(self):
        manifest = {
            "schema": "phase2-work5-statistics-v1", "joined_records": 160810,
            "conditions": list(report.CONDITIONS), "tracks": list(report.TRACKS),
            "seed": 20261014, "bootstrap_iterations": 2000,
            "paired_comparison_method": "exact McNemar; Holm-adjusted",
            "confidence_intervals": "cluster bootstrap 95 percent",
            "audit": {"source_files": [], "extraction_attempt_status_counts": [
                {"condition": "S0", "track": "vstar", "request_status": "INVALID_JUDGE_JSON",
                 "error_class": "evidence_not_exact", "attempts": 1},
            ]},
        }
        summaries = [{"scope": "track", "condition": c, "stratum": t, "n": "5000",
                      "native_accuracy": "0.5", "extraction_accuracy": "0.51"}
                     for c in report.CONDITIONS for t in report.TRACKS]
        summaries += [{"condition": c, "n": "32162", "native_accuracy": "0.5",
                       "extraction_accuracy": "0.51", "macro_track_native_accuracy": "0.5",
                       "macro_track_extraction_accuracy": "0.51", "native_extraction_agreement": "0.9",
                       "native_judge_exact_binary_rate": "1", "completion_tokens_mean": "20",
                       "completion_tokens_median": "15", "completion_tokens_coverage": "1"}
                      for c in report.CONDITIONS]
        paired = []
        for source in report.SOURCES:
            for scope, stratum in [("micro_all", "all_tracks")] + [("track", t) for t in report.TRACKS]:
                for c in report.PROMPTS:
                    paired.append({"score_source": source, "scope": scope, "stratum": stratum,
                                   "condition_a": "S0", "condition_b": c, "n_paired": "160810",
                                   "b_correct_a_wrong": "100", "a_correct_b_wrong": "90",
                                   "paired_delta_percentage_points": "0.1",
                                   "cluster_bootstrap_95ci_low_pp": "-0.1",
                                   "cluster_bootstrap_95ci_high_pp": "0.3",
                                   "mcnemar_exact_p": "0.5", "mcnemar_holm_p": "1"})
        tables = {name: [] for name in report.TABLES}
        tables["condition_summary"] = summaries
        tables["paired_comparisons"] = paired
        tables["score_disagreement"] = [
            {"condition": c, "native_correct": str(n), "extraction_correct": str(e), "n": "10"}
            for c in report.CONDITIONS for n, e in ((1, 1), (1, 0), (0, 1), (0, 0))]
        tables["length_accuracy_association"] = [
            {"scope": "micro_all", "score_source": source, "length_metric": metric,
             "condition": condition, "spearman_rho": "0.12"}
            for source in report.SOURCES for metric in
            ("response_chars", "response_words", "completion_tokens", "total_api_tokens")
            for condition in report.CONDITIONS]
        tables["response_length_by_score"] = [
            {"score_source": source, "metric": metric, "track": track,
             "condition": condition, "correct": str(correct), "median": "20"}
            for source in report.SOURCES for metric in ("completion_tokens", "response_chars", "response_words")
            for track in report.TRACKS for condition in report.CONDITIONS for correct in (0, 1)]
        charts = {
            "absolute": report.accuracy_heatmaps(summaries),
            "paired": report.paired_effect_forest(paired),
            "disagreement": report.disagreement_heatmap(tables["score_disagreement"]),
            "association": report.association_heatmaps(tables["length_accuracy_association"]),
            "tokens_native": report.token_medians(tables["response_length_by_score"], "native"),
            "tokens_extraction": report.token_medians(tables["response_length_by_score"], "extraction"),
        }
        document = report.build_html(manifest, tables, charts)
        self.assertIn("五条件全量评测与分层分析", document)
        self.assertIn("Score-source disagreement", document)
        self.assertIn("Holm-adjusted p / Holm校正p值", document)
        self.assertIn("evidence_not_exact", document)
        self.assertIn("All answer-extraction attempts / 全部答案抽取尝试", document)
        self.assertEqual(document.count("data:image/png;base64,"), 12)
        self.assertNotIn('src="http', document)
        self.assertNotIn("href=\"http", document)


if __name__ == "__main__":
    unittest.main()
