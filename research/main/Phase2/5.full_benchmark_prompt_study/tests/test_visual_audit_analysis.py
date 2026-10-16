import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_visual_audit import analyze_records, render_html
from full_campaign_layout import CONDITIONS, TRACKS
from run_visual_audit import make_candidate_excerpt


def fixture_data():
    sample_manifest = {"tracks_selection": {}}
    inputs, results, scores = [], [], []
    index = 0
    for track in TRACKS:
        selected = [{"sample_uid": f"{track}:id:{i}", "primary_stratum": "risk_a",
                     "all_flags": ["risk_a", "risk_b"]} for i in range(10)]
        sample_manifest["tracks_selection"][track] = {"selected": selected}
        for sample in selected:
            for condition in CONDITIONS:
                request_id = f"{track}|{sample['sample_uid']}|{condition}"
                verdict = ("CORRECT", "INCORRECT", "UNCERTAIN")[index % 3]
                candidate_response = "A"
                excerpt = make_candidate_excerpt(candidate_response)
                inputs.append({"request_id": request_id, "track": track,
                               "sample_uid": sample["sample_uid"], "condition": condition,
                               "reference_answer": "A", "candidate_response": "A",
                               "candidate_response_sha256": hashlib.sha256(candidate_response.encode()).hexdigest(),
                               "image_sha256": "img"})
                results.append({"request_id": request_id, "request_status": "OK",
                                "candidate_response_excerpt": excerpt["text"],
                                "candidate_response_excerpt_sha256": excerpt["sha256"],
                                "audit": {"verdict": verdict,
                                          "reference_assessment": "SUPPORTED" if verdict != "UNCERTAIN" else "UNVERIFIABLE",
                                          "candidate_final_answer": "A", "visual_evidence": "visible",
                                          "reason": "comparison"}})
                scores.append({"track": track, "sample_uid": sample["sample_uid"],
                               "condition": condition, "native_correct": str(index % 2),
                               "terminal_correct": str((index // 2) % 2),
                               "extraction_correct": str((index // 3) % 2)})
                index += 1
    return sample_manifest, inputs, results, scores


class VisualAuditAnalysisTests(unittest.TestCase):
    def test_summarizes_exact_paired_coverage_and_excludes_uncertain_from_agreement(self):
        manifest, inputs, results, scores = fixture_data()
        summary, rows = analyze_records(inputs, results, scores, manifest)
        self.assertEqual(summary["n_samples"], 60)
        self.assertEqual(summary["n_paired_requests"], 300)
        self.assertEqual(summary["verdict_counts"], {"CORRECT": 100, "INCORRECT": 100, "UNCERTAIN": 100})
        self.assertEqual(len(rows), 300)
        self.assertEqual(summary["truncated_candidate_responses"], 0)
        for comparison in summary["pairwise_agreement_among_definite_visual_verdicts"].values():
            self.assertEqual(comparison["definite_n"], 200)
            self.assertEqual(comparison["uncertain_n"], 100)
            self.assertEqual(comparison["agreement_n"] + comparison["disagreement_n"], 200)

    def test_rejects_partial_coverage_and_rendered_report_is_self_contained(self):
        manifest, inputs, results, scores = fixture_data()
        with self.assertRaises(ValueError):
            analyze_records(inputs, results[:-1], scores, manifest)
        summary, rows = analyze_records(inputs, results, scores, manifest)
        report = render_html(summary, rows, {"model": "test"})
        self.assertIn("not a population prevalence estimate", report)
        self.assertIn("exact excerpts and hashes are saved", report)
        self.assertIn("CORRECT × score 1", report)
        self.assertNotIn("src=\"http", report)
        self.assertEqual(report.count("<tr>"), 30)

    def test_rejects_audited_excerpt_mismatch(self):
        manifest, inputs, results, scores = fixture_data()
        results[0]["candidate_response_excerpt"] = "different text"
        with self.assertRaisesRegex(ValueError, "excerpt differs"):
            analyze_records(inputs, results, scores, manifest)

    def test_reports_conservative_verdict_normalizations(self):
        manifest, inputs, results, scores = fixture_data()
        results[0]["audit"]["verdict"] = "UNCERTAIN"
        results[0]["postprocessing"] = {
            "action": "conservative_verdict_normalization",
            "model_reported_verdict": "INCORRECT",
        }
        summary, rows = analyze_records(inputs, results, scores, manifest)
        self.assertEqual(summary["conservatively_normalized_rows"], 1)
        self.assertEqual(rows[0]["visual_verdict"], "UNCERTAIN")
        self.assertEqual(rows[0]["model_reported_verdict"], "INCORRECT")


if __name__ == "__main__":
    unittest.main()
