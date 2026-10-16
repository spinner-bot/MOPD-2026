import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from normalize_visual_audit import normalize_records


class VisualAuditNormalizationTests(unittest.TestCase):
    def test_inconsistent_definite_verdict_is_conservatively_downgraded(self):
        raw = json.dumps({
            "verdict": "INCORRECT", "reference_assessment": "CONTRADICTED",
            "candidate_final_answer": "B", "visual_evidence": "No referenced object is visible.",
            "reason": "The reference is contradicted by the image.",
        })
        source = [{"request_id": "r1", "request_status": "INVALID_AUDIT_JSON",
                   "parse_error": "ValueError: A definite candidate verdict requires a supported benchmark reference",
                   "raw_output": raw}]
        output, counts = normalize_records(source, {"r1"})
        self.assertEqual(output[0]["request_status"], "OK")
        self.assertEqual(output[0]["audit"]["verdict"], "UNCERTAIN")
        self.assertEqual(output[0]["postprocessing"]["model_reported_verdict"], "INCORRECT")
        self.assertEqual(output[0]["postprocessing"]["model_reported_reference_assessment"], "CONTRADICTED")
        self.assertEqual(output[0]["raw_output"], raw)
        self.assertEqual(source[0]["request_status"], "INVALID_AUDIT_JSON")
        self.assertEqual(counts["normalized_to_uncertain"], 1)

    def test_accepts_consistent_result_and_rejects_other_errors_or_partial_coverage(self):
        accepted = {"request_id": "r1", "request_status": "OK", "audit": {
            "verdict": "CORRECT", "reference_assessment": "SUPPORTED",
            "candidate_final_answer": "A", "visual_evidence": "Visible.", "reason": "Matches."}}
        normalized, counts = normalize_records([accepted], {"r1"})
        self.assertEqual(normalized[0]["audit"]["verdict"], "CORRECT")
        self.assertEqual(counts["accepted"], 1)
        with self.assertRaisesRegex(ValueError, "non-reviewable status"):
            normalize_records([{"request_id": "r1", "request_status": "TECHNICAL_ERROR",
                                "error": "capacity"}], {"r1"})
        with self.assertRaisesRegex(ValueError, "coverage"):
            normalize_records([], {"r1"})


if __name__ == "__main__":
    unittest.main()
