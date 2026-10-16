import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from full_campaign_layout import CONDITIONS, TRACKS
from validate_manual_review import validate_review_export


def fixture_export():
    manifest = {"schema": "phase2-work5-manual-audit-sample-v2", "tracks_selection": {}}
    rows = []
    case_index = 0
    for track in TRACKS:
        selected = [{"sample_uid": f"{track}:audit:{i}"} for i in range(10)]
        manifest["tracks_selection"][track] = {"selected": selected}
        for sample in selected:
            case_index += 1
            for condition in CONDITIONS:
                rows.append({
                    "case_title": f"Case {case_index} {track} · stratum",
                    "condition": f"{condition} · {track}",
                    "decisions": {
                        "extraction_faithful": "Yes / 是",
                        "semantic_correct": "Yes / 是",
                        "judge_valid": "Yes / 是",
                        "note": "",
                    },
                })
    return manifest, {"schema": "work5-manual-review-decisions-v1", "rows": rows}


class ManualReviewValidationTests(unittest.TestCase):
    def test_accepts_exact_completed_60_by_5_export(self):
        manifest, export = fixture_export()
        rows, summary = validate_review_export(export, manifest, "reviewer-1")
        self.assertEqual(len(rows), 300)
        self.assertEqual(summary["cases"], 60)
        self.assertEqual(summary["completed_rows"], 300)
        self.assertTrue(summary["not_an_automatic_score_replacement"])

    def test_rejects_missing_duplicate_pending_and_unexplained_uncertain(self):
        manifest, export = fixture_export()
        with self.assertRaisesRegex(ValueError, "exactly 300"):
            validate_review_export({**export, "rows": export["rows"][:-1]}, manifest, "r")
        duplicate = copy.deepcopy(export)
        duplicate["rows"][-1] = copy.deepcopy(duplicate["rows"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate review entry"):
            validate_review_export(duplicate, manifest, "r")
        pending = copy.deepcopy(export)
        pending["rows"][0]["decisions"]["judge_valid"] = "Pending / 待审"
        with self.assertRaisesRegex(ValueError, "still pending"):
            validate_review_export(pending, manifest, "r")
        unexplained = copy.deepcopy(export)
        unexplained["rows"][0]["decisions"]["semantic_correct"] = "Unclear / 不确定"
        with self.assertRaisesRegex(ValueError, "without an evidence note"):
            validate_review_export(unexplained, manifest, "r")


if __name__ == "__main__":
    unittest.main()
