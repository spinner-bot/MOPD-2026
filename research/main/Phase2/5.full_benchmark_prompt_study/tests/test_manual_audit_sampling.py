import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from create_manual_audit_sample import select_audit_sample
from full_campaign_layout import CONDITIONS, TRACKS


def synthetic_groups(n=40):
    groups = {}
    for track in TRACKS:
        samples = {}
        for i in range(n):
            rows = []
            for j, condition in enumerate(CONDITIONS):
                native = int(i % 4 != 0)
                terminal = int(not (i % 5 == 0 and j == 0))
                extraction = int(not (i % 3 == 0 and j == 2))
                status = "AMBIGUOUS" if i % 7 == 0 and j == 1 else "ANSWERED"
                rows.append({"track": track, "condition": condition, "sample_uid": f"{track}:{i}",
                             "native_correct": str(native), "terminal_correct": str(terminal),
                             "extraction_correct": str(extraction), "extraction_status": status,
                             "completion_tokens": str(1000 if i == n - 1 else i + j)})
            samples[f"{track}:{i}"] = rows
        groups[track] = samples
    return groups


class ManualAuditSamplingTests(unittest.TestCase):
    def test_selection_is_deterministic_and_paired(self):
        groups = synthetic_groups()
        first = select_audit_sample(groups, seed=12, per_track=10)
        second = select_audit_sample(groups, seed=12, per_track=10)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], "phase2-work5-manual-audit-sample-v2")
        self.assertEqual(first["paired_condition_rows"], 300)
        for track in TRACKS:
            selected = first["tracks_selection"][track]["selected"]
            self.assertEqual(len(selected), 10)
            self.assertEqual(len({row["sample_uid"] for row in selected}), 10)

    def test_allocates_defined_risk_strata_without_duplicate_samples(self):
        result = select_audit_sample(synthetic_groups(), seed=12, per_track=10)
        for track in TRACKS:
            entry = result["tracks_selection"][track]
            selected = entry["selected"]
            for stratum, quota in (("native_extraction_disagreement", 3),
                                   ("native_terminal_disagreement", 2),
                                   ("non_answer_extraction", 2),
                                   ("long_tail_output", 1),
                                   ("three_score_agreement", 2)):
                self.assertGreaterEqual(entry["strata_available"][stratum], quota)
                self.assertGreaterEqual(entry["strata_selected"][stratum], quota)
            self.assertEqual(len({row["sample_uid"] for row in selected}), len(selected))

    def test_rejects_wrong_tracks_and_too_small_quota(self):
        groups = synthetic_groups()
        with self.assertRaisesRegex(ValueError, "six frozen"):
            select_audit_sample({"vstar": groups["vstar"]}, seed=0)
        with self.assertRaisesRegex(ValueError, "at least"):
            select_audit_sample(groups, seed=0, per_track=9)


if __name__ == "__main__":
    unittest.main()
