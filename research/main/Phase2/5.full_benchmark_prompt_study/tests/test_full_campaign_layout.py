from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from full_campaign_layout import CONDITIONS, TRACKS, task_coordinates, task_id  # noqa: E402


class FullCampaignLayoutTests(unittest.TestCase):
    def test_array_mapping_covers_all_condition_track_pairs(self):
        pairs = [task_coordinates(i) for i in range(len(CONDITIONS) * len(TRACKS))]
        self.assertEqual(len(set(pairs)), 30)
        self.assertEqual(task_id("S0", "vstar"), 0)
        self.assertEqual(task_id("S0", "mme-realworld-cn"), 5)
        self.assertEqual(task_id("SC", "vstar"), 6)
        self.assertEqual(task_id("S3", "mme-realworld-cn"), 29)
        self.assertEqual([task_id(*pair) for pair in pairs], list(range(30)))

    def test_out_of_range_task_is_rejected(self):
        with self.assertRaises(ValueError):
            task_coordinates(30)


if __name__ == "__main__":
    unittest.main()
