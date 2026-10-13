from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from check_judgment_stage import missing_extraction, missing_native  # noqa: E402


class JudgmentStageCheckerTests(unittest.TestCase):
    def test_native_requires_marker_and_nonempty_native_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "official-scoring" / "vision-opd-native-v1" / "S0" / "tiny"
            (root / "judge" / "tiny").mkdir(parents=True)
            (root / "COMPLETE").write_text("ok\n", encoding="utf-8")
            score = root / "judge" / "tiny" / "OPD-V-Qwen3.5-4B-S0_answer.jsonl"
            score.write_text("[]", encoding="utf-8")
            self.assertEqual(missing_native(Path(tmp), ("S0",), ("tiny",)), [])
            score.write_text("", encoding="utf-8")
            self.assertEqual(missing_native(Path(tmp), ("S0",), ("tiny",)), [0])

    def test_extraction_accepts_last_ok_retry_record_and_exact_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "judgment" / "runs" / "answer-extraction-v1" / "tiny"
            root.mkdir(parents=True)
            path = root / "S0.jsonl"
            records = [
                {"sample_uid": "u1", "condition": "S0", "track": "tiny", "request_status": "TECHNICAL_ERROR"},
                {"sample_uid": "u1", "condition": "S0", "track": "tiny", "request_status": "OK"},
                {"sample_uid": "u2", "condition": "S0", "track": "tiny", "request_status": "OK"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            self.assertEqual(missing_extraction(Path(tmp), ("S0",), ("tiny",), {"tiny": 2}), [])

    def test_extraction_flags_error_or_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "judgment" / "runs" / "answer-extraction-v1" / "tiny"
            root.mkdir(parents=True)
            path = root / "S0.jsonl"
            path.write_text(json.dumps({
                "sample_uid": "u1", "condition": "S0", "track": "tiny", "request_status": "TECHNICAL_ERROR"
            }) + "\n", encoding="utf-8")
            self.assertEqual(missing_extraction(Path(tmp), ("S0",), ("tiny",), {"tiny": 2}), [0])


if __name__ == "__main__":
    unittest.main()
