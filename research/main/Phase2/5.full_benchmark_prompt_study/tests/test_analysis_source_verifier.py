import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_analysis_sources import verify_analysis_sources


def make_manifest(root: Path, source: Path, expected_hash: str) -> Path:
    analysis = root / "analysis"
    analysis.mkdir()
    manifest = {
        "schema": "phase2-work5-statistics-v4",
        "joined_records": 160810,
        "audit": {"source_files": [{"role": "fixture", "path": str(source), "sha256": expected_hash}]},
    }
    (analysis / "analysis-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return analysis


class AnalysisSourceVerifierTests(unittest.TestCase):
    def test_verifies_matching_registered_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text("immutable source", encoding="utf-8")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            result = verify_analysis_sources(make_manifest(root, source, expected), workers=2)
            self.assertEqual(result["verified_count"], 1)
            self.assertEqual(result["failures"], [])

    def test_reports_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text("changed source", encoding="utf-8")
            result = verify_analysis_sources(make_manifest(root, source, "0" * 64), workers=1)
            self.assertEqual(result["failures"][0]["status"], "mismatch")

    def test_reports_missing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "missing.jsonl"
            result = verify_analysis_sources(make_manifest(root, source, "0" * 64), workers=1)
            self.assertEqual(result["failures"][0]["status"], "missing")

    def test_rejects_duplicate_manifest_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text("source", encoding="utf-8")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            analysis = make_manifest(root, source, expected)
            path = analysis / "analysis-manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["audit"]["source_files"].append(manifest["audit"]["source_files"][0])
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate source paths"):
                verify_analysis_sources(analysis)


if __name__ == "__main__":
    unittest.main()
