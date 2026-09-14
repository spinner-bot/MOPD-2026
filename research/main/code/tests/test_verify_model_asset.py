"""CPU tests for the local model asset identity gate."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "utils"))

from verify_model_asset import verify_model_asset


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifyModelAssetTests(unittest.TestCase):
    def fixture(self, root: Path):
        model = root / "Qwen3.5-4B-base"
        model.mkdir()
        (model / "config.json").write_bytes(b"config")
        (model / "model.safetensors").write_bytes(b"weights")
        revisions = root / "revisions.tsv"
        revisions.write_text(
            "repo\trevision\tpath\n"
            "Qwen/Qwen3.5-4B\t" + "a" * 40 + "\t/old/Qwen3.5-4B-base\n",
            encoding="utf-8",
        )
        checksums = root / "model-files.sha256"
        checksums.write_text(
            "\n".join(
                f"{digest(model / name)}  /old/Qwen3.5-4B-base/{name}"
                for name in ("config.json", "model.safetensors")
            ) + "\n",
            encoding="utf-8",
        )
        kwargs = {
            "model_dir": model,
            "repo": "Qwen/Qwen3.5-4B",
            "revision": "a" * 40,
            "revisions_tsv": revisions,
            "checksums_path": checksums,
            "expected_revisions_sha256": digest(revisions),
            "expected_checksums_sha256": digest(checksums),
        }
        return model, revisions, checksums, kwargs

    def test_exact_relocated_inventory_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, kwargs = self.fixture(Path(directory))
            result = verify_model_asset(**kwargs)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["expected_files"], 2)
            self.assertEqual(result["validation_errors"], [])

    def test_revision_and_inventory_manifest_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            _, revisions, _, kwargs = self.fixture(Path(directory))
            wrong_revision = verify_model_asset(**{**kwargs, "revision": "b" * 40})
            self.assertIn("model_revision_mismatch", wrong_revision["validation_errors"])
            revisions.write_text(revisions.read_text() + "\n", encoding="utf-8")
            changed_inventory = verify_model_asset(**kwargs)
            self.assertIn(
                "revisions_inventory_sha256_mismatch",
                changed_inventory["validation_errors"],
            )

    def test_missing_extra_and_changed_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, _, kwargs = self.fixture(Path(directory))
            (model / "config.json").write_bytes(b"changed")
            (model / "extra.json").write_bytes(b"extra")
            (model / "model.safetensors").unlink()
            result = verify_model_asset(**kwargs)
            self.assertIn("missing_model_files", result["validation_errors"])
            self.assertIn("unexpected_model_files", result["validation_errors"])
            self.assertIn(
                "model_file_sha256_mismatch:config.json",
                result["validation_errors"],
            )


if __name__ == "__main__":
    unittest.main()
