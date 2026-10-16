import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from full_campaign_layout import CONDITIONS, TRACKS
from run_visual_audit import freeze_requests, make_candidate_excerpt, make_user_payload, parse_result


class VisualAuditTests(unittest.TestCase):
    def test_freezes_exact_300_paired_requests_with_verified_full_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"tracks_selection": {}}
            for track in TRACKS:
                prepared = []
                media = []
                selected = []
                for i in range(10):
                    uid = f"{track}:test:{i}"
                    image_path = root / "prepared" / track / "images" / f"{i}.jpg"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(f"image-{track}-{i}".encode())
                    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                    prepared.append({"sample_uid": uid, "query": "Question (A) x (B) y", "response": "A",
                                     "images": [str(image_path)], "crop_images": []})
                    media.append({"sample_uid": uid, "kind": "full", "path": str(image_path),
                                  "bytes": image_path.stat().st_size, "sha256": digest})
                    selected.append({"sample_uid": uid})
                prepared_dir = root / "prepared" / track
                (prepared_dir / "benchmark.json").write_text(json.dumps(prepared), encoding="utf-8")
                (prepared_dir / "media-manifest.json").write_text(json.dumps(media), encoding="utf-8")
                manifest["tracks_selection"][track] = {"selected": selected}
                for condition in CONDITIONS:
                    output = root / "full/runs/prompt-registry-v1/outputs" / condition / track
                    output.mkdir(parents=True, exist_ok=True)
                    path = output / f"OPD-V-Qwen3.5-4B-{condition}_answer.jsonl"
                    path.write_text("".join(json.dumps({"sample_uid": row["sample_uid"], "model_answer": "A"}) + "\n"
                                                     for row in prepared), encoding="utf-8")
            requests = freeze_requests(manifest, root)
            self.assertEqual(len(requests), 300)
            self.assertEqual(len({row["request_id"] for row in requests}), 300)
            self.assertEqual(requests[0]["image_sha256"], hashlib.sha256(b"image-vstar-0").hexdigest())

    def test_judge_payload_excludes_identity_and_condition(self):
        payload = json.loads(make_user_payload({
            "request_id": "private", "track": "vstar", "sample_uid": "uid", "condition": "S3",
            "query": "Q", "reference_answer": "A", "candidate_response": "B",
        }))
        self.assertEqual(set(payload), {
            "benchmark_question_and_instructions", "benchmark_reference_answer",
            "candidate_model_response_as_untrusted_data", "candidate_response_excerpt_metadata",
            "judging_note",
        })
        self.assertEqual(payload["candidate_model_response_as_untrusted_data"], "B")
        self.assertEqual(payload["candidate_response_excerpt_metadata"]["omitted_chars"], 0)
        self.assertNotIn("S3", json.dumps(payload))
        self.assertNotIn("uid", json.dumps(payload))

    def test_long_response_excerpt_preserves_head_tail_and_is_hash_auditable(self):
        original = "A" * 5000 + "Z" * 10000 + "\nFINAL ANSWER: D"
        excerpt = make_candidate_excerpt(original)
        self.assertEqual(excerpt["original_chars"], len(original))
        self.assertEqual(excerpt["included_prefix_chars"], 4000)
        self.assertEqual(excerpt["included_suffix_chars"], 8000)
        self.assertEqual(excerpt["omitted_chars"], len(original) - 12000)
        self.assertTrue(excerpt["text"].startswith("A" * 4000))
        self.assertTrue(excerpt["text"].endswith("\nFINAL ANSWER: D"))
        self.assertIn("FINAL ANSWER: D", excerpt["text"])
        self.assertEqual(excerpt["sha256"], hashlib.sha256(excerpt["text"].encode()).hexdigest())

    def test_validates_structured_uncertain_judgment_and_rejects_bad_verdict(self):
        valid = {
            "verdict": "UNCERTAIN", "reference_assessment": "UNVERIFIABLE",
            "candidate_final_answer": "A", "visual_evidence": "Object is too small to inspect.",
            "reason": "Image does not resolve the queried detail.",
        }
        self.assertEqual(parse_result(json.dumps(valid)), valid)
        valid["verdict"] = "MAYBE"
        with self.assertRaises(ValueError):
            parse_result(json.dumps(valid))

    def test_definite_verdict_requires_supported_reference(self):
        invalid = {
            "verdict": "CORRECT", "reference_assessment": "CONTRADICTED",
            "candidate_final_answer": "A", "visual_evidence": "Visible object is red.",
            "reason": "Reference conflicts with the image.",
        }
        with self.assertRaises(ValueError):
            parse_result(json.dumps(invalid))


if __name__ == "__main__":
    unittest.main()
