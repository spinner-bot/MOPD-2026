"""Contract tests for C2 build_m3vqa_teacher_gap_review_packet (1018 §5.3).

Coverage required by 1018 §5.3.5: join misses, duplicate source images,
expansion determinism/nesting, and the evidence-text leak ban. Additional
contracts: blank human_* fields, path-traversal rejection, fail-closed
shortlist validation, raw-field enrichment.

Pure stdlib; no network, no GPU, no model calls.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

import build_m3vqa_teacher_gap_review_packet as c2  # noqa: E402

SECRET_EVIDENCE = "WIKI_EVIDENCE_SECRET_TOKEN_ABCDEF_0123456789"
SEED = 20260911
TIERS = "8,16,24"
SHORTLIST_N = 8


def make_raw(data_id: str, hop: int, image_id: str) -> dict:
    return {
        "data_id": data_id,
        "image_id": image_id,
        "question": f"question for {data_id}?",
        "answers": ["yes"],
        "evidence": [[SECRET_EVIDENCE]] * hop,
        "evidence_urls": [f"https://en.wikipedia.org/wiki/{data_id}"] * hop,
        "question_hop": hop,
        "question_type": "IP1TP0S2",
        "entity_num": 3,
        "answer_evals": [["yes", "yep", "affirmative"]],
        "img_entity_names": ["Guava"],
        "evidence_img_ids": [f"wikiimg_{data_id}"],
        "evidence_section_ids": [f"sec_{data_id}"],
        "evidence_section_titles": [f"Section of {data_id}"],
        "evidence_url_titles": [f"Title of {data_id}"],
    }


def make_norm(data_id: str, hop: int, image_path: str, image_id: str) -> dict:
    return {
        "candidate_id": f"m3vqa-{data_id}",
        "source_dataset": "M3VQA",
        "source_record_id": data_id,
        "source_image_id": image_id,
        "image_original": image_path,
        "final_question_source": f"question for {data_id}?",
        "reference_answers": ["yes"],
        "evidence": [[SECRET_EVIDENCE]] * hop,
        "evidence_urls": [f"https://en.wikipedia.org/wiki/{data_id}"] * hop,
        "task_family_candidate": "R" if hop >= 2 else "P",
        "reasoning_depth_candidate": hop,
        "stage0_admission": "blocked_pending_annotation",
    }


class C2Fixture(unittest.TestCase):
    """Builds a deterministic synthetic pool and runs the generator."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        cls.root = base
        cls.image_root = base / "images"
        cls.image_root.mkdir()

        raws, norms, cls.good_ids = [], [], []
        n_good = 30
        for i in range(1, n_good + 1):
            data_id = f"data_{i:05d}"
            hop = 2 + (i % 4)  # hops 2..5
            image_id = f"img_{i:05d}.jpg"
            image_path = cls.image_root / image_id
            image_path.write_bytes(b"\xff\xd8fakejpeg")
            raws.append(make_raw(data_id, hop, image_id))
            norms.append(make_norm(data_id, hop, str(image_path), image_id))
            cls.good_ids.append(f"m3vqa-{data_id}")

        # defective records
        cls.defects = {
            "raw_join_miss": "m3vqa-data_99991",
            "duplicate_source_image_id": "m3vqa-data_99992",
            "missing_question": "m3vqa-data_99993",
            "path_traversal": "m3vqa-data_99994",
            "missing_image_file": "m3vqa-data_99995",
        }
        raws.append(make_raw("data_99990", 3, "img_99990.jpg"))  # no normalized twin
        # raw join miss: image exists, but source_record_id has no raw counterpart
        join_miss = make_norm("data_99991", 3, str(cls.image_root / "img_99991.jpg"), "img_99991.jpg")
        join_miss["source_record_id"] = "data_99999_absent"
        norms.append(join_miss)
        (cls.image_root / "img_99991.jpg").write_bytes(b"x")
        # duplicate of good image #2
        norms.append(make_norm("data_99992", 2, str(cls.image_root / "img_00002.jpg"), "img_00002.jpg"))
        raws.append(make_raw("data_99992", 2, "img_00002.jpg"))
        # missing question
        bad_q = make_norm("data_99993", 2, str(cls.image_root / "img_99993.jpg"), "img_99993.jpg")
        bad_q["final_question_source"] = ""
        norms.append(bad_q)
        raws.append(make_raw("data_99993", 2, "img_99993.jpg"))
        (cls.image_root / "img_99993.jpg").write_bytes(b"x")
        # path traversal
        norms.append(make_norm("data_99994", 2, "../evil.jpg", "img_99994.jpg"))
        raws.append(make_raw("data_99994", 2, "img_99994.jpg"))
        # missing image file (exists in neither root nor disk)
        norms.append(make_norm("data_99995", 2, str(cls.image_root / "absent_99995.jpg"), "img_99995.jpg"))
        raws.append(make_raw("data_99995", 2, "absent_99995.jpg"))

        cls.raw_path = base / "quesions.jsonl"
        cls.raw_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raws), encoding="utf-8")
        cls.norm_path = base / "source-normalized.jsonl"
        cls.norm_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in norms), encoding="utf-8")

        shortlist = [{"candidate_id": cid, "source_image_id": f"img_{i+1:05d}.jpg"}
                     for i, cid in enumerate(sorted(cls.good_ids)[:SHORTLIST_N])]
        cls.shortlist_path = base / "candidate-shortlist.jsonl"
        cls.shortlist_path.write_text(
            "".join(json.dumps(s) + "\n" for s in shortlist), encoding="utf-8")
        cls.shortlist_ids = [s["candidate_id"] for s in shortlist]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_c2(self, out_dir: Path, extra: list[str] | None = None) -> None:
        argv = [
            "--raw", str(self.raw_path),
            "--normalized", str(self.norm_path),
            "--shortlist", str(self.shortlist_path),
            "--output-dir", str(out_dir),
            "--seed", str(SEED),
            "--tiers", TIERS,
        ] + (extra or [])
        with contextlib.redirect_stdout(io.StringIO()):
            c2.main(argv)

    def load_outputs(self, out_dir: Path) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in sorted(out_dir.iterdir())}

    def enriched(self, out_dir: Path, tier: int) -> list[dict]:
        path = out_dir / f"enriched-tier-{tier}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summary(self, out_dir: Path) -> dict:
        return json.loads((out_dir / "packet-summary.json").read_text(encoding="utf-8"))


class TestC2Contract(C2Fixture):
    def test_double_run_hash_identical(self):
        out1, out2 = self.root / "out1", self.root / "out2"
        self.run_c2(out1)
        self.run_c2(out2)
        files1, files2 = self.load_outputs(out1), self.load_outputs(out2)
        self.assertEqual(sorted(files1), sorted(files2))
        for name in files1:
            self.assertEqual(
                hashlib.sha256(files1[name]).hexdigest(),
                hashlib.sha256(files2[name]).hexdigest(),
                f"non-deterministic output: {name}")

    def test_tier_counts_nesting_and_origins(self):
        out = self.root / "out3"
        self.run_c2(out)
        ids = {tier: {r["candidate_id"] for r in self.enriched(out, tier)} for tier in (8, 16, 24)}
        self.assertEqual(len(ids[8]), 8)
        self.assertEqual(len(ids[16]), 16)
        self.assertEqual(len(ids[24]), 24)
        self.assertTrue(ids[8] <= ids[16] <= ids[24], "expansion tiers must be nested supersets")
        self.assertEqual(ids[8], set(self.shortlist_ids), "tier 1 must equal the C1 shortlist")
        summary = self.summary(out)
        self.assertTrue(all(summary["nesting_verified"].values()))
        self.assertEqual(summary["tiers"]["24"]["count"], 24)

    def test_structural_reject_reason_codes(self):
        out = self.root / "out4"
        self.run_c2(out, extra=["--image-root", str(self.image_root)])
        rejects = [json.loads(line) for line in
                   (out / "structural-rejects.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        by_reason = {}
        for r in rejects:
            by_reason.setdefault(r["reason_code"], set()).add(str(r["key"]))
        for reason, key in self.defects.items():
            self.assertIn(reason, by_reason, f"missing reject reason {reason}")
            self.assertIn(key, by_reason[reason], f"{key} not rejected as {reason}")
        # rejected records never appear in any tier
        all_ids = set()
        for tier in (8, 16, 24):
            all_ids |= {r["candidate_id"] for r in self.enriched(out, tier)}
        for key in self.defects.values():
            self.assertNotIn(key, all_ids)

    def test_evidence_text_never_leaks(self):
        out = self.root / "out5"
        self.run_c2(out, extra=["--image-root", str(self.image_root)])
        for name, blob in self.load_outputs(out).items():
            self.assertNotIn(SECRET_EVIDENCE.encode("utf-8"), blob,
                             f"evidence text leaked into {name}")

    def test_human_fields_present_and_blank(self):
        out = self.root / "out6"
        self.run_c2(out)
        for tier in (8, 16, 24):
            rows = [json.loads(line) for line in
                    (out / f"review-tier-{tier}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), tier)
            for row in rows:
                for field in c2.HUMAN_FIELDS:
                    self.assertIn(field, row)
                    self.assertEqual(row[field], "", f"{field} must ship empty")
            csv_bytes = (out / f"review-tier-{tier}.csv").read_bytes()
            header = csv_bytes.decode("utf-8-sig").splitlines()[0]
            for field in c2.HUMAN_FIELDS:
                self.assertIn(field, header)

    def test_human_template_covers_every_finalizer_requirement(self):
        required = {
            "human_knowledge_label", "human_task_family", "human_unique_answer",
            "human_full_image_valid", "human_required_evidence", "human_reasoning_depth",
            "human_reasoning_graph", "human_guides_ok", "human_options_ok",
            "human_answer_leakage", "human_external_fact_injected", "human_decision",
            "human_reviewer", "human_reviewed_at", "human_notes",
        }
        self.assertEqual(set(c2.HUMAN_FIELDS), required)
        self.assertEqual(c2.RULES_VERSION, "c2-v3-answer-cardinality")

    def test_raw_field_enrichment_and_firewall_structure(self):
        out = self.root / "out7"
        self.run_c2(out)
        records = self.enriched(out, 8)
        self.assertTrue(records)
        for rec in records:
            screen = rec["screening_input"]
            self.assertEqual(screen["question_type"], "IP1TP0S2")
            self.assertEqual(screen["entity_num"], 3)
            self.assertEqual(screen["answer_evals"], [["yes", "yep", "affirmative"]])
            posthoc = rec["posthoc_analysis_only"]
            self.assertEqual(posthoc["img_entity_names"], ["Guava"])
            self.assertIn("evidence_urls", posthoc)
            self.assertIn("NOT for blind solver", posthoc["_warning"])
            self.assertNotIn("evidence", screen, "evidence text field forbidden in screening_input")
            self.assertEqual(rec["provenance"]["rules_version"], c2.RULES_VERSION)

    def test_fail_closed_when_shortlist_id_ineligible(self):
        broken = self.root / "broken-shortlist.jsonl"
        rows = [{"candidate_id": cid} for cid in self.shortlist_ids]
        rows.append({"candidate_id": self.defects["raw_join_miss"]})  # not eligible
        broken.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        argv = ["--raw", str(self.raw_path), "--normalized", str(self.norm_path),
                "--shortlist", str(broken), "--output-dir", str(self.root / "out8"),
                "--seed", str(SEED), "--tiers", "9,16,24"]
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()):
                c2.main(argv)

    def test_fail_closed_tier_shortlist_size_mismatch(self):
        argv = ["--raw", str(self.raw_path), "--normalized", str(self.norm_path),
                "--shortlist", str(self.shortlist_path), "--output-dir", str(self.root / "out9"),
                "--seed", str(SEED), "--tiers", "7,16,24"]
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()):
                c2.main(argv)

    def test_w2a_contract_compatibility(self):
        """Enriched records must be drop-in input for Codex's W2A runner
        (teacher_gap_candidate_contract.validate_candidate, allow-list renderer)."""
        from teacher_gap_candidate_contract import validate_candidate
        out = self.root / "out11"
        self.run_c2(out)
        for tier in (8, 16, 24):
            for rec in self.enriched(out, tier):
                errors = validate_candidate(rec)
                self.assertEqual(errors, [], f"tier {tier} {rec['candidate_id']}: {errors}")

    def test_aligned_packet_over_frozen_external_expansion(self):
        from teacher_gap_candidate_contract import validate_candidate
        aligned_ids = sorted(self.good_ids)[6:18]  # overlaps shortlist in exactly 2 ids
        aligned_path = self.root / "aligned-fixture.jsonl"
        aligned_path.write_text(
            "".join(json.dumps({"candidate_id": c}) + "\n" for c in aligned_ids), encoding="utf-8")
        out = self.root / "out12"
        self.run_c2(out, extra=["--aligned", str(aligned_path)])
        records = [json.loads(line) for line in
                   (out / "enriched-aligned-12.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(records), 12)
        self.assertEqual({r["candidate_id"] for r in records}, set(aligned_ids))
        expected_overlap = set(aligned_ids) & set(self.shortlist_ids)
        self.assertEqual({r["candidate_id"] for r in records if r["in_c1_shortlist"]}, expected_overlap)
        for rec in records:
            self.assertEqual(rec["tier_origin"], "codex_frozen_expansion")
            self.assertEqual(rec["tier"], "aligned-12")
            self.assertEqual(validate_candidate(rec), [])
        summary = self.summary(out)
        self.assertEqual(summary["aligned_packet"]["count"], 12)
        self.assertEqual(summary["aligned_packet"]["overlap_with_c1_shortlist"], len(expected_overlap))
        self.assertEqual(summary["aligned_packet"]["new_vs_c1_shortlist"], 12 - len(expected_overlap))
        rows = [json.loads(line) for line in
                (out / "review-aligned-12.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            for field in c2.HUMAN_FIELDS:
                self.assertEqual(row[field], "")
        for path in out.iterdir():
            self.assertNotIn(SECRET_EVIDENCE.encode("utf-8"), path.read_bytes(),
                             f"evidence text leaked into {path.name}")

    def test_aligned_fail_closed_unknown_id(self):
        aligned_path = self.root / "aligned-bad.jsonl"
        aligned_path.write_text(
            json.dumps({"candidate_id": self.shortlist_ids[0]}) + "\n"
            + json.dumps({"candidate_id": self.defects["raw_join_miss"]}) + "\n",
            encoding="utf-8")
        argv = ["--raw", str(self.raw_path), "--normalized", str(self.norm_path),
                "--shortlist", str(self.shortlist_path), "--output-dir", str(self.root / "out13"),
                "--seed", str(SEED), "--tiers", TIERS, "--aligned", str(aligned_path)]
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()):
                c2.main(argv)

    def test_summary_hashes_match_files(self):
        out = self.root / "out10"
        self.run_c2(out)
        summary = self.summary(out)
        manifest = json.loads((out / "manifest.sha256.json").read_text(encoding="utf-8"))
        for name, meta in manifest["files"].items():
            blob = (out / name).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), meta["sha256"], name)
        self.assertEqual(summary["seed"], SEED)
        self.assertEqual(summary["evidence_text_leak_check"], "passed")


if __name__ == "__main__":
    unittest.main()
