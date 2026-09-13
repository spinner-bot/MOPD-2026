"""Tests for the loopback-only B2a human review interface."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "review"))

from build_image_affordance_blind_calibration import build_blind_packet
from image_affordance_blind_review_app import (
    HTML_PAGE,
    ReviewStore,
    parse_args,
    validate_edit_updates,
    validate_loopback_host,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_packet(root: Path, count: int = 2) -> tuple[Path, Path, list[dict]]:
    image_root = root / "images"
    image_root.mkdir()
    sources = []
    for index in range(count):
        image_path = image_root / f"image-{index}.jpg"
        Image.new("RGB", (8 + index, 6 + index), (index, 20, 40)).save(image_path)
        sources.append({
            "image_candidate_id": f"candidate-{index}",
            "source_image_id": image_path.name,
            "source_dataset": "SyntheticSource",
            "image_original": str(image_path),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "source_questions_exposed_to_screen": False,
            "question_origin": "none_image_screening",
        })
    rows, _ = build_blind_packet(sources, count)
    packet = root / "packet.jsonl"
    write_jsonl(packet, rows)
    return packet, image_root, rows


def store_for(root: Path, count: int = 2, *, draft: Path | None = None) -> ReviewStore:
    packet, image_root, _ = make_packet(root, count)
    return ReviewStore(
        input_path=packet,
        image_root=image_root,
        draft_path=draft or root / "review.jsonl",
        export_prefix=root / "final-review-v1",
        expected_input_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
        expected_records=count,
    )


class ImageAffordanceBlindReviewAppTests(unittest.TestCase):
    def test_initial_state_is_blank_and_save_preserves_json_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = store_for(root)
            frozen_before = store.input_path.read_bytes()
            state = store.record(0)
            self.assertEqual(state["row"]["human_image_usable"], "")
            self.assertEqual(state["row"]["human_evidence_descriptions"], "")
            self.assertEqual(state["row"]["human_verified"], "")
            record_id = state["row"]["calibration_record_id"]
            saved = store.save(record_id, {
                "human_image_usable": False,
                "human_external_knowledge_needed": False,
                "human_evidence_descriptions": ["first visible fact", "second visible fact"],
            })
            self.assertIs(saved["row"]["human_image_usable"], False)
            draft = json.loads(store.draft_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIs(draft["human_image_usable"], False)
            self.assertIsInstance(draft["human_evidence_descriptions"], list)
            self.assertEqual(store.input_path.read_bytes(), frozen_before)

    def test_unset_false_and_empty_array_are_distinct(self):
        self.assertEqual(validate_edit_updates({"human_image_usable": ""})["human_image_usable"], "")
        self.assertIs(validate_edit_updates({"human_image_usable": False})["human_image_usable"], False)
        self.assertEqual(
            validate_edit_updates({"human_difficult_visual_evidence": []})[
                "human_difficult_visual_evidence"
            ],
            [],
        )
        with self.assertRaisesRegex(ValueError, "unset, true, or false"):
            validate_edit_updates({"human_image_usable": 0})

    def test_immutable_and_firewall_updates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = store_for(Path(directory))
            record_id = store.rows[0]["calibration_record_id"]
            with self.assertRaisesRegex(ValueError, "immutable or unknown"):
                store.save(record_id, {"image_path": "/different.jpg"})
            with self.assertRaisesRegex(ValueError, "immutable or unknown"):
                store.save(record_id, {"machine_proposal": {"decision": "candidate"}})

    def test_draft_cannot_replace_source_or_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet, image_root, rows = make_packet(root)
            digest = hashlib.sha256(packet.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "must not overwrite frozen"):
                ReviewStore(
                    input_path=packet,
                    image_root=image_root,
                    draft_path=packet,
                    export_prefix=root / "export",
                    expected_input_sha256=digest,
                    expected_records=2,
                )
            rows[0]["source_image_id"] = "changed.jpg"
            draft = root / "draft.jsonl"
            write_jsonl(draft, rows)
            with self.assertRaisesRegex(ValueError, "immutable field"):
                ReviewStore(
                    input_path=packet,
                    image_root=image_root,
                    draft_path=draft,
                    export_prefix=root / "export",
                    expected_input_sha256=digest,
                    expected_records=2,
                )

    def test_startup_rejects_firewall_path_and_hash_failures(self):
        def expect_failure(mutator, message: str) -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                packet, image_root, rows = make_packet(root, 1)
                mutator(root, rows)
                write_jsonl(packet, rows)
                with self.assertRaisesRegex(ValueError, message):
                    ReviewStore(
                        input_path=packet,
                        image_root=image_root,
                        draft_path=root / "draft.jsonl",
                        export_prefix=root / "export",
                        expected_input_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
                        expected_records=1,
                    )

        expect_failure(lambda root, rows: rows[0].update({"machine_proposal": {}}), "schema differs")
        expect_failure(lambda root, rows: rows[0].update({"source_question": "secret"}), "schema differs")
        expect_failure(lambda root, rows: rows[0].update({"question": "secret alias"}), "schema differs")
        expect_failure(lambda root, rows: rows[0].update({"image_sha256": "0" * 64}), "SHA-256")

        def escape(root: Path, rows: list[dict]) -> None:
            outside = root / "outside.jpg"
            Image.new("RGB", (4, 4)).save(outside)
            rows[0]["image_path"] = str(outside)
            rows[0]["image_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()

        expect_failure(escape, "escapes configured root")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet, image_root, rows = make_packet(root, 1)
            rows[0]["image_path"] = str(image_root / "missing.jpg")
            write_jsonl(packet, rows)
            with self.assertRaises(FileNotFoundError):
                ReviewStore(
                    input_path=packet,
                    image_root=image_root,
                    draft_path=root / "draft.jsonl",
                    export_prefix=root / "export",
                    expected_input_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
                    expected_records=1,
                )

    def test_complete_32_record_draft_exports_through_existing_finalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet, image_root, rows = make_packet(root, 32)
            reviewed = copy.deepcopy(rows)
            for row in reviewed:
                row.update({
                    "human_verified": True,
                    "review_status": "human_blind_affordance_review_complete",
                    "human_image_usable": True,
                    "human_affordance_decision": "reject",
                    "human_external_knowledge_needed": False,
                    "human_reviewer": "reviewer-01",
                    "human_reviewed_at": "2026-09-14T05:25:00+08:00",
                    "human_notes": "No valid closed-world R/PR target found.",
                })
            draft = root / "review.jsonl"
            write_jsonl(draft, reviewed)
            store = ReviewStore(
                input_path=packet,
                image_root=image_root,
                draft_path=draft,
                export_prefix=root / "final-review-v1",
                expected_input_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
                expected_records=32,
            )
            result = store.finalize_export()
            self.assertTrue(result["exported"])
            self.assertEqual(result["summary"]["valid_signed_reviews"], 32)
            self.assertEqual(
                len(Path(result["paths"]["accepted"]).read_text(encoding="utf-8").splitlines()),
                32,
            )
            self.assertEqual(
                result["summary"]["reviewed_packet_sha256"],
                hashlib.sha256(draft.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                store.finalize_export()

    def test_incomplete_review_does_not_create_final_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = store_for(root)
            result = store.finalize_export()
            self.assertFalse(result["exported"])
            self.assertFalse(Path(str(store.export_prefix) + ".summary.json").exists())

    def test_only_literal_ipv4_loopback_is_allowed(self):
        validate_loopback_host("127.0.0.1")
        for host in ("0.0.0.0", "localhost", "::1", "192.168.1.1"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "127.0.0.1"):
                validate_loopback_host(host)
        args = parse_args(["--draft", "draft.jsonl", "--export-prefix", "export"])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertFalse(hasattr(args, "share"))
        self.assertFalse(hasattr(args, "public"))

    def test_html_has_native_view_and_no_machine_reveal(self):
        self.assertIn("100% 原生像素", HTML_PAGE)
        self.assertIn("未选择", HTML_PAGE)
        self.assertNotIn("machine_proposal", HTML_PAGE)
        self.assertNotIn("source_question", HTML_PAGE)


if __name__ == "__main__":
    unittest.main()
