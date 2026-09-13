"""CPU contract tests for the 1016 candidate screening/construction stages."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from process_teacher_gap_candidate_stage import parse_records, select_construction
from run_teacher_gap_candidate_stage import main as run_candidate_stage
from finalize_teacher_gap_manifests import (
    arrange_options,
    duplicate_cluster_map,
    generic_context_for,
    validate_human_review,
)
from teacher_gap_candidate_contract import (
    audit_fallback_target,
    canonical_answer_text,
    decision_lock_errors,
    fallback_decision_fields,
    is_construction_candidate,
    render_compact_stage_prompt,
    render_stage_prompt,
    validate_audit,
    validate_blind,
    validate_candidate,
    validate_construct,
)


def candidate() -> dict:
    return {
        "candidate_id": "m3vqa-1",
        "source_image_id": "image-1",
        "image_original": "/work/image-1.jpg",
        "final_question_source": "Which shape is related to both marked objects?",
        "reference_answers": ["triangle"],
        "reference_answer_alias_groups": [["triangle"]],
        "canonical_answer_text": "triangle",
        "evidence": [["TOP SECRET OFFICIAL EVIDENCE"]],
        "evidence_urls": ["https://example.invalid/secret"],
    }


def blind() -> dict:
    return {
        "candidate_answer": "triangle",
        "visible_evidence": ["The left mark connects to the top vertex.", "The right mark connects to the lower vertex."],
        "image_is_necessary": True,
        "requires_unprovided_fact": False,
        "ambiguous": False,
        "uncertainty": "low",
    }


def audit() -> dict:
    return {
        "knowledge_dependency_label": "image_only",
        "reason_codes": ["closed_world_visible_relations"],
        "reference_entailed": True,
        "unique_answer": True,
        "image_is_necessary": True,
        "full_image_reference_valid": True,
        "task_family": "R",
        "reasoning_depth": 2,
        "required_evidence": ["left connection", "right connection"],
        "minimal_reasoning_graph": {
            "nodes": [
                {"id": "e1", "type": "evidence", "claim": "left connection"},
                {"id": "e2", "type": "evidence", "claim": "right connection"},
                {"id": "h1", "type": "intermediate", "claim": "shared endpoint"},
                {"id": "a", "type": "answer", "claim": "the required shape"},
            ],
            "edges": [["e1", "h1"], ["e2", "h1"], ["h1", "a"]],
        },
        "explanation": "Two visible connections determine the shape.",
    }


def construction() -> dict:
    return {
        "decision": "accept_candidate",
        "guides": [
            {"question": "Where does the left mark connect?", "answer": "At the top vertex.", "supports_next": "h1"},
            {"question": "Does the right mark share that figure?", "answer": "Yes, at a lower vertex.", "supports_next": "a"},
        ],
        "correct_answer_text": "triangle",
        "distractors": ["circle", "square", "pentagon"],
        "generic_context_seed": "inspect-compare-verify",
        "answer_leakage": False,
        "external_fact_injected": False,
        "rejection_reason": "",
    }


def parsed_row(stage: str, proposal: dict) -> dict:
    return {
        "candidate_id": "m3vqa-1",
        "source_image_id": "image-1",
        "stage": stage,
        "candidate": candidate(),
        "proposal": proposal,
        "raw_record_provenance": {"model_revision": "rev"},
    }


class TeacherGapCandidatePipelineTests(unittest.TestCase):
    def test_runner_refuses_a_commit_changed_after_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            source = root / "input.jsonl"
            source.write_text("", encoding="utf-8")
            argv = [
                "run_teacher_gap_candidate_stage.py",
                "--stage", "blind",
                "--input", str(source),
                "--output", str(root / "output.jsonl"),
                "--model", str(model),
                "--model-revision", "model-revision",
                "--code-commit", "submitted-commit",
                "--image-root", str(root),
            ]
            with patch("sys.argv", argv), patch(
                "run_teacher_gap_candidate_stage.git_commit", return_value="newer-commit"
            ):
                with self.assertRaisesRegex(SystemExit, "code commit changed after submission"):
                    run_candidate_stage()

    def test_prompts_never_include_official_evidence(self):
        for stage in ("blind", "audit", "construct"):
            prompt = render_stage_prompt(candidate(), stage, audit() if stage == "construct" else None)
            self.assertNotIn("TOP SECRET", prompt)
            self.assertNotIn("example.invalid", prompt)
            compact = render_compact_stage_prompt(
                candidate(), stage, audit() if stage == "construct" else None
            )
            self.assertNotIn("TOP SECRET", compact)
            self.assertNotIn("example.invalid", compact)

    def test_blind_prompt_bounds_evidence_output(self):
        prompt = render_stage_prompt(candidate(), "blind")
        self.assertIn("1-4 distinct observations", prompt)
        self.assertIn("Do not repeat an observation", prompt)

        too_many = blind()
        too_many["visible_evidence"] = ["one", "two", "three", "four", "five"]
        self.assertIn("invalid_visible_evidence", validate_blind(too_many))

        repeated = blind()
        repeated["visible_evidence"] = ["same observation", "Same   observation"]
        self.assertIn("duplicate_visible_evidence", validate_blind(repeated))
        self.assertEqual(fallback_decision_fields("blind", blind()), {})

    def test_audit_fallback_locks_external_decision_and_emits_json_null(self):
        primary = audit()
        primary.update({
            "knowledge_dependency_label": "external_required",
            "reference_entailed": False,
            "unique_answer": False,
            "task_family": "null",
            "reasoning_depth": 0,
            "required_evidence": [],
            "minimal_reasoning_graph": {"nodes": [], "edges": []},
        })
        locked = fallback_decision_fields("audit", primary)
        self.assertEqual(locked["knowledge_dependency_label"], "external_required")
        self.assertIsNone(locked["task_family"])
        target = audit_fallback_target(primary)
        self.assertEqual(validate_audit(target), [])
        prompt = render_compact_stage_prompt(candidate(), "audit", primary_proposal=primary)
        self.assertIn('"knowledge_dependency_label":"external_required"', prompt)
        self.assertIn('"task_family":null', prompt)
        self.assertNotIn('"task_family":"null"', prompt)
        flipped = dict(target)
        flipped["knowledge_dependency_label"] = "image_only"
        self.assertEqual(
            decision_lock_errors(flipped, locked),
            ["fallback_decision_mismatch"],
        )

    def test_unparseable_audit_fallback_is_conservatively_ambiguous(self):
        target = audit_fallback_target(None)
        self.assertEqual(target["knowledge_dependency_label"], "ambiguous")
        self.assertIsNone(target["task_family"])
        self.assertEqual(validate_audit(target), [])

    def test_valid_proposals_and_candidate_selection(self):
        self.assertEqual(validate_blind(blind()), [])
        self.assertEqual(validate_audit(audit()), [])
        self.assertEqual(validate_construct(construction(), candidate(), audit()), [])
        self.assertTrue(is_construction_candidate(blind(), audit()))

    def test_multi_answer_target_is_one_complete_canonical_option(self):
        multi = candidate()
        multi.update({
            "reference_answers": ["Golden Retriever", "Akita"],
            "reference_answer_alias_groups": [
                ["Golden Retriever", "Goldie"],
                ["Akita", "Akita Inu"],
            ],
            "canonical_answer_text": '["Golden Retriever","Akita"]',
        })
        self.assertEqual(validate_candidate(multi), [])
        self.assertEqual(canonical_answer_text(multi), '["Golden Retriever","Akita"]')
        proposal = construction()
        proposal["correct_answer_text"] = '["Golden Retriever","Akita"]'
        proposal["distractors"] = [
            '["Labrador","Akita"]',
            '["Golden Retriever","Shiba Inu"]',
            '["Labrador","Shiba Inu"]',
        ]
        proposal["guides"] = [
            {"question": "Which breed is the left dog?", "answer": "Golden Retriever", "supports_next": "h1"},
            {"question": "Which breed is the right dog?", "answer": "Akita", "supports_next": "a"},
        ]
        self.assertEqual(validate_construct(proposal, multi, audit()), [])
        proposal["correct_answer_text"] = "Golden Retriever"
        self.assertIn(
            "correct_answer_not_canonical_complete_answer",
            validate_construct(proposal, multi, audit()),
        )

    def test_multi_answer_distractors_preserve_target_cardinality(self):
        multi = candidate()
        multi.update({
            "reference_answers": ["Ford", "Acura"],
            "reference_answer_alias_groups": [["Ford"], ["Acura"]],
            "canonical_answer_text": '["Ford","Acura"]',
        })
        proposal = construction()
        proposal["correct_answer_text"] = multi["canonical_answer_text"]
        proposal["distractors"] = [
            '["Ford","Toyota"]', '["Nissan","Acura"]', '["Nissan","Toyota"]',
        ]
        self.assertEqual(validate_construct(proposal, multi, audit()), [])
        proposal["distractors"][0] = "Ford"
        self.assertIn("invalid_distractor_cardinality", validate_construct(proposal, multi, audit()))
        proposal["distractors"][0] = '["Ford",""]'
        self.assertIn("invalid_distractor_cardinality", validate_construct(proposal, multi, audit()))
        proposal["distractors"][0] = '["Acura","Ford"]'
        self.assertIn("options_not_distinct", validate_construct(proposal, multi, audit()))

    def test_construct_gate_rejects_dangling_guide_target(self):
        proposal = construction()
        proposal["guides"][0]["supports_next"] = "missing-node"
        self.assertTrue(any(
            "unknown_supports_next" in reason
            for reason in validate_construct(proposal, candidate(), audit())
        ))

    def test_repeated_answer_item_requires_repeated_mention_for_leakage(self):
        multi = candidate()
        multi.update({
            "reference_answers": ["Ford", "Ford"],
            "reference_answer_alias_groups": [["Ford"], ["Ford"]],
            "canonical_answer_text": '["Ford","Ford"]',
        })
        proposal = construction()
        proposal["correct_answer_text"] = multi["canonical_answer_text"]
        proposal["distractors"] = [
            '["Ford","Acura"]', '["Toyota","Ford"]', '["Toyota","Acura"]',
        ]
        proposal["guides"][0]["answer"] = "Ford is visible"
        self.assertNotIn("obvious_answer_leakage", validate_construct(proposal, multi, audit()))
        proposal["guides"][0]["answer"] = "Ford and Ford are visible"
        self.assertIn("obvious_answer_leakage", validate_construct(proposal, multi, audit()))

    def test_construct_fallback_locks_input_canonical_answer_not_invalid_primary(self):
        primary = construction()
        primary["correct_answer_text"] = "The answer is triangle."
        locked = fallback_decision_fields("construct", primary, candidate())
        self.assertEqual(locked, {
            "decision": "accept_candidate",
            "correct_answer_text": "triangle",
        })

    def test_audit_rejects_fallback_placeholders(self):
        placeholder = audit_fallback_target(audit())
        self.assertIn("placeholder_required_evidence", validate_audit(placeholder))
        self.assertIn("placeholder_reasoning_graph", validate_audit(placeholder))

    def test_audit_rejects_nonminimal_or_inconsistent_reasoning_graphs(self):
        one_evidence = audit()
        one_evidence["minimal_reasoning_graph"] = {
            "nodes": [
                {"id": "e1", "type": "evidence", "claim": "one observation"},
                {"id": "h1", "type": "intermediate", "claim": "derived relation"},
                {"id": "a", "type": "answer", "claim": "answer"},
            ],
            "edges": [["e1", "h1"], ["h1", "a"]],
        }
        self.assertTrue(any("requires_2_evidence" in error for error in validate_audit(one_evidence)))

        wrong_depth = audit()
        wrong_depth["reasoning_depth"] = 3
        self.assertTrue(any("depth_mismatch" in error for error in validate_audit(wrong_depth)))

        disconnected = audit()
        disconnected["minimal_reasoning_graph"]["nodes"].append(
            {"id": "e3", "type": "evidence", "claim": "unused observation"}
        )
        self.assertTrue(any("outside_answer_path" in error for error in validate_audit(disconnected)))

        cyclic = audit()
        cyclic["minimal_reasoning_graph"]["edges"].append(["h1", "e1"])
        self.assertTrue(validate_audit(cyclic))

    def test_external_audit_is_not_selected(self):
        external = audit()
        external.update({
            "knowledge_dependency_label": "external_required",
            "task_family": None,
            "reasoning_depth": 0,
            "required_evidence": [],
            "minimal_reasoning_graph": {"nodes": [], "edges": []},
        })
        self.assertEqual(validate_audit(external), [])
        selected, rejected, summary = select_construction(
            [parsed_row("blind", blind())], [parsed_row("audit", external)]
        )
        self.assertEqual(selected, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(summary["selected_for_construction"], 0)

    def test_parse_keeps_raw_and_parsed_outputs_separate(self):
        raw = {
            "candidate_id": "m3vqa-1",
            "source_image_id": "image-1",
            "stage": "blind",
            "candidate": candidate(),
            "model_revision": "rev",
            "code_commit": "commit",
            "image_checksum": "sha",
            "prompt_hash": "prompt",
            "chat_template_hash": "template",
            "thinking_mode": False,
            "eos_reached": True,
            "truncated": False,
            "raw_model_output": "prefix\n```json\n" + json.dumps(blind()) + "\n```",
        }
        parsed, rejected, summary = parse_records([raw], "blind")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(rejected, [])
        self.assertNotIn("raw_model_output", parsed[0])
        self.assertEqual(summary["parse_rate"], 1.0)

    def test_parse_rejects_fallback_that_changes_locked_decision(self):
        proposal = blind()
        proposal["ambiguous"] = True
        raw = {
            "candidate_id": "m3vqa-1",
            "source_image_id": "image-1",
            "stage": "blind",
            "candidate": candidate(),
            "eos_reached": True,
            "truncated": False,
            "raw_model_output": json.dumps(proposal),
            "attempts": [{}, {"preserved_decision_fields": {"ambiguous": False}}],
        }
        parsed, rejected, summary = parse_records([raw], "blind")
        self.assertEqual(parsed, [])
        self.assertEqual(rejected[0]["rejection_reasons"], ["fallback_decision_mismatch"])
        self.assertEqual(summary["rejection_reasons"], {"fallback_decision_mismatch": 1})

    def test_obvious_answer_leakage_fails_closed(self):
        leaked = construction()
        leaked["guides"][0]["answer"] = "The triangle is visible."
        self.assertIn("obvious_answer_leakage", validate_construct(leaked, candidate(), audit()))

    def test_duplicate_clusters_are_transitively_joined(self):
        payload = {
            "images": [
                {"image_original": "/a.jpg", "sha256": "a" * 64},
                {"image_original": "/b.jpg", "sha256": "b" * 64},
                {"image_original": "/c.jpg", "sha256": "c" * 64},
            ],
            "duplicate_clusters": [
                {"kind": "near", "members": [0, 1]},
                {"kind": "near", "members": [1, 2]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            mapping = duplicate_cluster_map(path)
        self.assertEqual(len(set(mapping.values())), 1)

    def test_human_review_cannot_be_implicitly_filled(self):
        self.assertIn("human_decision_not_accept", validate_human_review({"candidate_id": "x"}))

    def test_human_review_uses_same_depth_and_graph_contract(self):
        review = {
            "candidate_id": "m3vqa-1",
            "human_decision": "accept",
            "human_knowledge_label": "image_only",
            "human_task_family": "R",
            "human_unique_answer": True,
            "human_full_image_valid": True,
            "human_guides_ok": True,
            "human_options_ok": True,
            "human_answer_leakage": False,
            "human_external_fact_injected": False,
            "human_reviewer": "reviewer-1",
            "human_reviewed_at": "2026-09-13T22:29:00+08:00",
            "human_required_evidence": ["left connection", "right connection"],
            "human_reasoning_depth": 2,
            "human_reasoning_graph": audit()["minimal_reasoning_graph"],
        }
        self.assertEqual(validate_human_review(review), [])
        review["human_reasoning_depth"] = 3
        self.assertTrue(any("depth_mismatch" in error for error in validate_human_review(review)))

    def test_human_review_time_requires_iso_timezone(self):
        review = {
            "human_decision": "accept",
            "human_knowledge_label": "image_only",
            "human_task_family": "R",
            "human_unique_answer": True,
            "human_full_image_valid": True,
            "human_guides_ok": True,
            "human_options_ok": True,
            "human_answer_leakage": False,
            "human_external_fact_injected": False,
            "human_reviewer": "reviewer-1",
            "human_reviewed_at": "2026-09-13 22:29",
            "human_required_evidence": ["left connection", "right connection"],
            "human_reasoning_depth": 2,
            "human_reasoning_graph": audit()["minimal_reasoning_graph"],
        }
        self.assertIn("human_reviewed_at_requires_timezone", validate_human_review(review))

    def test_option_arrangement_places_one_correct_answer(self):
        arranged = arrange_options(construction(), "C", "m3vqa-1")
        self.assertEqual(arranged["C"], "triangle")
        self.assertEqual(len(set(arranged.values())), 4)

    def test_generic_context_matches_frozen_token_target(self):
        class FakeTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return text.split()

        sample = {
            "guides": [
                {"question": "one two three four five six seven eight", "answer": "unused"},
                {"question": "nine ten eleven twelve thirteen fourteen", "answer": "unused"},
            ]
        }
        context = generic_context_for([sample], FakeTokenizer())
        self.assertGreater(len(context.split()), 5)


if __name__ == "__main__":
    unittest.main()
