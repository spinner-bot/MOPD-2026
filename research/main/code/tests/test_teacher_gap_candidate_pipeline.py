"""CPU contract tests for the 1016 candidate screening/construction stages."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from process_teacher_gap_candidate_stage import parse_records, select_construction
from teacher_gap_candidate_contract import (
    is_construction_candidate,
    render_stage_prompt,
    validate_audit,
    validate_blind,
    validate_construct,
)


def candidate() -> dict:
    return {
        "candidate_id": "m3vqa-1",
        "source_image_id": "image-1",
        "image_original": "/work/image-1.jpg",
        "final_question_source": "Which shape is related to both marked objects?",
        "reference_answers": ["triangle"],
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
                {"id": "h1", "type": "intermediate", "claim": "shared endpoint"},
                {"id": "a", "type": "answer", "claim": "the required shape"},
            ],
            "edges": [["e1", "h1"], ["h1", "a"]],
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
    def test_prompts_never_include_official_evidence(self):
        for stage in ("blind", "audit", "construct"):
            prompt = render_stage_prompt(candidate(), stage, audit() if stage == "construct" else None)
            self.assertNotIn("TOP SECRET", prompt)
            self.assertNotIn("example.invalid", prompt)

    def test_valid_proposals_and_candidate_selection(self):
        self.assertEqual(validate_blind(blind()), [])
        self.assertEqual(validate_audit(audit()), [])
        self.assertEqual(validate_construct(construction(), ["triangle"]), [])
        self.assertTrue(is_construction_candidate(blind(), audit()))

    def test_external_audit_is_not_selected(self):
        external = audit()
        external.update({
            "knowledge_dependency_label": "external_required",
            "task_family": None,
            "reasoning_depth": 0,
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

    def test_obvious_answer_leakage_fails_closed(self):
        leaked = construction()
        leaked["guides"][0]["answer"] = "The triangle is visible."
        self.assertIn("obvious_answer_leakage", validate_construct(leaked, ["triangle"]))


if __name__ == "__main__":
    unittest.main()

