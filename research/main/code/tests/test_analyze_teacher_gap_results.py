"""Archive-layout and scientific-status checks for the W2 analyzer."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "code" / "data_pipeline", ROOT / "code" / "utils"):
    sys.path.insert(0, str(folder))

from analyze_teacher_gap_results import analyze_construct_output, analyze_screen_output


def candidate() -> dict:
    return {
        "candidate_id": "candidate-1",
        "source_image_id": "image-1",
        "image_original": "/work/image.jpg",
        "final_question_source": "Which shape completes the relation?",
        "reference_answers": ["triangle"],
        "reference_answer_alias_groups": [["triangle"]],
        "canonical_answer_text": "triangle",
    }


def blind() -> dict:
    return {
        "candidate_answer": "triangle",
        "visible_evidence": ["Two marked relations converge on one shape."],
        "image_is_necessary": True,
        "requires_unprovided_fact": False,
        "ambiguous": False,
        "uncertainty": "low",
    }


def audit() -> dict:
    return {
        "knowledge_dependency_label": "image_only",
        "reason_codes": ["visible_relation"],
        "reference_entailed": True,
        "unique_answer": True,
        "image_is_necessary": True,
        "full_image_reference_valid": True,
        "task_family": "R",
        "reasoning_depth": 2,
        "required_evidence": ["left relation", "right relation"],
        "minimal_reasoning_graph": {
            "nodes": [
                {"id": "e1", "type": "evidence", "claim": "left relation"},
                {"id": "e2", "type": "evidence", "claim": "right relation"},
                {"id": "h1", "type": "intermediate", "claim": "joint relation"},
                {"id": "a", "type": "answer", "claim": "triangle"},
            ],
            "edges": [["e1", "h1"], ["e2", "h1"], ["h1", "a"]],
        },
        "explanation": "The visible relation determines one answer.",
    }


def construct() -> dict:
    return {
        "decision": "accept_candidate",
        "guides": [{
            "question": "What relation is visible?",
            "answer": "Two marks converge.",
            "supports_next": "h1",
        }],
        "correct_answer_text": "triangle",
        "distractors": ["circle", "square", "pentagon"],
        "generic_context_seed": "inspect-compare-verify",
        "answer_leakage": False,
        "external_fact_injected": False,
        "rejection_reason": "",
    }


def raw(stage: str, proposal: dict) -> dict:
    return {
        "candidate_id": "candidate-1",
        "source_image_id": "image-1",
        "stage": stage,
        "candidate": candidate(),
        "model_revision": "revision",
        "code_commit": "commit",
        "image_checksum": "checksum",
        "prompt_hash": "prompt",
        "chat_template_hash": "template",
        "thinking_mode": False,
        "eos_reached": True,
        "truncated": False,
        "raw_model_output": json.dumps(proposal),
        "blind_proposal": blind() if stage == "construct" else None,
        "audit_proposal": audit() if stage == "construct" else None,
    }


def write_raw(run: Path, stage: str, proposal: dict) -> None:
    path = run / "raw" / stage / "rank-0.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw(stage, proposal)) + "\n", encoding="utf-8")


class AnalyzeTeacherGapResultsTests(unittest.TestCase):
    def test_screen_reads_actual_raw_layout_and_preserves_status_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "teacher-gap-screen-test"
            write_raw(run, "blind", blind())
            write_raw(run, "audit", audit())
            result = analyze_screen_output(run)
        self.assertEqual(result["stages"]["blind"]["observed"], 1)
        self.assertEqual(result["stages"]["audit"]["knowledge_labels"], {"image_only": 1})
        self.assertEqual(result["selection"]["selected_for_construction"], 1)
        self.assertFalse(result["teacher_gap_measured"])
        self.assertEqual(
            result["human_review_queue"][0]["status"],
            "model_candidate_pending_construction_and_human_review",
        )

    def test_missing_audit_is_not_reported_as_zero_completed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "teacher-gap-screen-test"
            write_raw(run, "blind", blind())
            result = analyze_screen_output(run)
        self.assertEqual(result["stages"]["audit"]["status"], "not_available")
        self.assertEqual(
            result["selection"]["status"],
            "not_available_until_both_full_stages_exist",
        )

    def test_construct_never_marks_model_output_human_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "teacher-gap-construct-test"
            write_raw(run, "construct", construct())
            result = analyze_construct_output(run)
        self.assertEqual(result["decisions"], {"accept_candidate": 1, "reject": 0})
        self.assertEqual(result["automatically_human_verified"], 0)
        self.assertFalse(result["teacher_gap_measured"])


if __name__ == "__main__":
    unittest.main()
