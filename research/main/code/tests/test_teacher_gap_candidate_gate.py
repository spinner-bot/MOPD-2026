"""Fail-closed provenance tests for the W2 candidate-stage gate."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "code" / "data_pipeline", ROOT / "code" / "utils"):
    sys.path.insert(0, str(folder))

from check_teacher_gap_candidate_gate import summarize_and_validate
from teacher_gap_candidate_contract import (
    render_compact_stage_prompt,
    render_stage_prompt,
    stable_hash,
)


def raw_blind() -> dict:
    proposal = {
        "candidate_answer": "triangle",
        "visible_evidence": ["left relation", "right relation"],
        "image_is_necessary": True,
        "requires_unprovided_fact": False,
        "ambiguous": False,
        "uncertainty": "low",
    }
    candidate = {
        "candidate_id": "candidate-1",
        "source_image_id": "image-1",
        "image_original": "/images/image-1.jpg",
        "final_question_source": "Which shape follows both relations?",
        "reference_answers": ["triangle"],
        "reference_answer_alias_groups": [["triangle"]],
        "canonical_answer_text": "triangle",
    }
    raw_output = json.dumps(proposal)
    prompt_hash = stable_hash(render_stage_prompt(candidate, "blind"))
    attempt = {
        "attempt_kind": "primary",
        "max_new_tokens": 768,
        "prompt_hash": prompt_hash,
        "chat_template_hash": "template-sha",
        "input_tokens": 100,
        "generated_tokens": 40,
        "eos_reached": True,
        "truncated": False,
        "raw_model_output": raw_output,
        "validation_errors": [],
        "schema_valid": True,
        "preserved_decision_fields": {},
    }
    return {
        "candidate_id": "candidate-1",
        "source_image_id": "image-1",
        "stage": "blind",
        "input_index": 0,
        "rank": 0,
        "world_size": 4,
        "candidate": candidate,
        "model_revision": "revision",
        "code_commit": "commit",
        "model_dtype": "torch.bfloat16",
        "thinking_mode": False,
        "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
        "generation_config": {
            "do_sample": False,
            "primary_max_new_tokens": 768,
            "fallback_max_new_tokens": 512,
            "max_attempts": 2,
            "structured_output_repetition_constraints": "disabled",
        },
        "prompt_fields_used": ["image_original", "final_question_source"],
        "image_checksum": "image-sha",
        "prompt_hash": prompt_hash,
        "chat_template_hash": "template-sha",
        "input_tokens": 100,
        "generated_tokens": 40,
        "eos_reached": True,
        "truncated": False,
        "raw_model_output": raw_output,
        "attempts": [attempt],
        "attempt_count": 1,
        "chosen_attempt": 0,
        "fallback_used": False,
        "fallback_recovered": False,
        "primary_failure_reasons": [],
    }


class CandidateGateTests(unittest.TestCase):
    def test_complete_frozen_provenance_passes(self):
        summary, failures = summarize_and_validate(
            [raw_blind()],
            "blind",
            expected_total=1,
            min_parse_rate=1.0,
            expected_world_size=4,
            expected_thinking=False,
            expected_primary_max_new_tokens=768,
            expected_fallback_max_new_tokens=512,
        )
        self.assertEqual(failures, [])
        self.assertEqual(summary["valid_parsed"], 1)

    def test_gate_accepts_auditable_compact_fallback_recovery(self):
        row = raw_blind()
        primary = dict(row["attempts"][0])
        primary.update({
            "generated_tokens": 768,
            "eos_reached": False,
            "truncated": True,
            "raw_model_output": "{",
            "validation_errors": ["invalid_json", "truncated_output"],
            "schema_valid": False,
        })
        fallback = dict(row["attempts"][0])
        fallback.update({
            "attempt_kind": "compact_fallback",
            "max_new_tokens": 512,
            "prompt_hash": stable_hash(render_compact_stage_prompt(row["candidate"], "blind")),
            "preserved_decision_fields": {},
        })
        row.update({
            "attempts": [primary, fallback],
            "attempt_count": 2,
            "chosen_attempt": 1,
            "fallback_used": True,
            "fallback_recovered": True,
            "primary_failure_reasons": ["invalid_json", "truncated_output"],
        })
        for field in (
            "prompt_hash", "chat_template_hash", "input_tokens", "generated_tokens",
            "eos_reached", "truncated", "raw_model_output",
        ):
            row[field] = fallback[field]

        summary, failures = summarize_and_validate(
            [row],
            "blind",
            expected_total=1,
            min_parse_rate=1.0,
            expected_world_size=4,
            expected_thinking=False,
            expected_primary_max_new_tokens=768,
            expected_fallback_max_new_tokens=512,
        )
        self.assertEqual(failures, [])
        self.assertEqual(summary["fallback_used"], 1)
        self.assertEqual(summary["fallback_recovered"], 1)
        self.assertEqual(summary["primary_truncated"], 1)

    def test_config_and_sharding_drift_fail_closed(self):
        row = raw_blind()
        row["thinking_mode"] = True
        row["rank"] = 1
        row["generation_config"]["structured_output_repetition_constraints"] = "enabled"
        _, failures = summarize_and_validate(
            [row],
            "blind",
            expected_total=1,
            expected_world_size=4,
            expected_thinking=False,
            expected_primary_max_new_tokens=768,
            expected_fallback_max_new_tokens=512,
        )
        joined = "\n".join(failures)
        self.assertIn("thinking_mode", joined)
        self.assertIn("generation_config", joined)
        self.assertIn("input sharding", joined)

    def test_full_gate_can_tolerate_one_explicitly_rejected_truncation(self):
        row = raw_blind()
        attempt = row["attempts"][0]
        attempt.update({
            "eos_reached": False,
            "truncated": True,
            "generated_tokens": 768,
            "validation_errors": ["truncated_output"],
            "schema_valid": False,
        })
        fallback = dict(attempt)
        fallback.update({
            "attempt_kind": "compact_fallback",
            "max_new_tokens": 512,
            "prompt_hash": stable_hash(render_compact_stage_prompt(row["candidate"], "blind")),
            "generated_tokens": 512,
        })
        row.update({
            "attempts": [attempt, fallback],
            "attempt_count": 2,
            "chosen_attempt": 1,
            "fallback_used": True,
            "fallback_recovered": False,
            "primary_failure_reasons": ["truncated_output"],
        })
        for field in (
            "prompt_hash", "chat_template_hash", "input_tokens", "generated_tokens",
            "eos_reached", "truncated", "raw_model_output",
        ):
            row[field] = fallback[field]
        _, strict_failures = summarize_and_validate([row], "blind", expected_total=1)
        self.assertTrue(any("max_truncated=0" in failure for failure in strict_failures))
        summary, bounded_failures = summarize_and_validate(
            [row], "blind", expected_total=1, min_parse_rate=0.0, max_truncated=1
        )
        self.assertEqual(bounded_failures, [])
        self.assertEqual(summary["valid_parsed"], 0)
        self.assertEqual(summary["truncated_or_missing_eos"], 1)


if __name__ == "__main__":
    unittest.main()
