from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "code" / "data_pipeline", ROOT / "code" / "utils"):
    sys.path.insert(0, str(folder))

from check_image_affordance_gate import validate_rows
from image_affordance_contract import render_affordance_prompt, stable_hash


def valid_raw():
    candidate = {"image_candidate_id": "i1", "source_image_id": "a.jpg", "image_original": "/images/a.jpg", "image_sha256": "a" * 64, "source_questions_exposed_to_screen": False, "question_origin": "none_image_screening"}
    proposal = {"decision": "ambiguous", "observable_entities": [], "observable_relations": [], "candidate_reasoning_paths": [], "p_score": 0, "r_score": 0, "pr_score": 0, "closed_world_question_feasible": False, "external_identification_needed": False, "uncertainty": "low resolution"}
    raw = json.dumps(proposal)
    attempt = {"attempt_kind": "primary", "max_new_tokens": 1024, "prompt_hash": stable_hash(render_affordance_prompt(candidate)), "chat_template_hash": "t", "input_tokens": 10, "generated_tokens": 20, "eos_reached": True, "truncated": False, "raw_model_output": raw, "validation_errors": [], "schema_valid": True, "preserved_decision_fields": {}}
    return {"image_candidate_id": "i1", "source_image_id": "a.jpg", "stage": "image_affordance", "input_index": 0, "rank": 0, "world_size": 4, "candidate": candidate, "model_revision": "rev", "code_commit": "commit", "thinking_mode": False, "chat_template": {"enable_thinking": False, "add_generation_prompt": True}, "model_dtype": "torch.bfloat16", "generation_config": {"do_sample": False, "primary_max_new_tokens": 1024, "fallback_max_new_tokens": 768, "max_attempts": 2, "structured_output_repetition_constraints": "disabled"}, "prompt_fields_used": ["image_original"], "image_checksum": "a" * 64, "attempts": [attempt], "attempt_count": 1, "chosen_attempt": 0, "fallback_used": False, "fallback_recovered": False, "primary_failure_reasons": [], **{key: attempt[key] for key in ("prompt_hash", "chat_template_hash", "input_tokens", "generated_tokens", "eos_reached", "truncated", "raw_model_output")}}


class ImageAffordancePipelineTests(unittest.TestCase):
    def test_complete_raw_record_passes_gate(self):
        summary, failures = validate_rows([valid_raw()], expected_total=1, min_parse_rate=1.0, expected_commit="commit")
        self.assertEqual(failures, [])
        self.assertEqual(summary["valid_parsed"], 1)

    def test_question_firewall_and_checksum_drift_fail(self):
        row = valid_raw()
        row["candidate"]["final_question_source"] = "leak"
        row["image_checksum"] = "bad"
        _, failures = validate_rows([row], expected_total=1, min_parse_rate=0.0)
        self.assertTrue(any("invalid image candidate" in error for error in failures))
        self.assertTrue(any("image checksum" in error for error in failures))


if __name__ == "__main__":
    unittest.main()
