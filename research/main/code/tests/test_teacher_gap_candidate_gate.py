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


def raw_blind() -> dict:
    proposal = {
        "candidate_answer": "triangle",
        "visible_evidence": ["left relation", "right relation"],
        "image_is_necessary": True,
        "requires_unprovided_fact": False,
        "ambiguous": False,
        "uncertainty": "low",
    }
    return {
        "candidate_id": "candidate-1",
        "source_image_id": "image-1",
        "stage": "blind",
        "input_index": 0,
        "rank": 0,
        "world_size": 4,
        "candidate": {
            "candidate_id": "candidate-1",
            "source_image_id": "image-1",
            "image_original": "/images/image-1.jpg",
            "final_question_source": "Which shape follows both relations?",
            "reference_answers": ["triangle"],
        },
        "model_revision": "revision",
        "code_commit": "commit",
        "model_dtype": "torch.bfloat16",
        "thinking_mode": False,
        "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 1536,
            "structured_output_repetition_constraints": "disabled",
        },
        "prompt_fields_used": ["image_original", "final_question_source"],
        "image_checksum": "image-sha",
        "prompt_hash": "prompt-sha",
        "chat_template_hash": "template-sha",
        "eos_reached": True,
        "truncated": False,
        "raw_model_output": json.dumps(proposal),
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
            expected_max_new_tokens=1536,
        )
        self.assertEqual(failures, [])
        self.assertEqual(summary["valid_parsed"], 1)

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
            expected_max_new_tokens=1536,
        )
        joined = "\n".join(failures)
        self.assertIn("thinking_mode", joined)
        self.assertIn("generation_config", joined)
        self.assertIn("input sharding", joined)


if __name__ == "__main__":
    unittest.main()
