from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data_pipeline"))

from image_affordance_contract import (
    affordance_fallback_lock,
    affordance_lock_errors,
    render_affordance_prompt,
    render_compact_affordance_prompt,
    validate_affordance_proposal,
    validate_image_candidate,
)


def candidate():
    return {"image_candidate_id": "i1", "source_image_id": "a.jpg", "image_original": "/images/a.jpg", "image_sha256": "a" * 64, "source_questions_exposed_to_screen": False, "question_origin": "none_image_screening"}


def proposal():
    return {"decision": "candidate", "observable_entities": [{"id": "e1", "description": "left person", "region": "left", "fine_detail_difficult": False}, {"id": "e2", "description": "right box", "region": "right", "fine_detail_difficult": False}], "observable_relations": [{"source_id": "e1", "relation": "points toward", "target_id": "e2"}], "candidate_reasoning_paths": [{"family": "R", "evidence_entity_ids": ["e1", "e2"], "transformations": ["follow the pointing direction", "select the referenced box"], "natural_question_type": "spatial"}], "p_score": 0, "r_score": 2, "pr_score": 0, "closed_world_question_feasible": True, "external_identification_needed": False, "uncertainty": "low"}


class ImageAffordanceContractTests(unittest.TestCase):
    def test_prompt_is_constant_and_question_firewalled(self):
        c = candidate()
        c["opaque_provenance"] = "allowed but never rendered"
        prompt = render_affordance_prompt(c)
        self.assertNotIn("a.jpg", prompt)
        self.assertNotIn("opaque", prompt)
        c["final_question_source"] = "secret"
        self.assertIn("question_firewall_violation", validate_image_candidate(c))

    def test_valid_r_candidate(self):
        self.assertEqual(validate_affordance_proposal(proposal()), [])

    def test_path_and_relations_must_resolve_to_visible_entities(self):
        p = proposal()
        p["observable_relations"][0]["target_id"] = "missing"
        p["candidate_reasoning_paths"][0]["evidence_entity_ids"] = ["e1", "missing"]
        errors = validate_affordance_proposal(p)
        self.assertIn("relation_0_unknown_entity", errors)
        self.assertIn("path_0_invalid_evidence_entities", errors)

    def test_pr_requires_difficult_local_node(self):
        p = proposal()
        p["candidate_reasoning_paths"][0]["family"] = "PR"
        p["pr_score"] = 2
        self.assertIn("pr_path_without_difficult_visual_node", validate_affordance_proposal(p))
        p["observable_entities"][0]["fine_detail_difficult"] = True
        self.assertNotIn("pr_path_without_difficult_visual_node", validate_affordance_proposal(p))

    def test_score_is_not_task_truth(self):
        p = proposal()
        p["decision"] = "ambiguous"
        p["observable_entities"] = []
        p["observable_relations"] = []
        p["candidate_reasoning_paths"] = []
        p["r_score"] = 0
        p["closed_world_question_feasible"] = False
        self.assertEqual(validate_affordance_proposal(p), [])

    def test_malformed_model_values_return_errors_instead_of_crashing(self):
        p = proposal()
        p["candidate_reasoning_paths"][0]["evidence_entity_ids"] = [{"bad": "type"}]
        self.assertIn("path_0_invalid_evidence_entities", validate_affordance_proposal(p))

    def test_fallback_never_promotes_unrecoverable_primary(self):
        lock = affordance_fallback_lock(None)
        self.assertEqual(lock["decision"], "ambiguous")
        prompt = render_compact_affordance_prompt(None)
        self.assertIn('"decision": "ambiguous"', prompt)
        rewritten = proposal()
        self.assertEqual(
            affordance_lock_errors(rewritten, lock), ["fallback_decision_mismatch"]
        )

    def test_valid_candidate_decision_is_locked(self):
        lock = affordance_fallback_lock(proposal())
        self.assertEqual(lock, {
            "decision": "candidate",
            "closed_world_question_feasible": True,
            "external_identification_needed": False,
        })

    def test_invalid_candidate_can_downgrade_during_fallback(self):
        invalid = proposal()
        invalid["r_score"] = 1
        self.assertIn("candidate_r_score_below_two", validate_affordance_proposal(invalid))
        self.assertEqual(affordance_fallback_lock(invalid), {})
        rewritten = proposal()
        rewritten.update({
            "decision": "reject",
            "observable_entities": [],
            "observable_relations": [],
            "candidate_reasoning_paths": [],
            "r_score": 0,
            "closed_world_question_feasible": False,
        })
        self.assertEqual(affordance_lock_errors(rewritten, {}), [])

    def test_compact_prompt_preserves_nested_object_schema(self):
        invalid = proposal()
        invalid["r_score"] = 1
        prompt = render_compact_affordance_prompt(invalid)
        self.assertIn('"id":"e1"', prompt)
        self.assertIn('"source_id":"e1"', prompt)
        self.assertIn('"family":"R"', prompt)
        self.assertIn("integers 0-3", prompt)


if __name__ == "__main__":
    unittest.main()
