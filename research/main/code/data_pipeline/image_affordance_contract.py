"""Question-independent machine-proposal contract for image affordance.

This stage describes visible structure that may support a future closed-world
R/PR question.  A valid proposal is neither a task-family label nor human
truth; both require a later question and independent review.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


DECISIONS = {"candidate", "reject", "ambiguous"}
FAMILIES = {"R", "PR"}
REGIONS = {
    "top-left", "top", "top-right", "left", "center", "right",
    "bottom-left", "bottom", "bottom-right", "distributed", "whole",
}
QUESTION_TYPES = {"compare", "spatial", "connection", "order", "state_change", "chart_relation"}

AFFORDANCE_PROMPT = """Inspect only this complete image. No source question, answer, dataset evidence, URL, or model result is available.
Describe visible structure that could naturally support a future closed-world visual reasoning question. Do not identify a person, place, brand, species, event, or hidden fact from memory. Do not invent a question or answer yet.
R requires at least two distinct visible entities/regions and at least two indispensable transformations. PR requires the same reasoning chain plus a genuinely difficult local visual detail on that chain. A long description is not a reasoning chain.
Return exactly one concise JSON object and no other text:
{"decision":"candidate|reject|ambiguous","observable_entities":[{"id":"e1","description":"visible description","region":"top-left|top|top-right|left|center|right|bottom-left|bottom|bottom-right|distributed|whole","fine_detail_difficult":false}],"observable_relations":[{"source_id":"e1","relation":"visible relation","target_id":"e2"}],"candidate_reasoning_paths":[{"family":"R|PR","evidence_entity_ids":["e1","e2"],"transformations":["derive step 1","derive step 2"],"natural_question_type":"compare|spatial|connection|order|state_change|chart_relation"}],"p_score":0,"r_score":0,"pr_score":0,"closed_world_question_feasible":false,"external_identification_needed":false,"uncertainty":"brief"}
Scores are integers 0-3. Use candidate only when at least one natural path satisfies the definitions and could have a unique answer from the image alone. Use reject when no such path is visible; ambiguous when image quality or structure prevents a reliable decision. Stop after the closing brace."""


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_affordance_prompt(candidate: dict) -> str:
    errors = validate_image_candidate(candidate)
    if errors:
        raise ValueError(",".join(errors))
    return AFFORDANCE_PROMPT


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def validate_image_candidate(candidate: dict) -> list[str]:
    errors = []
    for key in ("image_candidate_id", "source_image_id", "image_original", "image_sha256"):
        if not _text(candidate.get(key)):
            errors.append(f"missing_{key}")
    if candidate.get("source_questions_exposed_to_screen") is not False:
        errors.append("source_questions_must_not_be_exposed")
    if candidate.get("question_origin") != "none_image_screening":
        errors.append("invalid_question_origin")
    forbidden = {
        "final_question_source", "reference_answers", "reference_answer_alias_groups",
        "canonical_answer_text", "evidence", "evidence_urls", "question_hop",
        "reasoning_depth_candidate", "task_family_candidate",
    }
    if forbidden & candidate.keys():
        errors.append("question_firewall_violation")
    return errors


def _validate_entities(value: Any) -> tuple[list[str], set[str], set[str]]:
    if not isinstance(value, list):
        return ["observable_entities_not_list"], set(), set()
    errors, ids, difficult_ids = [], set(), set()
    for index, entity in enumerate(value):
        if not isinstance(entity, dict):
            errors.append(f"entity_{index}_not_object")
            continue
        entity_id = _text(entity.get("id"))
        if not re.fullmatch(r"e[1-9][0-9]*", entity_id):
            errors.append(f"entity_{index}_invalid_id")
        elif entity_id in ids:
            errors.append(f"duplicate_entity_id:{entity_id}")
        else:
            ids.add(entity_id)
        if not _text(entity.get("description")):
            errors.append(f"entity_{index}_missing_description")
        if entity.get("region") not in REGIONS:
            errors.append(f"entity_{index}_invalid_region")
        if not isinstance(entity.get("fine_detail_difficult"), bool):
            errors.append(f"entity_{index}_invalid_fine_detail")
        if entity.get("fine_detail_difficult") is True and entity_id in ids:
            difficult_ids.add(entity_id)
    return errors, ids, difficult_ids


def validate_affordance_proposal(proposal: dict) -> list[str]:
    if not isinstance(proposal, dict):
        return ["proposal_not_object"]
    errors = []
    decision = proposal.get("decision")
    if decision not in DECISIONS:
        errors.append("invalid_decision")
    entity_errors, entity_ids, difficult_ids = _validate_entities(proposal.get("observable_entities"))
    errors.extend(entity_errors)

    relations = proposal.get("observable_relations")
    if not isinstance(relations, list):
        errors.append("observable_relations_not_list")
        relations = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            errors.append(f"relation_{index}_not_object")
            continue
        source_id, target_id = _text(relation.get("source_id")), _text(relation.get("target_id"))
        if source_id not in entity_ids or target_id not in entity_ids:
            errors.append(f"relation_{index}_unknown_entity")
        if source_id == target_id:
            errors.append(f"relation_{index}_self_relation")
        if not _text(relation.get("relation")):
            errors.append(f"relation_{index}_missing_relation")

    paths = proposal.get("candidate_reasoning_paths")
    if not isinstance(paths, list):
        errors.append("candidate_reasoning_paths_not_list")
        paths = []
    for index, path in enumerate(paths):
        if not isinstance(path, dict):
            errors.append(f"path_{index}_not_object")
            continue
        if path.get("family") not in FAMILIES:
            errors.append(f"path_{index}_invalid_family")
        evidence_ids = path.get("evidence_entity_ids")
        valid_evidence_ids = (
            isinstance(evidence_ids, list)
            and all(isinstance(value, str) and _text(value) for value in evidence_ids)
        )
        normalized_evidence_ids = [_text(value) for value in evidence_ids] if valid_evidence_ids else []
        if not valid_evidence_ids or len(set(normalized_evidence_ids)) < 2 or any(
            value not in entity_ids for value in normalized_evidence_ids
        ):
            errors.append(f"path_{index}_invalid_evidence_entities")
        transformations = path.get("transformations")
        if not isinstance(transformations, list) or len(transformations) < 2 or any(not _text(value) for value in transformations):
            errors.append(f"path_{index}_invalid_transformations")
        if path.get("natural_question_type") not in QUESTION_TYPES:
            errors.append(f"path_{index}_invalid_question_type")
        if path.get("family") == "PR" and not difficult_ids.intersection(normalized_evidence_ids):
            errors.append(f"path_{index}_pr_missing_difficult_evidence")

    for key in ("p_score", "r_score", "pr_score"):
        value = proposal.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
            errors.append(f"invalid_{key}")
    for key in ("closed_world_question_feasible", "external_identification_needed"):
        if not isinstance(proposal.get(key), bool):
            errors.append(f"invalid_{key}")
    if not _text(proposal.get("uncertainty")):
        errors.append("missing_uncertainty")

    if decision == "candidate":
        if len(entity_ids) < 2:
            errors.append("candidate_requires_two_entities")
        if not relations:
            errors.append("candidate_requires_visible_relation")
        if not paths:
            errors.append("candidate_requires_reasoning_path")
        if proposal.get("closed_world_question_feasible") is not True:
            errors.append("candidate_not_closed_world_feasible")
        if proposal.get("external_identification_needed") is not False:
            errors.append("candidate_requires_external_identification")
        if not isinstance(proposal.get("r_score"), int) or proposal.get("r_score", -1) < 2:
            errors.append("candidate_r_score_below_two")
        if any(isinstance(path, dict) and path.get("family") == "PR" for path in paths):
            if not difficult_ids or not isinstance(proposal.get("pr_score"), int) or proposal.get("pr_score", -1) < 2:
                errors.append("pr_path_without_difficult_visual_node")
    return sorted(set(errors))
