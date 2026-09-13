"""Frozen candidate-stage contracts for the exploratory teacher-gap pilot.

The functions in this module are deliberately model-independent.  They render
only protocol-approved fields and validate model proposals without promoting
them to human truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from reasoning_graph_contract import validate_reasoning_graph


STAGES = ("blind", "audit", "construct")
KNOWLEDGE_LABELS = ("image_only", "external_required", "ambiguous")
TASK_FAMILIES = ("R", "PR")

BLIND_PROMPT = """You are the blind image solver in a closed-world visual reasoning audit.
Use only the supplied complete image and question. You do not have Wikipedia evidence or URLs.
Question: {question}

Return exactly one JSON object and no other text:
{{"candidate_answer":"brief answer","visible_evidence":["observation 1","observation 2"],"image_is_necessary":true,"requires_unprovided_fact":false,"ambiguous":false,"uncertainty":"brief"}}
Keep `visible_evidence` to 1-4 distinct observations, with at most 24 words per observation. Do not repeat an observation. Keep the complete JSON response concise.
Do not identify people or places from memory and do not guess facts absent from the image/question."""

AUDIT_PROMPT = """You are the entailment auditor for a closed-world visual reasoning pilot.
Use only the supplied complete image, question, and reference answer aliases. No official evidence text or URL is available.
Question: {question}
Reference answer aliases: {answers}

Classify whether the answer and every necessary intermediate claim are entailed by the image/question.
image_only means uniquely answerable without any unprovided fact. external_required means at least one necessary fact is absent. ambiguous means the boundary, reference, image, or uniqueness cannot be verified.
R requires at least two distinct visual entities/regions/states and at least two indispensable transformations. PR additionally requires a difficult local node on that same necessary chain.
Return exactly one JSON object and no other text:
{{"knowledge_dependency_label":"image_only|external_required|ambiguous","reason_codes":["code"],"reference_entailed":true,"unique_answer":true,"image_is_necessary":true,"full_image_reference_valid":true,"task_family":"R|PR|null","reasoning_depth":2,"required_evidence":["E1","E2"],"minimal_reasoning_graph":{{"nodes":[{{"id":"e1","type":"evidence","claim":"..."}},{{"id":"e2","type":"evidence","claim":"..."}},{{"id":"h1","type":"intermediate","claim":"..."}},{{"id":"a","type":"answer","claim":"..."}}],"edges":[["e1","h1"],["e2","h1"],["h1","a"]]}},"explanation":"brief"}}
For a non-qualifying sample use JSON null for task_family, reasoning_depth 0, an empty required_evidence list, and {{"nodes":[],"edges":[]}} for minimal_reasoning_graph."""

CONSTRUCT_PROMPT = """You are constructing candidate guides and multiple-choice options for a closed-world visual reasoning pilot.
Use only the supplied complete image, question, reference answer aliases, and audited visual evidence/path. Never use Wikipedia evidence or URLs.
Question: {question}
Reference answer aliases: {answers}
Required visual evidence: {required_evidence}
Audited reasoning graph: {reasoning_graph}

Create 1-4 indispensable guide QA pairs and three plausible same-type distractors. Guides must support the next reasoning node, must be answerable from the image or prior guide state, and must not state/paraphrase the final answer or reveal an option label. Distractors must be rejectable from the image without outside knowledge.
Return exactly one JSON object and no other text:
{{"decision":"accept_candidate|reject","guides":[{{"question":"...","answer":"...","supports_next":"node id"}}],"correct_answer_text":"...","distractors":["...","...","..."],"generic_context_seed":"inspect-compare-verify","answer_leakage":false,"external_fact_injected":false,"rejection_reason":""}}"""


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _answers(candidate: dict) -> list[str]:
    value = candidate.get("reference_answers")
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def validate_candidate(candidate: dict) -> list[str]:
    errors = []
    for key in ("candidate_id", "source_image_id", "image_original", "final_question_source"):
        if not _text(candidate.get(key)):
            errors.append(f"missing_{key}")
    if not _answers(candidate):
        errors.append("missing_reference_answers")
    return errors


def render_stage_prompt(candidate: dict, stage: str, audit_proposal: dict | None = None) -> str:
    """Render a prompt from an allow-list; candidate evidence fields are ignored."""

    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    errors = validate_candidate(candidate)
    if errors:
        raise ValueError(",".join(errors))
    question = _text(candidate["final_question_source"])
    answers = json.dumps(_answers(candidate), ensure_ascii=False)
    if stage == "blind":
        return BLIND_PROMPT.format(question=question)
    if stage == "audit":
        return AUDIT_PROMPT.format(question=question, answers=answers)
    if not isinstance(audit_proposal, dict):
        raise ValueError("construct stage requires audit_proposal")
    return CONSTRUCT_PROMPT.format(
        question=question,
        answers=answers,
        required_evidence=json.dumps(audit_proposal.get("required_evidence", []), ensure_ascii=False),
        reasoning_graph=json.dumps(audit_proposal.get("minimal_reasoning_graph", {}), ensure_ascii=False),
    )


def validate_blind(proposal: dict) -> list[str]:
    errors = []
    if not _text(proposal.get("candidate_answer")):
        errors.append("missing_candidate_answer")
    evidence = proposal.get("visible_evidence")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 4 or any(not _text(item) for item in evidence):
        errors.append("invalid_visible_evidence")
    elif any(len(_text(item).split()) > 24 for item in evidence):
        errors.append("visible_evidence_item_too_long")
    elif len({_normalize_for_leakage(_text(item)) for item in evidence}) != len(evidence):
        errors.append("duplicate_visible_evidence")
    for key in ("image_is_necessary", "requires_unprovided_fact", "ambiguous"):
        if not isinstance(proposal.get(key), bool):
            errors.append(f"invalid_{key}")
    if not _text(proposal.get("uncertainty")):
        errors.append("missing_uncertainty")
    return errors


def validate_audit(proposal: dict) -> list[str]:
    errors = []
    label = proposal.get("knowledge_dependency_label")
    if label not in KNOWLEDGE_LABELS:
        errors.append("invalid_knowledge_dependency_label")
    codes = proposal.get("reason_codes")
    if not isinstance(codes, list) or not codes or any(not _text(item) for item in codes):
        errors.append("invalid_reason_codes")
    for key in ("reference_entailed", "unique_answer", "image_is_necessary", "full_image_reference_valid"):
        if not isinstance(proposal.get(key), bool):
            errors.append(f"invalid_{key}")
    qualifying = label == "image_only"
    family = proposal.get("task_family")
    depth = proposal.get("reasoning_depth")
    evidence = proposal.get("required_evidence")
    if qualifying:
        if family not in TASK_FAMILIES:
            errors.append("invalid_task_family")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 2:
            errors.append("invalid_reasoning_depth")
        if not isinstance(evidence, list) or len(evidence) < 2 or any(not _text(item) for item in evidence):
            errors.append("invalid_required_evidence")
        elif len({_normalize_for_leakage(_text(item)) for item in evidence}) != len(evidence):
            errors.append("duplicate_required_evidence")
        graph_errors, _ = validate_reasoning_graph(
            proposal.get("minimal_reasoning_graph"),
            declared_depth=depth,
            require_two_evidence=True,
        )
        errors.extend(graph_errors)
    else:
        if family is not None:
            errors.append("nonqualifying_task_family_must_be_null")
        if depth != 0 or isinstance(depth, bool):
            errors.append("nonqualifying_reasoning_depth_must_be_zero")
        if not isinstance(evidence, list) or evidence:
            errors.append("nonqualifying_required_evidence_must_be_empty")
        graph = proposal.get("minimal_reasoning_graph")
        if not isinstance(graph, dict) or graph.get("nodes") != [] or graph.get("edges") != []:
            errors.append("nonqualifying_reasoning_graph_must_be_empty")
    if not _text(proposal.get("explanation")):
        errors.append("missing_explanation")
    return errors


def _normalize_for_leakage(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def obvious_answer_leakage(proposal: dict, reference_answers: list[str]) -> bool:
    answers = {_normalize_for_leakage(value) for value in reference_answers if _normalize_for_leakage(value)}
    if not answers:
        return True
    for guide in proposal.get("guides", []):
        if not isinstance(guide, dict):
            continue
        for key in ("question", "answer"):
            text = _normalize_for_leakage(_text(guide.get(key)))
            if any(answer == text or (len(answer) >= 4 and answer in text) for answer in answers):
                return True
    return False


def validate_construct(proposal: dict, reference_answers: list[str]) -> list[str]:
    errors = []
    if proposal.get("decision") not in {"accept_candidate", "reject"}:
        errors.append("invalid_decision")
        return errors
    if proposal.get("decision") == "reject":
        if not _text(proposal.get("rejection_reason")):
            errors.append("missing_rejection_reason")
        return errors
    guides = proposal.get("guides")
    if not isinstance(guides, list) or not 1 <= len(guides) <= 4:
        errors.append("invalid_guides")
    else:
        for guide in guides:
            if not isinstance(guide, dict) or any(not _text(guide.get(k)) for k in ("question", "answer", "supports_next")):
                errors.append("invalid_guide")
                break
    correct = _normalize_for_leakage(_text(proposal.get("correct_answer_text")))
    aliases = {_normalize_for_leakage(value) for value in reference_answers}
    if not correct or correct not in aliases:
        errors.append("correct_answer_not_reference_alias")
    distractors = proposal.get("distractors")
    if not isinstance(distractors, list) or len(distractors) != 3 or any(not _text(item) for item in distractors):
        errors.append("invalid_distractors")
    elif len({_normalize_for_leakage(item) for item in distractors} | {correct}) != 4:
        errors.append("options_not_distinct")
    for key in ("answer_leakage", "external_fact_injected"):
        if proposal.get(key) is not False:
            errors.append(f"{key}_not_false")
    if obvious_answer_leakage(proposal, reference_answers):
        errors.append("obvious_answer_leakage")
    if not _text(proposal.get("generic_context_seed")):
        errors.append("missing_generic_context_seed")
    if _text(proposal.get("rejection_reason")):
        errors.append("accepted_candidate_has_rejection_reason")
    return errors


def proposal_errors(stage: str, proposal: dict, candidate: dict) -> list[str]:
    if not isinstance(proposal, dict):
        return ["proposal_not_object"]
    if stage == "blind":
        return validate_blind(proposal)
    if stage == "audit":
        return validate_audit(proposal)
    if stage == "construct":
        return validate_construct(proposal, _answers(candidate))
    return ["unsupported_stage"]


def is_construction_candidate(blind: dict, audit: dict) -> bool:
    """Conservative model-candidate rule; human verification is still mandatory."""

    return (
        not validate_blind(blind)
        and not validate_audit(audit)
        and blind.get("image_is_necessary") is True
        and blind.get("requires_unprovided_fact") is False
        and blind.get("ambiguous") is False
        and audit.get("knowledge_dependency_label") == "image_only"
        and audit.get("reference_entailed") is True
        and audit.get("unique_answer") is True
        and audit.get("image_is_necessary") is True
        and audit.get("full_image_reference_valid") is True
        and audit.get("task_family") in TASK_FAMILIES
    )
