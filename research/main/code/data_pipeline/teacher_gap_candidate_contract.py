"""Frozen candidate-stage contracts for the exploratory teacher-gap pilot.

The functions in this module are deliberately model-independent.  They render
only protocol-approved fields and validate model proposals without promoting
them to human truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from reasoning_graph_contract import validate_guide_graph_alignment, validate_reasoning_graph


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
Canonical complete answer: {canonical_answer}
Accepted aliases grouped by required answer item: {answer_alias_groups}

The canonical answer may be a JSON array string. If so, every array item is required; the items are not interchangeable aliases.
Classify whether the complete answer and every necessary intermediate claim are entailed by the image/question. Every evidence-node claim must be a direct visual observation, never a statement about what the question asks or a general learned association. The answer-node claim must explicitly cover every canonical answer item without denying any required item.
image_only means uniquely answerable without any unprovided fact. external_required means at least one necessary fact is absent. ambiguous means the boundary, reference, image, or uniqueness cannot be verified.
R requires at least two distinct visual entities/regions/states and at least two indispensable transformations. PR additionally requires a difficult local node on that same necessary chain.
Return exactly one JSON object and no other text:
{{"knowledge_dependency_label":"image_only|external_required|ambiguous","reason_codes":["code"],"reference_entailed":true,"unique_answer":true,"image_is_necessary":true,"full_image_reference_valid":true,"task_family":"R|PR|null","reasoning_depth":2,"required_evidence":["E1","E2"],"minimal_reasoning_graph":{{"nodes":[{{"id":"e1","type":"evidence","claim":"..."}},{{"id":"e2","type":"evidence","claim":"..."}},{{"id":"h1","type":"intermediate","claim":"..."}},{{"id":"a","type":"answer","claim":"..."}}],"edges":[["e1","h1"],["e2","h1"],["h1","a"]]}},"explanation":"brief"}}
For a non-qualifying sample use JSON null for task_family, reasoning_depth 0, an empty required_evidence list, and {{"nodes":[],"edges":[]}} for minimal_reasoning_graph."""

CONSTRUCT_PROMPT = """You are constructing candidate guides and multiple-choice options for a closed-world visual reasoning pilot.
Use only the supplied complete image, question, reference answer aliases, and audited visual evidence/path. Never use Wikipedia evidence or URLs.
Question: {question}
Canonical complete answer (copy exactly into correct_answer_text): {canonical_answer}
Accepted aliases grouped by required answer item: {answer_alias_groups}
Required answer cardinality: {answer_cardinality}
Required visual evidence: {required_evidence}
Audited reasoning graph: {reasoning_graph}

Create 1-4 indispensable guide QA pairs and three plausible same-type distractors. Each supports_next must name the existing graph node whose claim the guide elicits or supports; evidence, intermediate, and answer nodes are all valid targets. Guides must be answerable from the image or prior guide state and must not state/paraphrase the final answer or reveal an option label. Distractors must be rejectable from the image without outside knowledge. When answer cardinality is greater than one, every distractor must be a JSON array of exactly that many non-empty string items (a JSON-array string with the same content is also accepted).
Return exactly one JSON object and no other text:
{{"decision":"accept_candidate|reject","guides":[{{"question":"...","answer":"...","supports_next":"node id"}}],"correct_answer_text":"...","distractors":["...","...","..."],"generic_context_seed":"inspect-compare-verify","answer_leakage":false,"external_fact_injected":false,"rejection_reason":""}}"""

COMPACT_BLIND_PROMPT = """Inspect the complete image and answer this closed-world audit question using no outside facts:
{question}
Return one compact JSON object, no markdown or analysis, with exactly these keys:
{{"candidate_answer":"under 12 words","visible_evidence":["observation under 12 words","optional second observation under 12 words"],"image_is_necessary":true,"requires_unprovided_fact":false,"ambiguous":false,"uncertainty":"low|medium|high"}}
Use 1-2 distinct visible_evidence items. Stop immediately after the closing brace."""

COMPACT_AUDIT_PROMPT = """Rewrite the first audit attempt as one schema-valid compact JSON object. Do not redo or reverse its recoverable high-level decision.
Question: {question}
Canonical complete answer: {canonical_answer}
Accepted aliases grouped by required answer item: {answer_alias_groups}
Locked decision fields: {decision_lock}
Target shape for this record: {target_shape}
Copy every locked value exactly. JSON null has no quotes. Include every key shown. For image_only only, replace short evidence/claim placeholders with image-grounded content while keeping a two-evidence DAG. Return no markdown or analysis and stop after the closing brace."""

COMPACT_CONSTRUCT_PROMPT = """Using only the complete image and audited visual path, construct short guides and options for this question.
Question: {question}
Canonical complete answer (copy exactly into correct_answer_text): {canonical_answer}
Accepted aliases grouped by required answer item: {answer_alias_groups}
Required answer cardinality: {answer_cardinality}
Required visual evidence: {required_evidence}
Audited reasoning graph: {reasoning_graph}
Recoverable first-attempt fields to preserve when present: {decision_lock}
Return one compact JSON object, no markdown or analysis, with exactly these keys:
{{"decision":"accept_candidate","guides":[{{"question":"under 16 words","answer":"under 12 words","supports_next":"existing node id"}}],"correct_answer_text":{canonical_answer_json},"distractors":{distractor_example},"generic_context_seed":"inspect-compare-verify","answer_leakage":false,"external_fact_injected":false,"rejection_reason":""}}
Use 1-3 guides. Every supports_next must name the existing graph node whose claim the guide elicits or supports. Never state the complete final answer in a guide. When answer cardinality is greater than one, every distractor must be a JSON array of exactly that many non-empty string items. Stop after the closing brace. If safe construction is impossible, return the same keys with decision "reject", empty guides/options/answer/seed, both booleans false, and a short rejection_reason."""


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _answers(candidate: dict) -> list[str]:
    value = candidate.get("reference_answers")
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _answer_alias_groups(candidate: dict) -> list[list[str]]:
    value = candidate.get("reference_answer_alias_groups")
    if not isinstance(value, list):
        return []
    groups = []
    for group in value:
        if not isinstance(group, list):
            return []
        rendered = [_text(item) for item in group if _text(item)]
        if not rendered:
            return []
        groups.append(rendered)
    return groups


def canonical_answer_text(candidate: dict) -> str:
    """Return the one frozen MC option representing the complete target.

    M3VQA is multi-answer: ``answers`` contains canonical answer items while
    each ``answer_evals`` sub-list contains aliases for one required item.
    A JSON array string keeps a multi-item answer distinct from an alias list.
    """

    answers = _answers(candidate)
    if len(answers) == 1:
        return answers[0]
    return json.dumps(answers, ensure_ascii=False, separators=(",", ":"))


def validate_candidate(candidate: dict) -> list[str]:
    errors = []
    for key in ("candidate_id", "source_image_id", "image_original", "final_question_source"):
        if not _text(candidate.get(key)):
            errors.append(f"missing_{key}")
    answers = _answers(candidate)
    aliases = _answer_alias_groups(candidate)
    if not answers:
        errors.append("missing_reference_answers")
    if not aliases:
        errors.append("missing_reference_answer_alias_groups")
    elif len(aliases) != len(answers):
        errors.append("answer_alias_group_count_mismatch")
    else:
        for answer, group in zip(answers, aliases):
            normalized_group = {_normalize_for_leakage(value) for value in group}
            if _normalize_for_leakage(answer) not in normalized_group:
                errors.append("canonical_answer_missing_from_alias_group")
                break
    expected = canonical_answer_text(candidate) if answers else ""
    if _text(candidate.get("canonical_answer_text")) != expected:
        errors.append("canonical_answer_text_mismatch")
    return errors


def render_stage_prompt(candidate: dict, stage: str, audit_proposal: dict | None = None) -> str:
    """Render a prompt from an allow-list; candidate evidence fields are ignored."""

    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    errors = validate_candidate(candidate)
    if errors:
        raise ValueError(",".join(errors))
    question = _text(candidate["final_question_source"])
    canonical_answer = json.dumps(canonical_answer_text(candidate), ensure_ascii=False)
    answer_alias_groups = json.dumps(_answer_alias_groups(candidate), ensure_ascii=False)
    if stage == "blind":
        return BLIND_PROMPT.format(question=question)
    if stage == "audit":
        return AUDIT_PROMPT.format(
            question=question,
            canonical_answer=canonical_answer,
            answer_alias_groups=answer_alias_groups,
        )
    if not isinstance(audit_proposal, dict):
        raise ValueError("construct stage requires audit_proposal")
    return CONSTRUCT_PROMPT.format(
        question=question,
        canonical_answer=canonical_answer,
        answer_alias_groups=answer_alias_groups,
        answer_cardinality=len(_answers(candidate)),
        required_evidence=json.dumps(audit_proposal.get("required_evidence", []), ensure_ascii=False),
        reasoning_graph=json.dumps(audit_proposal.get("minimal_reasoning_graph", {}), ensure_ascii=False),
    )


def render_compact_stage_prompt(
    candidate: dict,
    stage: str,
    audit_proposal: dict | None = None,
    primary_proposal: dict | None = None,
) -> str:
    """Render the recorded one-shot fallback prompt from the same allow-list."""

    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    errors = validate_candidate(candidate)
    if errors:
        raise ValueError(",".join(errors))
    question = _text(candidate["final_question_source"])
    canonical_answer = json.dumps(canonical_answer_text(candidate), ensure_ascii=False)
    answer_alias_groups = json.dumps(_answer_alias_groups(candidate), ensure_ascii=False)
    decision_lock = fallback_decision_fields(stage, primary_proposal, candidate)
    rendered_lock = json.dumps(decision_lock, ensure_ascii=False, sort_keys=True)
    if stage == "blind":
        return COMPACT_BLIND_PROMPT.format(question=question, decision_lock=rendered_lock)
    if stage == "audit":
        target = audit_fallback_target(primary_proposal)
        return COMPACT_AUDIT_PROMPT.format(
            question=question,
            canonical_answer=canonical_answer,
            answer_alias_groups=answer_alias_groups,
            decision_lock=rendered_lock,
            target_shape=json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        )
    if not isinstance(audit_proposal, dict):
        raise ValueError("construct stage requires audit_proposal")
    return COMPACT_CONSTRUCT_PROMPT.format(
        question=question,
        canonical_answer=canonical_answer,
        canonical_answer_json=canonical_answer,
        answer_alias_groups=answer_alias_groups,
        answer_cardinality=len(_answers(candidate)),
        required_evidence=json.dumps(audit_proposal.get("required_evidence", []), ensure_ascii=False),
        reasoning_graph=json.dumps(audit_proposal.get("minimal_reasoning_graph", {}), ensure_ascii=False),
        decision_lock=rendered_lock,
        distractor_example=json.dumps(
            [
                [f"wrong item {option + 1}.{item + 1}" for item in range(len(_answers(candidate)))]
                for option in range(3)
            ]
            if len(_answers(candidate)) > 1
            else ["wrong option 1", "wrong option 2", "wrong option 3"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def fallback_decision_fields(
    stage: str,
    proposal: dict | None,
    candidate: dict | None = None,
) -> dict:
    """Return only valid high-level fields a compact retry must preserve.

    Audit retries are conservative: an unparseable first attempt becomes
    ``ambiguous`` rather than being promoted to an image-only candidate.
    """

    proposal = proposal if isinstance(proposal, dict) else {}
    locked: dict[str, Any] = {}
    if stage == "blind":
        # Blind answers are open-ended and may themselves contain the external
        # premise that caused the schema failure.  Reusing them made one r5b
        # retry loop.  The independent audit lock is the semantic safeguard.
        return locked
    if stage == "audit":
        label = proposal.get("knowledge_dependency_label")
        if label not in KNOWLEDGE_LABELS:
            label = "ambiguous"
        locked["knowledge_dependency_label"] = label
        for key in ("reference_entailed", "unique_answer", "image_is_necessary", "full_image_reference_valid"):
            locked[key] = proposal[key] if isinstance(proposal.get(key), bool) else False
        family = proposal.get("task_family")
        locked["task_family"] = family if label == "image_only" and family in TASK_FAMILIES else None
        return locked
    if stage == "construct":
        # A retry may conservatively downgrade an invalid accept proposal to
        # reject. It must never promote a primary reject to accept.
        if proposal.get("decision") == "reject":
            locked["decision"] = "reject"
        return locked
    raise ValueError(f"unsupported stage: {stage}")


def audit_fallback_target(primary_proposal: dict | None) -> dict:
    """Build the record-specific audit shape shown to the compact retry."""

    proposal = primary_proposal if isinstance(primary_proposal, dict) else {}
    locked = fallback_decision_fields("audit", proposal)
    reasons = proposal.get("reason_codes")
    if not isinstance(reasons, list) or not reasons or any(not _text(value) for value in reasons):
        reasons = ["unresolved_primary_schema"]
    if locked["knowledge_dependency_label"] == "image_only":
        explanation = _text(proposal.get("explanation")) or "Visible evidence supports the locked image-only decision."
    else:
        explanation = (
            f"Primary audit marked this record {locked['knowledge_dependency_label']}; "
            "it is not a closed-world image-only candidate."
        )
    target = {
        "knowledge_dependency_label": locked["knowledge_dependency_label"],
        "reason_codes": reasons,
        "reference_entailed": locked["reference_entailed"],
        "unique_answer": locked["unique_answer"],
        "image_is_necessary": locked["image_is_necessary"],
        "full_image_reference_valid": locked["full_image_reference_valid"],
        "task_family": locked["task_family"],
        "reasoning_depth": 0,
        "required_evidence": [],
        "minimal_reasoning_graph": {"nodes": [], "edges": []},
        "explanation": explanation,
    }
    if locked["knowledge_dependency_label"] == "image_only":
        target.update({
            "task_family": locked["task_family"] or "R",
            "reasoning_depth": 2,
            "required_evidence": ["replace with first visible evidence", "replace with second visible evidence"],
            "minimal_reasoning_graph": {
                "nodes": [
                    {"id": "e1", "type": "evidence", "claim": "replace with first visible evidence"},
                    {"id": "e2", "type": "evidence", "claim": "replace with second visible evidence"},
                    {"id": "h1", "type": "intermediate", "claim": "replace with intermediate relation"},
                    {"id": "a", "type": "answer", "claim": "replace with answer claim"},
                ],
                "edges": [["e1", "h1"], ["e2", "h1"], ["h1", "a"]],
            },
        })
    return target


def decision_lock_errors(proposal: dict | None, locked: dict) -> list[str]:
    """Reject a retry that changed any recoverable first-attempt decision."""

    if not isinstance(proposal, dict):
        return []
    if any(proposal.get(key) != value for key, value in locked.items()):
        return ["fallback_decision_mismatch"]
    return []


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


def validate_audit(proposal: dict, candidate: dict | None = None) -> list[str]:
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
        elif any(_is_placeholder_text(item) for item in evidence):
            errors.append("placeholder_required_evidence")
        graph_errors, _ = validate_reasoning_graph(
            proposal.get("minimal_reasoning_graph"),
            declared_depth=depth,
            require_two_evidence=True,
        )
        errors.extend(graph_errors)
        graph = proposal.get("minimal_reasoning_graph")
        if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
            if any(
                isinstance(node, dict) and _is_placeholder_text(node.get("claim"))
                for node in graph["nodes"]
            ):
                errors.append("placeholder_reasoning_graph")
            evidence_claims = [_text(item) for item in evidence] + [
                _text(node.get("claim")) for node in graph["nodes"]
                if isinstance(node, dict) and node.get("type") == "evidence"
            ]
            if any(_is_meta_or_nonvisual_evidence(claim) for claim in evidence_claims):
                errors.append("nonvisual_or_meta_evidence_claim")
            if isinstance(candidate, dict):
                answer_claims = [
                    _text(node.get("claim")) for node in graph["nodes"]
                    if isinstance(node, dict) and node.get("type") == "answer"
                ]
                if len(answer_claims) == 1 and not _claim_covers_complete_answer(answer_claims[0], candidate):
                    errors.append("answer_node_missing_canonical_item")
                if len(answer_claims) == 1 and _claim_negates_required_answer(answer_claims[0], candidate):
                    errors.append("answer_node_negates_required_item")
                if _entailed_explanation_conflict(proposal.get("explanation"), candidate):
                    errors.append("reference_entailment_explanation_conflict")
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


def _is_placeholder_text(value: Any) -> bool:
    normalized = _normalize_for_leakage(_text(value))
    return normalized in {"e1", "e2", "evidence 1", "evidence 2"} or "replace with" in normalized


def _is_meta_or_nonvisual_evidence(value: Any) -> bool:
    normalized = _normalize_for_leakage(_text(value))
    phrases = (
        "the question asks", "question asks", "the question states",
        "reference answer", "canonical answer", "from the options",
        "well known", "is traditionally", "is a traditional",
        "is associated with", "are associated with",
        "which is indicative of", "which are indicative of",
        "is characteristic of", "are characteristic of",
    )
    return any(phrase in normalized for phrase in phrases)


def _claim_covers_complete_answer(claim: str, candidate: dict) -> bool:
    text = _normalize_for_leakage(claim)
    groups = [
        tuple(sorted({_normalize_for_leakage(alias) for alias in group if _normalize_for_leakage(alias)}))
        for group in _answer_alias_groups(candidate)
    ]
    required_groups = Counter(groups)
    return bool(groups) and all(
        max((_alias_occurrences(text, alias) for alias in group), default=0) >= multiplicity
        for group, multiplicity in required_groups.items()
    )


def _claim_negates_required_answer(value: Any, candidate: dict) -> bool:
    normalized = _normalize_for_leakage(_text(value))
    clauses = re.split(r"[.;:]|\bbut\b|\bhowever\b", normalized)
    absence = ("not present", "not shown", "not visible", "is absent", "are absent")
    aliases = {
        _normalize_for_leakage(alias)
        for group in _answer_alias_groups(candidate)
        for alias in group
        if _normalize_for_leakage(alias)
    }
    return any(
        any(_alias_occurrences(clause, alias) for alias in aliases)
        and any(phrase in clause for phrase in absence)
        for clause in clauses
    )


def _entailed_explanation_conflict(value: Any, candidate: dict) -> bool:
    normalized = _normalize_for_leakage(_text(value))
    absence = any(phrase in normalized for phrase in (
        "not visually represented", "not represented in the image",
        "not present in the image", "not shown in the image",
        "absent from the image",
    ))
    aliases = {
        _normalize_for_leakage(alias)
        for group in _answer_alias_groups(candidate)
        for alias in group
        if _normalize_for_leakage(alias)
    }
    target = (
        "reference answer" in normalized
        or "canonical answer" in normalized
        or any(_alias_occurrences(normalized, alias) for alias in aliases)
    )
    return absence and target


def obvious_answer_leakage(proposal: dict, candidate: dict) -> bool:
    """Fail if one guide directly contains the complete final answer.

    For a multi-item target, an atomic item may be a legitimate intermediate
    state. It is leakage only when a single guide states at least one alias
    from every required answer group. Human review still judges indirect and
    cross-guide leakage.
    """

    groups = [
        tuple(sorted({_normalize_for_leakage(alias) for alias in group if _normalize_for_leakage(alias)}))
        for group in _answer_alias_groups(candidate)
    ]
    if not groups:
        return True
    for guide in proposal.get("guides", []):
        if not isinstance(guide, dict):
            continue
        for key in ("question", "answer"):
            text = _normalize_for_leakage(_text(guide.get(key)))
            required_groups = Counter(groups)
            if all(
                max((_alias_occurrences(text, alias) for alias in group), default=0) >= multiplicity
                for group, multiplicity in required_groups.items()
            ):
                return True
    return False


def _alias_occurrences(text: str, alias: str) -> int:
    if not alias:
        return 0
    return len(re.findall(rf"(?<!\w){re.escape(alias)}(?!\w)", text))


def _multi_item_option_values(value: Any) -> list[str] | None:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
    if not isinstance(parsed, list):
        return None
    if not all(isinstance(item, str) and _text(item) for item in parsed):
        return None
    return parsed


def _valid_multi_item_option(value: Any, cardinality: int) -> bool:
    parsed = _multi_item_option_values(value)
    return (
        parsed is not None
        and len(parsed) == cardinality
    )


def _option_distinct_key(value: Any, cardinality: int) -> tuple[str, ...]:
    if cardinality <= 1:
        return (_normalize_for_leakage(_text(value)),)
    parsed = _multi_item_option_values(value)
    if parsed is None:
        return ("invalid_option", _text(value))
    return tuple(sorted(_normalize_for_leakage(_text(item)) for item in parsed))


def validate_construct(
    proposal: dict,
    candidate: dict,
    audit_proposal: dict | None = None,
) -> list[str]:
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
    if not isinstance(audit_proposal, dict):
        errors.append("missing_audit_proposal_for_construct")
    elif isinstance(guides, list):
        errors.extend(
            f"guide_graph:{error}"
            for error in validate_guide_graph_alignment(
                guides, audit_proposal.get("minimal_reasoning_graph")
            )
        )
    correct_raw = _text(proposal.get("correct_answer_text"))
    correct = _normalize_for_leakage(correct_raw)
    expected = canonical_answer_text(candidate)
    if not correct or correct_raw != expected:
        errors.append("correct_answer_not_canonical_complete_answer")
    distractors = proposal.get("distractors")
    cardinality = len(_answers(candidate))
    if not isinstance(distractors, list) or len(distractors) != 3:
        errors.append("invalid_distractors")
    elif cardinality == 1 and any(not isinstance(item, str) or not _text(item) for item in distractors):
        errors.append("invalid_distractors")
    elif cardinality > 1 and any(
        not _valid_multi_item_option(item, cardinality)
        for item in distractors
    ):
        errors.append("invalid_distractor_cardinality")
    elif len({
        _option_distinct_key(item, cardinality)
        for item in [correct_raw, *distractors]
    }) != 4:
        errors.append("options_not_distinct")
    for key in ("answer_leakage", "external_fact_injected"):
        if proposal.get(key) is not False:
            errors.append(f"{key}_not_false")
    if obvious_answer_leakage(proposal, candidate):
        errors.append("obvious_answer_leakage")
    if not _text(proposal.get("generic_context_seed")):
        errors.append("missing_generic_context_seed")
    if _text(proposal.get("rejection_reason")):
        errors.append("accepted_candidate_has_rejection_reason")
    return errors


def proposal_errors(
    stage: str,
    proposal: dict,
    candidate: dict,
    audit_proposal: dict | None = None,
) -> list[str]:
    if not isinstance(proposal, dict):
        return ["proposal_not_object"]
    if stage == "blind":
        return validate_blind(proposal)
    if stage == "audit":
        return validate_audit(proposal, candidate)
    if stage == "construct":
        return validate_construct(proposal, candidate, audit_proposal)
    return ["unsupported_stage"]


def is_construction_candidate(blind: dict, audit: dict, candidate: dict | None = None) -> bool:
    """Conservative model-candidate rule; human verification is still mandatory."""

    return (
        not validate_blind(blind)
        and not validate_audit(audit, candidate)
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
