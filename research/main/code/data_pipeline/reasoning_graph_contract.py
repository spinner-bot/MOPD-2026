"""Shared structural validation for 1017 minimal visual reasoning graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


NODE_TYPES = {"evidence", "intermediate", "answer"}


def validate_reasoning_graph(
    graph: Any,
    declared_depth: int | None = None,
    *,
    require_two_evidence: bool = True,
) -> tuple[list[str], int | None]:
    """Validate a minimal DAG and compute longest evidence-to-answer depth.

    Edges are directed ``[dependency, dependent]``. Every node must contribute
    to the unique answer, evidence nodes must be roots, and the answer must be a
    sink. R/PR graphs require at least two distinct evidence nodes.
    """

    if not isinstance(graph, dict):
        return ["reasoning_graph_not_object"], None
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return ["reasoning_graph_requires_nodes_and_edges"], None

    errors: list[str] = []
    node_by_id: dict[str, dict] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"graph_node_{index}_not_object")
            continue
        node_id = str(node.get("id", "")).strip()
        node_type = node.get("type")
        claim = str(node.get("claim", "")).strip()
        if not node_id:
            errors.append(f"graph_node_{index}_missing_id")
        elif node_id in node_by_id:
            errors.append(f"duplicate_graph_node_id:{node_id}")
        else:
            node_by_id[node_id] = node
        if node_type not in NODE_TYPES:
            errors.append(f"graph_node_{node_id or index}_invalid_type")
        if not claim:
            errors.append(f"graph_node_{node_id or index}_missing_claim")

    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    seen_edges = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            errors.append(f"graph_edge_{index}_invalid_shape")
            continue
        source, target = (str(value).strip() for value in edge)
        if source not in node_by_id or target not in node_by_id:
            errors.append(f"graph_edge_{index}_unknown_endpoint")
            continue
        if source == target:
            errors.append(f"graph_edge_{index}_self_loop")
            continue
        pair = (source, target)
        if pair in seen_edges:
            errors.append(f"duplicate_graph_edge:{source}->{target}")
            continue
        seen_edges.add(pair)
        incoming[target].append(source)
        outgoing[source].append(target)

    type_counts = Counter(node.get("type") for node in node_by_id.values())
    if type_counts["answer"] != 1:
        errors.append("reasoning_graph_requires_one_answer")
    if type_counts["intermediate"] < 1:
        errors.append("reasoning_graph_requires_intermediate")
    minimum_evidence = 2 if require_two_evidence else 1
    if type_counts["evidence"] < minimum_evidence:
        errors.append(f"reasoning_graph_requires_{minimum_evidence}_evidence_nodes")
    evidence_claims = [
        " ".join(str(node.get("claim", "")).casefold().split())
        for node in node_by_id.values()
        if node.get("type") == "evidence"
    ]
    if len(set(evidence_claims)) != len(evidence_claims):
        errors.append("duplicate_reasoning_graph_evidence_claim")

    answer_ids = [node_id for node_id, node in node_by_id.items() if node.get("type") == "answer"]
    for node_id, node in node_by_id.items():
        node_type = node.get("type")
        if node_type == "evidence" and incoming[node_id]:
            errors.append(f"evidence_node_has_dependency:{node_id}")
        if node_type != "evidence" and not incoming[node_id]:
            errors.append(f"non_evidence_node_has_no_dependency:{node_id}")
        if node_type == "answer" and outgoing[node_id]:
            errors.append(f"answer_node_not_sink:{node_id}")

    if errors:
        return sorted(set(errors)), None

    state: dict[str, int] = {}
    depth_cache: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if state.get(node_id) == 1:
            raise ValueError(f"reasoning_graph_cycle:{node_id}")
        if state.get(node_id) == 2:
            return depth_cache[node_id]
        state[node_id] = 1
        node_type = node_by_id[node_id]["type"]
        value = 0 if node_type == "evidence" else max(depth(parent) + 1 for parent in incoming[node_id])
        state[node_id] = 2
        depth_cache[node_id] = value
        return value

    answer_id = answer_ids[0]
    try:
        computed_depth = depth(answer_id)
    except ValueError as error:
        return [str(error)], None

    ancestors = set()

    def collect(node_id: str) -> None:
        if node_id in ancestors:
            return
        ancestors.add(node_id)
        for parent in incoming[node_id]:
            collect(parent)

    collect(answer_id)
    unused = sorted(set(node_by_id) - ancestors)
    if unused:
        errors.append("reasoning_graph_nodes_outside_answer_path:" + ",".join(unused))
    used_evidence = sum(node_by_id[node_id]["type"] == "evidence" for node_id in ancestors)
    if used_evidence < minimum_evidence:
        errors.append(f"answer_path_requires_{minimum_evidence}_evidence_nodes")
    if declared_depth is not None:
        if not isinstance(declared_depth, int) or isinstance(declared_depth, bool):
            errors.append("declared_reasoning_depth_not_integer")
        elif declared_depth != computed_depth:
            errors.append(
                f"declared_reasoning_depth_mismatch:{declared_depth}!={computed_depth}"
            )
    return sorted(set(errors)), computed_depth


def validate_guide_graph_alignment(guides: Any, graph: Any) -> list[str]:
    """Validate that guide targets name usable nodes in a valid graph.

    Semantic usefulness remains a human judgement. This check only prevents
    dangling/evidence-node references and a guide chain that never reaches an
    intermediate reasoning state.
    """

    graph_errors, _ = validate_reasoning_graph(graph, require_two_evidence=True)
    if graph_errors:
        return ["guide_alignment_graph_invalid"]
    if not isinstance(guides, list) or not guides:
        return ["guide_alignment_requires_guides"]
    node_types = {str(node["id"]).strip(): node["type"] for node in graph["nodes"]}
    errors = []
    targets = []
    for index, guide in enumerate(guides):
        if not isinstance(guide, dict):
            errors.append(f"guide_{index}_not_object")
            continue
        target = str(guide.get("supports_next", "")).strip()
        targets.append(target)
        if target not in node_types:
            errors.append(f"guide_{index}_unknown_supports_next:{target or '<empty>'}")
        elif node_types[target] == "evidence":
            errors.append(f"guide_{index}_targets_evidence:{target}")
    if not any(node_types.get(target) == "intermediate" for target in targets):
        errors.append("guides_do_not_support_intermediate_node")
    return sorted(set(errors))
