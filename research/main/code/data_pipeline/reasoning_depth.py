"""
推理深度计算工具

按实验方案 §11 定义 reasoning depth 的操作性定义：
"从原始视觉 evidence 到最终答案之间，至少需要多少个不可省略的中间信息转换"

Depth 定义:
- Depth 1: E1 → A
- Depth 2: E1 → H1 → A
- Depth 3: E1 → H1 → H2 → A
- Depth 4+: 三层以上中间状态

正式测试目标分布（方案 §11）:
| Depth | Diagnostic test |
|-------|----------------|
| 1     | ≥300           |
| 2     | ≥400           |
| 3     | ≥400           |
| 4+    | ≥400           |
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


class ReasoningDepth(IntEnum):
    """推理深度级别"""
    DEPTH_1 = 1  # E1 → A
    DEPTH_2 = 2  # E1 → H1 → A
    DEPTH_3 = 3  # E1 → H1 → H2 → A
    DEPTH_4 = 4  # E1 → H1 → H2 → H3 → A
    DEPTH_5_PLUS = 5  # 5+


@dataclass
class ReasoningNode:
    """推理链中的节点"""
    node_id: str
    node_type: str  # "evidence", "hypothesis", "answer"
    content: str
    depends_on: list[str] = field(default_factory=list)  # 依赖的节点 ID


@dataclass
class ReasoningGraph:
    """
    推理图（方案 §14 Step 3）

    表示从视觉证据到最终答案的推理路径。
    """
    nodes: list[ReasoningNode] = field(default_factory=list)
    final_answer_node_id: str = ""

    def add_node(self, node: ReasoningNode):
        self.nodes.append(node)

    def get_node(self, node_id: str) -> Optional[ReasoningNode]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_evidence_nodes(self) -> list[ReasoningNode]:
        return [n for n in self.nodes if n.node_type == "evidence"]

    def get_hypothesis_nodes(self) -> list[ReasoningNode]:
        return [n for n in self.nodes if n.node_type == "hypothesis"]


def compute_depth_from_graph(graph: ReasoningGraph) -> int:
    """
    从推理图计算推理深度（方案 §11）

    深度 = 从 evidence 到 answer 的最长路径中的中间状态数

    Args:
        graph: 推理图

    Returns:
        推理深度 (1-5+)
    """
    if not graph.nodes:
        return 1

    # 找到所有 evidence 节点
    evidence_ids = {n.node_id for n in graph.get_evidence_nodes()}

    # 找到 answer 节点
    answer_node = graph.get_node(graph.final_answer_node_id)
    if not answer_node:
        return 1

    # 回溯计算从 evidence 到 answer 的最长边数。直接 E → A 为 1，
    # 每个不可省略的中间状态再增加 1。
    def find_longest_path(node_id: str, path: set[str]) -> int:
        if node_id in evidence_ids:
            return 0

        node = graph.get_node(node_id)
        if not node or not node.depends_on:
            raise ValueError(f"Node '{node_id}' does not lead back to visual evidence")

        depths = []
        for dep_id in node.depends_on:
            if dep_id in path:
                raise ValueError(f"Cycle detected in reasoning graph at '{dep_id}'")
            depths.append(find_longest_path(dep_id, path | {dep_id}) + 1)

        return max(depths)

    # 从 answer 节点开始回溯
    depth = find_longest_path(graph.final_answer_node_id, {graph.final_answer_node_id})
    return min(depth, 5)  # 5 代表 5+


def compute_depth_from_guides(
    num_guides: int,
    has_intermediate_answers: bool = False
) -> int:
    """
    从引导问题数量估算推理深度

    简化版：如果只有 questions，深度 = guides 数量
    如果有 intermediate answers (T-QA)，深度 = guides 数量 + 1

    Args:
        num_guides: 引导问题数量
        has_intermediate_answers: 是否有中间答案

    Returns:
        推理深度
    """
    if num_guides == 0:
        return 1  # 无引导，直接从 evidence 到 answer

    if has_intermediate_answers:
        # T-QA: 每个 guide + answer 构成一个中间状态
        depth = num_guides + 1
    else:
        # T-Q: 只有引导问题
        depth = num_guides

    return min(depth, 5)


def classify_depth_category(depth: int) -> str:
    """
    分类深度类别

    Args:
        depth: 推理深度 (1-5+)

    Returns:
        类别名称
    """
    if depth == 1:
        return "D1"
    elif depth == 2:
        return "D2"
    elif depth == 3:
        return "D3"
    else:
        return "D4+"


def validate_depth_label(depth: int, reasoning_graph: Optional[ReasoningGraph] = None) -> list[str]:
    """
    验证推理深度标签是否合理

    Args:
        depth: 标注的推理深度
        reasoning_graph: 推理图（可选，用于交叉验证）

    Returns:
        警告列表
    """
    warnings = []

    if depth < 1:
        warnings.append(f"Invalid depth: {depth}. Must be >= 1.")

    if depth > 5:
        warnings.append(f"Unusual high depth: {depth}. Consider capping at 5+.")

    if reasoning_graph:
        computed = compute_depth_from_graph(reasoning_graph)
        if abs(depth - computed) > 1:
            warnings.append(
                f"Depth label ({depth}) differs from computed ({computed}) by more than 1."
            )

    return warnings


def compute_target_distribution(
    samples: list[dict],
    depth_key: str = "reasoning_depth"
) -> dict:
    """
    计算深度分布（方案 §11 目标）

    目标:
    | Depth | Diagnostic test |
    |-------|----------------|
    | 1     | ≥300           |
    | 2     | ≥400           |
    | 3     | ≥400           |
    | 4+    | ≥400           |

    Args:
        samples: 样本列表
        depth_key: 深度字段名

    Returns:
        深度分布统计
    """
    distribution = {"D1": 0, "D2": 0, "D3": 0, "D4+": 0}

    for sample in samples:
        depth = sample.get(depth_key, 1)
        category = classify_depth_category(depth)
        distribution[category] += 1

    # 计算与目标的差距
    targets = {"D1": 300, "D2": 400, "D3": 400, "D4+": 400}
    gaps = {}
    for cat, target in targets.items():
        gaps[cat] = distribution[cat] - target

    return {
        "distribution": distribution,
        "targets": targets,
        "gaps": gaps,
        "total": sum(distribution.values())
    }


def build_reasoning_graph_from_guides(
    guides: list[dict],
    final_question: str,
    final_answer: str
) -> ReasoningGraph:
    """
    从引导问题构建推理图

    Args:
        guides: 引导问题列表 [{"question": ..., "answer": ...}, ...]
        final_question: 最终问题
        final_answer: 最终答案

    Returns:
        推理图
    """
    graph = ReasoningGraph()

    # 添加 evidence 节点（第一个 guide 依赖的隐式证据）
    evidence_node = ReasoningNode(
        node_id="E1",
        node_type="evidence",
        content="Visual evidence from image"
    )
    graph.add_node(evidence_node)

    # 添加 guide 节点
    prev_id = "E1"
    for i, guide in enumerate(guides):
        # Hypothesis/intermediate node
        hypothesis_node = ReasoningNode(
            node_id=f"H{i+1}",
            node_type="hypothesis",
            content=f"Guide {i+1}: {guide['question']} -> {guide['answer']}",
            depends_on=[prev_id]
        )
        graph.add_node(hypothesis_node)
        prev_id = f"H{i+1}"

    # 添加 answer 节点
    answer_node = ReasoningNode(
        node_id="A",
        node_type="answer",
        content=f"{final_question} -> {final_answer}",
        depends_on=[prev_id]
    )
    graph.add_node(answer_node)
    graph.final_answer_node_id = "A"

    return graph
