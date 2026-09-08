"""
引导策略分类框架

按实验方案 §10 实现四种核心引导策略的判定标准和自动分类：
- A. Hierarchical Localization: Scene → Region → Entity → Attribute
- B. Relational / Entity Chain: Entity_A → Entity_B → Entity_C → Answer
- C. Logical Bridge: Evidence → Intermediate Conclusion → Answer
- D. Evidence Integration: E1, E2, E3 → Hypothesis → Answer
"""

from dataclasses import dataclass
from typing import Literal
from enum import Enum
import re


class GuidanceStrategy(str, Enum):
    """四种核心引导策略（方案 §10）"""
    LOCALIZATION = "localization"           # A. Hierarchical Localization
    ENTITY_TRACKING = "entity_tracking"     # B. Relational / Entity Chain
    LOGICAL_BRIDGE = "logical_bridge"       # C. Logical Bridge
    EVIDENCE_INTEGRATION = "evidence_integration"  # D. Evidence Integration


@dataclass
class StrategyFeatures:
    """用于策略分类的特征"""
    # 视觉特征
    target_area_ratio: float = 0.0      # 目标区域占整图比例
    num_required_regions: int = 1       # 需要观察的区域数
    num_required_entities: int = 1      # 需要追踪的实体数

    # 推理特征
    reasoning_depth: int = 1            # 推理深度
    has_spatial_relation: bool = False  # 是否涉及空间关系
    has_entity_chain: bool = False      # 是否涉及实体链
    has_logical_inference: bool = False # 是否涉及逻辑推断
    has_multi_evidence: bool = False    # 是否需要整合多证据

    # 问题关键词
    question_keywords: list[str] = None

    def __post_init__(self):
        if self.question_keywords is None:
            self.question_keywords = []


# === 策略判定规则 ===

# A. Hierarchical Localization（方案 §10.A）
# 核心: Scene → Region → Entity → Attribute
# 主要解决: "去哪里看"
LOCALIZATION_KEYWORDS = [
    "哪个区域", "什么位置", "在哪里", "哪个部分",
    "哪个物体", "什么颜色", "什么形状", "多大",
    "左边", "右边", "上面", "下面", "中间",
    "前景", "背景", "角落"
]

LOCALIZATION_FEATURES = {
    "target_area_small": True,      # 目标区域小
    "single_region": True,          # 单一区域
    "spatial_focus": True,          # 空间定位为主
    "shallow_reasoning": True       # 推理较浅 (depth 1-2)
}

# B. Relational / Entity Chain（方案 §10.B）
# 核心: Entity_A → Entity_B → Entity_C → Answer
# 主要解决: "应该沿着什么实体关系寻找目标"
ENTITY_CHAIN_KEYWORDS = [
    "与...相关", "属于", "正在使用", "正在做",
    "之间的关系", "互动", "接触",
    "是谁", "什么身份", "什么角色"
]

ENTITY_CHAIN_FEATURES = {
    "multi_entity": True,           # 多实体
    "entity_relation": True,        # 实体关系
    "chain_dependency": True,       # 链式依赖
    "medium_depth": True            # 中等深度 (depth 2-3)
}

# C. Logical Bridge（方案 §10.C）
# 核心: Evidence → Intermediate Conclusion → Answer
# 主要解决: "从视觉事实如何推出结论"
LOGICAL_BRIDGE_KEYWORDS = [
    "说明", "意味着", "表明", "推断",
    "为什么", "原因", "因为",
    "可能", "应该", "证明"
]

LOGICAL_BRIDGE_FEATURES = {
    "logical_inference": True,      # 逻辑推断
    "evidence_to_conclusion": True, # 从证据到结论
    "reasoning_required": True,     # 需要推理
    "deep_reasoning": True          # 较深推理 (depth 3+)
}

# D. Evidence Integration（方案 §10.D）
# 核心: E1, E2, E3 → Hypothesis → Answer
# 主要解决: "多个分散证据如何汇聚"
EVIDENCE_INTEGRATION_KEYWORDS = [
    "综合", "结合", "所有", "全部",
    " together", "整体", "汇总",
    "多个", "几个", "各种"
]

EVIDENCE_INTEGRATION_FEATURES = {
    "multi_evidence": True,         # 多证据
    "multi_region": True,           # 多区域
    "integration_required": True,   # 需要整合
    "deep_reasoning": True          # 较深推理 (depth 3+)
}


def classify_strategy(features: StrategyFeatures) -> GuidanceStrategy:
    """
    基于特征自动分类引导策略

    优先级：
    1. 如果有明确特征匹配，直接返回
    2. 否则基于综合评分选择最接近的策略

    Args:
        features: 样本特征

    Returns:
        最匹配的引导策略
    """
    scores = {
        GuidanceStrategy.LOCALIZATION: 0.0,
        GuidanceStrategy.ENTITY_TRACKING: 0.0,
        GuidanceStrategy.LOGICAL_BRIDGE: 0.0,
        GuidanceStrategy.EVIDENCE_INTEGRATION: 0.0,
    }

    # === Localization 评分 ===
    if features.target_area_ratio > 0 and features.target_area_ratio < 0.1:
        scores[GuidanceStrategy.LOCALIZATION] += 0.3
    if features.num_required_regions == 1:
        scores[GuidanceStrategy.LOCALIZATION] += 0.2
    if features.reasoning_depth <= 2:
        scores[GuidanceStrategy.LOCALIZATION] += 0.3
    if features.has_spatial_relation:
        scores[GuidanceStrategy.LOCALIZATION] += 0.2

    # === Entity Chain 评分 ===
    if features.num_required_entities >= 2:
        scores[GuidanceStrategy.ENTITY_TRACKING] += 0.3
    if features.has_entity_chain:
        scores[GuidanceStrategy.ENTITY_TRACKING] += 0.4
    if 2 <= features.reasoning_depth <= 3:
        scores[GuidanceStrategy.ENTITY_TRACKING] += 0.2

    # === Logical Bridge 评分 ===
    if features.has_logical_inference:
        scores[GuidanceStrategy.LOGICAL_BRIDGE] += 0.4
    if features.reasoning_depth >= 3:
        scores[GuidanceStrategy.LOGICAL_BRIDGE] += 0.3

    # === Evidence Integration 评分 ===
    if features.has_multi_evidence:
        scores[GuidanceStrategy.EVIDENCE_INTEGRATION] += 0.4
    if features.num_required_regions >= 2:
        scores[GuidanceStrategy.EVIDENCE_INTEGRATION] += 0.3
    if features.reasoning_depth >= 3:
        scores[GuidanceStrategy.EVIDENCE_INTEGRATION] += 0.2

    # 关键词加分
    for kw in (features.question_keywords or []):
        kw_lower = kw.lower()
        if any(k in kw_lower for k in LOCALIZATION_KEYWORDS):
            scores[GuidanceStrategy.LOCALIZATION] += 0.1
        if any(k in kw_lower for k in ENTITY_CHAIN_KEYWORDS):
            scores[GuidanceStrategy.ENTITY_TRACKING] += 0.1
        if any(k in kw_lower for k in LOGICAL_BRIDGE_KEYWORDS):
            scores[GuidanceStrategy.LOGICAL_BRIDGE] += 0.1
        if any(k in kw_lower for k in EVIDENCE_INTEGRATION_KEYWORDS):
            scores[GuidanceStrategy.EVIDENCE_INTEGRATION] += 0.1

    # 返回最高分
    return max(scores, key=scores.get)


def classify_from_question(question: str) -> GuidanceStrategy:
    """
    从问题文本直接分类（简单版）

    Args:
        question: 问题文本

    Returns:
        最匹配的引导策略
    """
    question_lower = question.lower()

    # 统计各策略关键词命中数
    scores = {
        GuidanceStrategy.LOCALIZATION: 0,
        GuidanceStrategy.ENTITY_TRACKING: 0,
        GuidanceStrategy.LOGICAL_BRIDGE: 0,
        GuidanceStrategy.EVIDENCE_INTEGRATION: 0,
    }

    for kw in LOCALIZATION_KEYWORDS:
        if kw in question_lower:
            scores[GuidanceStrategy.LOCALIZATION] += 1

    for kw in ENTITY_CHAIN_KEYWORDS:
        if kw in question_lower:
            scores[GuidanceStrategy.ENTITY_TRACKING] += 1

    for kw in LOGICAL_BRIDGE_KEYWORDS:
        if kw in question_lower:
            scores[GuidanceStrategy.LOGICAL_BRIDGE] += 1

    for kw in EVIDENCE_INTEGRATION_KEYWORDS:
        if kw in question_lower:
            scores[GuidanceStrategy.EVIDENCE_INTEGRATION] += 1

    # 返回最高分
    max_score = max(scores.values())
    if max_score == 0:
        # 无关键词命中，默认 localization（最常见）
        return GuidanceStrategy.LOCALIZATION

    return max(scores, key=scores.get)


def get_strategy_description(strategy: GuidanceStrategy) -> str:
    """获取策略描述（方案 §10）"""
    descriptions = {
        GuidanceStrategy.LOCALIZATION: (
            "Hierarchical Localization: Scene → Region → Entity → Attribute. "
            "主要解决: 去哪里看。"
        ),
        GuidanceStrategy.ENTITY_TRACKING: (
            "Relational / Entity Chain: Entity_A → Entity_B → Entity_C → Answer. "
            "主要解决: 应该沿着什么实体关系寻找目标。"
        ),
        GuidanceStrategy.LOGICAL_BRIDGE: (
            "Logical Bridge: Evidence → Intermediate Conclusion → Answer. "
            "主要解决: 从视觉事实如何推出结论。"
        ),
        GuidanceStrategy.EVIDENCE_INTEGRATION: (
            "Evidence Integration: E1, E2, E3 → Hypothesis → Answer. "
            "主要解决: 多个分散证据如何汇聚。"
        ),
    }
    return descriptions.get(strategy, "Unknown strategy")


def validate_strategy_assignment(
    strategy: GuidanceStrategy,
    features: StrategyFeatures
) -> list[str]:
    """
    验证策略分配是否合理

    Returns:
        警告列表
    """
    warnings = []

    if strategy == GuidanceStrategy.LOCALIZATION:
        if features.reasoning_depth > 2:
            warnings.append(
                f"Localization strategy with deep reasoning (depth={features.reasoning_depth}). "
                "Consider Logical Bridge or Evidence Integration."
            )
        if features.num_required_entities > 2:
            warnings.append(
                f"Localization strategy with multiple entities ({features.num_required_entities}). "
                "Consider Entity Chain."
            )

    elif strategy == GuidanceStrategy.ENTITY_TRACKING:
        if features.num_required_entities < 2:
            warnings.append(
                f"Entity Chain strategy but only {features.num_required_entities} entity. "
                "Consider Localization."
            )

    elif strategy == GuidanceStrategy.LOGICAL_BRIDGE:
        if not features.has_logical_inference:
            warnings.append(
                "Logical Bridge strategy but no logical inference feature detected."
            )

    elif strategy == GuidanceStrategy.EVIDENCE_INTEGRATION:
        if features.num_required_regions < 2 and not features.has_multi_evidence:
            warnings.append(
                "Evidence Integration strategy but no multi-evidence feature detected."
            )

    return warnings
