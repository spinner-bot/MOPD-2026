"""
数据 Schema 定义

按实验方案 §9 设计，定义每个正式样本的完整字段结构。
后期最有价值的实验往往不是重新训练模型，而是根据 metadata 重新切测试集。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from enum import Enum
import json


class TaskFamily(str, Enum):
    """任务类型（方案 §6）"""
    P = "P"      # Perception-Bottleneck: 视觉困难、推理较浅
    R = "R"      # Reasoning-Bottleneck: 视觉清晰、推理复杂
    PR = "PR"    # Dual Bottleneck: 视觉+推理双重困难


class GuidanceStrategy(str, Enum):
    """引导策略（方案 §10）"""
    LOCALIZATION = "localization"           # A. Hierarchical Localization
    ENTITY_TRACKING = "entity_tracking"     # B. Relational / Entity Chain
    LOGICAL_BRIDGE = "logical_bridge"       # C. Logical Bridge
    EVIDENCE_INTEGRATION = "evidence_integration"  # D. Evidence Integration
    ATTRIBUTE_DECOMPOSITION = "attribute_decomposition"  # 第二阶段
    DISAMBIGUATION = "disambiguation"      # 第二阶段
    CAUSAL_FUNCTIONAL = "causal_functional"  # 第二阶段


class VisualClarity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VisualDependency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class GuidingQuestion:
    """单个引导问题"""
    question: str
    answer: str
    # Legacy semantic-review score. Experimental-model deletion effects belong
    # in analysis artifacts and must never be written here for data selection.
    dependency_score: float = 0.0


@dataclass
class EvidenceRegion:
    """证据区域"""
    bbox: list[int]  # [x1, y1, x2, y2]
    description: str = ""


@dataclass
class SampleMetadata:
    """
    完整样本元数据（方案 §9）

    存储所有实验需要的字段，支持后期按各种维度重新切测试集。
    """

    # === 标识 ===
    sample_id: str
    source_image_id: str
    source_dataset: str

    # === 图像信息 ===
    image_original: str
    image_width: int
    image_height: int

    # === 最终问答 ===
    final_question: str
    option_A: str
    option_B: str
    option_C: str
    option_D: str
    final_answer: Literal["A", "B", "C", "D"]

    # === 任务分类 ===
    task_family: TaskFamily
    guidance_strategy: GuidanceStrategy

    # === 引导链（最多 4 个 guide）===
    guide_1_question: str = ""
    guide_1_answer: str = ""
    guide_2_question: str = ""
    guide_2_answer: str = ""
    guide_3_question: str = ""
    guide_3_answer: str = ""
    guide_4_question: str = ""
    guide_4_answer: str = ""

    # === 推理深度（方案 §11）===
    reasoning_depth: int = 1  # 1-4+

    # === 证据区域 ===
    evidence_region_1_bbox: list[int] = field(default_factory=list)
    evidence_region_2_bbox: list[int] = field(default_factory=list)
    evidence_region_3_bbox: list[int] = field(default_factory=list)

    # === Crop 图像路径 ===
    crop_primary: str = ""
    crop_secondary: str = ""

    # === 视觉属性 ===
    target_area_ratio: float = 0.0  # 目标区域占整图比例
    num_required_regions: int = 1   # 需要观察的区域数
    num_required_entities: int = 1  # 需要追踪的实体数
    visual_clarity: VisualClarity = VisualClarity.MEDIUM
    visual_dependency: VisualDependency = VisualDependency.MEDIUM

    # === 引导质量评分 ===
    # Legacy semantic-review score (D_sem), not model log-prob contribution.
    guide_dependency_score: float = 0.0
    guide_relevance_score: float = 0.0    # 引导相关性
    guide_answerability_score: float = 0.0  # 引导可回答性
    answer_leakage_score: float = 0.0     # 答案泄漏程度（0 = 无泄漏）

    # === Teacher Gap（结果后分析；不得参与样本准入）===
    # None means that no result has been attached. These legacy fields remain
    # readable for compatibility, but locked manifests should keep effects in
    # separate analysis artifacts linked by manifest hash (1082).
    teacher_gap_S_T: Optional[float] = None
    teacher_gap_S_V: Optional[float] = None
    teacher_gap_V_T: Optional[float] = None

    # === 版本与验证 ===
    human_verified: bool = False
    generation_model: str = ""
    verification_model: str = ""
    data_version: str = "v0.1"

    def to_dict(self) -> dict:
        """转为字典，Enum 转为字符串"""
        d = asdict(self)
        d["task_family"] = self.task_family.value
        d["guidance_strategy"] = self.guidance_strategy.value
        d["visual_clarity"] = self.visual_clarity.value
        d["visual_dependency"] = self.visual_dependency.value
        return d

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleMetadata":
        """从字典创建"""
        # 转换 Enum 字段
        if "task_family" in d:
            d["task_family"] = TaskFamily(d["task_family"])
        if "guidance_strategy" in d:
            d["guidance_strategy"] = GuidanceStrategy(d["guidance_strategy"])
        if "visual_clarity" in d:
            d["visual_clarity"] = VisualClarity(d["visual_clarity"])
        if "visual_dependency" in d:
            d["visual_dependency"] = VisualDependency(d["visual_dependency"])
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "SampleMetadata":
        """从 JSON 创建"""
        return cls.from_dict(json.loads(json_str))


def validate_sample(sample: SampleMetadata) -> list[str]:
    """
    验证样本完整性，返回错误列表

    按方案 §16 自动质检要求：
    - answerability 必须高
    - relevance 必须高
    - dependency 至少中等
    - leakage 必须为 0
    - final_answer 必须唯一
    """
    errors = []

    # 必填字段检查
    if not sample.sample_id:
        errors.append("缺少 sample_id")
    if not sample.source_image_id:
        errors.append("缺少 source_image_id")
    if not sample.final_question:
        errors.append("缺少 final_question")
    if not sample.final_answer:
        errors.append("缺少 final_answer")

    # 引导链检查
    has_guides = bool(sample.guide_1_question)
    if has_guides:
        # 有引导时检查质量
        if sample.guide_answerability_score < 0.5:
            errors.append(f"guide_answerability_score 过低: {sample.guide_answerability_score}")
        if sample.guide_relevance_score < 0.5:
            errors.append(f"guide_relevance_score 过低: {sample.guide_relevance_score}")
        if sample.guide_dependency_score < 0.3:
            errors.append(f"guide_dependency_score 过低: {sample.guide_dependency_score}")

    # 泄漏检查（方案 §69）
    if sample.answer_leakage_score > 0:
        errors.append(f"检测到答案泄漏: leakage_score = {sample.answer_leakage_score}")

    # 推理深度合理性
    if sample.reasoning_depth < 1 or sample.reasoning_depth > 5:
        errors.append(f"reasoning_depth 异常: {sample.reasoning_depth}")

    return errors


# === 数据集版本管理（方案 §80）===

DATA_VERSIONS = {
    "v0.1": "generator test",
    "v0.2": "Stage 0 screening",
    "v0.3": "Pilot",
    "v1.0": "Formal frozen dataset",
}


def get_version_description(version: str) -> str:
    """获取版本描述"""
    return DATA_VERSIONS.get(version, "Unknown version")
