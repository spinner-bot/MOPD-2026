"""
Guide 生成约束验证

按实验方案 §15 实现引导问题质量检查：
1. Answerable - 可回答
2. Path Relevant - 路径相关
3. Non-leaking - 不泄漏答案
4. Locally Useful - 局部有用
5. No Trivial Restatement - 非简单重述
6. No Generic Prompt - 非泛化提示
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    scores: dict[str, float]


# 泛化提示词黑名单（方案 §15.6）
GENERIC_PHRASES = [
    "请仔细查看", "请观察", "请分析", "注意看",
    "please look", "please observe", "please analyze",
    "carefully examine", "take a closer look",
    "the image", "the picture", "the photo",
    "相关区域", "相关部分", "有关内容"
]

# 泄漏检测 - 答案相关词
LEAKAGE_INDICATORS = [
    # 中文
    "答案是", "正确答案", "选项",
    # 英文
    "the answer is", "correct option", "choice"
]


def validate_answerable(
    guide_question: str,
    guide_answer: str,
    image_available: bool = True
) -> tuple[bool, str]:
    """
    检查引导问题是否可回答（方案 §15.1）

    Args:
        guide_question: 引导问题
        guide_answer: 引导答案
        image_available: 图像是否可用

    Returns:
        (是否通过, 原因)
    """
    if not guide_question or not guide_question.strip():
        return False, "Guide question is empty"

    if not guide_answer or not guide_answer.strip():
        return False, "Guide answer is empty"

    # 检查问题是否以问号结尾（中文或英文）
    question_stripped = guide_question.strip()
    if not (question_stripped.endswith("?") or question_stripped.endswith("？")):
        return False, "Guide question does not end with question mark"

    # 检查答案长度是否合理
    if len(guide_answer) < 2:
        return False, "Guide answer too short"

    if len(guide_answer) > 500:
        return False, "Guide answer too long (should be concise)"

    return True, "OK"


def validate_path_relevant(
    guide_question: str,
    final_question: str,
    reasoning_graph: Optional[dict] = None
) -> tuple[bool, str]:
    """
    检查引导问题是否在推理路径上（方案 §15.2）

    Args:
        guide_question: 引导问题
        final_question: 最终问题
        reasoning_graph: 推理图（可选）

    Returns:
        (是否通过, 原因)
    """
    if not guide_question or not final_question:
        return False, "Missing questions"

    # 简单检查：问题和最终问题应该有共同的话题领域
    # 更严格的检查需要语义分析

    guide_lower = guide_question.lower()
    final_lower = final_question.lower()

    # 提取关键词（简单版）
    guide_words = set(re.findall(r'\w+', guide_lower))
    final_words = set(re.findall(r'\w+', final_lower))

    # 如果完全没有共同词，可能是警告
    common_words = guide_words & final_words
    if len(common_words) == 0 and len(guide_words) > 3:
        return True, "Warning: No obvious lexical overlap (may still be valid)"

    return True, "OK"


def validate_non_leaking(
    guide_question: str,
    guide_answer: str,
    final_question: str,
    final_answer: str,
    options: dict[str, str] = None
) -> tuple[bool, str]:
    """
    检查是否泄漏最终答案（方案 §15.3, §69）

    这是最重要的约束！

    Args:
        guide_question: 引导问题
        guide_answer: 引导答案
        final_question: 最终问题
        final_answer: 最终答案
        options: 选项字典 {"A": ..., "B": ..., ...}

    Returns:
        (是否通过, 原因)
    """
    if not final_answer:
        return True, "No final answer to check"

    guide_text = f"{guide_question} {guide_answer}".lower()
    final_answer_lower = final_answer.lower()

    # 1. Lexical leakage: 最终答案直接出现在引导中
    if final_answer_lower in guide_text:
        return False, f"Lexical leakage: final answer '{final_answer}' appears in guide"

    # 2. 检查选项是否出现在引导中
    if options:
        for opt_key, opt_value in options.items():
            if opt_key == final_answer:
                continue  # 跳过正确答案
            opt_lower = opt_value.lower()
            if len(opt_lower) > 3 and opt_lower in guide_text:
                return False, f"Option leakage: wrong option '{opt_value}' appears in guide"

    # 3. 检查答案的关键属性是否泄漏
    # 例如：最终问颜色，引导中不应直接给出颜色
    if len(final_answer_lower) <= 10:  # 短答案（可能是颜色、形状等）
        if final_answer_lower in guide_text:
            return False, f"Attribute leakage: answer '{final_answer}' in guide"

    # 4. 检查否定形式的泄漏
    # 例如："它不是蓝色" + 最终问"是什么颜色" -> 泄漏
    for phrase in ["不是", "不是", "not the", "isn't the", "doesn't have"]:
        if phrase in guide_lower:
            # 可能有否定泄漏
            pass  # 需要更精细的检查

    return True, "OK"


def validate_not_trivial_restatement(
    guide_question: str,
    final_question: str
) -> tuple[bool, str]:
    """
    检查是否是对最终问题的简单重述（方案 §15.5）

    Args:
        guide_question: 引导问题
        final_question: 最终问题

    Returns:
        (是否通过, 原因)
    """
    if not guide_question or not final_question:
        return True, "OK"

    guide_lower = guide_question.lower().strip()
    final_lower = final_question.lower().strip()

    # 1. 完全相同
    if guide_lower == final_lower:
        return False, "Guide is identical to final question"

    # 2. 计算相似度（简单版：Jaccard）
    guide_words = set(re.findall(r'\w+', guide_lower))
    final_words = set(re.findall(r'\w+', final_lower))

    if len(guide_words) > 0 and len(final_words) > 0:
        intersection = guide_words & final_words
        union = guide_words | final_words
        jaccard = len(intersection) / len(union) if union else 0

        if jaccard > 0.8:
            return False, f"Guide is too similar to final question (Jaccard={jaccard:.2f})"

    # 3. 检查是否只是重新措辞
    # 例如："什么颜色？" vs "它的颜色是什么？"
    # 需要更复杂的语义检查

    return True, "OK"


def validate_not_generic(
    guide_question: str
) -> tuple[bool, str]:
    """
    检查是否为泛化提示（方案 §15.6）

    泛化提示如"请仔细观察图片"不属于 privileged reasoning information

    Args:
        guide_question: 引导问题

    Returns:
        (是否通过, 原因)
    """
    if not guide_question:
        return True, "OK"

    guide_lower = guide_question.lower()

    # 检查是否只包含泛化词汇
    for phrase in GENERIC_PHRASES:
        if phrase in guide_lower:
            # 进一步检查是否有具体内容
            # 如果去掉泛化词汇后几乎没有内容，则可能是泛化提示
            remaining = guide_lower
            for gp in GENERIC_PHRASES:
                remaining = remaining.replace(gp, "")
            remaining = remaining.strip()

            if len(remaining) < 10:
                return False, f"Guide appears to be generic: '{guide_question}'"

    return True, "OK"


def validate_guide(
    guide_question: str,
    guide_answer: str,
    final_question: str,
    final_answer: str,
    options: dict[str, str] = None,
    reasoning_graph: dict = None
) -> ValidationResult:
    """
    完整验证单个引导问题

    Args:
        guide_question: 引导问题
        guide_answer: 引导答案
        final_question: 最终问题
        final_answer: 最终答案
        options: 选项字典
        reasoning_graph: 推理图

    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    scores = {}

    # 1. Answerable
    ok, msg = validate_answerable(guide_question, guide_answer)
    if not ok:
        errors.append(f"Answerable: {msg}")
    scores["answerable"] = 1.0 if ok else 0.0

    # 2. Path Relevant
    ok, msg = validate_path_relevant(guide_question, final_question, reasoning_graph)
    if not ok:
        errors.append(f"Path Relevant: {msg}")
    elif "Warning" in msg:
        warnings.append(f"Path Relevant: {msg}")
    scores["path_relevant"] = 1.0 if ok else 0.0

    # 3. Non-leaking (最重要)
    ok, msg = validate_non_leaking(
        guide_question, guide_answer,
        final_question, final_answer,
        options
    )
    if not ok:
        errors.append(f"Non-leaking: {msg}")
    scores["non_leaking"] = 1.0 if ok else 0.0

    # 4. Not trivial restatement
    ok, msg = validate_not_trivial_restatement(guide_question, final_question)
    if not ok:
        errors.append(f"Not Trivial: {msg}")
    scores["not_trivial"] = 1.0 if ok else 0.0

    # 5. Not generic
    ok, msg = validate_not_generic(guide_question)
    if not ok:
        errors.append(f"Not Generic: {msg}")
    scores["not_generic"] = 1.0 if ok else 0.0

    # 综合判定
    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        scores=scores
    )


def validate_all_guides(
    guides: list[dict],
    final_question: str,
    final_answer: str,
    options: dict[str, str] = None
) -> dict:
    """
    验证所有引导问题

    Args:
        guides: [{"question": ..., "answer": ...}, ...]
        final_question: 最终问题
        final_answer: 最终答案
        options: 选项字典

    Returns:
        {
            "total": 总数,
            "valid": 通过数,
            "invalid": 失败数,
            "results": [ValidationResult, ...],
            "overall_valid": bool
        }
    """
    results = []
    valid_count = 0

    for guide in guides:
        result = validate_guide(
            guide_question=guide.get("question", ""),
            guide_answer=guide.get("answer", ""),
            final_question=final_question,
            final_answer=final_answer,
            options=options
        )
        results.append(result)
        if result.is_valid:
            valid_count += 1

    return {
        "total": len(guides),
        "valid": valid_count,
        "invalid": len(guides) - valid_count,
        "results": results,
        "overall_valid": valid_count == len(guides)
    }
