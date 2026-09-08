"""
泄漏检测工具

按实验方案 §69 实现答案泄漏检测：
- Lexical leakage: 最终 answer 是否直接出现在 guide answer 中
- Semantic leakage: 中间答案是否唯一编码 final answer
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class LeakageResult:
    """泄漏检测结果"""
    has_leakage: bool
    leakage_type: str  # "lexical", "semantic", "none"
    details: list[str]
    severity: str  # "high", "medium", "low", "none"


def detect_lexical_leakage(
    guide_text: str,
    final_answer: str,
    options: dict[str, str] = None,
    case_sensitive: bool = False
) -> tuple[bool, list[str]]:
    """
    检测词法泄漏（方案 §69）

    最终 answer 是否直接出现在 guide 中

    Args:
        guide_text: 引导文本（question + answer 合并）
        final_answer: 最终答案
        options: 选项字典 {"A": ..., "B": ..., ...}
        case_sensitive: 是否区分大小写

    Returns:
        (是否泄漏, 泄漏详情列表)
    """
    leakages = []

    if not final_answer:
        return False, []

    # 准备文本
    if case_sensitive:
        guide_check = guide_text
        answer_check = final_answer
    else:
        guide_check = guide_text.lower()
        answer_check = final_answer.lower()

    # 1. 直接匹配最终答案
    if answer_check in guide_check:
        leakages.append(f"Final answer '{final_answer}' appears in guide text")

    # 2. 检查错误选项是否出现（可能间接泄漏）
    if options:
        for opt_key, opt_value in options.items():
            if opt_key == final_answer:
                continue  # 跳过正确答案
            opt_lower = opt_value.lower() if not case_sensitive else opt_value
            if len(opt_lower) > 2 and opt_lower in guide_check:
                leakages.append(f"Wrong option '{opt_value}' appears in guide (may bias model)")

    # 3. 检查答案的部分匹配（可能泄漏关键属性）
    # 例如答案是 "blue"，guide 中提到 "blue"
    answer_words = re.findall(r'\w+', answer_check)
    for word in answer_words:
        if len(word) > 3 and word in guide_check:
            # 可能是有意义的词（非停用词）
            if word not in {'what', 'which', 'where', 'when', 'this', 'that', 'with', 'from'}:
                leakages.append(f"Answer word '{word}' appears in guide text")

    return len(leakages) > 0, leakages


def detect_semantic_leakage(
    guide_question: str,
    guide_answer: str,
    final_question: str,
    final_answer: str,
    options: dict[str, str] = None
) -> tuple[bool, list[str]]:
    """
    检测语义泄漏（方案 §69）

    中间答案是否唯一编码 final answer
    例如：
    - Final: "耳机是什么颜色？"
    - Guide: "它不是蓝色" -> 泄漏（排除了一个选项）

    Args:
        guide_question: 引导问题
        guide_answer: 引导答案
        final_question: 最终问题
        final_answer: 最终答案
        options: 选项字典

    Returns:
        (是否泄漏, 泄漏详情列表)
    """
    leakages = []

    guide_lower = guide_answer.lower()
    final_lower = final_question.lower()

    # 1. 检查否定形式泄漏
    # 如果 guide 否定某个选项，而最终问题是问该属性
    negative_patterns = [
        r'不是(\w+)',
        r'not\s+(\w+)',
        r"doesn't\s+(\w+)",
        r'没有(\w+)',
        r'excluding\s+(\w+)'
    ]

    for pattern in negative_patterns:
        matches = re.findall(pattern, guide_lower)
        for match in matches:
            if options:
                # 检查否定的内容是否是某个选项
                for opt_key, opt_value in options.items():
                    if match in opt_value.lower():
                        leakages.append(
                            f"Guide negates option {opt_key} ('{opt_value}'), "
                            "which may leak information about the final answer"
                        )

    # 2. 检查引导答案是否直接给出最终属性
    # 例如最终问颜色，引导直接给出颜色
    # 这需要更复杂的语义理解，简单检查：
    # 如果 final_question 包含"颜色/什么颜色"，而 guide_answer 包含颜色词
    color_words = {'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange',
                   'purple', 'pink', 'brown', 'gray', '红色', '蓝色', '绿色',
                   '黄色', '黑色', '白色', '橙色', '紫色', '粉色', '棕色', '灰色'}

    if any(w in final_lower for w in ['颜色', '什么色', 'color']):
        answer_words = set(re.findall(r'\w+', guide_lower))
        found_colors = answer_words & color_words
        if found_colors:
            leakages.append(
                f"Guide answer contains color word(s) {found_colors}, "
                "which may leak the final answer about color"
            )

    # 3. 检查引导问题是否等同于最终问题
    if guide_question.lower().strip() == final_question.lower().strip():
        leakages.append("Guide question is identical to final question")

    return len(leakages) > 0, leakages


def detect_answer_leakage(
    guides: list[dict],
    final_question: str,
    final_answer: str,
    options: dict[str, str] = None
) -> LeakageResult:
    """
    检测所有引导问题的答案泄漏

    Args:
        guides: [{"question": ..., "answer": ...}, ...]
        final_question: 最终问题
        final_answer: 最终答案
        options: 选项字典

    Returns:
        LeakageResult
    """
    all_details = []
    has_any_leakage = False
    max_severity = "none"

    for i, guide in enumerate(guides):
        guide_text = f"{guide.get('question', '')} {guide.get('answer', '')}"

        # Lexical leakage
        has_lex, lex_details = detect_lexical_leakage(
            guide_text, final_answer, options
        )
        if has_lex:
            has_any_leakage = True
            all_details.extend([f"Guide {i+1} lexical: {d}" for d in lex_details])
            max_severity = "high"

        # Semantic leakage
        has_sem, sem_details = detect_semantic_leakage(
            guide.get('question', ''),
            guide.get('answer', ''),
            final_question,
            final_answer,
            options
        )
        if has_sem:
            has_any_leakage = True
            all_details.extend([f"Guide {i+1} semantic: {d}" for d in sem_details])
            if max_severity != "high":
                max_severity = "medium"

    return LeakageResult(
        has_leakage=has_any_leakage,
        leakage_type="lexical" if "lexical" in " ".join(all_details) else
                    "semantic" if "semantic" in " ".join(all_details) else
                    "none",
        details=all_details,
        severity=max_severity
    )


def compute_leakage_score(
    guides: list[dict],
    final_question: str,
    final_answer: str,
    options: dict[str, str] = None
) -> float:
    """
    计算泄漏分数（方案 §9: answer_leakage_score）

    Returns:
        0.0 = 无泄漏（通过）
        1.0 = 严重泄漏（失败）
    """
    result = detect_answer_leakage(guides, final_question, final_answer, options)

    if not result.has_leakage:
        return 0.0

    # 根据严重程度返回分数
    severity_scores = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.3,
        "none": 0.0
    }
    return severity_scores.get(result.severity, 0.5)
