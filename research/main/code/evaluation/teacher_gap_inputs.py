"""Render the four frozen conditions in the 1017 teacher-gap protocol.

The final task shell is byte-identical across conditions.  Only the contents of
the ``<PRIVILEGED_CONTEXT>`` block may change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


TEACHER_GAP_CONDITIONS = ("S", "T_Q", "T_QA", "T_generic")

COMMON_SHELL = """You must answer the multiple-choice question using the image and the information below.

<PRIVILEGED_CONTEXT>
{privileged_context}
</PRIVILEGED_CONTEXT>

Question: {question}
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Return exactly one letter: A, B, C, or D."""


@dataclass(frozen=True)
class RenderedTeacherGapInput:
    condition: str
    prompt: str
    privileged_context: str
    image_path: str
    shell_hash: str
    prompt_hash: str


def _require_text(sample: dict, key: str) -> str:
    value = str(sample.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def _guides(sample: dict) -> list[dict[str, str]]:
    guides = sample.get("guides")
    if not isinstance(guides, list) or not 1 <= len(guides) <= 4:
        raise ValueError("guides must be a list containing 1..4 items")
    normalized = []
    for index, guide in enumerate(guides, start=1):
        if not isinstance(guide, dict):
            raise ValueError(f"guide {index} must be an object")
        question = str(guide.get("question", "")).strip()
        answer = str(guide.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"guide {index} requires question and answer")
        normalized.append({"question": question, "answer": answer})
    return normalized


def privileged_context(sample: dict, condition: str) -> str:
    """Return the sole condition-dependent text block."""

    if condition not in TEACHER_GAP_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    if condition == "S":
        return ""
    if condition == "T_generic":
        return _require_text(sample, "generic_context")
    guides = _guides(sample)
    lines = []
    for index, guide in enumerate(guides, start=1):
        lines.append(f"Guide {index}: {guide['question']}")
        if condition == "T_QA":
            lines.append(f"Intermediate answer {index}: {guide['answer']}")
    return "\n".join(lines)


def render_teacher_gap_input(sample: dict, condition: str) -> RenderedTeacherGapInput:
    """Render one condition while preserving the common prompt shell."""

    context = privileged_context(sample, condition)
    values = {
        "privileged_context": context,
        "question": _require_text(sample, "final_question"),
        "option_A": _require_text(sample, "option_A"),
        "option_B": _require_text(sample, "option_B"),
        "option_C": _require_text(sample, "option_C"),
        "option_D": _require_text(sample, "option_D"),
    }
    prompt = COMMON_SHELL.format(**values)
    return RenderedTeacherGapInput(
        condition=condition,
        prompt=prompt,
        privileged_context=context,
        image_path=_require_text(sample, "image_original"),
        shell_hash=hashlib.sha256(COMMON_SHELL.encode("utf-8")).hexdigest(),
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def validate_generic_length(
    samples: list[dict],
    token_count,
    tolerance: float = 0.10,
) -> dict:
    """Check the 1017 corpus-level Generic-T/T-Q median token contract.

    ``token_count`` is injected so the frozen experiment tokenizer, rather than
    an approximate whitespace count, defines length.
    """

    if not samples:
        raise ValueError("samples must not be empty")
    tq_lengths = sorted(token_count(privileged_context(row, "T_Q")) for row in samples)
    generic_lengths = sorted(token_count(privileged_context(row, "T_generic")) for row in samples)

    def median(values: list[int]) -> float:
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return (values[middle - 1] + values[middle]) / 2

    tq_median = median(tq_lengths)
    generic_median = median(generic_lengths)
    if tq_median <= 0:
        raise ValueError("T-Q median token length must be positive")
    relative_difference = abs(generic_median - tq_median) / tq_median
    return {
        "t_q_median_tokens": tq_median,
        "generic_median_tokens": generic_median,
        "relative_difference": relative_difference,
        "tolerance": tolerance,
        "passes": relative_difference <= tolerance,
    }
