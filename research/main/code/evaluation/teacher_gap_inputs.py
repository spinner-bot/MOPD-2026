"""Render frozen conditions in the 1017 teacher-gap protocol.

The final task shell is byte-identical across conditions.  Only the contents of
the ``<PRIVILEGED_CONTEXT>`` block may change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


TEACHER_GAP_CONDITIONS = ("S", "T_Q", "T_QA", "T_generic")
NATURAL_ROTATION_CONDITIONS = ("S", "T_Q", "T_QA", "T_generic_Q", "T_generic_QA")

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

    if condition not in set(TEACHER_GAP_CONDITIONS) | set(NATURAL_ROTATION_CONDITIONS):
        raise ValueError(f"unsupported condition: {condition}")
    if condition == "S":
        return ""
    if condition == "T_generic":
        return _require_text(sample, "generic_context")
    if condition == "T_generic_Q":
        return _require_text(sample, "generic_context_t_q")
    if condition == "T_generic_QA":
        return _require_text(sample, "generic_context_t_qa")
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
    dual_control: bool = False,
) -> dict:
    """Check generic/structured token-length contracts.

    ``token_count`` is injected so the frozen experiment tokenizer, rather than
    an approximate whitespace count, defines length.  Historical four-condition
    profiles retain their corpus-median contract.  The natural dual-control
    profile requires every paired sample to pass independently.
    """

    if not samples:
        raise ValueError("samples must not be empty")
    def median(values: list[int]) -> float:
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return (values[middle - 1] + values[middle]) / 2

    pairs = (
        (("T_Q", "T_generic_Q"), ("T_QA", "T_generic_QA"))
        if dual_control
        else (("T_Q", "T_generic"),)
    )
    comparisons = {}
    for treatment, control in pairs:
        sample_checks = []
        treatment_lengths = []
        control_lengths = []
        for index, row in enumerate(samples):
            treatment_tokens = token_count(privileged_context(row, treatment))
            control_tokens = token_count(privileged_context(row, control))
            if treatment_tokens <= 0:
                raise ValueError(f"{treatment} token length must be positive at row {index}")
            relative_difference = abs(control_tokens - treatment_tokens) / treatment_tokens
            sample_checks.append({
                "sample_id": str(row.get("base_sample_id") or row.get("sample_id") or f"row-{index}"),
                "treatment_tokens": treatment_tokens,
                "control_tokens": control_tokens,
                "relative_difference": relative_difference,
                "passes": relative_difference <= tolerance,
            })
            treatment_lengths.append(treatment_tokens)
            control_lengths.append(control_tokens)
        treatment_median = median(sorted(treatment_lengths))
        control_median = median(sorted(control_lengths))
        if treatment_median <= 0:
            raise ValueError(f"{treatment} median token length must be positive")
        median_relative_difference = abs(control_median - treatment_median) / treatment_median
        if dual_control:
            relative_differences = sorted(check["relative_difference"] for check in sample_checks)
            failed_sample_ids = list(dict.fromkeys(
                check["sample_id"] for check in sample_checks if not check["passes"]
            ))
            passes = not failed_sample_ids
        else:
            relative_differences = [median_relative_difference]
            failed_sample_ids = []
            passes = median_relative_difference <= tolerance
        comparisons[f"{treatment}_vs_{control}"] = {
            "treatment_median_tokens": treatment_median,
            "control_median_tokens": control_median,
            "relative_difference": median_relative_difference,
            "per_sample_required": dual_control,
            "sample_checks": sample_checks if dual_control else [],
            "sample_relative_difference_median": median(relative_differences),
            "sample_relative_difference_max": max(relative_differences),
            "failed_sample_ids": failed_sample_ids,
            "passes": passes,
        }
    result = {
        "dual_control": dual_control,
        "tolerance": tolerance,
        "comparisons": comparisons,
        "passes": all(value["passes"] for value in comparisons.values()),
    }
    if not dual_control:
        only = next(iter(comparisons.values()))
        result.update({
            "t_q_median_tokens": only["treatment_median_tokens"],
            "generic_median_tokens": only["control_median_tokens"],
            "relative_difference": only["relative_difference"],
        })
    return result
