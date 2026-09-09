from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class PromptCase:
    name: str
    use_original_question: bool = True
    use_separator_question: bool = False
    use_grounding_separator: bool = False


CASES: dict[str, PromptCase] = {
    "orig": PromptCase(name="orig"),
    "separator": PromptCase(name="separator", use_separator_question=True),
    "grounding": PromptCase(name="grounding", use_grounding_separator=True),
    "all_text": PromptCase(name="all_text", use_separator_question=True, use_grounding_separator=True),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def outputs_dir() -> Path:
    out = project_root() / "outputs"
    out.mkdir(exist_ok=True)
    return out


def _clean_problem(question: str, problem: str | None = None) -> str:
    source = (problem or question).replace("<image>", "", 1).strip()
    source = re.sub(r"\n\s*Only focus on the objects inside the red bounding box in the image to answer this question\.\s*", "\n", source).strip()
    return source


def _stem(question: str, problem: str | None = None) -> str:
    source = _clean_problem(question, problem)
    return re.split(r"\n\s*A\.\s", source, maxsplit=1)[0].strip()


def build_separator_question(question: str, problem: str | None = None) -> str:
    """Create a text-only intermediate question from the original dataset item.

    This is deliberately answer-neutral: it asks the model to identify the
    queried target and the requested attribute before selecting an option.
    """
    return (
        "Separator question: before answering the multiple-choice question, "
        f"what visual target does this question ask you to inspect, and what property of that target must be determined? "
        f"Question: {_stem(question, problem)}"
    )


def build_grounding_separator(question: str, bbox: list[int] | None, problem: str | None = None) -> str:
    """Create the dataset-aware grounding separator without adding an image."""
    target = build_separator_question(question, problem)
    region = "the evidence region selected for this dataset item"
    if bbox and len(bbox) == 4:
        region += f" (native annotation box [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}])"
    return (
        "Grounding separator question: which specific visual evidence inside "
        f"{region} is needed to answer the question? Inspect that evidence first, "
        "then answer the original question using only the given choices.\n" + target
    )


def build_prompt(question: str, separator: str | None, grounding: str | None, case: PromptCase) -> str:
    parts = [question.strip()]
    if case.use_separator_question and separator:
        parts.append(separator.strip())
    if case.use_grounding_separator and grounding:
        parts.append(grounding.strip())
    return "\n".join(parts).strip()
