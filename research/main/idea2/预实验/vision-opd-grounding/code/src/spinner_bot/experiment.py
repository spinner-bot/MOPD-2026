from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptCase:
    name: str
    use_original_question: bool = True
    use_bbox_image: bool = False
    use_grounding_question: bool = False
    use_teacher_image: bool = False


CASES: dict[str, PromptCase] = {
    "orig": PromptCase(name="orig"),
    "orig_bbox": PromptCase(name="orig_bbox", use_bbox_image=True),
    "orig_ground": PromptCase(name="orig_ground", use_grounding_question=True),
    "all": PromptCase(name="all", use_bbox_image=True, use_grounding_question=True, use_teacher_image=True),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def outputs_dir() -> Path:
    out = project_root() / "outputs"
    out.mkdir(exist_ok=True)
    return out


def build_grounding_question(question: str, bbox: list[int] | None, problem: str | None = None) -> str:
    """Use the dataset's native grounding instruction, without leaking labels."""
    if problem:
        grounded = problem.replace("<image>", "", 1).strip()
        marker = "\n\nAnswer with the option's letter from the given choices."
        if marker in grounded:
            return grounded
        return grounded
    return question


def build_prompt(question: str, grounding_question: str | None, case: PromptCase) -> str:
    q = grounding_question if case.use_grounding_question and grounding_question else question
    parts = [q.strip()]
    if case.use_bbox_image:
        parts.append("The second image is the same scene with a red box marking the relevant region.")
    if case.use_teacher_image:
        parts.append("The last image is a crop of the relevant region.")
    return "\n".join(parts).strip()
