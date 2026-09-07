from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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
    """Build a grounding-aware variant from the dataset's original problem.

    The dataset already defines the target semantics: only the object inside the
    red box should be used. We preserve the original stem and choices, remove
    the image placeholder, and add normalized spatial/box metadata derived only
    from the native bbox and image-independent coordinates.
    """
    source = (problem or question).replace("<image>", "", 1).strip()
    source = re.sub(r"\n\s*Only focus on the objects inside the red bounding box in the image to answer this question\.\s*", "\n", source).strip()
    if not bbox or len(bbox) != 4:
        return source
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # Coordinates are retained as the exact native annotation; coarse labels
    # provide a resolution-agnostic grounding cue without inventing content.
    horiz = "left" if cx < 0.33 * max(x2, 1) else "right" if cx > 0.66 * max(x2, 1) else "center"
    vert = "upper" if cy < 0.33 * max(y2, 1) else "lower" if cy > 0.66 * max(y2, 1) else "middle"
    lead = (
        "Ground the answer only in the object or region specified by the dataset annotation. "
        f"The annotated box is [{x1}, {y1}, {x2}, {y2}], approximately in the {vert}-{horiz} part of the image. "
        "Ignore evidence outside this region.\n\n"
    )
    return lead + source


def build_prompt(question: str, grounding_question: str | None, case: PromptCase) -> str:
    q = grounding_question if case.use_grounding_question and grounding_question else question
    parts = [q.strip()]
    if case.use_bbox_image:
        parts.append("The second image is the same scene with a red box marking the relevant region.")
    if case.use_teacher_image:
        parts.append("The last image is a crop of the relevant region.")
    return "\n".join(parts).strip()
