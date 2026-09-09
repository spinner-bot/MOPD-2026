"""Build and validate the nine frozen Stage 0 contexts."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

STAGE0_CONDITIONS = ("S", "V_crop", "V_add", "T_Q", "T_QA", "VT", "T_random", "T_shuffle", "T_generic")
PROMPT_FILES = {"S": "student.txt", "V_crop": "v_crop.txt", "V_add": "v_add.txt", "T_Q": "t_q.txt", "T_QA": "t_qa.txt", "VT": "vt.txt", "T_random": "t_random.txt", "T_shuffle": "t_shuffle.txt", "T_generic": "t_generic.txt"}

@dataclass(frozen=True)
class RenderedCondition:
    condition: str
    prompt: str
    image_paths: tuple[str, ...]

def extract_guides(sample: dict) -> list[dict[str, str]]:
    guides = []
    for index in range(1, 5):
        question = str(sample.get(f"guide_{index}_question", "")).strip()
        answer = str(sample.get(f"guide_{index}_answer", "")).strip()
        if not question and not answer:
            continue
        if not question or not answer:
            raise ValueError(f"guide_{index} must contain both question and answer")
        guides.append({"question": question, "answer": answer})
    if not guides:
        raise ValueError("sample has no guiding QA chain")
    return guides

def _format_guides(guides: list[dict[str, str]], answers: bool) -> str:
    result = []
    for index, guide in enumerate(guides, start=1):
        result.append(f"Guide {index}: {guide['question']}")
        if answers:
            result.append(f"Intermediate answer: {guide['answer']}")
    return "\n".join(result)

def validate_stage0_sample(sample: dict, project_root: str | Path | None = None) -> list[str]:
    errors = []
    required = ("sample_id", "source_image_id", "final_question", "option_A", "option_B", "option_C", "option_D", "final_answer", "image_original")
    for key in required:
        if not str(sample.get(key, "")).strip():
            errors.append(f"missing required field: {key}")
    if sample.get("final_answer") not in {"A", "B", "C", "D"}:
        errors.append("final_answer must be A, B, C, or D")
    try:
        extract_guides(sample)
    except ValueError as error:
        errors.append(str(error))
    if not str(sample.get("crop_primary", "")).strip():
        errors.append("missing crop_primary required by visual conditions")
    if not str(sample.get("random_questions", "")).strip() or not sample.get("random_questions_frozen", False):
        errors.append("missing frozen random_questions required by T_random")
    if project_root:
        for key in ("image_original", "crop_primary"):
            path = Path(sample.get(key, ""))
            path = path if path.is_absolute() else Path(project_root) / path
            if not path.is_file():
                errors.append(f"missing image asset for {key}: {path}")
    return errors

def render_condition(sample: dict, condition: str, prompts_dir: str | Path) -> RenderedCondition:
    if condition not in STAGE0_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    errors = validate_stage0_sample(sample)
    if errors:
        raise ValueError("; ".join(errors))
    guides = extract_guides(sample)
    shuffled = guides.copy()
    random.Random(int(hashlib.sha256(sample["sample_id"].encode()).hexdigest()[:16], 16)).shuffle(shuffled)
    template = (Path(prompts_dir) / PROMPT_FILES[condition]).read_text(encoding="utf-8")
    prompt = template.format(
        image="<image>", crop_image="<image>", original_image="<image>",
        question=sample["final_question"], option_A=sample["option_A"], option_B=sample["option_B"],
        option_C=sample["option_C"], option_D=sample["option_D"],
        guiding_questions=_format_guides(guides, False), guiding_qa_chain=_format_guides(guides, True),
        random_questions=sample["random_questions"], shuffled_questions=_format_guides(shuffled, True),
    )
    if condition == "V_crop":
        images = (sample["crop_primary"],)
    elif condition in {"V_add", "VT"}:
        images = (sample["image_original"], sample["crop_primary"])
    else:
        images = (sample["image_original"],)
    return RenderedCondition(condition, prompt, images)
