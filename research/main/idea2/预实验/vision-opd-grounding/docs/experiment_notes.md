# Vision-OPD / OPD-V Four-Way Inference Test

Inputs:
1. `orig`: original image + original dataset question
2. `separator`: original image + original question + text separator question
3. `grounding`: original image + original question + dataset-aware grounding separator question
4. `all_text`: original image + original question + both text separator questions

Here, text grounding means privileged guiding text, not the Vision-OPD crop or
red-box image. The separator is generated from each item's original problem,
preserves its multiple-choice stem and is answer-neutral. The bbox is used only
to identify the dataset-selected evidence region; no ground-truth answer is
inserted into the prompt.

Dataset facts from the official data card: 6,241 train records with `images`, `teacher_images`, `original_images`, `bbox`, `problem`, `answer`, and `extra_info.question`.
