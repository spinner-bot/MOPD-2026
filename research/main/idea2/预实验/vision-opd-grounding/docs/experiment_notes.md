# Vision-OPD / OPD-V Four-Way Inference Test

Inputs:
1. `orig`: `original_images` + `extra_info.question`
2. `orig_bbox`: `original_images` + `images` red-box/bbox overlay + original question
3. `orig_ground`: `original_images` + original question prefixed with bbox grounding coordinates
4. `all`: `original_images` + red-box/bbox overlay + `teacher_images` crop + grounding question

Dataset facts from the official data card: 6,241 train records with `images`, `teacher_images`, `original_images`, `bbox`, `problem`, `answer`, and `extra_info.question`.
