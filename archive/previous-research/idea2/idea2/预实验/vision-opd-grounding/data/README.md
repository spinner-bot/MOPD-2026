---
language:
- en
license: apache-2.0
tags:
- multimodal
- vision-language-model
- fine-grained-perception
- self-distillation
- on-policy-distillation
- vqa
size_categories:
- 1K<n<10K
---

# Vision-OPD-6K: Training Data for Vision-OPD

## Overview

**Vision-OPD** proposes a regional-to-global self-distillation framework that transfers the model's own privileged regional perception to its full-image policy, without external teacher models, ground-truth labels, reward verifiers, or inference-time tool use.

Vision-OPD instantiates two conditional policies from the same MLLM:

- A **crop-conditioned teacher** that observes the evidence-centered crop as a privileged input
- A **full-image-conditioned student** that observes the full image as in standard inference

The student generates on-policy rollouts, and Vision-OPD minimizes token-level divergence between the teacher and student next-token distributions along these rollouts.

This dataset contains **6.2K** fine-grained visual question answering samples used for Vision-OPD training. Each sample includes a full image with red bounding box overlay (student input), a cropped region image (teacher input), and the original image with bounding box coordinates.

## Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Total Samples** | 6,241 |
| **Image Types** | Full image with bbox overlay (student) + cropped region (teacher) + original image |

## Data Structure & Extraction

### 1. Extract Student Images

```bash
cd images/
cat images.tar.gz* | tar -xvf - -C ./
```

### 2. Extract Teacher Images

```bash
cd teacher_images/
tar -xvf teacher_images.tar.gz -C ./
```

### 3. Extract Original Images (Optional)

```bash
cd original_images/
tar -xvf original_images.tar.gz -C ./
```

## Data Format

Each record in `train.jsonl` contains:

| Field | Type | Description |
|---|---|---|
| `images` | `list[str]` | Relative path to bbox overlay image (student input) |
| `teacher_images` | `list[str]` | Relative path to cropped region image (teacher input) |
| `original_images` | `list[str]` | Relative path to original image (without bbox overlay) |
| `bbox` | `list[int]` | Bounding box coordinates `[x1, y1, x2, y2]` in the original image |
| `problem` | `str` | Question prompt with `<image>` token |
| `answer` | `str` | Ground truth answer |
| `extra_info` | `dict` | Contains `answer` and `question` (without `<image>` prefix) |

## Intended Use

- **On-policy self-distillation** training of MLLMs for fine-grained visual understanding
- **Research on regional-to-global perception** transfer without inference-time tool use

## Related Resources

| Resource | Link |
|----------|------|
| **Paper** | [arXiv:2605.18740](https://arxiv.org/abs/2605.18740) |
| **Code** | [GitHub: VisionOPD/Vision-OPD](https://github.com/VisionOPD/Vision-OPD) |

## Citation

```bibtex
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```

## License

Apache-2.0 License
