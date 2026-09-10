"""Deterministic bbox coordinate-space normalization for T02 proposals.

Qwen-VL family models natively emit bbox coordinates normalized to [0,1000]
regardless of prompt instructions (observed in pilot 20260909T: 95/95 bboxes
capped at exactly 1000 while source images were 1500-2672 px). Proposals whose
coordinates exceed 1000 on any axis are treated as already pixel-space.

The transform is deterministic and auditable: both the normalized input and the
pixel-space output are preserved on each region, together with a provenance tag.
"""
from __future__ import annotations

NORM_SPACE_MAX = 1000


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def normalize_bbox(box, width, height):
    """Return (pixel_box, coord_space, norm_box_or_None) for one bbox_xyxy."""
    ints = [int(x) for x in box]
    if not width or not height:
        return ints, "unknown_dims", None
    if any(x > NORM_SPACE_MAX for x in ints):
        return ints, "pixel", None
    x1, y1, x2, y2 = ints
    pixel = [
        round(x1 / NORM_SPACE_MAX * width),
        round(y1 / NORM_SPACE_MAX * height),
        round(x2 / NORM_SPACE_MAX * width),
        round(y2 / NORM_SPACE_MAX * height),
    ]
    return pixel, "norm1000_to_pixel", ints


def normalize_proposal_geometry(proposal, width, height):
    """Rewrite evidence_regions bboxes of one parsed proposal into pixel space."""
    if not isinstance(proposal, dict) or not isinstance(proposal.get("evidence_regions"), list):
        return proposal
    for region in proposal["evidence_regions"]:
        if not isinstance(region, dict):
            continue
        box = region.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(_is_number(x) for x in box):
            continue
        pixel, space, norm = normalize_bbox(box, width, height)
        region["bbox_xyxy"] = pixel
        region["bbox_coord_space"] = space
        if norm is not None:
            region["bbox_xyxy_norm1000"] = norm
    return proposal
