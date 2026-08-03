"""Pure math for building/composing 2x3 affine transforms. No image or
Streamlit dependencies.

The reference frame is used exactly as shot: no repositioning, no rotation
leveling -- its own original framing and orientation are the series'
canonical composition, respected as-is. Only its SIZE is adjusted (scaled to
a canvas-filling crop, chosen by pipeline.py to maximize how much of the
aligned series' shared area survives). Every other frame is then aligned to
the reference frame's background (features.py), not repositioned to any
"ideal" subject placement -- this module never decides where a subject
should sit in frame, only how to align/scale.

Coordinate convention: standard image pixel space, x right, y down. All
matrices map src -> dst (forward), which is what cv2.warpAffine expects
without the WARP_INVERSE_MAP flag.
"""

from __future__ import annotations

import numpy as np

from .presets import PresetSpec


def compute_reference_transform(
    ref_shape: tuple[int, int],
    preset: PresetSpec,
    scale: float,
) -> np.ndarray:
    """Center-crop transform for the reference frame: maps the frame's own
    geometric center to the output canvas's center, scaled by `scale`. No
    rotation, no repositioning -- the photographer's original framing is
    preserved exactly; only how much of it fits in the canvas changes.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")

    height, width = ref_shape[:2]
    center = np.array([width / 2.0, height / 2.0])
    target_center = np.array([preset.output_width / 2.0, preset.output_height / 2.0])

    linear = scale * np.eye(2)
    translation = target_center - linear @ center

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = translation
    return matrix


def apply_transform(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """Map a single (x, y) point through a 2x3 affine matrix. Useful for tests/overlays."""
    x, y = point
    dx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    dy = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return (dx, dy)


def compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine matrices: result(p) = outer(inner(p)).

    Used to chain "this frame's original pixels -> the reference frame's
    pixel space" (from background feature matching) with "the reference
    frame's pixel space -> the output canvas" (its center-crop) into a
    single direct transform.
    """
    outer3 = np.vstack([outer, [0.0, 0.0, 1.0]])
    inner3 = np.vstack([inner, [0.0, 0.0, 1.0]])
    combined = outer3 @ inner3
    return combined[:2, :]
