"""Pure math: turn an AlignmentRecord + preset + a shared rotation angle into a
2x3 affine transform matrix. No image, MediaPipe, or Streamlit dependencies --
this is the module the rest of the app's "does it actually align" behavior
hinges on, and the easiest to unit test with synthetic landmark data.

Coordinate convention: standard image pixel space, x right, y down.

The transform is built directly (not via cv2.getRotationMatrix2D, to avoid
that function's angle-sign convention) as:

    dst_point = L @ (src_point - anchor) + target_anchor

where L = scale * R(theta) is a rotation-then-scale linear map about the
anchor. This guarantees the anchor always lands exactly on the preset's
target_anchor_frac, regardless of rotation/scale.

The resulting 2x3 matrix maps src -> dst (forward), which is exactly what
cv2.warpAffine expects when called without the WARP_INVERSE_MAP flag.
"""

from __future__ import annotations

import math

import numpy as np

from .models import AlignmentRecord
from .presets import PresetSpec


def torso_vector(record: AlignmentRecord) -> tuple[float, float]:
    """The (dx, dy) vector from the anchor to the scale/rotation reference point."""
    (ax, ay), (rx, ry) = record.anchor, record.reference_point
    return (rx - ax, ry - ay)


def reference_angle(record: AlignmentRecord) -> float:
    """The rotation (radians) that would make this record's torso vector vertical.

    Solves for theta such that R(theta) @ (dx, dy) = (0, -r), i.e. the
    reference point ends up directly above the anchor. theta = atan2(-dx, -dy).
    """
    dx, dy = torso_vector(record)
    if dx == 0 and dy == 0:
        return 0.0
    return math.atan2(-dx, -dy)


def compute_transform(
    record: AlignmentRecord,
    preset: PresetSpec,
    global_rotation: float,
) -> np.ndarray:
    """Build the 2x3 affine matrix aligning `record` onto `preset`'s canonical framing.

    `global_rotation` is a single angle (radians) shared across the whole
    frame sequence -- computed once from a designated reference frame via
    `reference_angle`, not recomputed per-frame. Pass 0.0 to disable rotation
    correction entirely.
    """
    length = record.length
    if length <= 0:
        raise ValueError("AlignmentRecord has zero-length scale reference; cannot compute scale")

    target_length = preset.target_scale_frac * preset.output_height
    scale = target_length / length

    cos_t = math.cos(global_rotation)
    sin_t = math.sin(global_rotation)
    # L = scale * R(theta), R(theta) = [[cos, -sin], [sin, cos]]
    linear = scale * np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    anchor = np.array(record.anchor, dtype=np.float64)
    target_anchor = np.array(
        [
            preset.target_anchor_frac[0] * preset.output_width,
            preset.target_anchor_frac[1] * preset.output_height,
        ],
        dtype=np.float64,
    )

    translation = target_anchor - linear @ anchor

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
