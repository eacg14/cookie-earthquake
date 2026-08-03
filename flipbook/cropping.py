"""Applies an alignment.py affine matrix to a real image: warps/crops it onto
a preset's output canvas, and detects when the canvas reaches outside the
original photo's bounds (the photographer's composition was already too tight
to hit the target scale) so the UI can surface a warning instead of silently
degrading the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from .presets import PresetSpec

# Canvas edge is filled with the nearest source pixel when the affine maps
# outside the original photo -- avoids black bars, at the cost of a smeared
# edge that should be flagged rather than hidden (see out_of_bounds_fraction).
BORDER_MODE = cv2.BORDER_REPLICATE

# Fraction of the source image's own dimension that the canvas is allowed to
# reach past the photo's edge before we surface a "tight crop" warning.
TIGHT_CROP_WARNING_THRESHOLD = 0.05


def warp_frame(image: np.ndarray, matrix: np.ndarray, preset: PresetSpec) -> np.ndarray:
    """Warp `image` through `matrix` onto the preset's output canvas."""
    return cv2.warpAffine(
        image,
        matrix,
        (preset.output_width, preset.output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=BORDER_MODE,
    )


def out_of_bounds_fraction(
    matrix: np.ndarray,
    preset: PresetSpec,
    src_shape: tuple[int, int],
) -> float:
    """How far the output canvas reaches past the source image's edges.

    Maps the four output-canvas corners back into source-image space and
    measures the worst overhang, as a fraction of the source image's own
    width/height. 0.0 means the canvas is fully covered by real source
    pixels; e.g. 0.1 means a corner sampled 10% of the source's dimension
    beyond its edge (and had to be border-replicated).
    """
    src_h, src_w = src_shape[:2]
    out_w, out_h = preset.output_width, preset.output_height

    inverse = cv2.invertAffineTransform(matrix)
    corners = [(0, 0), (out_w, 0), (0, out_h), (out_w, out_h)]

    max_frac = 0.0
    for x, y in corners:
        sx = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
        sy = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
        overhang_x = max(0.0, -sx, sx - src_w)
        overhang_y = max(0.0, -sy, sy - src_h)
        frac = max(overhang_x / src_w, overhang_y / src_h)
        max_frac = max(max_frac, frac)
    return max_frac


def warp_and_crop(
    image: np.ndarray,
    matrix: np.ndarray,
    preset: PresetSpec,
) -> tuple[np.ndarray, float]:
    """Warp the frame and report how tight the resulting crop was."""
    output = warp_frame(image, matrix, preset)
    fraction = out_of_bounds_fraction(matrix, preset, image.shape)
    return output, fraction


def is_tight_crop(fraction: float) -> bool:
    return fraction > TIGHT_CROP_WARNING_THRESHOLD
