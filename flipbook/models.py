"""Shared dataclasses passed between alignment, cropping, and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

AlignmentSource = Literal["auto", "manual"]


@dataclass
class FramePlan:
    """Everything needed to render one output frame.

    The reference frame's matrix is a plain center-crop
    (alignment.compute_reference_transform) -- no repositioning, no rotation
    leveling, just the photographer's original framing at whatever size the
    series' shared area allows. Every other frame's matrix comes from
    matching its static background to the reference frame (features.py),
    composed with the reference frame's own crop (alignment.compose_affine).
    Either way, by the time it reaches a FramePlan it's just "the transform
    that maps this frame's original pixels to the output canvas" -- uniform
    regardless of how it was derived.
    """

    frame_index: int
    matrix: np.ndarray | None = None  # 2x3 affine, original pixels -> output canvas
    source: AlignmentSource = "auto"
    confidence: float = 1.0
    needs_manual: bool = False
    match_info: str = ""  # e.g. "58 matches, 92% inliers" -- diagnostic, shown in the UI
    output_image: np.ndarray | None = None  # warped/cropped result, once rendered
    tight_crop_fraction: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """True while this frame still needs manual correction before it can be exported."""
        return self.needs_manual
