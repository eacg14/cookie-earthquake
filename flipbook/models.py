"""Shared dataclasses passed between detection, alignment, cropping, and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

AlignmentSource = Literal["auto", "manual"]


@dataclass(frozen=True)
class Landmark:
    """A single pose landmark in pixel coordinates (image space)."""

    x: float
    y: float
    visibility: float = 1.0
    presence: float = 1.0


@dataclass
class PoseResult:
    """Raw MediaPipe PoseLandmarker output for one frame, kept around for overlay drawing."""

    landmarks: dict[int, Landmark] = field(default_factory=dict)
    detected: bool = False


@dataclass
class AlignmentRecord:
    """The anchor + scale/rotation reference used to align one frame.

    `anchor` is the point that gets translated to the preset's
    target_anchor_frac (hip midpoint, or a fallback point -- see detection.py).
    `reference_point` is the other end of the scale/rotation reference segment
    (torso length: shoulder midpoint down to the anchor). Its distance from
    `anchor` drives scaling; its angle off vertical, relative to the sequence's
    designated reference frame, drives the shared rotation correction.
    """

    anchor: tuple[float, float]
    reference_point: tuple[float, float]
    source: AlignmentSource = "auto"
    confidence: float = 1.0
    needs_manual: bool = False

    @property
    def length(self) -> float:
        (ax, ay), (rx, ry) = self.anchor, self.reference_point
        return ((rx - ax) ** 2 + (ry - ay) ** 2) ** 0.5


@dataclass
class FramePlan:
    """Everything needed to render one output frame."""

    frame_index: int
    alignment: AlignmentRecord
    pose_result: PoseResult | None = None
    transform_matrix: np.ndarray | None = None  # 2x3 affine matrix once computed
    output_image: np.ndarray | None = None  # warped/cropped result, once rendered
    tight_crop_fraction: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """True while this frame still needs manual correction before it can be exported."""
        return self.alignment.needs_manual
