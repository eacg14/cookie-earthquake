"""Orchestrates a full photo series: run pose detection on every frame,
determine the sequence's shared rotation-correction angle from the
designated reference frame, merge in any manual per-frame overrides, then
compute and apply each frame's alignment + crop.

This is the module app.py calls into; it has no Streamlit dependency itself
so it can be exercised directly in tests/scripts too.
"""

from __future__ import annotations

import numpy as np

from . import alignment, cropping
from .detection import PoseDetector, detect_frame
from .models import AlignmentRecord, FramePlan
from .presets import PresetSpec


def build_frame_plans(
    frames: list[np.ndarray],
    preset: PresetSpec,
    reference_frame_idx: int,
    detector: PoseDetector,
    manual_overrides: dict[int, AlignmentRecord] | None = None,
    rotation_enabled: bool = True,
) -> list[FramePlan]:
    """Detect + align + crop every frame in `frames` (RGB, EXIF-oriented arrays).

    `manual_overrides` replaces the auto-detected AlignmentRecord for the
    given frame indices (source="manual" records supplied by the review UI).
    """
    if not frames:
        raise ValueError("build_frame_plans requires at least one frame")
    if not 0 <= reference_frame_idx < len(frames):
        raise ValueError(f"reference_frame_idx {reference_frame_idx} out of range")

    manual_overrides = manual_overrides or {}
    plans: list[FramePlan] = []

    for index, image in enumerate(frames):
        if index in manual_overrides:
            record = manual_overrides[index]
            pose_result = None
            warnings: list[str] = []
        else:
            pose_result, record, warnings = detect_frame(detector, image)
        plans.append(FramePlan(frame_index=index, alignment=record, pose_result=pose_result, warnings=warnings))

    reference_record = plans[reference_frame_idx].alignment
    if rotation_enabled and not reference_record.needs_manual:
        global_rotation = alignment.reference_angle(reference_record)
    else:
        global_rotation = 0.0
        if rotation_enabled and reference_record.needs_manual:
            plans[reference_frame_idx].warnings.append(
                "Reference frame needs manual correction; rotation correction "
                "disabled until it is fixed."
            )

    for plan, image in zip(plans, frames):
        if plan.blocked:
            plan.warnings.append("Blocked from export until this frame's anchor is set manually.")
            continue

        matrix = alignment.compute_transform(plan.alignment, preset, global_rotation)
        output_image, tight_fraction = cropping.warp_and_crop(image, matrix, preset)

        plan.transform_matrix = matrix
        plan.output_image = output_image
        plan.tight_crop_fraction = tight_fraction
        if cropping.is_tight_crop(tight_fraction):
            plan.warnings.append(
                f"Tight crop: canvas reaches {tight_fraction:.0%} past this photo's edge."
            )

    return plans


def exportable_plans(plans: list[FramePlan]) -> list[FramePlan]:
    """Plans that are ready to export (not blocked on a manual correction)."""
    return [p for p in plans if not p.blocked]
