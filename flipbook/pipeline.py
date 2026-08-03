"""Orchestrates a full photo series: pose-detect only the designated
reference frame (to establish the crop's framing), then align every other
frame by matching its static background to that reference frame rather than
tracking the moving subject -- the crop only has to cancel out camera
shake/pan between shots, not chase wherever the golfer is mid-swing, so it
can stay far looser and more consistent across the series.

This is the module app.py calls into; it has no Streamlit dependency itself
so it can be exercised directly in tests/scripts too.
"""

from __future__ import annotations

import numpy as np

from . import alignment, cropping, features
from .detection import PoseDetector, detect_frame
from .models import AlignmentRecord, FramePlan
from .presets import PresetSpec


def build_frame_plans(
    frames: list[np.ndarray],
    preset: PresetSpec,
    reference_frame_idx: int,
    detector: PoseDetector,
    reference_record: AlignmentRecord | None = None,
    manual_frame_matrices: dict[int, np.ndarray] | None = None,
    rotation_enabled: bool = True,
) -> list[FramePlan]:
    """Detect + align + crop every frame in `frames` (RGB, EXIF-oriented arrays).

    `reference_record` is the AlignmentRecord to use for the reference
    frame's crop framing. Pass None to have this function pose-detect it
    directly (fine for one-off/test calls); callers that rerun this
    repeatedly (e.g. the Streamlit app, on every widget interaction) should
    cache the detection result themselves -- whether auto-detected or
    manually corrected -- and pass it in here to avoid re-running MediaPipe
    each time.

    `manual_frame_matrices` supplies a final (original-pixels -> output
    canvas) transform directly for specific non-reference frame indices,
    from manually-clicked point correspondences, overriding auto background
    matching for those frames.
    """
    if not frames:
        raise ValueError("build_frame_plans requires at least one frame")
    if not 0 <= reference_frame_idx < len(frames):
        raise ValueError(f"reference_frame_idx {reference_frame_idx} out of range")

    manual_frame_matrices = manual_frame_matrices or {}

    if reference_record is not None:
        ref_record = reference_record
        ref_warnings: list[str] = []
    else:
        _, ref_record, ref_warnings = detect_frame(detector, frames[reference_frame_idx])

    if ref_record.needs_manual:
        # Without a usable reference framing, nothing downstream can be
        # positioned -- every frame is blocked until the reference is fixed.
        message = (
            "Reference frame's pose could not be detected; set its anchor "
            "and scale points manually to establish the crop framing."
        )
        return [
            FramePlan(frame_index=i, matrix=None, source="auto", confidence=0.0, needs_manual=True, warnings=[message])
            for i in range(len(frames))
        ]

    global_rotation = alignment.reference_angle(ref_record) if rotation_enabled else 0.0
    matrix_ref = alignment.compute_transform(ref_record, preset, global_rotation)

    ref_gray = features.to_gray(frames[reference_frame_idx])
    ref_keypoints, ref_descriptors = features.detect_features(ref_gray)

    plans: list[FramePlan] = []
    for i, image in enumerate(frames):
        if i == reference_frame_idx:
            plans.append(
                FramePlan(
                    frame_index=i,
                    matrix=matrix_ref,
                    source=ref_record.source,
                    confidence=ref_record.confidence,
                    needs_manual=False,
                    match_info="reference frame",
                    warnings=list(ref_warnings),
                )
            )
            continue

        if i in manual_frame_matrices:
            plans.append(
                FramePlan(
                    frame_index=i,
                    matrix=manual_frame_matrices[i],
                    source="manual",
                    confidence=1.0,
                    needs_manual=False,
                )
            )
            continue

        frame_gray = features.to_gray(image)
        t_matrix, num_matches, inlier_ratio = features.estimate_transform_to_reference(
            frame_gray, ref_keypoints, ref_descriptors
        )
        if t_matrix is None:
            plans.append(
                FramePlan(
                    frame_index=i,
                    matrix=None,
                    source="auto",
                    confidence=inlier_ratio,
                    needs_manual=True,
                    warnings=[
                        f"Couldn't reliably match this frame's background to the reference "
                        f"frame ({num_matches} candidate matches, {inlier_ratio:.0%} agreeing) -- "
                        "click 2 matching static points in both photos."
                    ],
                )
            )
            continue

        combined = alignment.compose_affine(matrix_ref, t_matrix)
        plans.append(
            FramePlan(
                frame_index=i,
                matrix=combined,
                source="auto",
                confidence=inlier_ratio,
                needs_manual=False,
                match_info=f"{num_matches} matches, {inlier_ratio:.0%} inliers",
            )
        )

    for plan, image in zip(plans, frames):
        if plan.blocked:
            continue
        output_image, tight_fraction = cropping.warp_and_crop(image, plan.matrix, preset)
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
