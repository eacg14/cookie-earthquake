"""Orchestrates a full photo series: pose-detect only the designated
reference frame (to establish the crop's anchor position + rotation), then
align every other frame by matching its static background to that reference
frame rather than tracking the moving subject -- the crop only has to cancel
out camera shake/pan between shots, not chase wherever the golfer is
mid-swing.

Crop SIZE is not derived from pose data at all (torso length says nothing
about how much real background is actually shared across the series, and
using it just threw off the framing). Instead, once every frame is
registered into the reference frame's coordinate space, we binary-search for
the largest crop that still stays within every resolved frame's real pixel
bounds -- i.e. the maximum area shared by all the aligned photos, with zero
fabricated (border-replicated) pixels.

This is the module app.py calls into; it has no Streamlit dependency itself
so it can be exercised directly in tests/scripts too.
"""

from __future__ import annotations

import numpy as np

from . import alignment, cropping, features
from .detection import PoseDetector, detect_frame
from .models import AlignmentRecord, FramePlan
from .presets import PresetSpec

# Bounds for the max-area scale search, in output-canvas pixels per
# reference-frame pixel. Generic and resolution-independent: SCALE_SEARCH_LO
# is loose enough to be unsafe for virtually any real photo (forcing the
# search to tighten from there), SCALE_SEARCH_HI is tight enough to be safe
# for virtually any real photo (used as a fallback ceiling if even that
# doesn't fit for some frame).
SCALE_SEARCH_LO = 0.02
SCALE_SEARCH_HI = 5.0
SCALE_SEARCH_ITERATIONS = 30
SCALE_SEARCH_EPSILON = 1e-6


def _max_oob_across_frames(
    scale: float,
    ref_record: AlignmentRecord,
    preset: PresetSpec,
    global_rotation: float,
    ref_shape: tuple[int, int],
    frame_transforms: dict[int, np.ndarray],
    frame_shapes: dict[int, tuple[int, int]],
) -> float:
    matrix_ref = alignment.compute_transform(ref_record, preset, global_rotation, scale)
    worst = cropping.out_of_bounds_fraction(matrix_ref, preset, ref_shape)
    for idx, t_matrix in frame_transforms.items():
        combined = alignment.compose_affine(matrix_ref, t_matrix)
        worst = max(worst, cropping.out_of_bounds_fraction(combined, preset, frame_shapes[idx]))
    return worst


def _find_max_area_scale(
    ref_record: AlignmentRecord,
    preset: PresetSpec,
    global_rotation: float,
    ref_shape: tuple[int, int],
    frame_transforms: dict[int, np.ndarray],
    frame_shapes: dict[int, tuple[int, int]],
) -> float:
    """The smallest (loosest/largest-area) scale such that every frame in
    `frame_transforms` (plus the reference frame itself) stays within its
    own real pixel bounds -- i.e. the biggest crop with zero fabricated
    pixels. Falls back to SCALE_SEARCH_HI (tightest allowed) if even that
    doesn't fit for some frame; per-frame tight-crop warnings still catch
    that case downstream.
    """

    def max_oob(scale: float) -> float:
        return _max_oob_across_frames(
            scale, ref_record, preset, global_rotation, ref_shape, frame_transforms, frame_shapes
        )

    low, high = SCALE_SEARCH_LO, SCALE_SEARCH_HI
    if max_oob(high) > SCALE_SEARCH_EPSILON:
        return high
    if max_oob(low) <= SCALE_SEARCH_EPSILON:
        return low

    for _ in range(SCALE_SEARCH_ITERATIONS):
        mid = (low + high) / 2
        if max_oob(mid) <= SCALE_SEARCH_EPSILON:
            high = mid
        else:
            low = mid
    return high


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
    frame's crop framing (anchor position + rotation reference; no longer
    scale). Pass None to have this function pose-detect it directly (fine
    for one-off/test calls); callers that rerun this repeatedly (e.g. the
    Streamlit app, on every widget interaction) should cache the detection
    result themselves -- whether auto-detected or manually corrected -- and
    pass it in here to avoid re-running MediaPipe each time.

    `manual_frame_matrices` supplies a final (original-pixels -> output
    canvas) transform directly for specific non-reference frame indices,
    from manually-clicked point correspondences, overriding auto background
    matching for those frames. These frames are excluded from the max-area
    scale search (their transform is fixed regardless of scale) but are
    still rendered and flagged if they end up needing a tight crop.
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

    ref_gray = features.to_gray(frames[reference_frame_idx])
    ref_keypoints, ref_descriptors = features.detect_features(ref_gray)

    # First pass: resolve every non-reference frame's transform-to-reference
    # (auto background match, or note that it's blocked) before choosing a
    # scale, since the max-area search needs every resolved frame's geometry.
    frame_transforms: dict[int, np.ndarray] = {}
    match_infos: dict[int, str] = {}
    blocked_warnings: dict[int, str] = {}

    for i, image in enumerate(frames):
        if i == reference_frame_idx or i in manual_frame_matrices:
            continue
        frame_gray = features.to_gray(image)
        t_matrix, num_matches, inlier_ratio = features.estimate_transform_to_reference(
            frame_gray, ref_keypoints, ref_descriptors
        )
        if t_matrix is None:
            blocked_warnings[i] = (
                f"Couldn't reliably match this frame's background to the reference "
                f"frame ({num_matches} candidate matches, {inlier_ratio:.0%} agreeing) -- "
                "click 2 matching static points in both photos."
            )
        else:
            frame_transforms[i] = t_matrix
            match_infos[i] = f"{num_matches} matches, {inlier_ratio:.0%} inliers"

    frame_shapes = {i: frames[i].shape[:2] for i in frame_transforms}
    ref_shape = frames[reference_frame_idx].shape[:2]

    scale = _find_max_area_scale(
        ref_record, preset, global_rotation, ref_shape, frame_transforms, frame_shapes
    )
    matrix_ref = alignment.compute_transform(ref_record, preset, global_rotation, scale)

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
        elif i in manual_frame_matrices:
            plans.append(
                FramePlan(
                    frame_index=i,
                    matrix=manual_frame_matrices[i],
                    source="manual",
                    confidence=1.0,
                    needs_manual=False,
                )
            )
        elif i in frame_transforms:
            combined = alignment.compose_affine(matrix_ref, frame_transforms[i])
            plans.append(
                FramePlan(
                    frame_index=i,
                    matrix=combined,
                    source="auto",
                    confidence=1.0,
                    needs_manual=False,
                    match_info=match_infos[i],
                )
            )
        else:
            plans.append(
                FramePlan(
                    frame_index=i,
                    matrix=None,
                    source="auto",
                    confidence=0.0,
                    needs_manual=True,
                    warnings=[blocked_warnings[i]],
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
