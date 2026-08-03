"""Wraps MediaPipe's PoseLandmarker (Tasks API, IMAGE mode) and implements the
anchor/scale-reference fallback chain that turns raw pose landmarks into an
AlignmentRecord.

The fallback logic (`alignment_record_from_landmarks`) is a pure function of
a landmark dict + image size -- it does not touch MediaPipe -- so it can be
unit tested with synthetic landmarks without loading a model.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .models import AlignmentRecord, Landmark, PoseResult

# BlazePose 33-point topology indices used here.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

VISIBILITY_THRESHOLD = 0.5

# Fallback-of-last-resort calibration: when neither hip is usable, we estimate
# an equivalent "torso length" from the shoulder-midpoint-to-nose distance.
# This ratio is an approximation (typical neck+lower-face proportion relative
# to torso length) meant to be tuned against real photos, not a biomechanical
# constant -- frames using it get a reduced confidence score.
NOSE_TO_TORSO_RATIO = 0.55


def _visible(landmarks: dict[int, Landmark], index: int) -> bool:
    lm = landmarks.get(index)
    return lm is not None and lm.visibility >= VISIBILITY_THRESHOLD


def _midpoint(landmarks: dict[int, Landmark], i: int, j: int) -> tuple[float, float]:
    a, b = landmarks[i], landmarks[j]
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def alignment_record_from_landmarks(
    landmarks: dict[int, Landmark],
    image_width: int,
    image_height: int,
) -> tuple[AlignmentRecord, list[str]]:
    """Pick an anchor + scale/rotation reference point from detected landmarks.

    Fallback chain, most to least preferred:
    1. Both hips + both shoulders visible -> anchor = hip midpoint,
       reference = shoulder midpoint (true torso length).
    2. One hip + both shoulders visible -> anchor = that hip, reference =
       shoulder midpoint (still a torso-length measure, just off-center).
    3. Both shoulders visible, no usable hip -> anchor = shoulder midpoint,
       reference synthesized along the shoulder->nose direction at an
       estimated torso-equivalent distance (NOSE_TO_TORSO_RATIO). Approximate,
       flagged with reduced confidence.
    4. Nothing usable -> needs_manual record centered on the image.
    """
    warnings: list[str] = []

    hips_ok = _visible(landmarks, LEFT_HIP) and _visible(landmarks, RIGHT_HIP)
    shoulders_ok = _visible(landmarks, LEFT_SHOULDER) and _visible(landmarks, RIGHT_SHOULDER)
    left_hip_ok = _visible(landmarks, LEFT_HIP)
    right_hip_ok = _visible(landmarks, RIGHT_HIP)

    if hips_ok and shoulders_ok:
        anchor = _midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
        reference = _midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        confidence = min(
            landmarks[LEFT_HIP].visibility,
            landmarks[RIGHT_HIP].visibility,
            landmarks[LEFT_SHOULDER].visibility,
            landmarks[RIGHT_SHOULDER].visibility,
        )
        return AlignmentRecord(anchor, reference, "auto", confidence, False), warnings

    if (left_hip_ok or right_hip_ok) and shoulders_ok:
        hip_index = LEFT_HIP if left_hip_ok else RIGHT_HIP
        anchor = (landmarks[hip_index].x, landmarks[hip_index].y)
        reference = _midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        confidence = 0.9 * min(
            landmarks[hip_index].visibility,
            landmarks[LEFT_SHOULDER].visibility,
            landmarks[RIGHT_SHOULDER].visibility,
        )
        warnings.append("Only one hip landmark was reliable; anchored off a single hip point.")
        return AlignmentRecord(anchor, reference, "auto", confidence, False), warnings

    if shoulders_ok and _visible(landmarks, NOSE):
        anchor = _midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        nose = landmarks[NOSE]
        dx, dy = nose.x - anchor[0], nose.y - anchor[1]
        neck_length = math.hypot(dx, dy)
        torso_estimate = neck_length * NOSE_TO_TORSO_RATIO
        if neck_length > 0 and torso_estimate > 0:
            ux, uy = dx / neck_length, dy / neck_length
            reference = (anchor[0] + ux * torso_estimate, anchor[1] + uy * torso_estimate)
            confidence = 0.4 * min(
                landmarks[LEFT_SHOULDER].visibility,
                landmarks[RIGHT_SHOULDER].visibility,
                landmarks[NOSE].visibility,
            )
            warnings.append(
                "No reliable hip landmark; estimated scale/rotation from shoulders and nose. "
                "Review this frame's alignment."
            )
            return AlignmentRecord(anchor, reference, "auto", confidence, False), warnings

    warnings.append("Could not reliably detect a pose in this frame; manual correction required.")
    center = (image_width / 2.0, image_height / 2.0)
    fallback_reference = (center[0], center[1] - image_height * 0.1)
    return AlignmentRecord(center, fallback_reference, "auto", 0.0, True), warnings


def pose_result_from_task_output(raw_landmarks, image_width: int, image_height: int) -> PoseResult:
    """Convert a MediaPipe PoseLandmarker result's landmark list (normalized
    0-1 coords) for one detected pose into a PoseResult in pixel coordinates.
    """
    landmarks: dict[int, Landmark] = {}
    for i, lm in enumerate(raw_landmarks):
        landmarks[i] = Landmark(
            x=lm.x * image_width,
            y=lm.y * image_height,
            visibility=getattr(lm, "visibility", 1.0),
            presence=getattr(lm, "presence", 1.0),
        )
    return PoseResult(landmarks=landmarks, detected=True)


class PoseDetector:
    """Loads a MediaPipe PoseLandmarker model once and detects poses in stills.

    Requires the `.task` model asset to already be downloaded (see
    scripts/download_models.py) -- the pip package does not bundle it.
    """

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Run scripts/download_models.py first."
            )

        # Imported lazily so importing this module (e.g. for the pure
        # fallback-chain function above) doesn't require mediapipe to be
        # installed/importable in unit tests.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> PoseResult:
        """`image_rgb` must be an HxWx3 RGB uint8 array."""
        height, width = image_rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return PoseResult(landmarks={}, detected=False)
        return pose_result_from_task_output(result.pose_landmarks[0], width, height)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def detect_frame(
    detector: PoseDetector,
    image_rgb: np.ndarray,
) -> tuple[PoseResult, AlignmentRecord, list[str]]:
    """Run pose detection on one frame and derive its auto AlignmentRecord."""
    height, width = image_rgb.shape[:2]
    pose_result = detector.detect(image_rgb)
    if not pose_result.detected:
        record, warnings = alignment_record_from_landmarks({}, width, height)
        return pose_result, record, warnings
    record, warnings = alignment_record_from_landmarks(pose_result.landmarks, width, height)
    return pose_result, record, warnings
