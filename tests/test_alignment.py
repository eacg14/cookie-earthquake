import math

import pytest

from flipbook.alignment import apply_transform, compute_transform, reference_angle, torso_vector
from flipbook.detection import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    alignment_record_from_landmarks,
)
from flipbook.models import AlignmentRecord, Landmark
from flipbook.presets import PresetSpec

TEST_PRESET = PresetSpec(
    key="test",
    label="test",
    platform="test",
    aspect_w=1,
    aspect_h=1,
    output_width=1000,
    output_height=1000,
    target_anchor_frac=(0.5, 0.6),
    target_scale_frac=0.3,
)


def make_record(anchor, reference_point):
    return AlignmentRecord(anchor=anchor, reference_point=reference_point)


# --- torso_vector / reference_angle -----------------------------------------------------------


def test_torso_vector():
    record = make_record((100.0, 200.0), (100.0, 100.0))
    assert torso_vector(record) == (0.0, -100.0)


def test_reference_angle_zero_when_already_vertical():
    record = make_record((100.0, 200.0), (100.0, 100.0))
    assert reference_angle(record) == pytest.approx(0.0, abs=1e-9)


def test_reference_angle_degenerate_zero_length():
    record = make_record((50.0, 50.0), (50.0, 50.0))
    assert reference_angle(record) == 0.0


def test_reference_angle_verticalizes_tilted_torso():
    # Reference point tilted 30 degrees to the right of vertical.
    dx, dy = math.sin(math.radians(30)), -math.cos(math.radians(30))
    record = make_record((0.0, 0.0), (dx, dy))
    theta = reference_angle(record)

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotated = (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)
    assert rotated[0] == pytest.approx(0.0, abs=1e-9)
    assert rotated[1] < 0  # still points up


# --- compute_transform -------------------------------------------------------------------------


def test_compute_transform_maps_anchor_to_target():
    record = make_record((300.0, 500.0), (300.0, 400.0))
    matrix = compute_transform(record, TEST_PRESET, global_rotation=0.0)
    mapped_anchor = apply_transform(matrix, record.anchor)
    expected = (
        TEST_PRESET.target_anchor_frac[0] * TEST_PRESET.output_width,
        TEST_PRESET.target_anchor_frac[1] * TEST_PRESET.output_height,
    )
    assert mapped_anchor[0] == pytest.approx(expected[0], abs=1e-6)
    assert mapped_anchor[1] == pytest.approx(expected[1], abs=1e-6)


def test_compute_transform_scales_to_target_length():
    record = make_record((300.0, 500.0), (300.0, 350.0))  # length 150
    matrix = compute_transform(record, TEST_PRESET, global_rotation=0.0)
    mapped_anchor = apply_transform(matrix, record.anchor)
    mapped_reference = apply_transform(matrix, record.reference_point)
    mapped_length = math.hypot(
        mapped_reference[0] - mapped_anchor[0], mapped_reference[1] - mapped_anchor[1]
    )
    expected_length = TEST_PRESET.target_scale_frac * TEST_PRESET.output_height
    assert mapped_length == pytest.approx(expected_length, abs=1e-6)


def test_compute_transform_no_rotation_keeps_tilt():
    # A tilted torso, with global_rotation=0, should stay tilted post-transform
    # (only scale/translate applied, direction preserved).
    record = make_record((300.0, 500.0), (350.0, 400.0))
    matrix = compute_transform(record, TEST_PRESET, global_rotation=0.0)
    mapped_anchor = apply_transform(matrix, record.anchor)
    mapped_reference = apply_transform(matrix, record.reference_point)
    assert mapped_reference[0] > mapped_anchor[0]  # still tilted right


def test_compute_transform_global_rotation_verticalizes_reference_frame():
    record = make_record((300.0, 500.0), (350.0, 400.0))
    theta = reference_angle(record)
    matrix = compute_transform(record, TEST_PRESET, global_rotation=theta)
    mapped_anchor = apply_transform(matrix, record.anchor)
    mapped_reference = apply_transform(matrix, record.reference_point)
    assert mapped_reference[0] == pytest.approx(mapped_anchor[0], abs=1e-6)
    assert mapped_reference[1] < mapped_anchor[1]


def test_compute_transform_zero_length_raises():
    record = make_record((100.0, 100.0), (100.0, 100.0))
    with pytest.raises(ValueError):
        compute_transform(record, TEST_PRESET, global_rotation=0.0)


# --- detection.py fallback chain (pure function, no MediaPipe needed) -----------------------


def test_fallback_hips_and_shoulders_reliable():
    landmarks = {
        LEFT_HIP: Landmark(100, 300, visibility=0.9),
        RIGHT_HIP: Landmark(140, 300, visibility=0.9),
        LEFT_SHOULDER: Landmark(105, 150, visibility=0.9),
        RIGHT_SHOULDER: Landmark(135, 150, visibility=0.9),
    }
    record, warnings = alignment_record_from_landmarks(landmarks, 400, 600)
    assert record.anchor == pytest.approx((120.0, 300.0))
    assert record.reference_point == pytest.approx((120.0, 150.0))
    assert not record.needs_manual
    assert record.source == "auto"
    assert warnings == []


def test_fallback_single_hip_and_shoulders():
    landmarks = {
        LEFT_HIP: Landmark(100, 300, visibility=0.9),
        RIGHT_HIP: Landmark(140, 300, visibility=0.1),  # occluded
        LEFT_SHOULDER: Landmark(105, 150, visibility=0.9),
        RIGHT_SHOULDER: Landmark(135, 150, visibility=0.9),
    }
    record, warnings = alignment_record_from_landmarks(landmarks, 400, 600)
    assert record.anchor == pytest.approx((100.0, 300.0))
    assert record.reference_point == pytest.approx((120.0, 150.0))
    assert not record.needs_manual
    assert len(warnings) == 1


def test_fallback_shoulders_and_nose_only():
    landmarks = {
        LEFT_HIP: Landmark(100, 300, visibility=0.1),
        RIGHT_HIP: Landmark(140, 300, visibility=0.1),
        LEFT_SHOULDER: Landmark(105, 150, visibility=0.9),
        RIGHT_SHOULDER: Landmark(135, 150, visibility=0.9),
        NOSE: Landmark(120, 80, visibility=0.9),
    }
    record, warnings = alignment_record_from_landmarks(landmarks, 400, 600)
    assert not record.needs_manual
    assert record.confidence < 0.5
    assert record.anchor == pytest.approx((120.0, 150.0))
    assert record.length > 0
    assert len(warnings) == 1


def test_fallback_nothing_visible_needs_manual():
    landmarks = {
        LEFT_HIP: Landmark(100, 300, visibility=0.1),
        RIGHT_HIP: Landmark(140, 300, visibility=0.1),
        LEFT_SHOULDER: Landmark(105, 150, visibility=0.1),
        RIGHT_SHOULDER: Landmark(135, 150, visibility=0.1),
        NOSE: Landmark(120, 80, visibility=0.1),
    }
    record, warnings = alignment_record_from_landmarks(landmarks, 400, 600)
    assert record.needs_manual
    assert record.confidence == 0.0
    assert record.anchor == (200.0, 300.0)
    assert len(warnings) == 1


def test_fallback_no_landmarks_at_all_needs_manual():
    record, warnings = alignment_record_from_landmarks({}, 400, 600)
    assert record.needs_manual
    assert len(warnings) == 1
