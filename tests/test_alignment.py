import pytest

import numpy as np

from flipbook.alignment import apply_transform, compose_affine, compute_reference_transform
from flipbook.presets import PresetSpec

TEST_PRESET = PresetSpec(
    key="test",
    label="test",
    platform="test",
    aspect_w=1,
    aspect_h=1,
    output_width=1000,
    output_height=1000,
)


# --- compute_reference_transform ----------------------------------------------------------------


def test_compute_reference_transform_maps_center_to_center():
    ref_shape = (800, 600)  # (height, width)
    matrix = compute_reference_transform(ref_shape, TEST_PRESET, scale=1.0)
    mapped = apply_transform(matrix, (300.0, 400.0))  # source's own center (x, y)
    assert mapped[0] == pytest.approx(TEST_PRESET.output_width / 2, abs=1e-6)
    assert mapped[1] == pytest.approx(TEST_PRESET.output_height / 2, abs=1e-6)


def test_compute_reference_transform_no_rotation():
    ref_shape = (800, 600)
    matrix = compute_reference_transform(ref_shape, TEST_PRESET, scale=2.0)
    # A horizontal offset from center should stay purely horizontal (no rotation mixed in).
    center = (300.0, 400.0)
    offset = (350.0, 400.0)
    mapped_center = apply_transform(matrix, center)
    mapped_offset = apply_transform(matrix, offset)
    assert mapped_offset[1] == pytest.approx(mapped_center[1], abs=1e-6)
    assert mapped_offset[0] > mapped_center[0]


def test_compute_reference_transform_scales_distances():
    ref_shape = (800, 600)
    matrix = compute_reference_transform(ref_shape, TEST_PRESET, scale=3.0)
    p1 = apply_transform(matrix, (300.0, 400.0))
    p2 = apply_transform(matrix, (310.0, 400.0))
    assert p2[0] - p1[0] == pytest.approx(30.0, abs=1e-6)


def test_compute_reference_transform_rejects_non_positive_scale():
    with pytest.raises(ValueError):
        compute_reference_transform((800, 600), TEST_PRESET, scale=0.0)
    with pytest.raises(ValueError):
        compute_reference_transform((800, 600), TEST_PRESET, scale=-1.0)


# --- compose_affine (chains a background-match transform with the reference frame's crop) -----


def test_compose_affine_identity():
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    point = (37.0, -12.5)
    result = apply_transform(compose_affine(identity, identity), point)
    assert result[0] == pytest.approx(point[0])
    assert result[1] == pytest.approx(point[1])


def test_compose_affine_translations_add():
    outer = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0]])
    inner = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, -50.0]])
    combined = compose_affine(outer, inner)
    result = apply_transform(combined, (0.0, 0.0))
    assert result[0] == pytest.approx(110.0)
    assert result[1] == pytest.approx(-45.0)


def test_compose_affine_matches_sequential_application():
    outer = np.array([[0.5, -0.2, 30.0], [0.2, 0.5, -10.0]])
    inner = np.array([[1.2, 0.1, -5.0], [-0.1, 1.2, 8.0]])
    combined = compose_affine(outer, inner)

    point = (42.0, 17.0)
    expected = apply_transform(outer, apply_transform(inner, point))
    actual = apply_transform(combined, point)
    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])
