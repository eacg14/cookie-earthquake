import cv2
import numpy as np
import pytest

from flipbook import features
from flipbook.alignment import apply_transform


def _synthetic_textured_image(width=500, height=400, seed=42) -> np.ndarray:
    """A grayscale image with plenty of ORB-detectable corners: random
    circles and rectangles, standing in for a textured real-world background
    (trees, grass, equipment) without needing a real photo pair.
    """
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 40, dtype=np.uint8)
    for _ in range(120):
        x, y = rng.integers(0, width), rng.integers(0, height)
        radius = int(rng.integers(3, 12))
        color = int(rng.integers(80, 255))
        cv2.circle(image, (x, y), radius, color, -1)
    for _ in range(40):
        x, y = rng.integers(0, width - 20), rng.integers(0, height - 20)
        w, h = int(rng.integers(5, 20)), int(rng.integers(5, 20))
        color = int(rng.integers(80, 255))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    return image


def test_estimate_transform_to_reference_recovers_known_camera_shake():
    """Simulates handheld camera shake between two shots of the same static
    scene: warp a synthetic textured image by a small known rotation/scale/
    translation and confirm background matching recovers that motion.
    """
    ref = _synthetic_textured_image()

    angle_deg, scale, tx, ty = 4.0, 0.98, 12.0, -7.0
    center = (ref.shape[1] / 2, ref.shape[0] / 2)
    applied = cv2.getRotationMatrix2D(center, angle_deg, scale)
    applied[0, 2] += tx
    applied[1, 2] += ty

    frame = cv2.warpAffine(ref, applied, (ref.shape[1], ref.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    ref_kp, ref_desc = features.detect_features(ref)
    matrix, num_matches, inlier_ratio = features.estimate_transform_to_reference(frame, ref_kp, ref_desc)

    assert matrix is not None
    assert num_matches >= features.MIN_MATCH_COUNT
    assert inlier_ratio > 0.5

    expected = cv2.invertAffineTransform(applied)
    assert matrix == pytest.approx(expected, abs=0.5)


def test_estimate_transform_to_reference_fails_on_featureless_image():
    ref = _synthetic_textured_image()
    ref_kp, ref_desc = features.detect_features(ref)

    blank_frame = np.full_like(ref, 128)
    matrix, num_matches, inlier_ratio = features.estimate_transform_to_reference(
        blank_frame, ref_kp, ref_desc
    )
    assert matrix is None


def test_fit_transform_from_point_pairs_recovers_translation():
    src = [(10.0, 10.0), (50.0, 20.0)]
    dst = [(15.0, 25.0), (55.0, 35.0)]  # pure translation by (5, 15)
    matrix = features.fit_transform_from_point_pairs(src, dst)
    assert matrix is not None

    mapped = apply_transform(matrix, (10.0, 10.0))
    assert mapped[0] == pytest.approx(15.0, abs=1e-3)
    assert mapped[1] == pytest.approx(25.0, abs=1e-3)


def test_fit_transform_from_point_pairs_needs_two_points():
    assert features.fit_transform_from_point_pairs([(0.0, 0.0)], [(1.0, 1.0)]) is None
