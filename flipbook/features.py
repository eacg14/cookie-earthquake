"""Background registration: aligns a frame to a reference frame's static
scene (trees, golf bag, ground texture, etc.) via ORB keypoint matching +
RANSAC, instead of tracking the moving subject.

The moving golfer does produce keypoints too, but as long as the static
background has reasonable texture, its keypoints vastly outnumber the
subject's across the frame -- RANSAC naturally fits the dominant (background)
motion and treats the subject's keypoints as outliers. This is the same
technique used for video stabilization and panorama stitching.
"""

from __future__ import annotations

import cv2
import numpy as np

ORB_FEATURES = 2000
RATIO_TEST_THRESHOLD = 0.75
MIN_MATCH_COUNT = 12
RANSAC_REPROJ_THRESHOLD = 4.0
MIN_INLIER_RATIO = 0.35


def to_gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def detect_features(image_gray: np.ndarray):
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    return orb.detectAndCompute(image_gray, None)


def _good_matches(frame_desc, ref_desc) -> list:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(frame_desc, ref_desc, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            good.append(m)
    return good


def estimate_transform_to_reference(
    frame_gray: np.ndarray,
    ref_keypoints,
    ref_descriptors,
) -> tuple[np.ndarray | None, int, float]:
    """Estimate the 2x3 affine mapping frame_gray's pixels -> the reference
    frame's pixel space, from matched background keypoints.

    Returns (matrix_or_none, num_good_matches, inlier_ratio). matrix is None
    if there weren't enough good matches or RANSAC couldn't find a
    sufficiently-agreeing (high inlier ratio) transform -- both signal the
    frame needs manual point correspondence instead.
    """
    frame_kp, frame_desc = detect_features(frame_gray)
    if frame_desc is None or ref_descriptors is None or len(frame_kp) < 2:
        return None, 0, 0.0

    good = _good_matches(frame_desc, ref_descriptors)
    if len(good) < MIN_MATCH_COUNT:
        return None, len(good), 0.0

    src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
    )
    if matrix is None:
        return None, len(good), 0.0

    inlier_ratio = float(inliers.sum()) / len(good) if inliers is not None else 0.0
    if inlier_ratio < MIN_INLIER_RATIO:
        return None, len(good), inlier_ratio

    return matrix, len(good), inlier_ratio


def fit_transform_from_point_pairs(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
) -> np.ndarray | None:
    """Solve for the similarity transform (rotation+scale+translation) mapping
    src_points -> dst_points exactly, from manually-clicked correspondences
    (2 point pairs). Used as the manual fallback when auto background
    matching fails for a frame.
    """
    if len(src_points) < 2 or len(dst_points) < 2:
        return None
    src = np.float32(src_points).reshape(-1, 1, 2)
    dst = np.float32(dst_points).reshape(-1, 1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return matrix
