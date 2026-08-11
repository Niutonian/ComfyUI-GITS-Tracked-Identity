import math

import numpy as np

from nodes.tracking_core import Pose, alpha_composite_numpy, face_region_mask, place_overlay_in_frame, pose_from_landmarks


def landmarks(tilt=0.0):
    points = [[0.5, 0.5] for _ in range(468)]
    points[33] = [0.4, 0.45 - tilt]
    points[263] = [0.6, 0.45 + tilt]
    points[234], points[454] = [0.3, 0.5], [0.7, 0.5]
    points[10], points[152] = [0.5, 0.3], [0.5, 0.7]
    return points


def test_centered_face_and_scale_offset():
    base = pose_from_landmarks(landmarks(), 200, 100, y_offset=0.0)
    changed = pose_from_landmarks(landmarks(), 200, 100, face_scale=1.5, y_offset=0.1)
    assert base.x == 100
    assert base.roll == 0
    assert changed.size == base.size * 1.5
    assert changed.y == base.y + changed.size * 0.1


def test_tilt_roll_sign_and_magnitude():
    pose = pose_from_landmarks(landmarks(0.05), 200, 100)
    assert pose.roll > 0
    assert pose.roll == pytest.approx(math.atan2(10, 40))


def test_offscreen_overlay_clips():
    layer = np.full((10, 10, 4), 255, np.uint8)
    result = place_overlay_in_frame(layer, 8, 8, 0, 0)
    assert result.shape == (8, 8, 4)
    assert np.count_nonzero(result[..., 3]) == 25


def test_alpha_composite_rgb():
    background = np.zeros((1, 1, 3), np.uint8)
    overlay = np.array([[[200, 100, 50, 128]]], np.uint8)
    result = alpha_composite_numpy(background, overlay)
    assert result[0, 0].tolist() == [100, 50, 25]


def test_face_region_mask_is_full_and_clips():
    mask = face_region_mask(32, 40, Pose(2, 16, 20, 0), scale=1.0, feather_px=0)
    assert mask.shape == (32, 40)
    assert mask.max() == 1.0
    assert mask.sum() > 100


def test_profile_mask_covers_more_than_frontal():
    """Side shots must expand the LaMa region so half-faces are not left behind."""
    frontal = face_region_mask(64, 96, Pose(48, 32, 28, 0, 0.0), scale=1.15, feather_px=0)
    profile = face_region_mask(64, 96, Pose(48, 32, 28, 0, 1.1), scale=1.15, feather_px=0)
    assert profile.sum() > frontal.sum() * 1.08


def test_profile_landmarks_inflate_size():
    """Collapsed temple width (profile) should not under-size the face diameter."""
    frontal = landmarks()
    profile = landmarks()
    # Pull temples/eyes toward center like a yawed face silhouette.
    profile[234], profile[454] = [0.42, 0.5], [0.55, 0.5]
    profile[33], profile[263] = [0.44, 0.45], [0.52, 0.45]
    profile[1] = [0.58, 0.5]  # nose shifted right
    frontal_pose = pose_from_landmarks(frontal, 200, 200, face_scale=1.0, y_offset=0.0)
    profile_pose = pose_from_landmarks(profile, 200, 200, face_scale=1.0, y_offset=0.0)
    assert abs(profile_pose.yaw) > abs(frontal_pose.yaw)
    assert profile_pose.size >= frontal_pose.size * 0.85


def test_merge_pose_prefers_larger_box_on_profile():
    from nodes.tracking_core import merge_pose_pair

    landmark = Pose(40, 40, 20, 0.0, 0.9)
    yunet = Pose(42, 41, 34, 0.0, 1.0)
    merged = merge_pose_pair(landmark, yunet)
    assert merged.size >= 30
    assert abs(merged.yaw) >= 0.9


import pytest
