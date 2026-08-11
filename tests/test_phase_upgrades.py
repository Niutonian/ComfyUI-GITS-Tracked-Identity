"""Tests for v0.2 tracking, effects, appearance matching, and temporal blend."""

import numpy as np
import torch

from nodes.appearance import appearance_distance, bipartite_match, extract_appearance, spatial_match_cost
from nodes.effects import apply_glitch_rgba, apply_signal_flicker, yaw_scale_x
from nodes.multi_face_tracking import MultiFaceIdentityTracker
from nodes.presets import PRESET_NAMES, apply_preset
from nodes.temporal_lama import TemporalLamaProcessor
from nodes.tracking_core import LostTracker, Pose, fuse_poses


def test_presets_named_and_overlay():
    assert "classic" in PRESET_NAMES
    merged = apply_preset("aggressive_track", {"glitch_intensity": 0.0, "hold_frames": 1})
    assert merged["hold_frames"] == 18
    assert merged["glitch_intensity"] > 0.0
    custom = apply_preset("custom", {"hold_frames": 3})
    assert custom["hold_frames"] == 3


def test_signal_flicker_only_when_weak():
    assert apply_signal_flicker(1.0, 10, 1.0) == 1.0
    weak = apply_signal_flicker(0.5, 10, 0.8)
    assert 0.0 <= weak <= 0.5


def test_glitch_preserves_shape():
    rgba = np.full((32, 32, 4), 200, dtype=np.uint8)
    out = apply_glitch_rgba(rgba, 0.6, frame_index=4, track_id=1)
    assert out.shape == rgba.shape
    assert out.dtype == np.uint8


def test_yaw_scale_contracts():
    assert yaw_scale_x(0.0, 1.0) == 1.0
    assert yaw_scale_x(1.0, 1.0) < 1.0


def test_lost_tracker_predicts_motion():
    tracker = LostTracker(hold_frames=3, fade_frames=0, predict=True)
    a = Pose(10, 20, 30, 0, 0)
    b = Pose(14, 24, 30, 0, 0)
    tracker.update(a)
    tracker.update(b)
    predicted, alpha, detected = tracker.update(None)
    assert detected is False
    assert alpha == 1.0
    assert predicted is not None
    assert predicted.x > b.x
    assert predicted.y > b.y


def test_appearance_and_bipartite_match_cross_swap():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[10:40, 10:40] = (200, 40, 40)
    frame[10:40, 70:100] = (40, 40, 200)
    left = Pose(25, 25, 24, 0)
    right = Pose(85, 25, 24, 0)
    left_app = extract_appearance(frame, left)
    right_app = extract_appearance(frame, right)
    assert appearance_distance(left_app, left_app) < 0.05
    assert appearance_distance(left_app, right_app) > appearance_distance(left_app, left_app)

    # Crossing: previous left matches current left by appearance even if positions swap labels.
    cost = np.array(
        [
            [
                spatial_match_cost(left, right) * 0.55 + appearance_distance(left_app, right_app) * 0.45,
                spatial_match_cost(left, left) * 0.55 + appearance_distance(left_app, left_app) * 0.45,
            ],
            [
                spatial_match_cost(right, right) * 0.55 + appearance_distance(right_app, right_app) * 0.45,
                spatial_match_cost(right, left) * 0.55 + appearance_distance(right_app, left_app) * 0.45,
            ],
        ],
        dtype=np.float32,
    )
    matches = bipartite_match(cost, max_cost=2.0)
    assert (0, 1) in matches
    assert (1, 0) in matches


def test_fuse_poses_adds_non_overlapping():
    primary = [Pose(20, 20, 20, 0)]
    secondary = [Pose(20, 20, 18, 0), Pose(70, 20, 18, 0)]
    fused = fuse_poses(primary, secondary, iou_threshold=0.35)
    assert len(fused) == 2


class EmptyDetector:
    def detect_all(self, _frame, _timestamp):
        return []

    def close(self):
        pass


class PoseSequenceFallback:
    def __init__(self, frames):
        self.frames = list(frames)

    def detect_poses(self, _frame, _scale, _offset, small_face_boost=False):
        return self.frames.pop(0)

    def close(self):
        pass


def test_multi_face_survives_one_frame_gap_with_prediction():
    fallback = PoseSequenceFallback([
        [Pose(20, 30, 24, 0)],
        [],
        [Pose(22, 31, 24, 0)],
    ])
    tracker = MultiFaceIdentityTracker(
        EmptyDetector(), fallback, 30, 1.2, 0.035, 4, 2, track_prediction=True, fuse_detectors=False
    )
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    first = tracker.update(frame, 0, 1, 0)
    gap = tracker.update(frame, 33, 1, 0)
    back = tracker.update(frame, 66, 1, 0)
    assert first[0].track_id == gap[0].track_id == back[0].track_id
    assert gap[0].detected is False
    assert back[0].detected is True


def test_temporal_blend_crossfades_refresh():
    calls = []

    def remove_callback(images, masks, strength, edge):
        calls.append(True)
        # Alternating fill values so the temporal blend is observable.
        value = 0.9 if len(calls) == 1 else 0.1
        return torch.full_like(images, value)

    processor = TemporalLamaProcessor(remove_callback, temporal_blend=0.5)
    images = torch.zeros((2, 32, 32, 3))
    mask = np.ones((32, 32), dtype=np.float32)
    faces = [[(1, mask)], [(1, mask)]]
    out = processor.fill(
        images, faces, resolution=16, strength=200, edge_smoothness=4,
        start_frame=0, interval=1, temporal_flow=False, temporal_blend=0.5,
    )
    assert out.shape == images.shape
    assert len(calls) == 2
    # Second refresh is blended toward the prior 0.9 instead of pure 0.1.
    assert float(out[1, 16, 16, 0]) > 0.1
    assert float(out[1, 16, 16, 0]) < 0.9
