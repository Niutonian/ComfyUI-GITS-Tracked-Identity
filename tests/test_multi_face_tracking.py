import numpy as np

from nodes.multi_face_tracking import MultiFaceIdentityTracker
from nodes.tracking_core import Pose


class EmptyDetector:
    def detect_all(self, _frame, _timestamp):
        return []

    def close(self):
        pass


class PoseSequenceFallback:
    def __init__(self, frames):
        self.frames = list(frames)

    def detect_poses(self, _frame, _scale, _offset):
        return self.frames.pop(0)

    def close(self):
        pass


def test_locked_face_keeps_identity_when_other_face_becomes_larger():
    fallback = PoseSequenceFallback([
        [Pose(20, 30, 24, 0), Pose(70, 30, 16, 0)],
        [Pose(21, 30, 18, 0), Pose(69, 30, 30, 0)],
    ])
    tracker = MultiFaceIdentityTracker(EmptyDetector(), fallback, 30, 1.2, 0.035, 2, 2)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    first = tracker.select(tracker.update(frame, 0, 1, 0), "locked_face")
    second_active = tracker.update(frame, 33, 1, 0)
    second_locked = tracker.select(second_active, "locked_face")
    second_largest = tracker.select(second_active, "largest_face")

    assert first[0].track_id == second_locked[0].track_id
    assert second_largest[0].track_id != second_locked[0].track_id


def test_all_faces_returns_every_stable_track():
    fallback = PoseSequenceFallback([[Pose(20, 30, 20, 0), Pose(70, 30, 20, 0)]])
    tracker = MultiFaceIdentityTracker(EmptyDetector(), fallback, 30, 1.2, 0.035, 2, 2)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    selected = tracker.select(tracker.update(frame, 0, 1, 0), "all_faces")

    assert len(selected) == 2
    assert len({face.track_id for face in selected}) == 2
