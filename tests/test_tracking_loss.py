import pytest

from nodes.tracking_core import LostTracker, Pose


def test_hold_and_fade_exact_counts():
    tracker = LostTracker(hold_frames=2, fade_frames=3)
    pose = Pose(1, 2, 3, 0)
    assert tracker.update(pose) == (pose, 1.0, True)
    results = [tracker.update(None) for _ in range(6)]
    assert [r[1] for r in results[:5]] == pytest.approx([1, 1, 2 / 3, 1 / 3, 0])
    assert results[0][0] == pose
    assert results[4][0] == pose
    assert results[5] == (None, 0.0, False)
