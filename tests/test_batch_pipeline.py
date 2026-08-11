import json

import torch

import nodes.track_and_guide as track_module
from nodes.track_and_guide import GITSTrackAndGuide
from nodes.simple_face_replacement import GITSSimpleFaceReplacement
from nodes.webcam_face_replacement import GITSWebcamFaceReplacement
from nodes.tracking_core import Pose


def synthetic_landmarks():
    points = [[0.5, 0.5] for _ in range(468)]
    points[33], points[263] = [0.4, 0.45], [0.6, 0.45]
    points[234], points[454] = [0.3, 0.5], [0.7, 0.5]
    points[10], points[152] = [0.5, 0.3], [0.5, 0.7]
    return points


class MockTracker:
    def __init__(self, _path, _confidence):
        self.timestamps = []

    def detect(self, _frame, timestamp):
        self.timestamps.append(timestamp)
        return synthetic_landmarks()

    def close(self):
        pass


class MockFallback:
    def __init__(self, _path, _confidence):
        pass

    def detect_pose(self, _frame, _face_scale, _y_offset):
        return None

    def close(self):
        pass


class StreamingTracker(MockTracker):
    instances = []

    def __init__(self, *args):
        super().__init__(*args)
        self.closed = False
        type(self).instances.append(self)

    def close(self):
        self.closed = True


class EmptyStreamingTracker:
    def __init__(self, *_args):
        pass

    def detect_all(self, _frame, _timestamp):
        return []

    def close(self):
        pass


class TwoFaceFallback(MockFallback):
    def detect_poses(self, _frame, _scale, _offset):
        return [Pose(18, 32, 18, 0), Pose(46, 32, 18, 0)]


def test_mocked_batch_contract(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", MockTracker)
    images = torch.zeros((3, 32, 48, 3))
    logo = torch.ones((1, 12, 12, 3))
    mask = torch.ones((1, 12, 12))
    outputs = GITSTrackAndGuide().track_and_guide(
        images, static_logo=logo, static_logo_mask=mask,
        ring_logo=logo, ring_logo_mask=mask, fallback_detector="disabled",
        face_mode="single_face", face_removal_mode="disabled",
    )
    final, removed, guide, overlay, out_mask, preview, metadata, face_mask = outputs
    assert guide.shape == images.shape == overlay.shape == preview.shape
    assert removed.shape == images.shape == final.shape
    assert out_mask.shape == (3, 32, 48)
    assert face_mask.shape == (3, 32, 48)
    for tensor in (*outputs[:6], outputs[7]):
        assert tensor.min() >= 0 and tensor.max() <= 1
    assert len(json.loads(metadata)["frames"]) == 3
    records = json.loads(metadata)["frames"]
    assert [r["ring_angle"] for r in records] == [0.0, 1.75, 3.5]


class MockLamaRemover:
    calls = []

    def remove_object(self, images, masks, removal_strength, edge_smoothness):
        type(self).calls.append((masks.clone(), removal_strength, edge_smoothness))
        return (torch.full_like(images, 0.25),)


def test_integrated_lama_and_final_composite(monkeypatch, tmp_path):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", MockTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "lama_remover_class", MockLamaRemover)
    local_lama = tmp_path / "big-lama.pt"
    local_lama.write_bytes(b"test")
    monkeypatch.setattr(track_module, "_lama_model_path", lambda: local_lama)
    MockLamaRemover.calls.clear()
    images = torch.zeros((2, 32, 48, 3))
    logo = torch.ones((1, 12, 12, 3))
    mask = torch.ones((1, 12, 12))

    node = GITSTrackAndGuide()
    outputs = node.track_and_guide(
        images, static_logo=logo, static_logo_mask=mask,
        ring_logo=logo, ring_logo_mask=mask,
        fallback_detector="disabled",
        face_removal_mode="lama",
        lama_removal_strength=222,
        lama_edge_smoothness=5,
        face_mode="single_face",
    )

    final, removed, overlay, overlay_mask = outputs[0], outputs[1], outputs[3], outputs[4]
    assert len(MockLamaRemover.calls) == 1
    assert MockLamaRemover.calls[0][0].shape == (2, 32, 48)
    assert MockLamaRemover.calls[0][1:] == (222, 5)
    assert node._lama_remover.model_path == str(local_lama)
    assert torch.allclose(removed, torch.full_like(images, 0.25))
    alpha = overlay_mask[..., None]
    assert torch.allclose(final, overlay * alpha + removed * (1.0 - alpha))


def test_unconnected_artwork_inputs_do_not_block(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", MockTracker)
    images = torch.rand((2, 32, 48, 3))
    outputs = GITSTrackAndGuide().track_and_guide(
        images, fallback_detector="disabled", face_removal_mode="disabled"
    )
    assert torch.allclose(outputs[0], images)
    assert torch.allclose(outputs[1], images)
    assert torch.count_nonzero(outputs[4]) == 0


def test_no_logos_still_runs_lama_advanced(monkeypatch):
    """Global rule: unconnected logo sockets must not skip face removal."""
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", MockTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "lama_remover_class", MockLamaRemover)
    MockLamaRemover.calls.clear()
    images = torch.zeros((2, 32, 48, 3))

    final, removed, *_rest = GITSTrackAndGuide().track_and_guide(
        images,
        fallback_detector="disabled",
        face_removal_mode="lama",
        temporal_lama=True,
        lama_resolution="256",
        face_mode="locked_face",
        # No static_logo / ring_logo / masks connected.
    )

    assert final.shape == removed.shape == images.shape
    assert len(MockLamaRemover.calls) >= 1
    # With no artwork, final equals the LaMa-processed background.
    assert torch.allclose(final, removed)
    # Face occlusion mask is still produced from tracking alone.
    face_occlusion = _rest[-1]
    assert torch.count_nonzero(face_occlusion) > 0
    # Cropped Temporal LaMa writes the mock fill into the face ROI (not the whole frame).
    assert float(removed.max()) > 0.0


def test_no_logos_still_runs_lama_simple(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", MockTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", MockFallback)
    monkeypatch.setattr(GITSTrackAndGuide, "lama_remover_class", MockLamaRemover)
    MockLamaRemover.calls.clear()
    images = torch.zeros((2, 32, 48, 3))

    final, removed = GITSSimpleFaceReplacement().replace_face(
        images, face_removal="lama", temporal_lama=True, lama_resolution="256",
        effect_preset="custom",
    )

    assert final.shape == removed.shape == images.shape
    assert len(MockLamaRemover.calls) >= 1
    assert torch.allclose(final, removed)
    assert float(removed.max()) > 0.0


def test_no_logos_still_runs_lama_webcam(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", StreamingTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", MockFallback)
    monkeypatch.setattr(GITSTrackAndGuide, "lama_remover_class", MockLamaRemover)
    MockLamaRemover.calls.clear()
    StreamingTracker.instances.clear()
    node = GITSWebcamFaceReplacement()
    image = torch.rand((1, 64, 64, 3))

    final, removed = node.process_webcam(
        image,
        performance_mode="quality",
        lama_resolution="256",
        effect_preset="custom",
        # No logo inputs.
    )

    assert final.shape == removed.shape == image.shape
    assert len(MockLamaRemover.calls) >= 1


def test_simple_node_accepts_missing_ring(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", MockTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", MockFallback)
    images = torch.zeros((2, 32, 48, 3))
    logo = torch.ones((1, 12, 12, 3))
    comfy_transparency_mask = torch.zeros((1, 12, 12))
    final, removed = GITSSimpleFaceReplacement().replace_face(
        images,
        static_logo=logo,
        static_logo_mask=comfy_transparency_mask,
        face_removal="disabled",
    )
    assert final.shape == removed.shape == images.shape
    assert torch.count_nonzero(final) > 0
    assert torch.allclose(removed, images)


def test_webcam_node_keeps_tracker_and_timeline(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", StreamingTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", MockFallback)
    StreamingTracker.instances.clear()
    node = GITSWebcamFaceReplacement()
    image = torch.rand((1, 64, 64, 3))

    first = node.process_webcam(image, performance_mode="fast", fps=30.0)
    second = node.process_webcam(image, performance_mode="fast", fps=30.0)

    assert first[0].shape == second[0].shape == image.shape
    assert len(StreamingTracker.instances) == 1
    assert StreamingTracker.instances[0].timestamps == [0, 33]
    assert node.frame_index == 2

    node.process_webcam(image, performance_mode="fast", fps=30.0, reset_tracking=True)
    assert len(StreamingTracker.instances) == 2
    assert StreamingTracker.instances[0].closed
    assert StreamingTracker.instances[1].timestamps == [0]


def test_webcam_balanced_mode_reuses_roi_lama_patch(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", StreamingTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", MockFallback)
    monkeypatch.setattr(GITSTrackAndGuide, "lama_remover_class", MockLamaRemover)
    MockLamaRemover.calls.clear()
    node = GITSWebcamFaceReplacement()
    image = torch.rand((1, 64, 64, 3))

    flow_calls = []
    original_warp = node._warp_patch
    monkeypatch.setattr(node, "_warp_patch", lambda *args: (flow_calls.append(True), original_warp(*args))[1])
    node.process_webcam(
        image, performance_mode="balanced", lama_resolution="256", lama_every_n_frames=2,
        effect_preset="custom",
    )
    node.process_webcam(
        image, performance_mode="balanced", lama_resolution="256", lama_every_n_frames=2,
        effect_preset="custom",
    )

    assert len(MockLamaRemover.calls) == 1
    lama_mask = MockLamaRemover.calls[0][0]
    assert lama_mask.shape == (1, 256, 256)
    assert flow_calls == [True]


def test_webcam_balanced_interval_applies_per_frame_in_large_batch(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", StreamingTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", MockFallback)
    monkeypatch.setattr(GITSTrackAndGuide, "lama_remover_class", MockLamaRemover)
    MockLamaRemover.calls.clear()
    node = GITSWebcamFaceReplacement()
    images = torch.rand((5, 64, 64, 3))

    node.process_webcam(
        images, performance_mode="balanced", lama_resolution="256", lama_every_n_frames=2,
        effect_preset="custom",
    )

    assert len(MockLamaRemover.calls) == 3


def test_webcam_all_faces_builds_two_independent_masks(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", EmptyStreamingTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", TwoFaceFallback)
    node = GITSWebcamFaceReplacement()
    image = torch.zeros((1, 64, 64, 3))

    _overlay, _art_mask, union_mask, face_entries = node._track_faces(
        image, None, None, None, None, 30, 0, 1.0, 0.0, 0.0, 1.0, "all_faces", 4,
        0.0, 0.0, 0.0, 0.0,
    )

    assert len(face_entries[0]) == 2
    assert len({track_id for track_id, _mask in face_entries[0]}) == 2
    assert union_mask[0, 32, 18] > 0.9
    assert union_mask[0, 32, 46] > 0.9


def test_advanced_node_all_faces_outputs_combined_mask_and_metadata(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", EmptyStreamingTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", TwoFaceFallback)
    image = torch.zeros((1, 64, 64, 3))

    outputs = GITSTrackAndGuide().track_and_guide(
        image, face_mode="all_faces", max_faces=4, face_removal_mode="disabled"
    )

    metadata = json.loads(outputs[6])
    assert metadata["frames"][0]["face_count"] == 2
    assert len(metadata["frames"][0]["faces"]) == 2
    assert outputs[7][0, 32, 18] > 0.9
    assert outputs[7][0, 32, 46] > 0.9


def test_simple_node_all_faces_uses_per_identity_temporal_lama(monkeypatch):
    monkeypatch.setattr(GITSTrackAndGuide, "tracker_factory", EmptyStreamingTracker)
    monkeypatch.setattr(GITSTrackAndGuide, "fallback_factory", TwoFaceFallback)
    monkeypatch.setattr(GITSTrackAndGuide, "lama_remover_class", MockLamaRemover)
    MockLamaRemover.calls.clear()
    image = torch.zeros((1, 64, 64, 3))

    final, removed = GITSSimpleFaceReplacement().replace_face(
        image, face_mode="all_faces", max_faces=4, face_removal="lama",
        temporal_lama=True, lama_resolution="256",
    )

    assert final.shape == removed.shape == image.shape
    assert len(MockLamaRemover.calls) == 2
