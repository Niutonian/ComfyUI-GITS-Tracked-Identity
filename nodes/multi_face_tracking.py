"""Persistent identity association for multiple detected faces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .appearance import appearance_distance, bipartite_match, extract_appearance, spatial_match_cost
from .tracking_core import LostTracker, Pose, PoseSmoother, fuse_poses, pose_from_landmarks


@dataclass
class TrackedFace:
    track_id: int
    pose: Pose
    alpha: float
    detected: bool


class _TrackState:
    def __init__(
        self,
        track_id: int,
        pose: Pose,
        fps: float,
        min_cutoff: float,
        beta: float,
        hold: int,
        fade: int,
        appearance: np.ndarray | None,
        predict: bool,
    ):
        self.track_id = track_id
        self.smoother = PoseSmoother(fps, min_cutoff, beta)
        self.lost = LostTracker(hold, fade, predict=predict)
        self.pose = pose
        self.appearance = appearance
        self.is_new = True


class MultiFaceIdentityTracker:
    """Appearance-aware bipartite association with stable IDs and per-face filters."""

    def __init__(
        self,
        detector,
        fallback,
        fps: float,
        min_cutoff: float,
        beta: float,
        hold: int,
        fade: int,
        *,
        track_prediction: bool = True,
        fuse_detectors: bool = True,
        small_face_boost: bool = True,
        appearance_weight: float = 0.45,
        match_threshold: float = 1.45,
    ):
        self.detector = detector
        self.fallback = fallback
        self.fps = float(fps)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.hold = int(hold)
        self.fade = int(fade)
        self.track_prediction = bool(track_prediction)
        self.fuse_detectors = bool(fuse_detectors)
        self.small_face_boost = bool(small_face_boost)
        self.appearance_weight = float(appearance_weight)
        self.match_threshold = float(match_threshold)
        self.tracks: dict[int, _TrackState] = {}
        self.next_id = 1
        self.locked_id: int | None = None

    def _fallback_poses(self, frame: np.ndarray, face_scale: float, y_offset: float) -> list[Pose]:
        if self.fallback is None:
            return []
        detect_poses = getattr(self.fallback, "detect_poses", None)
        if detect_poses is not None:
            try:
                return detect_poses(frame, face_scale, y_offset, small_face_boost=self.small_face_boost)
            except TypeError:
                return detect_poses(frame, face_scale, y_offset)
        single = self.fallback.detect_pose(frame, face_scale, y_offset)
        return [] if single is None else [single]

    def _detect(self, frame: np.ndarray, timestamp_ms: int, face_scale: float, y_offset: float) -> list[Pose]:
        height, width = frame.shape[:2]
        detect_all = getattr(self.detector, "detect_all", None)
        if detect_all is not None:
            landmark_sets = detect_all(frame, timestamp_ms)
        else:
            single = self.detector.detect(frame, timestamp_ms)
            landmark_sets = [] if single is None else [single]
        poses = [pose_from_landmarks(points, width, height, face_scale, y_offset) for points in landmark_sets]
        fallback_poses = self._fallback_poses(frame, face_scale, y_offset)
        if self.fuse_detectors and fallback_poses:
            poses = fuse_poses(poses, fallback_poses)
        elif not poses:
            poses = fallback_poses
        return poses

    def _match_cost(self, state: _TrackState, pose: Pose, appearance: np.ndarray) -> float:
        spatial = spatial_match_cost(state.pose, pose)
        visual = appearance_distance(state.appearance, appearance)
        return spatial * (1.0 - self.appearance_weight) + visual * (1.0 + spatial) * self.appearance_weight

    def update(self, frame: np.ndarray, timestamp_ms: int, face_scale: float, y_offset: float) -> list[TrackedFace]:
        detections = self._detect(frame, timestamp_ms, face_scale, y_offset)
        appearances = [extract_appearance(frame, pose) for pose in detections]

        track_ids = list(self.tracks.keys())
        matched_detection: dict[int, int] = {}
        if track_ids and detections:
            cost = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for r, track_id in enumerate(track_ids):
                state = self.tracks[track_id]
                for c, pose in enumerate(detections):
                    cost[r, c] = self._match_cost(state, pose, appearances[c])
            for r, c in bipartite_match(cost, self.match_threshold):
                matched_detection[track_ids[r]] = c

        used_detections = set(matched_detection.values())
        for detection_index, pose in enumerate(detections):
            if detection_index in used_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = _TrackState(
                track_id,
                pose,
                self.fps,
                self.min_cutoff,
                self.beta,
                self.hold,
                self.fade,
                appearances[detection_index],
                self.track_prediction,
            )
            matched_detection[track_id] = detection_index

        active: list[TrackedFace] = []
        expired: list[int] = []
        for track_id, state in self.tracks.items():
            if track_id in matched_detection:
                idx = matched_detection[track_id]
                if state.is_new:
                    smoothed = state.smoother.apply(detections[idx])
                    pose, alpha, detected = state.lost.update(smoothed)
                    state.is_new = False
                else:
                    smoothed = state.smoother.apply(detections[idx])
                    pose, alpha, detected = state.lost.update(smoothed)
                new_app = appearances[idx]
                if state.appearance is None:
                    state.appearance = new_app
                else:
                    blended = 0.8 * state.appearance + 0.2 * new_app
                    norm = float(np.linalg.norm(blended))
                    state.appearance = blended / norm if norm > 1e-8 else new_app
            else:
                pose, alpha, detected = state.lost.update(None)
            if pose is None or alpha <= 0.0:
                expired.append(track_id)
                continue
            state.pose = pose
            active.append(TrackedFace(track_id, pose, float(alpha), bool(detected)))
        for track_id in expired:
            del self.tracks[track_id]
            if self.locked_id == track_id:
                self.locked_id = None
        return active

    def select(self, faces: list[TrackedFace], mode: str) -> list[TrackedFace]:
        if not faces:
            return []
        if mode == "all_faces":
            return faces
        if mode == "largest_face":
            return [max(faces, key=lambda face: face.pose.size)]
        if self.locked_id is not None:
            locked = next((face for face in faces if face.track_id == self.locked_id), None)
            if locked is not None:
                return [locked]
        selected = max(faces, key=lambda face: face.pose.size)
        self.locked_id = selected.track_id
        return [selected]

    def close(self) -> None:
        for item in (self.detector, self.fallback):
            close = getattr(item, "close", None)
            if close is not None:
                close()
        self.tracks.clear()
        self.locked_id = None
