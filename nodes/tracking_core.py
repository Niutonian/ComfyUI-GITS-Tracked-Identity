"""Framework-independent temporal tracking and overlay utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    size: float
    roll: float
    yaw: float = 0.0


class LowPassFilter:
    def __init__(self) -> None:
        self.value: float | None = None

    def apply(self, value: float, alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.value = value if self.value is None else alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    def __init__(self, frequency: float, min_cutoff: float = 1.2, beta: float = 0.035, d_cutoff: float = 1.0) -> None:
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.signal = LowPassFilter()
        self.derivative = LowPassFilter()
        self.previous: float | None = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        dt = 1.0 / self.frequency
        return 1.0 / (1.0 + tau / dt)

    def apply(self, value: float) -> float:
        derivative = 0.0 if self.previous is None else (value - self.previous) * self.frequency
        self.previous = value
        filtered_derivative = self.derivative.apply(derivative, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        return self.signal.apply(value, self._alpha(cutoff))


class PoseSmoother:
    def __init__(self, fps: float, min_cutoff: float, beta: float) -> None:
        self.filters = [OneEuroFilter(fps, min_cutoff, beta) for _ in range(5)]

    def apply(self, pose: Pose) -> Pose:
        values = [pose.x, pose.y, pose.size, pose.roll, pose.yaw]
        return Pose(*(f.apply(v) for f, v in zip(self.filters, values)))


class LostTracker:
    """Hold the last pose, optionally predict motion, then fade out."""

    def __init__(self, hold_frames: int, fade_frames: int, predict: bool = True) -> None:
        self.hold_frames = max(0, int(hold_frames))
        self.fade_frames = max(0, int(fade_frames))
        self.predict = bool(predict)
        self.last_pose: Pose | None = None
        self.velocity = (0.0, 0.0, 0.0, 0.0, 0.0)
        self.missing = 0

    def _blend_velocity(self, previous: Pose, current: Pose) -> None:
        raw = (
            current.x - previous.x,
            current.y - previous.y,
            current.size - previous.size,
            current.roll - previous.roll,
            current.yaw - previous.yaw,
        )
        damp = 0.65
        self.velocity = tuple(damp * r + (1.0 - damp) * v for r, v in zip(raw, self.velocity))

    def _predict(self, steps: int) -> Pose | None:
        if self.last_pose is None:
            return None
        if not self.predict or steps <= 0:
            return self.last_pose
        # Decay prediction so long holds do not drift off-screen aggressively.
        scale = sum((0.85 ** i) for i in range(steps))
        vx, vy, vs, vr, vyaw = self.velocity
        return Pose(
            self.last_pose.x + vx * scale,
            self.last_pose.y + vy * scale,
            max(1.0, self.last_pose.size + vs * scale),
            self.last_pose.roll + vr * scale,
            self.last_pose.yaw + vyaw * scale,
        )

    def update(self, pose: Pose | None) -> tuple[Pose | None, float, bool]:
        if pose is not None:
            if self.last_pose is not None:
                self._blend_velocity(self.last_pose, pose)
            else:
                self.velocity = (0.0, 0.0, 0.0, 0.0, 0.0)
            self.last_pose, self.missing = pose, 0
            return pose, 1.0, True
        self.missing += 1
        if self.last_pose is None:
            return None, 0.0, False
        if self.missing <= self.hold_frames:
            return self._predict(self.missing), 1.0, False
        fade_index = self.missing - self.hold_frames
        if self.fade_frames and fade_index <= self.fade_frames:
            return self._predict(self.missing), max(0.0, 1.0 - fade_index / self.fade_frames), False
        return None, 0.0, False


def _xy(landmarks: Sequence[Any], index: int) -> tuple[float, float]:
    point = landmarks[index]
    if hasattr(point, "x"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def pose_from_landmarks(
    landmarks: Sequence[Any], width: int, height: int, face_scale: float = 1.0, y_offset: float = -0.02
) -> Pose:
    left_eye, right_eye = _xy(landmarks, 33), _xy(landmarks, 263)
    left_temple, right_temple = _xy(landmarks, 234), _xy(landmarks, 454)
    forehead, chin = _xy(landmarks, 10), _xy(landmarks, 152)
    nose = _xy(landmarks, 1)
    eye_mid = ((left_eye[0] + right_eye[0]) * width / 2, (left_eye[1] + right_eye[1]) * height / 2)
    face_mid = ((forehead[0] + chin[0]) * width / 2, (forehead[1] + chin[1]) * height / 2)
    center_x = (eye_mid[0] + face_mid[0]) / 2
    center_y = (eye_mid[1] + face_mid[1]) / 2
    temple_width = math.hypot((right_temple[0] - left_temple[0]) * width, (right_temple[1] - left_temple[1]) * height)
    face_height = math.hypot((chin[0] - forehead[0]) * width, (chin[1] - forehead[1]) * height)
    size = max(temple_width, face_height) * float(face_scale)
    center_y += float(y_offset) * size
    roll = math.atan2((right_eye[1] - left_eye[1]) * height, (right_eye[0] - left_eye[0]) * width)
    # Approximate yaw from nose offset vs. face center, normalized by inter-temple width.
    nose_x = nose[0] * width
    denom = max(temple_width, 1.0)
    yaw = float(np.clip((nose_x - center_x) / (0.5 * denom), -1.5, 1.5))
    return Pose(center_x, center_y, size, roll, yaw)


def combine_rgba_layers(static_rgba: np.ndarray, ring_rgba: np.ndarray) -> np.ndarray:
    if static_rgba.shape != ring_rgba.shape or static_rgba.ndim != 3 or static_rgba.shape[2] != 4:
        raise ValueError("RGBA layers must have identical [H,W,4] shapes")
    sa = static_rgba[..., 3:4].astype(np.float32) / 255.0
    ra = ring_rgba[..., 3:4].astype(np.float32) / 255.0
    out_a = sa + ra * (1.0 - sa)
    premul = static_rgba[..., :3] * sa + ring_rgba[..., :3] * ra * (1.0 - sa)
    rgb = np.divide(premul, out_a, out=np.zeros_like(premul), where=out_a > 1e-6)
    return np.concatenate([rgb, out_a * 255.0], axis=2).clip(0, 255).astype(np.uint8)


def transform_rgba(
    rgba: np.ndarray,
    size: float,
    angle_degrees: float,
    scale_x: float = 1.0,
) -> np.ndarray:
    import cv2

    side = max(1, int(round(size)))
    resized = cv2.resize(
        rgba,
        (side, side),
        interpolation=cv2.INTER_AREA if side < rgba.shape[0] else cv2.INTER_LINEAR,
    )
    scale_x = float(np.clip(scale_x, 0.4, 1.0))
    if scale_x < 0.999:
        new_w = max(1, int(round(side * scale_x)))
        squeezed = cv2.resize(resized, (new_w, side), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((side, side, 4), dtype=np.uint8)
        x0 = (side - new_w) // 2
        canvas[:, x0 : x0 + new_w] = squeezed
        resized = canvas
    center = ((side - 1) / 2.0, (side - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(
        resized, matrix, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )


def place_overlay_in_frame(
    layer: np.ndarray, frame_height: int, frame_width: int, center_x: float, center_y: float
) -> np.ndarray:
    output = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
    h, w = layer.shape[:2]
    x0, y0 = int(round(center_x - w / 2)), int(round(center_y - h / 2))
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(frame_width, x0 + w), min(frame_height, y0 + h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return output
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    output[dst_y0:dst_y1, dst_x0:dst_x1] = layer[src_y0 : src_y0 + dst_y1 - dst_y0, src_x0 : src_x0 + dst_x1 - dst_x0]
    return output


def alpha_composite_numpy(background_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    alpha = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    return (
        overlay_rgba[..., :3].astype(np.float32) * alpha + background_rgb.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)


def face_region_mask(
    frame_height: int,
    frame_width: int,
    pose: Pose,
    scale: float = 1.15,
    feather_px: int = 8,
) -> np.ndarray:
    """Create a full-face elliptical mask for censoring or diffusion inpainting."""
    import cv2

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    # Mild horizontal squeeze when yaw is large so the removal region follows the head turn.
    yaw_squeeze = float(np.clip(1.0 - abs(pose.yaw) * 0.18, 0.7, 1.0))
    radius_x = max(1, int(round(pose.size * float(scale) * 0.5 * yaw_squeeze)))
    radius_y = max(1, int(round(pose.size * float(scale) * 0.5 * 1.08)))
    center = (int(round(pose.x)), int(round(pose.y)))
    cv2.ellipse(mask, center, (radius_x, radius_y), math.degrees(pose.roll), 0, 360, 255, -1, cv2.LINE_AA)
    feather = max(0, int(feather_px))
    if feather:
        kernel = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask.astype(np.float32) / 255.0


def serialize_tracking(fps: float, width: int, height: int, frames: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"version": 2, "fps": float(fps), "width": int(width), "height": int(height), "frames": frames},
        separators=(",", ":"),
    )


def _pose_iou_proxy(a: Pose, b: Pose) -> float:
    """Approximate face-box IoU from center/size for detector fusion."""
    ax0, ay0 = a.x - a.size * 0.5, a.y - a.size * 0.5
    ax1, ay1 = a.x + a.size * 0.5, a.y + a.size * 0.5
    bx0, by0 = b.x - b.size * 0.5, b.y - b.size * 0.5
    bx1, by1 = b.x + b.size * 0.5, b.y + b.size * 0.5
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


def fuse_poses(primary: list[Pose], secondary: list[Pose], iou_threshold: float = 0.35) -> list[Pose]:
    """Keep MediaPipe landmarks when present; add YuNet-only faces that do not overlap."""
    fused = list(primary)
    for candidate in secondary:
        if any(_pose_iou_proxy(candidate, existing) >= iou_threshold for existing in fused):
            continue
        fused.append(candidate)
    return fused


class MediaPipeFaceTracker:
    def __init__(self, model_path: Path, confidence: float, max_faces: int = 1) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Install this package's requirements.txt in ComfyUI's Python environment."
            ) from exc
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model not found at {model_path}. Run scripts/download_face_landmarker.py."
            )
        base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max(1, int(max_faces)),
            min_face_detection_confidence=float(confidence),
            min_face_presence_confidence=float(confidence),
            min_tracking_confidence=float(confidence),
        )
        self.mp = mp
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> Sequence[Any] | None:
        faces = self.detect_all(rgb, timestamp_ms)
        return faces[0] if faces else None

    def detect_all(self, rgb: np.ndarray, timestamp_ms: int) -> list[Sequence[Any]]:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        return list(result.face_landmarks)

    def close(self) -> None:
        self.landmarker.close()


class YuNetFaceDetector:
    """OpenCV YuNet fallback for small or blurred faces that lack full landmarks."""

    def __init__(self, model_path: Path, confidence: float) -> None:
        import cv2

        if not model_path.is_file():
            raise FileNotFoundError(
                f"YuNet fallback model not found at {model_path}. Run scripts/download_face_landmarker.py."
            )
        self.cv2 = cv2
        self.confidence = max(0.05, float(confidence))
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), self.confidence, 0.3, 5000
        )

    def detect_pose(self, rgb: np.ndarray, face_scale: float, y_offset: float) -> Pose | None:
        poses = self.detect_poses(rgb, face_scale, y_offset)
        return max(poses, key=lambda pose: pose.size) if poses else None

    def _detect_at_scale(self, rgb: np.ndarray, face_scale: float, y_offset: float, scale: float) -> list[Pose]:
        import cv2

        height, width = rgb.shape[:2]
        if scale != 1.0:
            scaled = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            scaled = rgb
        sh, sw = scaled.shape[:2]
        self.detector.setInputSize((sw, sh))
        _, faces = self.detector.detect(self.cv2.cvtColor(scaled, self.cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            return []
        inv = 1.0 / scale
        poses = []
        for face in faces:
            x, y, w, h = (float(v) * inv for v in face[:4])
            eye_a = (float(face[4]) * inv, float(face[5]) * inv)
            eye_b = (float(face[6]) * inv, float(face[7]) * inv)
            left_eye, right_eye = sorted((eye_a, eye_b), key=lambda point: point[0])
            roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            size = max(w, h) * 1.15 * float(face_scale)
            poses.append(Pose(x + w / 2.0, y + h / 2.0 + float(y_offset) * size, size, roll, 0.0))
        return poses

    def detect_poses(
        self,
        rgb: np.ndarray,
        face_scale: float,
        y_offset: float,
        small_face_boost: bool = False,
    ) -> list[Pose]:
        poses = self._detect_at_scale(rgb, face_scale, y_offset, 1.0)
        if not small_face_boost:
            return poses
        # Second pass on a 1.5x upscale recovers distant / tiny faces (movie stills, webcam far).
        height, width = rgb.shape[:2]
        if min(height, width) >= 96:
            boosted = self._detect_at_scale(rgb, face_scale, y_offset, 1.5)
            poses = fuse_poses(poses, boosted, iou_threshold=0.4)
        return poses
