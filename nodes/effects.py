"""Local visual effects for the identity graphic (MIT, no external models)."""

from __future__ import annotations

import math

import numpy as np


def apply_signal_flicker(alpha: float, frame_index: int, intensity: float, track_id: int = 0) -> float:
    """Modulate opacity when tracking is weak so the plate feels digital."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0.0 or alpha >= 0.999:
        return float(alpha)
    phase = frame_index * 0.73 + track_id * 1.7
    pulse = 0.55 + 0.45 * math.sin(phase)
    drop = (1.0 - float(alpha)) * intensity * pulse
    # Occasional hard drop frames for a censor-glitch feel.
    if (frame_index * 17 + track_id * 13) % 23 == 0:
        drop = min(1.0, drop + 0.35 * intensity)
    return float(np.clip(alpha * (1.0 - 0.35 * intensity * (1.0 - alpha)) - drop * 0.25, 0.0, 1.0))


def apply_glitch_rgba(
    rgba: np.ndarray,
    intensity: float,
    frame_index: int,
    track_id: int = 0,
) -> np.ndarray:
    """Scanlines, micro RGB shifts, and block noise on artwork RGBA."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0.0 or rgba.size == 0:
        return rgba
    out = rgba.astype(np.float32).copy()
    height, width = out.shape[:2]
    if height < 2 or width < 2:
        return rgba

    # Horizontal scanlines.
    line_period = max(2, int(round(3 + 4 * (1.0 - intensity))))
    rows = np.arange(height) % line_period == 0
    out[rows, :, :3] *= 1.0 - 0.35 * intensity
    out[rows, :, 3] *= 1.0 - 0.15 * intensity

    # Occasional chromatic channel shift.
    shift = max(1, int(round(1 + 3 * intensity)))
    if (frame_index + track_id) % 5 == 0:
        r = out[..., 0]
        b = out[..., 2]
        out[..., 0] = np.roll(r, shift, axis=1)
        out[..., 2] = np.roll(b, -shift, axis=1)

    # Sparse block noise on alpha edges.
    if intensity > 0.15:
        rng = np.random.default_rng((frame_index + 1) * 10007 + track_id * 97)
        edge = out[..., 3] > 8
        noise_mask = (rng.random((height, width)) < 0.02 * intensity) & edge
        noise = rng.integers(0, 255, size=(height, width, 3), dtype=np.int16)
        for c in range(3):
            channel = out[..., c]
            channel[noise_mask] = np.clip(
                channel[noise_mask] * (1.0 - intensity) + noise[..., c][noise_mask] * intensity,
                0,
                255,
            )
            out[..., c] = channel

    return out.clip(0, 255).astype(np.uint8)


def edge_aware_face_mask(
    frame_rgb: np.ndarray,
    base_mask: np.ndarray,
    edge_strength: float,
) -> np.ndarray:
    """Slightly grow the face mask along strong edges (hair / silhouette)."""
    import cv2

    strength = float(np.clip(edge_strength, 0.0, 1.0))
    if strength <= 0.0:
        return base_mask.astype(np.float32)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)
    if edges.max() > 1e-6:
        edges = edges / edges.max()
    else:
        return base_mask.astype(np.float32)
    base_u8 = (base_mask.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    dilated = cv2.dilate(base_u8, np.ones((5, 5), np.uint8), iterations=1).astype(np.float32) / 255.0
    ring = np.clip(dilated - base_mask.astype(np.float32), 0.0, 1.0)
    boosted = base_mask.astype(np.float32) + ring * edges * strength
    # Soft blur to keep compositing stable.
    boosted_u8 = (boosted.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    boosted_u8 = cv2.GaussianBlur(boosted_u8, (5, 5), 0)
    return boosted_u8.astype(np.float32) / 255.0


def yaw_scale_x(yaw: float, amount: float) -> float:
    """Cheap foreshortening scale from approximate head yaw in [-1, 1]."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return 1.0
    yaw = float(np.clip(yaw, -1.5, 1.5))
    return float(np.clip(1.0 - abs(yaw) * 0.55 * amount, 0.55, 1.0))
