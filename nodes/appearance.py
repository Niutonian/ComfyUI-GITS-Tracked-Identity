"""Lightweight appearance signatures for multi-face identity association.

Uses only NumPy (BSD) and local RGB crops — no network models.
"""

from __future__ import annotations

import math

import numpy as np

from .tracking_core import Pose


def extract_appearance(frame_rgb: np.ndarray, pose: Pose, bins: int = 8) -> np.ndarray:
    """Return a unit L2-normalized histogram signature for a face crop."""
    height, width = frame_rgb.shape[:2]
    half = max(4.0, pose.size * 0.42)
    x0 = max(0, int(round(pose.x - half)))
    y0 = max(0, int(round(pose.y - half)))
    x1 = min(width, int(round(pose.x + half)))
    y1 = min(height, int(round(pose.y + half)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(bins * 3 + 3, dtype=np.float32)
    crop = frame_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros(bins * 3 + 3, dtype=np.float32)
    flat = crop.reshape(-1, 3).astype(np.float32)
    means = flat.mean(axis=0) / 255.0
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(flat[:, channel], bins=bins, range=(0, 256), density=True)
        hist_parts.append(hist.astype(np.float32))
    feature = np.concatenate([means, *hist_parts]).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm > 1e-8:
        feature /= norm
    return feature


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.5
    # Cosine distance in [0, 2] mapped softly into [0, 1].
    sim = float(np.dot(a, b))
    return float(np.clip(0.5 * (1.0 - sim), 0.0, 1.0))


def bipartite_match(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Greedy global matching for small N (faces <= 12). Prefer low cost.

    Not full Hungarian, but sorts all edges once and commits uniquely — stable
    and O(n^2 log n), enough for multi-face webcam/video.
    """
    if cost.size == 0:
        return []
    rows, cols = cost.shape
    pairs: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            value = float(cost[r, c])
            if value <= max_cost:
                pairs.append((value, r, c))
    pairs.sort()
    used_r: set[int] = set()
    used_c: set[int] = set()
    matches: list[tuple[int, int]] = []
    for value, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        matches.append((r, c))
    return matches


def spatial_match_cost(previous: Pose, current: Pose) -> float:
    scale = max(previous.size, current.size, 1.0)
    distance = math.hypot(previous.x - current.x, previous.y - current.y) / scale
    size_change = abs(math.log(max(current.size, 1.0) / max(previous.size, 1.0)))
    roll_delta = abs(previous.roll - current.roll) / math.pi
    return distance + 0.35 * size_change + 0.15 * roll_delta
