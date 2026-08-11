"""Named effect presets for single-image, video batch, and webcam paths.

Presets only override keys they define. Explicit node widgets still win when
the selected preset is ``custom``.
"""

from __future__ import annotations

from typing import Any

# Keys map to engine kwargs used by Advanced / Simple / Webcam wrappers.
PRESET_NAMES = (
    "custom",
    "classic",
    "subtle",
    "aggressive_track",
    "live_balanced",
    "cinematic_glitch",
)

PRESETS: dict[str, dict[str, Any]] = {
    "custom": {},
    "classic": {
        "face_scale": 1.1,
        "y_offset": -0.04,
        "rotation_speed": 35.0,
        "follow_head_roll": 0.15,
        "min_cutoff": 1.2,
        "beta": 0.035,
        "hold_frames": 10,
        "fade_frames": 8,
        "tracker_confidence": 0.45,
        "occlusion_scale": 1.15,
        "occlusion_feather_px": 10,
        "logo_opacity": 1.0,
        "glitch_intensity": 0.0,
        "signal_flicker": 0.15,
        "edge_aware_mask": 0.35,
        "yaw_foreshorten": 0.45,
        "temporal_blend": 0.35,
        "track_prediction": True,
        "fuse_detectors": True,
        "small_face_boost": True,
    },
    "subtle": {
        "face_scale": 1.0,
        "y_offset": -0.02,
        "rotation_speed": 22.0,
        "follow_head_roll": 0.08,
        "min_cutoff": 0.9,
        "beta": 0.02,
        "hold_frames": 6,
        "fade_frames": 10,
        "occlusion_scale": 1.05,
        "occlusion_feather_px": 14,
        "glitch_intensity": 0.0,
        "signal_flicker": 0.05,
        "edge_aware_mask": 0.5,
        "yaw_foreshorten": 0.25,
        "temporal_blend": 0.45,
        "track_prediction": True,
        "fuse_detectors": True,
        "small_face_boost": True,
    },
    "aggressive_track": {
        "face_scale": 1.15,
        "y_offset": -0.04,
        "rotation_speed": 42.0,
        "follow_head_roll": 0.35,
        "min_cutoff": 2.0,
        "beta": 0.12,
        "hold_frames": 18,
        "fade_frames": 12,
        "tracker_confidence": 0.35,
        "occlusion_scale": 1.25,
        "occlusion_feather_px": 8,
        "glitch_intensity": 0.08,
        "signal_flicker": 0.35,
        "edge_aware_mask": 0.4,
        "yaw_foreshorten": 0.55,
        "temporal_blend": 0.4,
        "track_prediction": True,
        "fuse_detectors": True,
        "small_face_boost": True,
    },
    "live_balanced": {
        "face_scale": 1.1,
        "y_offset": -0.04,
        "rotation_speed": 35.0,
        "follow_head_roll": 0.12,
        "min_cutoff": 1.5,
        "beta": 0.05,
        "hold_frames": 6,
        "fade_frames": 4,
        "tracker_confidence": 0.4,
        "occlusion_scale": 1.15,
        "occlusion_feather_px": 8,
        "glitch_intensity": 0.0,
        "signal_flicker": 0.1,
        "edge_aware_mask": 0.25,
        "yaw_foreshorten": 0.35,
        "temporal_blend": 0.3,
        "track_prediction": True,
        "fuse_detectors": True,
        "small_face_boost": True,
        "lama_every_n_frames": 2,
        "lama_resolution": "384",
        "temporal_flow": True,
    },
    "cinematic_glitch": {
        "face_scale": 1.12,
        "y_offset": -0.04,
        "rotation_speed": 48.0,
        "follow_head_roll": 0.2,
        "min_cutoff": 1.3,
        "beta": 0.05,
        "hold_frames": 12,
        "fade_frames": 10,
        "occlusion_scale": 1.2,
        "occlusion_feather_px": 12,
        "glitch_intensity": 0.45,
        "signal_flicker": 0.55,
        "edge_aware_mask": 0.45,
        "yaw_foreshorten": 0.5,
        "temporal_blend": 0.4,
        "track_prediction": True,
        "fuse_detectors": True,
        "small_face_boost": True,
    },
}


def apply_preset(name: str, values: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied values dict with named preset overlays applied."""
    merged = dict(values)
    preset = PRESETS.get(str(name or "custom"), {})
    for key, value in preset.items():
        merged[key] = value
    return merged
