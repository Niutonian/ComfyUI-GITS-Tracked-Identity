"""Shared rendering for selected persistent face identities."""

from __future__ import annotations

import math

import numpy as np
import torch

from .effects import apply_glitch_rgba, apply_signal_flicker, edge_aware_face_mask, yaw_scale_x
from .track_and_guide import numpy_images_to_tensor, optional_logo_to_rgba, tensor_images_to_numpy
from .tracking_core import combine_rgba_layers, face_region_mask, place_overlay_in_frame, transform_rgba


def render_multi_face_batch(
    tracker,
    images: torch.Tensor,
    static_logo,
    static_logo_mask,
    ring_logo,
    ring_logo_mask,
    fps: float,
    start_frame: int,
    logo_scale: float,
    vertical_position: float,
    ring_speed: float,
    removal_area: float,
    face_mode: str,
    follow_head_roll: float = 0.15,
    logo_opacity: float = 1.0,
    occlusion_feather_px: int = 8,
    glitch_intensity: float = 0.0,
    signal_flicker: float = 0.0,
    edge_aware_mask: float = 0.0,
    yaw_foreshorten: float = 0.0,
    rotation_phase: float = 0.0,
    profile_boost: float = 0.55,
):
    frames = tensor_images_to_numpy(images)
    _, height, width, _ = frames.shape
    static_rgba = optional_logo_to_rgba(static_logo, static_logo_mask)
    ring_rgba = optional_logo_to_rgba(ring_logo, ring_logo_mask)
    overlays, art_masks, union_masks, per_frame_faces, records = [], [], [], [], []
    for offset, frame in enumerate(frames):
        frame_number = int(start_frame) + offset
        timestamp_ms = round(frame_number / float(fps) * 1000.0)
        active = tracker.update(frame, timestamp_ms, float(logo_scale), float(vertical_position))
        selected = tracker.select(active, face_mode)
        combined_frame = np.zeros((height, width, 4), dtype=np.uint8)
        union_mask = np.zeros((height, width), dtype=np.float32)
        face_entries, face_records = [], []
        ring_angle = float(rotation_phase) + frame_number / float(fps) * float(ring_speed)
        for face in selected:
            pose = face.pose
            alpha = apply_signal_flicker(face.alpha, frame_number, float(signal_flicker), face.track_id)
            roll_degrees = math.degrees(pose.roll) * float(follow_head_roll)
            scale_x = yaw_scale_x(pose.yaw, float(yaw_foreshorten))
            static_layer = transform_rgba(static_rgba, pose.size, roll_degrees, scale_x=scale_x)
            ring_layer = transform_rgba(ring_rgba, pose.size, ring_angle + roll_degrees, scale_x=scale_x)
            artwork = combine_rgba_layers(static_layer, ring_layer)
            artwork = apply_glitch_rgba(artwork, float(glitch_intensity), frame_number, face.track_id)
            artwork[..., 3] = np.rint(
                artwork[..., 3].astype(np.float32) * alpha * float(logo_opacity)
            ).clip(0, 255).astype(np.uint8)
            placed = place_overlay_in_frame(artwork, height, width, pose.x, pose.y)
            combined_frame = combine_rgba_layers(placed, combined_frame)
            individual_mask = face_region_mask(
                height,
                width,
                pose,
                float(removal_area),
                int(occlusion_feather_px),
                profile_boost=float(profile_boost),
            ) * alpha
            if float(edge_aware_mask) > 0.0:
                individual_mask = edge_aware_face_mask(frame, individual_mask, float(edge_aware_mask))
            union_mask = np.maximum(union_mask, individual_mask)
            face_entries.append((face.track_id, individual_mask.astype(np.float32)))
            face_records.append({
                "track_id": face.track_id,
                "detected": face.detected,
                "alpha": float(alpha),
                "x": pose.x,
                "y": pose.y,
                "size": pose.size,
                "roll": pose.roll,
                "yaw": pose.yaw,
            })
        overlays.append(combined_frame[..., :3])
        art_masks.append(combined_frame[..., 3].astype(np.float32) / 255.0)
        union_masks.append(union_mask.clip(0.0, 1.0))
        per_frame_faces.append(face_entries)
        records.append({
            "frame": frame_number,
            "time": frame_number / float(fps),
            "face_count": len(face_records),
            "faces": face_records,
        })
    device = images.device
    return (
        numpy_images_to_tensor(np.stack(overlays), device),
        torch.from_numpy(np.stack(art_masks)).to(device=device, dtype=torch.float32),
        torch.from_numpy(np.stack(union_masks)).to(device=device, dtype=torch.float32),
        per_frame_faces,
        records,
    )
