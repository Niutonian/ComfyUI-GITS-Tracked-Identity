"""ComfyUI temporal face tracking and pre-diffusion guide node."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .effects import apply_glitch_rgba, apply_signal_flicker, edge_aware_face_mask, yaw_scale_x
from .multi_face_tracking import MultiFaceIdentityTracker
from .presets import PRESET_NAMES, apply_preset
from .temporal_lama import TemporalLamaProcessor
from .tracking_core import (
    LostTracker,
    MediaPipeFaceTracker,
    PoseSmoother,
    YuNetFaceDetector,
    alpha_composite_numpy,
    combine_rgba_layers,
    face_region_mask,
    place_overlay_in_frame,
    pose_from_landmarks,
    serialize_tracking,
    transform_rgba,
)


def tensor_images_to_numpy(images: torch.Tensor) -> np.ndarray:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("IMAGE must be [B,H,W,3]")
    if images.shape[0] == 0:
        raise ValueError("IMAGE batch cannot be empty")
    return (images.detach().to("cpu", torch.float32).clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)


def numpy_images_to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("NumPy images must be [B,H,W,3]")
    return torch.from_numpy(np.ascontiguousarray(images)).to(device=device, dtype=torch.float32).div_(255.0)


class _NullProgressBar:
    def update_absolute(self, _value, _total=None):
        pass


def _progress_bar(total: int):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(total)
    except ImportError:
        return _NullProgressBar()


def logo_to_rgba(image: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    if image.ndim != 4 or image.shape[-1] != 3 or image.shape[0] == 0:
        raise ValueError("Logo IMAGE must be a non-empty [B,H,W,3] tensor")
    if mask.ndim != 3 or mask.shape[0] == 0:
        raise ValueError("Logo MASK must be a non-empty [B,H,W] tensor")
    rgb = image[0].detach().to("cpu", torch.float32).clamp(0, 1)
    alpha = mask[0].detach().to("cpu", torch.float32).clamp(0, 1)
    if alpha.shape != rgb.shape[:2]:
        alpha = F.interpolate(alpha[None, None], size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    rgba = torch.cat([rgb, alpha[..., None]], dim=-1)
    return (rgba.numpy() * 255.0 + 0.5).astype(np.uint8)


def _is_missing_image(image) -> bool:
    """True when an optional IMAGE socket is unconnected or unusable."""
    if image is None:
        return True
    if not isinstance(image, torch.Tensor):
        return True
    if image.numel() == 0 or image.ndim != 4 or image.shape[0] == 0:
        return True
    if image.shape[-1] not in (3, 4):
        return True
    return False


def _is_missing_mask(mask) -> bool:
    """True when an optional MASK socket is unconnected or unusable."""
    if mask is None:
        return True
    if not isinstance(mask, torch.Tensor):
        return True
    if mask.numel() == 0 or mask.ndim < 2 or mask.shape[0] == 0:
        return True
    return False


def optional_logo_to_rgba(image: torch.Tensor | None, mask: torch.Tensor | None) -> np.ndarray:
    """Convert an optional artwork layer.

    Artwork is never required for tracking or LaMa face removal. Unconnected
    static/ring image or mask sockets become a fully transparent 1x1 layer so
    the rest of the pipeline (detect → hold/fade → face mask → LaMa → composite)
    still runs on every node in the family.
    """
    if _is_missing_image(image):
        return np.zeros((1, 1, 4), dtype=np.uint8)
    if _is_missing_mask(mask):
        # Non-blocking RGB-derived alpha when only the image is connected.
        derived_mask = image[0].detach().to("cpu", torch.float32).clamp(0, 1).amax(dim=-1)[None]
        return logo_to_rgba(image, derived_mask)
    return logo_to_rgba(image, mask)


def wants_lama_removal(mode) -> bool:
    """Global gate for face removal — independent of logo connectivity."""
    return str(mode or "disabled").lower() in {"lama", "lama_rmbg", "true", "1", "on"}


def _model_path() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "gits_tracking" / "face_landmarker.task"
    except ImportError:
        return Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def _yunet_model_path() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "gits_tracking" / "face_detection_yunet_2023mar.onnx"
    except ImportError:
        return Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet_2023mar.onnx"


def _lama_model_path() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "gits_tracking" / "big-lama.pt"
    except ImportError:
        return Path(__file__).resolve().parents[1] / "models" / "big-lama.pt"


class GITSTrackAndGuide:
    tracker_factory: Callable[[Path, float], object] = MediaPipeFaceTracker
    fallback_factory: Callable[[Path, float], object] = YuNetFaceDetector
    lama_remover_class = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "face_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.01}),
                "y_offset": ("FLOAT", {"default": -0.02, "min": -1.0, "max": 1.0, "step": 0.01}),
                "rotation_speed": ("FLOAT", {"default": 42.0, "min": -720.0, "max": 720.0, "step": 1.0}),
                "follow_head_roll": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_cutoff": ("FLOAT", {"default": 1.2, "min": 0.01, "max": 20.0, "step": 0.01}),
                "beta": ("FLOAT", {"default": 0.035, "min": 0.0, "max": 2.0, "step": 0.001}),
                "hold_frames": ("INT", {"default": 8, "min": 0, "max": 120}),
                "fade_frames": ("INT", {"default": 8, "min": 0, "max": 120}),
                "tracker_confidence": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "static_logo": ("IMAGE",),
                "static_logo_mask": ("MASK",),
                "ring_logo": ("IMAGE",),
                "ring_logo_mask": ("MASK",),
                "effect_preset": (list(PRESET_NAMES), {"default": "custom"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "rotation_phase": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0}),
                "logo_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fallback_detector": (["yunet", "disabled"], {"default": "yunet"}),
                "occlusion_scale": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 2.5, "step": 0.01}),
                "occlusion_feather_px": ("INT", {"default": 8, "min": 0, "max": 128}),
                "guide_occlusion_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_removal_mode": (["lama", "disabled"], {"default": "lama"}),
                "lama_removal_strength": ("INT", {"default": 230, "min": 0, "max": 255}),
                "lama_edge_smoothness": ("INT", {"default": 8, "min": 0, "max": 20}),
                "final_overlay_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_mode": (["locked_face", "largest_face", "all_faces", "single_face"], {"default": "locked_face"}),
                "max_faces": ("INT", {"default": 4, "min": 1, "max": 12}),
                "temporal_lama": ("BOOLEAN", {"default": True}),
                "lama_resolution": (["256", "384", "512"], {"default": "384"}),
                "lama_every_n_frames": ("INT", {"default": 2, "min": 1, "max": 30}),
                "temporal_flow": ("BOOLEAN", {"default": True}),
                "temporal_blend": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.95, "step": 0.01}),
                "glitch_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "signal_flicker": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_aware_mask": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "yaw_foreshorten": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "profile_boost": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Extra LaMa/face-mask expansion on side shots only. 0 keeps tight frontal masks; raise toward 1 if profiles leave half a face.",
                    },
                ),
                "partial_face_boost": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Extra mask expansion when only part of the face is visible (frame edge crop / partial face). Full faces are almost unchanged.",
                    },
                ),
                "track_prediction": ("BOOLEAN", {"default": True}),
                "fuse_detectors": ("BOOLEAN", {"default": True}),
                "small_face_boost": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = (
        "final_images",
        "removed_background_images",
        "guide_images",
        "aligned_overlay_images",
        "overlay_masks",
        "tracking_preview",
        "tracking_json",
        "face_occlusion_masks",
    )
    FUNCTION = "track_and_guide"
    CATEGORY = "GITS/Tracking"

    def reset_streaming_state(self) -> None:
        state = getattr(self, "_streaming_state", None)
        if state is not None:
            for item in (state.get("tracker"), state.get("fallback")):
                close = getattr(item, "close", None)
                if close is not None:
                    close()
            del self._streaming_state

    def _get_lama_remover(self):
        remover_class = type(self).lama_remover_class
        if remover_class is None:
            comfy_nodes = sys.modules.get("nodes")
            mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
            remover_class = mappings.get("AILab_LamaRemover")
        if remover_class is None:
            raise RuntimeError(
                "LaMa removal is enabled, but ComfyUI-RMBG's 'Lama Remover (RMBG)' "
                "node is not installed or did not load. Install/enable ComfyUI-RMBG and restart ComfyUI."
            )
        if not hasattr(self, "_lama_remover") or not isinstance(self._lama_remover, remover_class):
            self._lama_remover = remover_class()
            local_lama = _lama_model_path()
            if local_lama.is_file():
                self._lama_remover.cache_dir = str(local_lama.parent)
                self._lama_remover.model_path = str(local_lama)
        return self._lama_remover

    def remove_with_lama(self, images, masks, strength=230, edge_smoothness=8):
        """Run LaMa and restore original colors outside the remove mask.

        AILab Big-LaMa rewrites the *entire* frame (not only the hole). Using that
        tensor as-is tints/shifts unmasked pixels. Composite back so only the
        face-remove region keeps the inpaint result.
        """
        source = images.to(dtype=torch.float32).clamp(0.0, 1.0)
        remover = self._get_lama_remover()
        results = []
        progress = _progress_bar(int(source.shape[0]))
        chunk_size = 4
        for start in range(0, int(source.shape[0]), chunk_size):
            end = min(start + chunk_size, int(source.shape[0]))
            chunk = remover.remove_object(
                source[start:end], masks[start:end], int(strength), int(edge_smoothness)
            )[0]
            results.append(chunk.to(device=source.device, dtype=torch.float32))
            progress.update_absolute(end, int(source.shape[0]))
        result = torch.cat(results, dim=0).clamp(0.0, 1.0)
        if result.shape != source.shape:
            raise ValueError("LaMa remover returned an image batch with an unexpected shape")
        alpha = masks.to(device=source.device, dtype=source.dtype).clamp(0.0, 1.0)
        if alpha.ndim == 3:
            alpha = alpha[..., None]
        return (result * alpha + source * (1.0 - alpha)).clamp(0.0, 1.0)

    def _track_and_guide_multi(
        self,
        images,
        static_logo,
        static_logo_mask,
        ring_logo,
        ring_logo_mask,
        fps,
        face_scale,
        y_offset,
        rotation_speed,
        follow_head_roll,
        min_cutoff,
        beta,
        hold_frames,
        fade_frames,
        tracker_confidence,
        start_frame,
        logo_opacity,
        fallback_detector,
        occlusion_scale,
        occlusion_feather_px,
        guide_occlusion_opacity,
        face_removal_mode,
        lama_removal_strength,
        lama_edge_smoothness,
        final_overlay_opacity,
        face_mode,
        max_faces,
        temporal_lama,
        lama_resolution,
        lama_every_n_frames,
        temporal_flow,
        temporal_blend,
        glitch_intensity,
        signal_flicker,
        edge_aware_mask,
        yaw_foreshorten,
        track_prediction,
        fuse_detectors,
        small_face_boost,
        rotation_phase,
        profile_boost,
        partial_face_boost,
    ):
        from .multi_face_rendering import render_multi_face_batch

        try:
            detector = type(self).tracker_factory(_model_path(), float(tracker_confidence), int(max_faces))
        except TypeError:
            detector = type(self).tracker_factory(_model_path(), float(tracker_confidence))
        fallback = None
        if fallback_detector == "yunet":
            fallback = type(self).fallback_factory(_yunet_model_path(), float(tracker_confidence))
        tracker = MultiFaceIdentityTracker(
            detector,
            fallback,
            float(fps),
            float(min_cutoff),
            float(beta),
            int(hold_frames),
            int(fade_frames),
            track_prediction=bool(track_prediction),
            fuse_detectors=bool(fuse_detectors),
            small_face_boost=bool(small_face_boost),
        )
        try:
            overlay, art_mask, face_mask, per_frame_faces, records = render_multi_face_batch(
                tracker,
                images,
                static_logo,
                static_logo_mask,
                ring_logo,
                ring_logo_mask,
                float(fps),
                int(start_frame),
                float(face_scale),
                float(y_offset),
                float(rotation_speed),
                float(occlusion_scale),
                face_mode,
                float(follow_head_roll),
                float(logo_opacity),
                int(occlusion_feather_px),
                float(glitch_intensity),
                float(signal_flicker),
                float(edge_aware_mask),
                float(yaw_foreshorten),
                float(rotation_phase),
                float(profile_boost),
                float(partial_face_boost),
            )
        finally:
            tracker.close()
        source = images.to(dtype=torch.float32).clamp(0.0, 1.0)
        plate_alpha = (face_mask * float(guide_occlusion_opacity)).clamp(0.0, 1.0)[..., None]
        plate_color = torch.tensor(
            [7 / 255.0, 18 / 255.0, 29 / 255.0], device=source.device, dtype=source.dtype
        )
        plated = plate_color * plate_alpha + source * (1.0 - plate_alpha)
        art_alpha = art_mask.clamp(0.0, 1.0)[..., None]
        guide = (overlay * art_alpha + plated * (1.0 - art_alpha)).clamp(0.0, 1.0)
        # LaMa / face removal never depends on static_logo / ring_logo being connected.
        # With no artwork, final_images is simply the reconstructed background.
        removed = source
        if wants_lama_removal(face_removal_mode):
            if temporal_lama:
                processor = TemporalLamaProcessor(self.remove_with_lama, temporal_blend=float(temporal_blend))
                removed = processor.fill(
                    source,
                    per_frame_faces,
                    int(lama_resolution),
                    int(lama_removal_strength),
                    int(lama_edge_smoothness),
                    int(start_frame),
                    int(lama_every_n_frames),
                    bool(temporal_flow),
                    temporal_blend=float(temporal_blend),
                )
            else:
                removed = self.remove_with_lama(
                    source, face_mask, lama_removal_strength, lama_edge_smoothness
                )
        final_alpha = (art_mask * float(final_overlay_opacity)).clamp(0.0, 1.0)[..., None]
        final = (overlay * final_alpha + removed * (1.0 - final_alpha)).clamp(0.0, 1.0)
        height, width = images.shape[1:3]
        metadata = serialize_tracking(float(fps), int(width), int(height), records)
        return final, removed, guide, overlay, art_mask, guide.clone(), metadata, face_mask

    def track_and_guide(
        self,
        images,
        fps=24.0,
        face_scale=1.0,
        y_offset=-0.02,
        rotation_speed=42.0,
        follow_head_roll=0.2,
        min_cutoff=1.2,
        beta=0.035,
        hold_frames=8,
        fade_frames=8,
        tracker_confidence=0.45,
        start_frame=0,
        rotation_phase=0.0,
        logo_opacity=1.0,
        fallback_detector="yunet",
        occlusion_scale=1.15,
        occlusion_feather_px=8,
        guide_occlusion_opacity=1.0,
        face_removal_mode="lama",
        lama_removal_strength=230,
        lama_edge_smoothness=8,
        final_overlay_opacity=1.0,
        face_mode="locked_face",
        max_faces=4,
        temporal_lama=True,
        lama_resolution="384",
        lama_every_n_frames=2,
        temporal_flow=True,
        temporal_blend=0.35,
        glitch_intensity=0.0,
        signal_flicker=0.1,
        edge_aware_mask=0.35,
        yaw_foreshorten=0.4,
        profile_boost=0.55,
        partial_face_boost=0.55,
        track_prediction=True,
        fuse_detectors=True,
        small_face_boost=True,
        effect_preset="custom",
        static_logo=None,
        static_logo_mask=None,
        ring_logo=None,
        ring_logo_mask=None,
        persist_tracking=False,
    ):
        if float(fps) <= 0:
            raise ValueError("fps must be positive")
        # Explicit None-normalization so unconnected optional sockets never block LaMa.
        if _is_missing_image(static_logo):
            static_logo = None
            static_logo_mask = None
        elif _is_missing_mask(static_logo_mask):
            static_logo_mask = None
        if _is_missing_image(ring_logo):
            ring_logo = None
            ring_logo_mask = None
        elif _is_missing_mask(ring_logo_mask):
            ring_logo_mask = None
        cfg = apply_preset(
            effect_preset,
            {
                "face_scale": face_scale,
                "y_offset": y_offset,
                "rotation_speed": rotation_speed,
                "follow_head_roll": follow_head_roll,
                "min_cutoff": min_cutoff,
                "beta": beta,
                "hold_frames": hold_frames,
                "fade_frames": fade_frames,
                "tracker_confidence": tracker_confidence,
                "occlusion_scale": occlusion_scale,
                "occlusion_feather_px": occlusion_feather_px,
                "logo_opacity": logo_opacity,
                "lama_every_n_frames": lama_every_n_frames,
                "lama_resolution": lama_resolution,
                "temporal_flow": temporal_flow,
                "temporal_blend": temporal_blend,
                "glitch_intensity": glitch_intensity,
                "signal_flicker": signal_flicker,
                "edge_aware_mask": edge_aware_mask,
                "yaw_foreshorten": yaw_foreshorten,
                "profile_boost": profile_boost,
                "partial_face_boost": partial_face_boost,
                "track_prediction": track_prediction,
                "fuse_detectors": fuse_detectors,
                "small_face_boost": small_face_boost,
            },
        )
        face_scale = cfg["face_scale"]
        y_offset = cfg["y_offset"]
        rotation_speed = cfg["rotation_speed"]
        follow_head_roll = cfg["follow_head_roll"]
        min_cutoff = cfg["min_cutoff"]
        beta = cfg["beta"]
        hold_frames = cfg["hold_frames"]
        fade_frames = cfg["fade_frames"]
        tracker_confidence = cfg["tracker_confidence"]
        occlusion_scale = cfg["occlusion_scale"]
        occlusion_feather_px = cfg["occlusion_feather_px"]
        logo_opacity = cfg["logo_opacity"]
        lama_every_n_frames = cfg["lama_every_n_frames"]
        lama_resolution = cfg["lama_resolution"]
        temporal_flow = cfg["temporal_flow"]
        temporal_blend = cfg["temporal_blend"]
        glitch_intensity = cfg["glitch_intensity"]
        signal_flicker = cfg["signal_flicker"]
        edge_aware_mask = cfg["edge_aware_mask"]
        yaw_foreshorten = cfg["yaw_foreshorten"]
        profile_boost = cfg["profile_boost"]
        partial_face_boost = cfg["partial_face_boost"]
        track_prediction = cfg["track_prediction"]
        fuse_detectors = cfg["fuse_detectors"]
        small_face_boost = cfg["small_face_boost"]

        if face_mode != "single_face":
            return self._track_and_guide_multi(
                images,
                static_logo,
                static_logo_mask,
                ring_logo,
                ring_logo_mask,
                fps,
                face_scale,
                y_offset,
                rotation_speed,
                follow_head_roll,
                min_cutoff,
                beta,
                hold_frames,
                fade_frames,
                tracker_confidence,
                start_frame,
                logo_opacity,
                fallback_detector,
                occlusion_scale,
                occlusion_feather_px,
                guide_occlusion_opacity,
                face_removal_mode,
                lama_removal_strength,
                lama_edge_smoothness,
                final_overlay_opacity,
                face_mode,
                max_faces,
                temporal_lama,
                lama_resolution,
                lama_every_n_frames,
                temporal_flow,
                temporal_blend,
                glitch_intensity,
                signal_flicker,
                edge_aware_mask,
                yaw_foreshorten,
                track_prediction,
                fuse_detectors,
                small_face_boost,
                rotation_phase,
                profile_boost,
                partial_face_boost,
            )
        frames = tensor_images_to_numpy(images)
        batch, height, width, _ = frames.shape
        static_rgba = optional_logo_to_rgba(static_logo, static_logo_mask)
        ring_rgba = optional_logo_to_rgba(ring_logo, ring_logo_mask)
        state_key = (
            float(fps),
            float(min_cutoff),
            float(beta),
            int(hold_frames),
            int(fade_frames),
            float(tracker_confidence),
            fallback_detector,
            bool(track_prediction),
            width,
            height,
        )
        state = getattr(self, "_streaming_state", None) if persist_tracking else None
        if state is not None and state.get("key") != state_key:
            self.reset_streaming_state()
            state = None
        if state is None:
            tracker = type(self).tracker_factory(_model_path(), float(tracker_confidence))
            fallback = None
            if fallback_detector == "yunet":
                fallback = type(self).fallback_factory(_yunet_model_path(), float(tracker_confidence))
            smoother = PoseSmoother(float(fps), float(min_cutoff), float(beta))
            lost = LostTracker(int(hold_frames), int(fade_frames), predict=bool(track_prediction))
            previous_timestamp = -1
            if persist_tracking:
                state = {
                    "key": state_key,
                    "tracker": tracker,
                    "fallback": fallback,
                    "smoother": smoother,
                    "lost": lost,
                    "previous_timestamp": -1,
                }
                self._streaming_state = state
        else:
            tracker = state["tracker"]
            fallback = state["fallback"]
            smoother = state["smoother"]
            lost = state["lost"]
            previous_timestamp = int(state["previous_timestamp"])
        guides, overlays_rgb, masks, face_masks, records = [], [], [], [], []
        tracking_progress = _progress_bar(batch)
        try:
            for index, frame in enumerate(frames):
                frame_number = int(start_frame) + index
                time_seconds = frame_number / float(fps)
                timestamp_ms = round(time_seconds * 1000.0)
                if timestamp_ms <= previous_timestamp:
                    raise ValueError("Frame timestamps are not strictly monotonic; check fps and start_frame")
                previous_timestamp = timestamp_ms
                landmarks = tracker.detect(frame, timestamp_ms)
                detector_name = "mediapipe"
                raw_pose = (
                    None
                    if landmarks is None
                    else pose_from_landmarks(landmarks, width, height, float(face_scale), float(y_offset))
                )
                if raw_pose is None and fallback is not None:
                    try:
                        poses = fallback.detect_poses(
                            frame, float(face_scale), float(y_offset), small_face_boost=bool(small_face_boost)
                        )
                    except TypeError:
                        poses = fallback.detect_poses(frame, float(face_scale), float(y_offset))
                    raw_pose = max(poses, key=lambda pose: pose.size) if poses else None
                    detector_name = "yunet" if raw_pose is not None else "none"
                smoothed = None if raw_pose is None else smoother.apply(raw_pose)
                pose, alpha, detected = lost.update(smoothed)
                if pose is not None:
                    alpha = apply_signal_flicker(alpha, frame_number, float(signal_flicker), 0)
                ring_angle = float(rotation_phase) + time_seconds * float(rotation_speed)
                full_rgba = np.zeros((height, width, 4), dtype=np.uint8)
                face_mask = np.zeros((height, width), dtype=np.float32)
                record = {
                    "frame": frame_number,
                    "time": float(time_seconds),
                    "detected": bool(detected),
                    "alpha": float(alpha),
                }
                if pose is not None and alpha > 0.0:
                    roll_degrees = math.degrees(pose.roll) * float(follow_head_roll)
                    scale_x = yaw_scale_x(pose.yaw, float(yaw_foreshorten))
                    static_layer = transform_rgba(static_rgba, pose.size, roll_degrees, scale_x=scale_x)
                    ring_layer = transform_rgba(
                        ring_rgba, pose.size, ring_angle + roll_degrees, scale_x=scale_x
                    )
                    combined = combine_rgba_layers(static_layer, ring_layer)
                    combined = apply_glitch_rgba(combined, float(glitch_intensity), frame_number, 0)
                    combined[..., 3] = np.rint(
                        combined[..., 3].astype(np.float32) * alpha * float(logo_opacity)
                    ).clip(0, 255).astype(np.uint8)
                    full_rgba = place_overlay_in_frame(combined, height, width, pose.x, pose.y)
                    face_mask = face_region_mask(
                        height,
                        width,
                        pose,
                        float(occlusion_scale),
                        int(occlusion_feather_px),
                        profile_boost=float(profile_boost),
                        partial_face_boost=float(partial_face_boost),
                    ) * float(alpha)
                    if float(edge_aware_mask) > 0.0:
                        face_mask = edge_aware_face_mask(frame, face_mask, float(edge_aware_mask))
                    record.update(
                        {
                            "x": float(pose.x),
                            "y": float(pose.y),
                            "size": float(pose.size),
                            "roll": float(pose.roll),
                            "yaw": float(pose.yaw),
                            "ring_angle": float(ring_angle),
                            "detector": detector_name,
                        }
                    )
                overlays_rgb.append(full_rgba[..., :3])
                masks.append(full_rgba[..., 3].astype(np.float32) / 255.0)
                face_masks.append(face_mask.clip(0.0, 1.0))
                plate = np.zeros((height, width, 4), dtype=np.uint8)
                plate[..., :3] = (7, 18, 29)
                plate[..., 3] = np.rint(face_mask * float(guide_occlusion_opacity) * 255.0).clip(0, 255).astype(
                    np.uint8
                )
                guide_overlay = combine_rgba_layers(full_rgba, plate)
                guides.append(alpha_composite_numpy(frame, guide_overlay))
                records.append(record)
                tracking_progress.update_absolute(index + 1, batch)
            if persist_tracking:
                state["previous_timestamp"] = previous_timestamp
        except Exception:
            if persist_tracking:
                self.reset_streaming_state()
            raise
        finally:
            if not persist_tracking:
                close = getattr(tracker, "close", None)
                if close is not None:
                    close()
                fallback_close = getattr(fallback, "close", None)
                if fallback_close is not None:
                    fallback_close()
        device = images.device
        guide_tensor = numpy_images_to_tensor(np.stack(guides), device)
        overlay_tensor = numpy_images_to_tensor(np.stack(overlays_rgb), device)
        mask_tensor = torch.from_numpy(np.stack(masks)).to(device=device, dtype=torch.float32)
        face_mask_tensor = torch.from_numpy(np.stack(face_masks)).to(device=device, dtype=torch.float32)
        removed_tensor = images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        # Artwork sockets are optional; LaMa uses face_occlusion masks only.
        if wants_lama_removal(face_removal_mode):
            removed_tensor = self.remove_with_lama(
                images, face_mask_tensor, lama_removal_strength, lama_edge_smoothness
            )
        final_alpha = (mask_tensor * float(final_overlay_opacity)).clamp(0.0, 1.0)[..., None]
        final_tensor = (overlay_tensor * final_alpha + removed_tensor * (1.0 - final_alpha)).clamp(0.0, 1.0)
        metadata = serialize_tracking(float(fps), width, height, records)
        return (
            final_tensor,
            removed_tensor,
            guide_tensor,
            overlay_tensor,
            mask_tensor,
            guide_tensor.clone(),
            metadata,
            face_mask_tensor,
        )
