"""Low-latency stateful webcam face replacement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .multi_face_rendering import render_multi_face_batch
from .multi_face_tracking import MultiFaceIdentityTracker
from .presets import PRESET_NAMES, apply_preset
from .simple_face_replacement import _prepare_load_image_mask
from .temporal_lama import TemporalLamaProcessor, warp_patch
from .track_and_guide import (
    GITSTrackAndGuide,
    _is_missing_image,
    _model_path,
    _yunet_model_path,
)


class GITSWebcamFaceReplacement:
    tracker_factory = None
    fallback_factory = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "effect_preset": (list(PRESET_NAMES), {"default": "live_balanced"}),
                "performance_mode": (["balanced", "fast", "quality"], {"default": "balanced"}),
                "lama_resolution": (["256", "384", "512"], {"default": "384"}),
                "lama_every_n_frames": ("INT", {"default": 2, "min": 1, "max": 30}),
                "logo_scale": ("FLOAT", {"default": 1.1, "min": 0.25, "max": 3.0, "step": 0.01}),
                "vertical_position": ("FLOAT", {"default": -0.04, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ring_speed": ("FLOAT", {"default": 35.0, "min": -720.0, "max": 720.0, "step": 1.0}),
                "removal_area": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 2.5, "step": 0.01}),
                "lama_strength": ("INT", {"default": 230, "min": 0, "max": 255}),
                "edge_smoothness": ("INT", {"default": 8, "min": 0, "max": 20}),
                "mask_input_mode": (["comfy_load_image", "alpha_mask"], {"default": "comfy_load_image"}),
                "reset_tracking": ("BOOLEAN", {"default": False}),
                "face_mode": (["locked_face", "largest_face", "all_faces"], {"default": "locked_face"}),
                "max_faces": ("INT", {"default": 4, "min": 1, "max": 12}),
                "temporal_flow": ("BOOLEAN", {"default": True}),
                "temporal_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.95, "step": 0.01}),
                "glitch_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "signal_flicker": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_aware_mask": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "yaw_foreshorten": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "profile_boost": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Extra LaMa mask expansion on side shots only. Raise if profiles leave half a face.",
                    },
                ),
                "partial_face_boost": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Extra mask expansion when only part of the face is visible (edge crop). Full faces almost unchanged.",
                    },
                ),
            },
            "optional": {
                "static_logo": ("IMAGE",),
                "static_logo_mask": ("MASK",),
                "ring_logo": ("IMAGE",),
                "ring_logo_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_images", "removed_background_images")
    FUNCTION = "process_webcam"
    CATEGORY = "GITS/Live"

    def __init__(self):
        self.engine = GITSTrackAndGuide()
        self.frame_index = 0
        self.temporal_processor = TemporalLamaProcessor(self.engine.remove_with_lama, temporal_blend=0.3)
        self.temporal_cache = self.temporal_processor.cache
        self.multi_tracker: MultiFaceIdentityTracker | None = None
        self.multi_tracker_key = None
        self._cfg = {
            "min_cutoff": 1.2,
            "beta": 0.035,
            "hold_frames": 6,
            "fade_frames": 4,
            "tracker_confidence": 0.45,
            "track_prediction": True,
            "fuse_detectors": True,
            "small_face_boost": True,
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Webcam frames must never be suppressed by ComfyUI's output cache.
        return float("nan")

    def _reset(self) -> None:
        self.engine.reset_streaming_state()
        if self.multi_tracker is not None:
            self.multi_tracker.close()
            self.multi_tracker = None
            self.multi_tracker_key = None
        self.frame_index = 0
        self.temporal_processor.reset()

    def _ensure_multi_tracker(self, fps: float, max_faces: int, width: int, height: int):
        key = (
            float(fps),
            int(max_faces),
            int(width),
            int(height),
            float(self._cfg["min_cutoff"]),
            float(self._cfg["beta"]),
            int(self._cfg["hold_frames"]),
            int(self._cfg["fade_frames"]),
            bool(self._cfg["track_prediction"]),
            bool(self._cfg["fuse_detectors"]),
            bool(self._cfg["small_face_boost"]),
            float(self._cfg["tracker_confidence"]),
        )
        if self.multi_tracker is not None and self.multi_tracker_key == key:
            return self.multi_tracker
        if self.multi_tracker is not None:
            self.multi_tracker.close()
        tracker_factory = type(self).tracker_factory or GITSTrackAndGuide.tracker_factory
        fallback_factory = type(self).fallback_factory or GITSTrackAndGuide.fallback_factory
        confidence = float(self._cfg["tracker_confidence"])
        try:
            detector = tracker_factory(_model_path(), confidence, int(max_faces))
        except TypeError:
            detector = tracker_factory(_model_path(), confidence)
        fallback = fallback_factory(_yunet_model_path(), confidence)
        self.multi_tracker = MultiFaceIdentityTracker(
            detector,
            fallback,
            fps,
            float(self._cfg["min_cutoff"]),
            float(self._cfg["beta"]),
            int(self._cfg["hold_frames"]),
            int(self._cfg["fade_frames"]),
            track_prediction=bool(self._cfg["track_prediction"]),
            fuse_detectors=bool(self._cfg["fuse_detectors"]),
            small_face_boost=bool(self._cfg["small_face_boost"]),
        )
        self.multi_tracker_key = key
        self.temporal_cache.clear()
        return self.multi_tracker

    def _track_faces(
        self,
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
        max_faces: int,
        glitch_intensity: float,
        signal_flicker: float,
        edge_aware_mask: float,
        yaw_foreshorten: float,
        profile_boost: float = 0.55,
        partial_face_boost: float = 0.55,
    ):
        height, width = images.shape[1:3]
        tracker = self._ensure_multi_tracker(fps, max_faces, width, height)
        overlay, art_mask, union_mask, per_frame_faces, _records = render_multi_face_batch(
            tracker,
            images,
            static_logo,
            static_logo_mask,
            ring_logo,
            ring_logo_mask,
            fps,
            start_frame,
            logo_scale,
            vertical_position,
            ring_speed,
            removal_area,
            face_mode,
            0.15,
            1.0,
            8,
            glitch_intensity,
            signal_flicker,
            edge_aware_mask,
            yaw_foreshorten,
            0.0,
            profile_boost,
            partial_face_boost,
        )
        return overlay, art_mask, union_mask, per_frame_faces

    def _fast_fill(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        image_cf = images.permute(0, 3, 1, 2)
        shortest = min(images.shape[1], images.shape[2])
        radius = max(4, min(31, shortest // 30))
        kernel = radius * 2 + 1
        blurred = F.avg_pool2d(image_cf, kernel, stride=1, padding=radius).permute(0, 2, 3, 1)
        alpha = masks.clamp(0.0, 1.0)[..., None]
        return blurred * alpha + images * (1.0 - alpha)

    def _warp_patch(self, current: torch.Tensor, previous_source: torch.Tensor, previous_removed: torch.Tensor) -> torch.Tensor:
        return warp_patch(current, previous_source, previous_removed)

    def _temporal_lama_fill(
        self,
        images: torch.Tensor,
        per_frame_faces,
        resolution: int,
        strength: int,
        edge_smoothness: int,
        start_frame: int,
        interval: int,
        temporal_flow: bool,
        temporal_blend: float,
    ) -> torch.Tensor:
        self.temporal_processor.temporal_blend = float(temporal_blend)
        return self.temporal_processor.fill(
            images,
            per_frame_faces,
            resolution,
            strength,
            edge_smoothness,
            start_frame,
            interval,
            temporal_flow,
            self._warp_patch,
            temporal_blend=float(temporal_blend),
        )

    def process_webcam(
        self,
        images,
        fps=30.0,
        effect_preset="live_balanced",
        performance_mode="balanced",
        lama_resolution="384",
        lama_every_n_frames=2,
        logo_scale=1.1,
        vertical_position=-0.04,
        ring_speed=35.0,
        removal_area=1.15,
        lama_strength=230,
        edge_smoothness=8,
        mask_input_mode="comfy_load_image",
        reset_tracking=False,
        face_mode="locked_face",
        max_faces=4,
        temporal_flow=True,
        temporal_blend=0.3,
        glitch_intensity=0.0,
        signal_flicker=0.1,
        edge_aware_mask=0.25,
        yaw_foreshorten=0.35,
        profile_boost=0.55,
        partial_face_boost=0.55,
        static_logo=None,
        static_logo_mask=None,
        ring_logo=None,
        ring_logo_mask=None,
    ):
        if reset_tracking:
            self._reset()
        cfg = apply_preset(
            effect_preset,
            {
                "face_scale": logo_scale,
                "y_offset": vertical_position,
                "rotation_speed": ring_speed,
                "occlusion_scale": removal_area,
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
                "min_cutoff": 1.2,
                "beta": 0.035,
                "hold_frames": 6,
                "fade_frames": 4,
                "tracker_confidence": 0.45,
                "track_prediction": True,
                "fuse_detectors": True,
                "small_face_boost": True,
            },
        )
        self._cfg = {
            "min_cutoff": cfg["min_cutoff"],
            "beta": cfg["beta"],
            "hold_frames": cfg["hold_frames"],
            "fade_frames": cfg["fade_frames"],
            "tracker_confidence": cfg.get("tracker_confidence", 0.45),
            "track_prediction": cfg["track_prediction"],
            "fuse_detectors": cfg["fuse_detectors"],
            "small_face_boost": cfg["small_face_boost"],
        }
        logo_scale = cfg["face_scale"]
        vertical_position = cfg["y_offset"]
        ring_speed = cfg["rotation_speed"]
        removal_area = cfg["occlusion_scale"]
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

        # Logos are fully optional: tracking + LaMa / blur fill still run without them.
        if _is_missing_image(static_logo):
            static_logo, static_logo_mask = None, None
        if _is_missing_image(ring_logo):
            ring_logo, ring_logo_mask = None, None
        static_alpha = _prepare_load_image_mask(static_logo_mask, mask_input_mode)
        ring_alpha = _prepare_load_image_mask(ring_logo_mask, mask_input_mode)
        start_frame = self.frame_index
        overlay, overlay_mask, face_mask, per_frame_faces = self._track_faces(
            images,
            static_logo,
            static_alpha,
            ring_logo,
            ring_alpha,
            fps,
            start_frame,
            logo_scale,
            vertical_position,
            ring_speed,
            removal_area,
            face_mode,
            max_faces,
            glitch_intensity,
            signal_flicker,
            edge_aware_mask,
            yaw_foreshorten,
            profile_boost,
            partial_face_boost,
        )
        self.frame_index += int(images.shape[0])
        source = images.to(dtype=torch.float32).clamp(0.0, 1.0)
        if performance_mode == "fast":
            removed = self._fast_fill(source, face_mask)
        else:
            interval = 1 if performance_mode == "quality" else max(1, int(lama_every_n_frames))
            removed = self._temporal_lama_fill(
                source,
                per_frame_faces,
                int(lama_resolution),
                int(lama_strength),
                int(edge_smoothness),
                start_frame,
                interval,
                bool(temporal_flow),
                float(temporal_blend),
            )
            current_ids = {track_id for entries in per_frame_faces[-1:] for track_id, _mask in entries}
            for stale_id in set(self.temporal_cache) - current_ids:
                del self.temporal_cache[stale_id]
        alpha = overlay_mask.clamp(0.0, 1.0)[..., None]
        final = (overlay * alpha + removed * (1.0 - alpha)).clamp(0.0, 1.0)
        return final, removed

    def __del__(self):
        try:
            self.engine.reset_streaming_state()
        except Exception:
            pass
