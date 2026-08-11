"""Streamlined all-in-one GITS face-removal and artwork node."""

from __future__ import annotations

import torch

from .presets import PRESET_NAMES
from .track_and_guide import GITSTrackAndGuide, _is_missing_image, _is_missing_mask


def _prepare_load_image_mask(mask: torch.Tensor | None, mode: str) -> torch.Tensor | None:
    if _is_missing_mask(mask):
        return None
    value = mask.to(dtype=torch.float32).clamp(0.0, 1.0)
    return 1.0 - value if mode == "comfy_load_image" else value


class GITSSimpleFaceReplacement:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "effect_preset": (list(PRESET_NAMES), {"default": "classic"}),
                "logo_scale": ("FLOAT", {"default": 1.1, "min": 0.25, "max": 3.0, "step": 0.01}),
                "vertical_position": ("FLOAT", {"default": -0.04, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ring_speed": ("FLOAT", {"default": 35.0, "min": -720.0, "max": 720.0, "step": 1.0}),
                "face_removal": (["lama", "disabled"], {"default": "lama"}),
                "removal_area": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 2.5, "step": 0.01}),
                "lama_strength": ("INT", {"default": 230, "min": 0, "max": 255}),
                "edge_smoothness": ("INT", {"default": 8, "min": 0, "max": 20}),
                "mask_input_mode": (["comfy_load_image", "alpha_mask"], {"default": "comfy_load_image"}),
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
    FUNCTION = "replace_face"
    CATEGORY = "GITS"

    def __init__(self):
        self.engine = GITSTrackAndGuide()

    def replace_face(
        self,
        images,
        fps=24.0,
        effect_preset="classic",
        logo_scale=1.1,
        vertical_position=-0.04,
        ring_speed=35.0,
        face_removal="lama",
        removal_area=1.15,
        lama_strength=230,
        edge_smoothness=8,
        mask_input_mode="comfy_load_image",
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
        static_logo=None,
        static_logo_mask=None,
        ring_logo=None,
        ring_logo_mask=None,
    ):
        # Logos are fully optional: unconnected sockets still allow LaMa face removal.
        if _is_missing_image(static_logo):
            static_logo, static_logo_mask = None, None
        if _is_missing_image(ring_logo):
            ring_logo, ring_logo_mask = None, None
        static_alpha = _prepare_load_image_mask(static_logo_mask, mask_input_mode)
        ring_alpha = _prepare_load_image_mask(ring_logo_mask, mask_input_mode)
        outputs = self.engine.track_and_guide(
            images=images,
            static_logo=static_logo,
            static_logo_mask=static_alpha,
            ring_logo=ring_logo,
            ring_logo_mask=ring_alpha,
            fps=fps,
            face_scale=logo_scale,
            y_offset=vertical_position,
            rotation_speed=ring_speed,
            follow_head_roll=0.15,
            min_cutoff=1.2,
            beta=0.035,
            hold_frames=8,
            fade_frames=8,
            tracker_confidence=0.45,
            fallback_detector="yunet",
            occlusion_scale=removal_area,
            occlusion_feather_px=8,
            guide_occlusion_opacity=1.0,
            face_removal_mode=face_removal,
            lama_removal_strength=lama_strength,
            lama_edge_smoothness=edge_smoothness,
            final_overlay_opacity=1.0,
            face_mode=face_mode,
            max_faces=max_faces,
            temporal_lama=temporal_lama,
            lama_resolution=lama_resolution,
            lama_every_n_frames=lama_every_n_frames,
            temporal_flow=temporal_flow,
            temporal_blend=temporal_blend,
            glitch_intensity=glitch_intensity,
            signal_flicker=signal_flicker,
            edge_aware_mask=edge_aware_mask,
            yaw_foreshorten=yaw_foreshorten,
            track_prediction=True,
            fuse_detectors=True,
            small_face_boost=True,
            effect_preset=effect_preset,
        )
        return outputs[0], outputs[1]
