"""Post-diffusion exact overlay compositing node."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _shape_mask(mask: torch.Tensor, expand: int, feather: int) -> torch.Tensor:
    value = mask[:, None]
    if expand:
        radius = abs(int(expand))
        kernel = radius * 2 + 1
        value = F.max_pool2d(value, kernel, stride=1, padding=radius) if expand > 0 else -F.max_pool2d(-value, kernel, stride=1, padding=radius)
    if feather:
        radius = int(feather)
        kernel = radius * 2 + 1
        value = F.avg_pool2d(value, kernel, stride=1, padding=radius)
    return value[:, 0].clamp(0.0, 1.0)


class GITSCompositeOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "stylized_images": ("IMAGE",), "aligned_overlay_images": ("IMAGE",), "overlay_masks": ("MASK",),
            "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "mask_expand_px": ("INT", {"default": 0, "min": -128, "max": 128}),
            "mask_feather_px": ("INT", {"default": 0, "min": 0, "max": 128}),
            "backplate_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "backplate_color": ("INT", {"default": 463389, "min": 0, "max": 16777215, "step": 1}),
        }, "optional": {"face_occlusion_masks": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("final_images", "final_masks")
    FUNCTION = "composite"
    CATEGORY = "GITS/Compositing"

    def composite(self, stylized_images, aligned_overlay_images, overlay_masks, opacity=1.0, mask_expand_px=0, mask_feather_px=0, backplate_opacity=0.0, backplate_color=463389, face_occlusion_masks=None):
        if stylized_images.ndim != 4 or aligned_overlay_images.ndim != 4 or overlay_masks.ndim != 3:
            raise ValueError("Expected IMAGE [B,H,W,C] and MASK [B,H,W] tensors")
        if stylized_images.shape != aligned_overlay_images.shape or stylized_images.shape[:3] != overlay_masks.shape:
            raise ValueError("Stylized images, overlays, and masks must have matching batch and spatial dimensions")
        overlay = aligned_overlay_images.to(device=stylized_images.device, dtype=stylized_images.dtype)
        base = stylized_images
        if face_occlusion_masks is not None and float(backplate_opacity) > 0.0:
            if face_occlusion_masks.shape != overlay_masks.shape:
                raise ValueError("Face occlusion mask must match the image batch and spatial dimensions")
            plate_alpha = face_occlusion_masks.to(device=base.device, dtype=base.dtype).clamp(0, 1)[..., None] * float(backplate_opacity)
            color = int(backplate_color)
            plate_rgb = torch.tensor([(color >> 16 & 255) / 255.0, (color >> 8 & 255) / 255.0, (color & 255) / 255.0], device=base.device, dtype=base.dtype)
            base = plate_rgb * plate_alpha + base * (1.0 - plate_alpha)
        mask = overlay_masks.to(device=stylized_images.device, dtype=stylized_images.dtype)
        mask = _shape_mask(mask, int(mask_expand_px), int(mask_feather_px)) * float(opacity)
        alpha = mask.clamp(0.0, 1.0)[..., None]
        result = overlay * alpha + base * (1.0 - alpha)
        return (result.clamp(0.0, 1.0), mask.clamp(0.0, 1.0))
