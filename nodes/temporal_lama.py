"""Shared per-identity cropped LaMa and optical-flow propagation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


def mask_box(mask: torch.Tensor, padding: float = 0.35) -> tuple[int, int, int, int] | None:
    points = torch.nonzero(mask > 0.02, as_tuple=False)
    if points.numel() == 0:
        return None
    height, width = mask.shape
    y0, x0 = points.amin(dim=0).tolist()
    y1, x1 = points.amax(dim=0).tolist()
    box_w, box_h = x1 - x0 + 1, y1 - y0 + 1
    side = max(box_w, box_h) * (1.0 + 2.0 * float(padding))
    cx, cy = (x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0
    x0 = max(0, round(cx - side / 2.0))
    y0 = max(0, round(cy - side / 2.0))
    x1 = min(width, round(cx + side / 2.0))
    y1 = min(height, round(cy + side / 2.0))
    return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))


def resize_image(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(
        image.permute(0, 3, 1, 2), (height, width), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)


def resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(mask[:, None], (height, width), mode="bilinear", align_corners=False)[:, 0]


def _optical_flow(previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
    """Backward flow current->previous using DIS when available, else Farneback.

    OpenCV DIS optical flow is Apache-2.0 and typically more stable than Farneback
    for face-sized crops between LaMa refresh frames.
    """
    import cv2

    create_dis = getattr(cv2, "DISOpticalFlow_create", None)
    if create_dis is not None:
        try:
            dis = create_dis(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            # DIS compute(I0, I1) estimates flow that maps I0 coords toward I1.
            # We need current -> previous for remapping previous pixels into current.
            return dis.calc(current_gray, previous_gray, None)
        except Exception:
            pass
    return cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0
    )


def warp_patch(
    current: torch.Tensor,
    previous_source: torch.Tensor,
    previous_removed: torch.Tensor,
) -> torch.Tensor:
    import cv2

    current_np = (current[0].detach().to("cpu").numpy() * 255.0).clip(0, 255).astype(np.uint8)
    previous_np = (previous_source[0].detach().to("cpu").numpy() * 255.0).clip(0, 255).astype(np.uint8)
    removed_np = (previous_removed[0].detach().to("cpu").numpy() * 255.0).clip(0, 255).astype(np.uint8)
    current_gray = cv2.cvtColor(current_np, cv2.COLOR_RGB2GRAY)
    previous_gray = cv2.cvtColor(previous_np, cv2.COLOR_RGB2GRAY)
    backward = _optical_flow(previous_gray, current_gray)
    height, width = current_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    warped = cv2.remap(
        removed_np,
        grid_x + backward[..., 0],
        grid_y + backward[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return (
        torch.from_numpy(np.ascontiguousarray(warped))
        .to(device=current.device, dtype=current.dtype)[None]
        .div(255.0)
    )


def edge_soft_alpha(mask: torch.Tensor, edge_px: int = 2) -> torch.Tensor:
    """Soften mask edges so inpainted crops blend into the surrounding frame."""
    if edge_px <= 0:
        return mask.clamp(0.0, 1.0)
    value = mask[:, None]
    # Average pool approximates a cheap feather without SciPy.
    kernel = int(edge_px) * 2 + 1
    soft = F.avg_pool2d(value, kernel, stride=1, padding=int(edge_px))
    return (soft[:, 0] * mask.clamp(0.0, 1.0)).clamp(0.0, 1.0)


class TemporalLamaProcessor:
    def __init__(self, remove_callback: Callable, temporal_blend: float = 0.35):
        self.remove_callback = remove_callback
        self.temporal_blend = float(np.clip(temporal_blend, 0.0, 0.95))
        self.cache: dict[int, dict[str, torch.Tensor]] = {}

    def reset(self) -> None:
        self.cache.clear()

    def fill(
        self,
        images: torch.Tensor,
        per_frame_faces,
        resolution: int,
        strength: int,
        edge_smoothness: int,
        start_frame: int,
        interval: int,
        temporal_flow: bool,
        warp_callback: Callable = warp_patch,
        temporal_blend: float | None = None,
    ) -> torch.Tensor:
        blend = self.temporal_blend if temporal_blend is None else float(np.clip(temporal_blend, 0.0, 0.95))
        results = []
        active_ids: set[int] = set()
        for offset, (image, face_entries) in enumerate(zip(images, per_frame_faces)):
            output = image.clone()
            frame_ids: set[int] = set()
            for track_id, mask_np in face_entries:
                frame_ids.add(int(track_id))
                active_ids.add(int(track_id))
                mask = torch.from_numpy(mask_np).to(device=image.device, dtype=image.dtype)
                box = mask_box(mask)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                crop = output[y0:y1, x0:x1][None]
                crop_mask = mask[y0:y1, x0:x1][None]
                resized_image = resize_image(crop, resolution, resolution)
                resized_mask = resize_mask(crop_mask, resolution, resolution)
                cache = self.cache.get(track_id)
                run_lama = cache is None or (start_frame + offset) % max(1, interval) == 0
                if run_lama:
                    patch = self.remove_callback(
                        resized_image, resized_mask, strength, edge_smoothness
                    ).detach()
                    # Cross-fade a fresh LaMa result with the flow-warped prior to kill refresh flash.
                    if cache is not None and blend > 0.0 and temporal_flow:
                        try:
                            warped_prior = warp_callback(
                                resized_image, cache["source"], cache["removed"]
                            ).to(device=patch.device, dtype=patch.dtype)
                            patch = patch * (1.0 - blend) + warped_prior * blend
                        except Exception:
                            pass
                    elif cache is not None and blend > 0.0:
                        prior = cache["removed"].to(device=patch.device, dtype=patch.dtype)
                        if prior.shape == patch.shape:
                            patch = patch * (1.0 - blend * 0.5) + prior * (blend * 0.5)
                elif temporal_flow:
                    patch = warp_callback(resized_image, cache["source"], cache["removed"])
                else:
                    patch = cache["removed"].to(device=image.device, dtype=image.dtype)
                self.cache[track_id] = {
                    "source": resized_image.detach(),
                    "removed": patch.detach(),
                }
                restored = resize_image(patch, y1 - y0, x1 - x0)[0]
                alpha = edge_soft_alpha(crop_mask, edge_px=2)[0][..., None]
                output[y0:y1, x0:x1] = restored * alpha + crop[0] * (1.0 - alpha)
            # Drop caches for identities that left this frame's active set after hold window.
            for stale_id in list(self.cache.keys()):
                if stale_id not in frame_ids and stale_id not in active_ids:
                    # Keep until fully absent from the whole batch pass; cleaned after loop.
                    pass
            results.append(output)
        # Prune tracks never seen in this batch (caller may also prune for streaming).
        live = {track_id for entries in per_frame_faces for track_id, _ in entries}
        for stale_id in list(self.cache.keys()):
            if stale_id not in live:
                del self.cache[stale_id]
        return torch.stack(results)
