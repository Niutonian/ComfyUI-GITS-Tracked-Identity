import numpy as np
import torch

from nodes.composite_overlay import GITSCompositeOverlay
from nodes.track_and_guide import logo_to_rgba, numpy_images_to_tensor, tensor_images_to_numpy


def test_tensor_numpy_roundtrip_shape_channel_and_range():
    source = torch.tensor([[[[1.0, 0.5, 0.0]]]])
    array = tensor_images_to_numpy(source)
    assert array.shape == (1, 1, 1, 3)
    assert array.tolist() == [[[[255, 128, 0]]]]
    restored = numpy_images_to_tensor(array, source.device)
    assert restored.min() >= 0 and restored.max() <= 1


def test_logo_mask_resizes_and_becomes_alpha():
    image = torch.ones((1, 4, 5, 3))
    mask = torch.zeros((1, 2, 2))
    rgba = logo_to_rgba(image, mask)
    assert rgba.shape == (4, 5, 4)
    assert np.all(rgba[..., :3] == 255) and np.all(rgba[..., 3] == 0)


def test_final_composite_broadcasting():
    base = torch.zeros((2, 3, 4, 3))
    overlay = torch.ones_like(base)
    mask = torch.full((2, 3, 4), 0.25)
    result, final_mask = GITSCompositeOverlay().composite(base, overlay, mask)
    assert torch.allclose(result, torch.full_like(result, 0.25))
    assert final_mask.shape == mask.shape


def test_final_composite_optional_backplate_hides_region():
    base = torch.ones((1, 2, 2, 3))
    overlay = torch.zeros_like(base)
    art_mask = torch.zeros((1, 2, 2))
    face_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    result, _ = GITSCompositeOverlay().composite(
        base, overlay, art_mask, backplate_opacity=1.0, backplate_color=0, face_occlusion_masks=face_mask
    )
    assert torch.all(result[0, 0, 0] == 0)
    assert torch.all(result[0, 1, 1] == 1)
