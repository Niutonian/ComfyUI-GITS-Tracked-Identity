# ComfyUI GITS Tracked Identity

A ComfyUI custom-node package that places a tracked, camera-facing identity
graphic over one or more faces — for **single images**, **ordered video
batches**, and **live webcam**. The name describes the visual technique; this
repository contains no copyrighted franchise artwork. Bundled cyan placeholders
are original geometric art.

**Version 0.2.0** — multi-face IDs, Temporal LaMa, webcam, glitch/signal
effects, appearance matching, motion prediction, and improved temporal fill.

Everything runs **locally**. Dependencies and models use MIT/Apache-style
licenses only (see [Licenses](#licenses)).

## Why the overlay is applied twice

```text
video/image batch -> Advanced Tracking -> preprocessing / diffusion -> Final Artwork Overlay
                         |                                      ^
                         +-- exact RGB overlay + alpha mask ----+
```

`GITS Face Tracking + Removal (Advanced)` puts an opaque proxy plate plus the
graphic into the images sent toward diffusion. It also returns the exact aligned
RGB graphic and alpha separately. `GITS Final Artwork Overlay` applies those
exact pixels after VAE decoding so letters, circles, thin lines, and ring
rotation do not deform. Do not expect KSampler to reproduce logo art.

For a fast path with no diffusion, use **Simple** or **Webcam**: they track,
optionally remove the face with LaMa, and composite the exact graphic in one
node.

**Artwork is optional on every node.** If `static_logo`, `static_logo_mask`,
`ring_logo`, and `ring_logo_mask` are all unconnected, tracking and **LaMa face
removal still run**. The outputs are then the reconstructed background (face
removed) with no graphic overlay.

## Installation

Reference environment: **Python 3.10 or 3.11**.

### Clone into ComfyUI

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/Niutonian/ComfyUI-GITS-Tracked-Identity.git
cd ComfyUI-GITS-Tracked-Identity
```

Or copy this folder to `ComfyUI/custom_nodes/ComfyUI-GITS-Tracked-Identity`.

### Dependencies

Use the same interpreter that starts ComfyUI:

```powershell
# Embedded ComfyUI Easy Install (Windows example)
..\..\python_embeded\python.exe -m pip install -r requirements.txt

# Or a normal venv
python -m pip install -r requirements.txt
```

OpenCV and PyTorch are normally already provided by ComfyUI. Face removal also
needs **ComfyUI-RMBG** (or an equivalent Big-LaMa provider) loaded in ComfyUI.

### Models (explicit download only)

Nodes never download weights at queue time:

```powershell
python scripts\download_face_landmarker.py
```

Destination: `ComfyUI/models/gits_tracking/`  
(`face_landmarker.task`, `face_detection_yunet_2023mar.onnx`, `big-lama.pt`)

Restart ComfyUI. Search the node menu for `GITS`.

### Optional: sync from a development checkout

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy_to_comfyui.ps1 `
  -Target "C:\path\to\ComfyUI\custom_nodes\ComfyUI-GITS-Tracked-Identity"
```

Or set `GITS_COMFYUI_CUSTOM_NODES` and run the script without `-Target`.

## Nodes

| Node | Role |
|------|------|
| **GITS Face Replacement (Simple)** | One-node image/video path: track → LaMa → exact overlay |
| **GITS Webcam Face Replacement** | Stateful live path; keeps tracker + LaMa cache across prompts |
| **GITS Face Tracking + Removal (Advanced)** | Full outputs for diffusion / ControlNet / diagnostics |
| **GITS Final Artwork Overlay** | Exact post-decode composite |

### Effect presets

Available on Simple, Advanced, and Webcam (`effect_preset`):

| Preset | Intent |
|--------|--------|
| `custom` | Use widget values only |
| `classic` | Default censor look, mild signal flicker |
| `subtle` | Softer edges, slower ring, more smoothing |
| `aggressive_track` | Longer hold, lower confidence, more prediction |
| `live_balanced` | Webcam-friendly hold/fade and Temporal LaMa |
| `cinematic_glitch` | Scanlines / RGB shift / stronger flicker |

### Face modes

- `locked_face` — keep the first selected identity
- `largest_face` — always follow the largest face
- `all_faces` — process up to `max_faces` identities
- `single_face` — legacy Advanced one-tracker path

### Tracking robustness (v0.2)

- MediaPipe FaceLandmarker (VIDEO mode) + OpenCV YuNet fallback
- **Detector fusion**: YuNet faces that do not overlap landmarks are kept
- **Small-face boost**: extra YuNet pass on a 1.5× upscale
- **Appearance matching**: local RGB histogram signatures reduce ID swaps
- **Bipartite association** for multi-face frames
- **Motion prediction** during hold/fade after brief occlusions
- One Euro smoothing + configurable hold/fade

### Temporal face removal

- Cropped Big-LaMa at 256 / 384 / 512
- `lama_every_n_frames` + optical flow (OpenCV DIS when available, else Farneback)
- `temporal_blend` cross-fades fresh LaMa with the warped prior (less flash)
- Webcam: `fast` (blur), `balanced` (interval LaMa), `quality` (every frame)

### Visual options

- `glitch_intensity` — scanlines, channel shift, block noise on artwork
- `signal_flicker` — digital opacity noise when tracking is weak
- `edge_aware_mask` — grow removal mask along image edges (hair)
- `yaw_foreshorten` — cheap horizontal squash from estimated head yaw

## Workflows

Packaged under `workflows/`:

- `gits_simple_demo.json` — Simple node, single image or batch
- `gits_core_image_batch.json` — Advanced + Final Overlay (core nodes only)
- `gits_integrated_working_demo.json` / `gits_working_demo.json` — LaMa path
- `gits_video_integration_example.json` — VideoHelperSuite template
- `gits_webcam_template.json` — live template

For long videos, load **chunks** with VideoHelperSuite to limit RAM.

## Performance tips

| Use case | Suggestion |
|----------|------------|
| Single image | Simple + `classic`, LaMa on, Temporal LaMa irrelevant for B=1 |
| Video batch | Simple/Advanced, `temporal_lama=true`, interval 2–3, blend ~0.35 |
| Webcam live | Webcam node, `live_balanced`, `balanced` mode, resolution 256–384 |
| Many faces | Lower `max_faces`, raise interval, or use `fast` mode live |

MediaPipe is CPU-side. Unchanged logos are converted once per call.

## Troubleshooting

- **Model not found:** run `scripts/download_face_landmarker.py`, restart ComfyUI
- **MediaPipe import error:** install `requirements.txt` into ComfyUI’s Python
- **Logo inverted:** Simple/Webcam default `comfy_load_image` fixes core Load Image masks; Advanced may need `InvertMask`
- **Track loss:** try `aggressive_track`, lower `tracker_confidence`, raise `hold_frames`, keep `small_face_boost` / `fuse_detectors` on
- **ID swaps:** `locked_face` + appearance matching; avoid extreme motion blur
- **LaMa flicker:** raise `temporal_blend`, keep `temporal_flow` on, lower interval only if quality allows
- **Tensor size mismatch (Final Overlay):** match batch/H/W after decode

## Tests

```powershell
cd ComfyUI-GITS-Tracked-Identity
python -m pytest -q
```

## Licenses

| Component | License |
|-----------|---------|
| This package | MIT (`LICENSE`) |
| MediaPipe | Apache-2.0 |
| OpenCV / YuNet | Apache-2.0 |
| Big-LaMa weights (local) | Apache-2.0 |

No telemetry, no cloud inference. MediaPipe may log harmless
`portable_clearcut_uploader` messages; they are not network model downloads
from this package.

## Limitations

- Artwork is a camera-facing billboard (roll + optional yaw squash), not a full 3D mesh sticker
- LaMa is an image inpainter; temporal cache/flow/blend improve continuity but are not a dedicated video-inpaint network
- Extreme profiles, heavy occlusion, and tiny faces remain difficult
- `all_faces` cost scales roughly with selected face count × LaMa
