# Changelog

## Unreleased

- **Color-safe LaMa composite:** after Big-LaMa (full-frame or crop), restore
  original RGB outside the face-remove mask so unmasked pixels keep the input
  colors. Fixes global tint / “color profile” shifts from AILab’s full-frame
  rewrite, especially on `single_face` and when `temporal_lama` is off.

## 0.2.0 — 2026-08-11

All four improvement phases for single-image, video-batch, and live webcam.

- Artwork sockets are optional on every node; unconnected logos still allow LaMa face removal.
- Packaging ready for git publish (`.gitignore`, metadata, deploy script without hardcoded local paths).

### Phase 1 — Ship polish
- Named `effect_preset` options: `custom`, `classic`, `subtle`, `aggressive_track`, `live_balanced`, `cinematic_glitch`
- Deploy script: `scripts/deploy_to_comfyui.ps1`
- README rewritten to match implemented multi-face, Temporal LaMa, webcam, and effects
- Package version bumped to 0.2.0

### Phase 2 — Visual fidelity
- Optional glitch layer (scanlines, RGB shift, block noise)
- Signal flicker when track alpha is weak
- Edge-aware face mask growth along hair/silhouette edges
- Yaw-based foreshortening of the billboard artwork

### Phase 3 — Tracking robustness
- Appearance signatures (local RGB histogram) for multi-face association
- Global bipartite matching instead of pure first-greedy pairing
- Detector fusion: MediaPipe landmarks + non-overlapping YuNet faces
- Small-face boost: YuNet second pass on 1.5× upscale
- Motion prediction during hold/fade to reduce track loss pops

### Phase 4 — Temporal fill
- DIS optical flow when OpenCV provides it (Farneback fallback)
- Temporal blend on LaMa refresh to reduce flash between full inpaint frames
- Soft edge compositing of restored face crops
- Stale per-identity cache pruning

### Licenses
- Package code: MIT
- MediaPipe Face Landmarker: Apache-2.0
- OpenCV YuNet / optical flow: Apache-2.0
- Big-LaMa weights: Apache-2.0 (via existing local / RMBG install path)
- No cloud APIs; models are installed explicitly and used offline

## 0.1.0

Initial multi-face tracking, Temporal LaMa, Simple and Webcam nodes.
