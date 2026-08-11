# GITS models

The nodes look for `face_landmarker.task`, the movie fallback
`face_detection_yunet_2023mar.onnx`, and `big-lama.pt` in
`ComfyUI/models/gits_tracking/`.
Run `python scripts/download_face_landmarker.py` after placing this package in
`ComfyUI/custom_nodes`. Models are deliberately not downloaded during node execution.
