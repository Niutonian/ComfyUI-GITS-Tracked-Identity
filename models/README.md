# GITS models

Place downloaded weights in **`ComfyUI/models/gits_tracking/`** (not inside this
package’s empty `models/` folder when the node is installed under ComfyUI).

| File | Manual download URL |
|------|---------------------|
| `face_landmarker.task` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |
| `face_detection_yunet_2023mar.onnx` | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx |
| `big-lama.pt` | https://huggingface.co/1038lab/Lama/resolve/main/big-lama.pt |

Or run from the package root after install:

```powershell
python scripts\download_face_landmarker.py
```

Models are deliberately not downloaded during node execution.
