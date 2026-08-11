"""Explicitly install all models used by GITS Tracked Identity."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
LAMA_URL = "https://huggingface.co/1038lab/Lama/resolve/main/big-lama.pt"


def default_output() -> Path:
    script = Path(__file__).resolve()
    if len(script.parents) > 3 and script.parents[2].name.lower() == "custom_nodes":
        return script.parents[3] / "models" / "gits_tracking" / "face_landmarker.task"
    return script.parents[1] / "models" / "face_landmarker.task"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--skip-lama", action="store_true", help="Install only the two face-detection models")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Face Landmarker to {destination}")
    urllib.request.urlretrieve(MODEL_URL, destination)
    print(f"Done ({destination.stat().st_size:,} bytes)")
    yunet = destination.with_name("face_detection_yunet_2023mar.onnx")
    print(f"Downloading YuNet movie fallback to {yunet}")
    urllib.request.urlretrieve(YUNET_URL, yunet)
    print(f"Done ({yunet.stat().st_size:,} bytes)")
    if not args.skip_lama:
        lama = destination.with_name("big-lama.pt")
        existing_rmbg = destination.parent.parent / "RMBG" / "Lama" / "big-lama.pt"
        if existing_rmbg.is_file():
            print(f"Copying existing Big-LaMa model from {existing_rmbg} to {lama}")
            shutil.copy2(existing_rmbg, lama)
        else:
            print(f"Downloading Big-LaMa to {lama}")
            urllib.request.urlretrieve(LAMA_URL, lama)
        print(f"Done ({lama.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
