from .composite_overlay import GITSCompositeOverlay
from .track_and_guide import GITSTrackAndGuide
from .simple_face_replacement import GITSSimpleFaceReplacement
from .webcam_face_replacement import GITSWebcamFaceReplacement

NODE_CLASS_MAPPINGS = {
    "GITSTrackAndGuide": GITSTrackAndGuide,
    "GITSCompositeOverlay": GITSCompositeOverlay,
    "GITSSimpleFaceReplacement": GITSSimpleFaceReplacement,
    "GITSWebcamFaceReplacement": GITSWebcamFaceReplacement,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GITSTrackAndGuide": "GITS Face Tracking + Removal (Advanced)",
    "GITSCompositeOverlay": "GITS Final Artwork Overlay",
    "GITSSimpleFaceReplacement": "GITS Face Replacement (Simple)",
    "GITSWebcamFaceReplacement": "GITS Webcam Face Replacement",
}
