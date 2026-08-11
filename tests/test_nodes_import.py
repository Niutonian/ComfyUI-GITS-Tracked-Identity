import importlib.util
from pathlib import Path


def test_package_exposes_mappings():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gits_nodes", root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "GITSTrackAndGuide" in module.NODE_CLASS_MAPPINGS
    assert "GITSCompositeOverlay" in module.NODE_CLASS_MAPPINGS
    assert "GITSSimpleFaceReplacement" in module.NODE_CLASS_MAPPINGS
    assert "GITSWebcamFaceReplacement" in module.NODE_CLASS_MAPPINGS
    assert module.NODE_DISPLAY_NAME_MAPPINGS["GITSSimpleFaceReplacement"] == "GITS Face Replacement (Simple)"
    assert module.NODE_DISPLAY_NAME_MAPPINGS["GITSTrackAndGuide"] == "GITS Face Tracking + Removal (Advanced)"
    assert module.NODE_DISPLAY_NAME_MAPPINGS["GITSCompositeOverlay"] == "GITS Final Artwork Overlay"
    assert module.NODE_DISPLAY_NAME_MAPPINGS["GITSWebcamFaceReplacement"] == "GITS Webcam Face Replacement"
