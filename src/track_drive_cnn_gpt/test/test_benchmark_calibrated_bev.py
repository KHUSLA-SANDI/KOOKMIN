import importlib.util
from pathlib import Path

import numpy as np
import pytest

from track_drive_cnn_gpt.bev_geometry import GRID_H, GRID_W


ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark_module():
    path = ROOT / "tools" / "benchmark_yolo_backends_gpt.py"
    spec = importlib.util.spec_from_file_location("benchmark_yolo_backends_gpt", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibrated_bev_iou_handles_empty_and_changed_cells() -> None:
    module = _load_benchmark_module()
    left = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    right = np.zeros_like(left)
    assert module.bev_iou(left, right) == 1.0

    left[10, 20] = 1
    right[10, 20] = 1
    right[10, 21] = 1
    assert module.bev_iou(left, right) == pytest.approx(0.5)


def test_calibrated_bev_iou_refuses_native_mask_shape_comparison() -> None:
    module = _load_benchmark_module()
    # A/B must project each native mask first; resizing one backend to the
    # other's native output shape would hide letterbox/calibration errors.
    with pytest.raises(ValueError, match="BEV comparison requires"):
        module.bev_iou(
            np.zeros((384, 640), dtype=np.uint8),
            np.zeros((640, 640), dtype=np.uint8),
        )


def test_frame_selection_covers_the_whole_recording(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    for index in range(10):
        (tmp_path / f"{index:02d}.jpg").write_bytes(b"not decoded here")
    selected = module.select_image_files(tmp_path, 4)
    assert [path.name for path in selected] == [
        "00.jpg",
        "03.jpg",
        "06.jpg",
        "09.jpg",
    ]
