"""Pure camera-mask to BEV geometry used by the runtime perception node.

This module deliberately has no ROS or Ultralytics imports.  The constants and
mapping equations mirror ``bev_dataset_gpt.py`` so that online inference and
the training dataset use one coordinate contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

import cv2
import numpy as np
import yaml


X_MIN = -0.20
X_MAX = 3.0
Y_MIN = -1.5
Y_MAX = 1.5
RES = 0.025
GRID_H = 128
GRID_W = 120

CH_MID = 0
CH_LANE = 1
CH_LIDAR = 2

DEFAULT_EXPECTED_MODEL_SHA256 = (
    "8ab529d985af95489c46fbd8293a3c1677150bc518e5c9f59f4fe51d0e514c89"
)

OPENVINO_MANIFEST_SCHEMA = "yolo_openvino_export_v2_gpt"


def normalize_imgsz(value: Any) -> tuple[int, int]:
    """Normalize an Ultralytics image size to an explicit ``(height, width)``.

    A scalar remains supported for PyTorch compatibility, but manifests and
    runtime comparisons always use the unambiguous two-dimensional form.
    """

    if isinstance(value, (str, bytes)) or isinstance(value, bool):
        raise ValueError("imgsz must be one integer or a height/width pair")
    if isinstance(value, (int, np.integer)):
        values = (int(value), int(value))
    elif isinstance(value, Sequence) or isinstance(value, np.ndarray):
        raw = list(value)
        if len(raw) == 1:
            values = (int(raw[0]), int(raw[0]))
        elif len(raw) == 2:
            values = (int(raw[0]), int(raw[1]))
        else:
            raise ValueError("imgsz must contain one or two integers")
    else:
        raise ValueError("imgsz must be one integer or a height/width pair")
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError("imgsz height and width must be positive")
    return values


def _matrix(data: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in data:
        raise KeyError(f"camera calibration is missing '{key}'")
    value = np.asarray(data[key], dtype=np.float64)
    if value.size != int(np.prod(shape)):
        raise ValueError(
            f"camera calibration '{key}' has {value.size} values; expected {shape}"
        )
    value = value.reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"camera calibration '{key}' contains non-finite values")
    return value


def load_camera(path: str | Path) -> dict[str, Any]:
    """Load and validate the camera calibration consumed by the BEV mapper."""

    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"camera calibration must be a YAML mapping: {path}")

    data = raw.get("camera", raw)
    if not isinstance(data, Mapping):
        raise ValueError(f"camera calibration 'camera' entry must be a mapping: {path}")

    model = str(data.get("distortion_model", data.get("model", "plumb_bob"))).lower()
    if model not in {"plumb_bob", "radtan"}:
        raise ValueError(
            f"unsupported distortion_model '{model}'; only plumb_bob/radtan is supported"
        )

    try:
        image_width = int(data["image_width"])
        image_height = int(data["image_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "camera calibration must contain integer image_width and image_height"
        ) from exc
    if image_width <= 0 or image_height <= 0:
        raise ValueError("camera image_width and image_height must be positive")

    K = _matrix(data, "K", (3, 3))
    D = np.asarray(data.get("D", []), dtype=np.float64).reshape(-1)
    if D.size < 4 or not np.isfinite(D).all():
        raise ValueError("camera calibration 'D' must contain at least four finite values")
    new_K = _matrix(data, "new_K", (3, 3))
    H = _matrix(data, "H", (3, 3))
    if abs(float(np.linalg.det(H))) < 1.0e-12:
        raise ValueError("camera calibration 'H' is singular")

    car_mask_v = float(data.get("car_mask_v", np.inf))
    if not np.isfinite(car_mask_v) and not np.isinf(car_mask_v):
        raise ValueError("camera calibration 'car_mask_v' must be numeric")

    return {
        "K": K,
        "D": D,
        "new_K": new_K,
        "H": H,
        "car_mask_v": car_mask_v,
        "distortion_model": model,
        "image_width": image_width,
        "image_height": image_height,
        "raw": raw,
    }


def class_ids(names_or_model: Any) -> tuple[int, int]:
    """Return ``(lane_id, mid_id)`` from Ultralytics-style class names."""

    names = getattr(names_or_model, "names", names_or_model)
    if isinstance(names, (list, tuple)):
        pairs = enumerate(names)
    elif isinstance(names, Mapping):
        pairs = names.items()
    else:
        raise TypeError("class names must be a mapping/list or an object with .names")

    normalized = {str(name).strip().lower(): int(index) for index, name in pairs}
    missing = [name for name in ("lane", "mid") if name not in normalized]
    if missing:
        raise ValueError(
            "segmentation model is missing required classes: " + ", ".join(missing)
        )
    return normalized["lane"], normalized["mid"]


def model_class_names(model_path: str | Path, model: Any) -> Any:
    """Read class names without pre-initializing an exported backend.

    Ultralytics 8.3.x implements ``YOLO.names`` by constructing a predictor
    when the loaded model is an exported directory. That construction uses
    the generic predict batch (16) before ``metadata.yaml`` is read, selecting
    OpenVINO throughput mode for a static batch-1 segmentation model. The
    export manifest is already part of our validated model contract, so use it
    directly. PyTorch models can use the underlying nn.Module names without
    creating a predictor.
    """

    path = Path(model_path).expanduser()
    if path.is_dir():
        manifest_path = path / "export_manifest_gpt.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"OpenVINO export manifest does not exist: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        classes = manifest.get("classes")
        _manifest_class_names(classes)
        return classes

    inner_model = getattr(model, "model", None)
    names = getattr(inner_model, "names", None)
    if isinstance(names, (Mapping, list, tuple)):
        return names
    raise TypeError("PyTorch model does not expose class names on model.model.names")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_class_names(classes: Any) -> set[str]:
    if isinstance(classes, Mapping):
        values = classes.values()
    elif isinstance(classes, (list, tuple)):
        values = classes
    else:
        raise ValueError("OpenVINO manifest 'classes' must be a mapping or list")
    return {str(value).strip().lower() for value in values}


def validate_model_contract(
    path: str | Path, expected_sha256: str, runtime_imgsz: Any
) -> str:
    """Validate the exact production PT or its traceable OpenVINO export."""

    model_path = Path(path).expanduser()
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("expected_model_sha256 must be exactly 64 hexadecimal characters")
    runtime_hw = normalize_imgsz(runtime_imgsz)

    if model_path.is_file():
        if model_path.suffix.lower() != ".pt":
            raise ValueError(f"model file must be a .pt checkpoint: {model_path}")
        actual = _sha256_file(model_path)
        if actual != expected:
            raise ValueError(
                f"model SHA256 mismatch: expected {expected}, got {actual} ({model_path})"
            )
        return "pytorch"

    if model_path.is_dir():
        if not any(model_path.glob("*.xml")) or not any(model_path.glob("*.bin")):
            raise ValueError(
                f"OpenVINO model directory must contain .xml and .bin files: {model_path}"
            )
        manifest_path = model_path / "export_manifest_gpt.json"
        if not manifest_path.is_file():
            raise ValueError(f"OpenVINO export manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read OpenVINO export manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError("OpenVINO export manifest root must be an object")
        if manifest.get("schema_version") != OPENVINO_MANIFEST_SCHEMA:
            raise ValueError(
                "unsupported OpenVINO export manifest schema_version; "
                "re-export the static rectangular model"
            )
        source_sha = str(manifest.get("source_sha256", "")).strip().lower()
        if source_sha != expected:
            raise ValueError(
                f"OpenVINO source SHA256 mismatch: expected {expected}, got {source_sha or '<missing>'}"
            )
        try:
            exported_hw = normalize_imgsz(manifest["imgsz"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "OpenVINO manifest has no valid [height, width] imgsz"
            ) from exc
        if exported_hw != runtime_hw:
            raise ValueError(
                "OpenVINO imgsz mismatch: "
                f"export={list(exported_hw)}, runtime={list(runtime_hw)}"
            )
        expected_input_shape = [1, 3, exported_hw[0], exported_hw[1]]
        if manifest.get("input_shape") != expected_input_shape:
            raise ValueError(
                "OpenVINO manifest input_shape mismatch: "
                f"expected {expected_input_shape}, got {manifest.get('input_shape')!r}"
            )
        if manifest.get("dynamic") is not False:
            raise ValueError("OpenVINO manifest must declare dynamic=false")
        if manifest.get("batch") != 1:
            raise ValueError("OpenVINO manifest must declare batch=1")
        if str(manifest.get("precision", "")).upper() != "FP32":
            raise ValueError("OpenVINO manifest must declare precision=FP32")
        if str(manifest.get("task", "")).lower() != "segment":
            raise ValueError("OpenVINO manifest must declare task=segment")
        names = _manifest_class_names(manifest.get("classes"))
        missing = {"lane", "mid"} - names
        if missing:
            raise ValueError(
                "OpenVINO manifest is missing required classes: "
                + ", ".join(sorted(missing))
            )
        return "openvino"

    raise FileNotFoundError(f"model_path does not exist: {model_path}")


def validate_classifier_model_contract(
    path: str | Path, expected_sha256: str, runtime_imgsz: Any
) -> str:
    """Validate the four-class traffic-light classifier or its OpenVINO export."""

    model_path = Path(path).expanduser()
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(
            "expected classifier SHA256 must be exactly 64 hexadecimal characters"
        )
    runtime_hw = normalize_imgsz(runtime_imgsz)

    if model_path.is_file():
        if model_path.suffix.lower() != ".pt":
            raise ValueError(f"classifier file must be a .pt checkpoint: {model_path}")
        actual = _sha256_file(model_path)
        if actual != expected:
            raise ValueError(
                "classifier SHA256 mismatch: "
                f"expected {expected}, got {actual} ({model_path})"
            )
        return "pytorch"

    if model_path.is_dir():
        if not any(model_path.glob("*.xml")) or not any(model_path.glob("*.bin")):
            raise ValueError(
                "OpenVINO classifier directory must contain .xml and .bin files: "
                f"{model_path}"
            )
        manifest_path = model_path / "export_manifest_gpt.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"OpenVINO classifier export manifest is missing: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read OpenVINO classifier manifest: {manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ValueError("OpenVINO classifier manifest root must be an object")
        if manifest.get("schema_version") != OPENVINO_MANIFEST_SCHEMA:
            raise ValueError("unsupported OpenVINO classifier manifest schema_version")
        source_sha = str(manifest.get("source_sha256", "")).strip().lower()
        if source_sha != expected:
            raise ValueError(
                "OpenVINO classifier source SHA256 mismatch: "
                f"expected {expected}, got {source_sha or '<missing>'}"
            )
        try:
            exported_hw = normalize_imgsz(manifest["imgsz"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "OpenVINO classifier manifest has no valid [height, width] imgsz"
            ) from exc
        if exported_hw != runtime_hw:
            raise ValueError(
                "OpenVINO classifier imgsz mismatch: "
                f"export={list(exported_hw)}, runtime={list(runtime_hw)}"
            )
        expected_input_shape = [1, 3, exported_hw[0], exported_hw[1]]
        if manifest.get("input_shape") != expected_input_shape:
            raise ValueError(
                "OpenVINO classifier input_shape mismatch: "
                f"expected {expected_input_shape}, got {manifest.get('input_shape')!r}"
            )
        if manifest.get("dynamic") is not False or manifest.get("batch") != 1:
            raise ValueError("OpenVINO classifier must be static batch=1")
        if str(manifest.get("precision", "")).upper() != "FP32":
            raise ValueError("OpenVINO classifier must declare precision=FP32")
        if str(manifest.get("task", "")).lower() != "classify":
            raise ValueError("OpenVINO classifier must declare task=classify")
        names = _manifest_class_names(manifest.get("classes"))
        missing = {"green", "left", "red", "yellow"} - names
        if missing:
            raise ValueError(
                "OpenVINO classifier is missing required classes: "
                + ", ".join(sorted(missing))
            )
        return "openvino"

    raise FileNotFoundError(f"classifier_model_path does not exist: {model_path}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def native_lane_mid_unions(
    result: Any, lane_id: int, mid_id: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Union native instances independently; a missing class remains ``None``."""

    if result.masks is None or result.boxes is None:
        return None, None
    masks = result.masks.data
    if len(getattr(masks, "shape", ())) != 3 or int(masks.shape[0]) == 0:
        return None, None
    classes = _to_numpy(result.boxes.cls).astype(np.int64, copy=False).reshape(-1)
    if classes.size != int(masks.shape[0]):
        raise RuntimeError(
            f"YOLO returned {masks.shape[0]} masks but {classes.size} class IDs"
        )

    def union_for(class_id: int) -> np.ndarray | None:
        indices = np.flatnonzero(classes == int(class_id))
        if indices.size == 0:
            return None
        selected = masks[indices.tolist()]
        if hasattr(selected, "amax") and hasattr(selected, "detach"):
            union = selected.amax(dim=0)
        else:
            union = np.max(_to_numpy(selected), axis=0)
        return (_to_numpy(union) > 0.5).astype(np.uint8)

    return union_for(lane_id), union_for(mid_id)


def letterbox_shape(
    orig_hw: tuple[int, int], imgsz: Any, stride: int = 32
) -> tuple[int, int, float, float, float]:
    """Reproduce Ultralytics ``LetterBox(auto=True)`` geometry.

    Returns ``(final_h, final_w, scale, pad_x, pad_y)`` exactly like the
    training-dataset extractor.
    """

    oh, ow = (int(orig_hw[0]), int(orig_hw[1]))
    target_h, target_w = normalize_imgsz(imgsz)
    stride = int(stride)
    if oh <= 0 or ow <= 0 or stride <= 0:
        raise ValueError("orig_hw, imgsz and stride must be positive")

    scale = min(target_h / oh, target_w / ow)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    dw = (target_w - nw) % stride
    dh = (target_h - nh) % stride
    return nh + dh, nw + dw, scale, dw / 2.0, dh / 2.0


def _ground_to_undistorted(
    H_inv: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.stack((x, y, np.ones_like(x)), axis=0)
    projected = H_inv @ points
    denom = projected[2]
    safe = np.where(np.abs(denom) < 1.0e-12, np.nan, denom)
    return projected[0] / safe, projected[1] / safe


def _redistort(
    u: np.ndarray,
    v: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    new_K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map undistorted pixel coordinates back into the raw camera image."""

    x = (u - new_K[0, 2]) / new_K[0, 0]
    y = (v - new_K[1, 2]) / new_K[1, 1]
    k1, k2, p1, p2 = D[:4]
    k3 = D[4] if D.size > 4 else 0.0
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]


class BevMaskRemapper:
    """Precomputed inverse maps from native YOLO mask pixels to the BEV grid."""

    def __init__(
        self,
        camera: Mapping[str, Any],
        orig_hw: tuple[int, int] = (1080, 1920),
        imgsz: Any = (384, 640),
        mask_hw: tuple[int, int] | None = None,
        supersample: int = 3,
        stride: int = 32,
    ) -> None:
        self.orig_hw = (int(orig_hw[0]), int(orig_hw[1]))
        self.imgsz = normalize_imgsz(imgsz)
        self.supersample = int(supersample)
        if self.orig_hw[0] <= 0 or self.orig_hw[1] <= 0:
            raise ValueError("orig_hw must be positive")
        if self.supersample <= 0:
            raise ValueError("supersample must be positive")

        self.camera = camera

        if mask_hw is None:
            mh, mw, scale, dw, dh = letterbox_shape(self.orig_hw, self.imgsz, stride)
            mask_hw = (mh, mw)
        else:
            mask_hw = (int(mask_hw[0]), int(mask_hw[1]))
            if mask_hw[0] <= 0 or mask_hw[1] <= 0:
                raise ValueError("mask_hw must be positive")
            oh, ow = self.orig_hw
            mh, mw = mask_hw
            scale = min(mw / ow, mh / oh)
            dw = (mw - ow * scale) / 2.0
            dh = (mh - oh * scale) / 2.0
        self.mask_hw = mask_hw
        self.letterbox = (float(scale), float(dw), float(dh))
        self.maps = self._build_maps()
        self._maps = self.maps

    def _build_maps(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        camera = self.camera
        K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(camera["D"], dtype=np.float64).reshape(-1)
        new_K = np.asarray(camera["new_K"], dtype=np.float64).reshape(3, 3)
        H = np.asarray(camera["H"], dtype=np.float64).reshape(3, 3)
        H_inv = np.linalg.inv(H)
        car_mask_v = float(camera.get("car_mask_v", np.inf))
        scale, dw, dh = self.letterbox

        offsets = (np.arange(self.supersample, dtype=np.float64) + 0.5) / self.supersample
        maps: list[tuple[np.ndarray, np.ndarray]] = []
        oh, ow = self.orig_hw

        for row_offset in offsets:
            x = X_MAX - (np.arange(GRID_H, dtype=np.float64) + row_offset) * RES
            for col_offset in offsets:
                y = Y_MAX - (np.arange(GRID_W, dtype=np.float64) + col_offset) * RES
                xx, yy = np.meshgrid(x, y, indexing="ij")
                uu, vv = _ground_to_undistorted(H_inv, xx.ravel(), yy.ravel())
                ur, vr = _redistort(uu, vv, K, D, new_K)

                map_x = (ur + 0.5) * scale - 0.5 + dw
                map_y = (vr + 0.5) * scale - 0.5 + dh
                valid = (
                    np.isfinite(map_x)
                    & np.isfinite(map_y)
                    & (ur >= 0.0)
                    & (ur <= ow - 1)
                    & (vr >= 0.0)
                    & (vr <= oh - 1)
                    & (vv < car_mask_v)
                )
                map_x = np.where(valid, map_x, -1000.0).reshape(GRID_H, GRID_W)
                map_y = np.where(valid, map_y, -1000.0).reshape(GRID_H, GRID_W)
                maps.append((map_x.astype(np.float32), map_y.astype(np.float32)))

        return tuple(maps)

    def apply(self, native_mask: np.ndarray) -> np.ndarray:
        """Remap one binary native-resolution segmentation mask to ``128x120``."""

        mask = np.asarray(native_mask)
        if mask.shape != self.mask_hw:
            raise ValueError(
                f"native mask shape {mask.shape} does not match remapper {self.mask_hw}"
            )
        source = (mask > 0).astype(np.uint8, copy=False)
        output = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        for map_x, map_y in self._maps:
            sampled = cv2.remap(
                source,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            output |= sampled.astype(np.uint8, copy=False)
        return output


def optional_native_mask_hw(
    lane_native: np.ndarray | None, mid_native: np.ndarray | None
) -> tuple[int, int] | None:
    """Find native mask geometry without requiring both classes to exist."""

    available = lane_native if lane_native is not None else mid_native
    if available is None:
        return None
    if available.ndim != 2:
        raise ValueError("native segmentation masks must be two-dimensional")
    mask_hw = (int(available.shape[0]), int(available.shape[1]))
    if lane_native is not None and tuple(lane_native.shape) != mask_hw:
        raise ValueError("lane native mask has inconsistent shape")
    if mid_native is not None and tuple(mid_native.shape) != mask_hw:
        raise ValueError("mid native mask has inconsistent shape")
    return mask_hw


def remap_optional_native_masks(
    lane_native: np.ndarray | None,
    mid_native: np.ndarray | None,
    remapper: BevMaskRemapper | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap each present class independently; absence zeros only that channel."""

    lane_grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    mid_grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    if lane_native is None and mid_native is None:
        return lane_grid, mid_grid
    if remapper is None:
        raise ValueError("a remapper is required when either native mask exists")
    if lane_native is not None:
        lane_grid = remapper.apply(lane_native)
    if mid_native is not None:
        mid_grid = remapper.apply(mid_native)
    return lane_grid, mid_grid


def full_mask_to_grid(
    native_mask: np.ndarray,
    camera: Mapping[str, Any],
    supersample: int = 3,
) -> np.ndarray:
    """Reference conversion for an already original-resolution retina mask."""

    mask = np.asarray(native_mask)
    if mask.ndim != 2:
        raise ValueError("native_mask must be a two-dimensional array")
    oh, ow = mask.shape
    remapper = BevMaskRemapper(
        camera,
        (oh, ow),
        max(oh, ow),
        mask_hw=mask.shape,
        supersample=supersample,
    )
    return remapper.apply(mask)


def make_bev_image(mid_grid: np.ndarray, lane_grid: np.ndarray) -> np.ndarray:
    """Pack segmentation into ROS ``8UC3`` order: mid, lane, reserved LiDAR."""

    mid = np.asarray(mid_grid)
    lane = np.asarray(lane_grid)
    expected = (GRID_H, GRID_W)
    if mid.shape != expected or lane.shape != expected:
        raise ValueError(f"BEV channel shapes must both be {expected}")
    output = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    output[..., CH_MID] = mid > 0
    output[..., CH_LANE] = lane > 0
    return output


def grid_xy_to_rc(
    x: np.ndarray | float, y: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized dataset conversion from metres to row, column and validity."""

    row = np.floor((X_MAX - np.asarray(x)) / RES).astype(np.int32)
    col = np.floor((Y_MAX - np.asarray(y)) / RES).astype(np.int32)
    valid = (row >= 0) & (row < GRID_H) & (col >= 0) & (col < GRID_W)
    return row, col, valid


def lidar_grid(points_xy: np.ndarray, radius_cells: int = 1) -> np.ndarray:
    """Rasterize forward LiDAR points under the same training-data contract."""

    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    radius_cells = int(radius_cells)
    if radius_cells < 0:
        raise ValueError("radius_cells cannot be negative")
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return grid
    rows, cols, valid = grid_xy_to_rc(points[:, 0], points[:, 1])
    for row, col in zip(rows[valid], cols[valid]):
        row_start = max(0, row - radius_cells)
        row_end = min(GRID_H, row + radius_cells + 1)
        col_start = max(0, col - radius_cells)
        col_end = min(GRID_W, col + radius_cells + 1)
        grid[row_start:row_end, col_start:col_end] = 1
    return grid


__all__ = [
    "X_MIN",
    "X_MAX",
    "Y_MIN",
    "Y_MAX",
    "RES",
    "GRID_H",
    "GRID_W",
    "CH_MID",
    "CH_LANE",
    "CH_LIDAR",
    "DEFAULT_EXPECTED_MODEL_SHA256",
    "BevMaskRemapper",
    "class_ids",
    "full_mask_to_grid",
    "grid_xy_to_rc",
    "letterbox_shape",
    "lidar_grid",
    "load_camera",
    "make_bev_image",
    "native_lane_mid_unions",
    "normalize_imgsz",
    "OPENVINO_MANIFEST_SCHEMA",
    "optional_native_mask_hw",
    "remap_optional_native_masks",
    "validate_model_contract",
    "validate_classifier_model_contract",
]
