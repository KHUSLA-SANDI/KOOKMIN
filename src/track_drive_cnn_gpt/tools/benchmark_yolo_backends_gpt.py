#!/usr/bin/env python3
"""Compare PyTorch and OpenVINO on identical frames in calibrated BEV space.

Native segmentation masks are never resized to one another. Each backend's
mask is independently projected through the production camera calibration to
the frozen 128x120 BEV grid, so IoU remains meaningful if native shapes differ.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from track_drive_cnn_gpt.bev_geometry import (
    DEFAULT_EXPECTED_MODEL_SHA256,
    GRID_H,
    GRID_W,
    BevMaskRemapper,
    class_ids,
    load_camera,
    model_class_names,
    native_lane_mid_unions,
    normalize_imgsz,
    optional_native_mask_hw,
    remap_optional_native_masks,
    validate_model_contract,
)


def decode(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def select_image_files(root: Path, limit: int) -> list[Path]:
    """Select frames across the full recording instead of one short burst."""

    supported = {".jpg", ".jpeg", ".png"}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    )
    if len(files) <= limit:
        return files
    if limit == 1:
        return [files[len(files) // 2]]
    indices = [index * (len(files) - 1) // (limit - 1) for index in range(limit)]
    return [files[index] for index in indices]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), q))


def bev_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Binary IoU for two already calibrated BEV channels."""

    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if left.shape != (GRID_H, GRID_W) or right.shape != (GRID_H, GRID_W):
        raise ValueError(
            f"BEV comparison requires {(GRID_H, GRID_W)}, got "
            f"{left.shape} and {right.shape}"
        )
    occupied_union = left | right
    if not bool(occupied_union.any()):
        return 1.0
    return float(np.count_nonzero(left & right)) / int(
        np.count_nonzero(occupied_union)
    )


def _predict_bev(
    model: Any,
    image: np.ndarray,
    imgsz: tuple[int, int],
    lane_id: int,
    mid_id: int,
    camera: dict[str, Any],
    remappers: dict[tuple[tuple[int, int], tuple[int, int]], BevMaskRemapper],
    device: str,
) -> tuple[dict[str, np.ndarray], tuple[int, int] | None, float, float]:
    started = time.perf_counter()
    result = model.predict(
        image,
        imgsz=list(imgsz),
        # Keep both backends in the production single-frame latency mode.
        # Ultralytics 8.3.x otherwise passes its generic batch=16 default to
        # OpenVINO before reading the exported batch-1 metadata.
        batch=1,
        conf=0.25,
        iou=0.70,
        device=device,
        retina_masks=False,
        verbose=False,
    )[0]
    predict_ms = (time.perf_counter() - started) * 1000.0

    lane_native, mid_native = native_lane_mid_unions(result, lane_id, mid_id)
    mask_hw = optional_native_mask_hw(lane_native, mid_native)
    remapper = None
    if mask_hw is not None:
        orig_hw = (int(image.shape[0]), int(image.shape[1]))
        key = (orig_hw, mask_hw)
        remapper = remappers.get(key)
        if remapper is None:
            remapper = BevMaskRemapper(
                camera,
                orig_hw,
                imgsz,
                mask_hw=mask_hw,
                supersample=3,
            )
            remappers[key] = remapper
    lane_grid, mid_grid = remap_optional_native_masks(
        lane_native, mid_native, remapper
    )
    to_bev_ms = (time.perf_counter() - started) * 1000.0
    return {"lane": lane_grid, "mid": mid_grid}, mask_hw, predict_ms, to_bev_ms


def measure(
    model_path: Path,
    images: list[np.ndarray],
    imgsz: tuple[int, int],
    camera: dict[str, Any],
    warmup: int,
    repeats: int,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="segment")
    names = model_class_names(model_path, model)
    lane_id, mid_id = class_ids(names)
    remappers: dict[
        tuple[tuple[int, int], tuple[int, int]], BevMaskRemapper
    ] = {}

    # Warm up both model execution and the shape-keyed calibrated remapper.
    for index in range(warmup):
        _predict_bev(
            model,
            images[index % len(images)],
            imgsz,
            lane_id,
            mid_id,
            camera,
            remappers,
            device,
        )

    predict_times: list[float] = []
    pipeline_times: list[float] = []
    outputs: list[dict[str, Any]] = []
    comparison_count = min(len(images), repeats)
    for index in range(repeats):
        bev, mask_hw, predict_ms, to_bev_ms = _predict_bev(
            model,
            images[index % len(images)],
            imgsz,
            lane_id,
            mid_id,
            camera,
            remappers,
            device,
        )
        predict_times.append(predict_ms)
        pipeline_times.append(to_bev_ms)
        if index < comparison_count:
            outputs.append(
                {
                    "bev": bev,
                    "native_mask_shape": list(mask_hw) if mask_hw else None,
                }
            )

    return {
        "predict_p50_ms": percentile(predict_times, 50),
        "predict_p95_ms": percentile(predict_times, 95),
        "predict_mean_ms": float(statistics.mean(predict_times)),
        "yolo_to_bev_p50_ms": percentile(pipeline_times, 50),
        "yolo_to_bev_p95_ms": percentile(pipeline_times, 95),
        "yolo_to_bev_mean_ms": float(statistics.mean(pipeline_times)),
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--openvino", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=(384, 640),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument(
        "--expected-sha256", default=DEFAULT_EXPECTED_MODEL_SHA256
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--limit-images", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("benchmark_yolo_gpt.json"))
    args = parser.parse_args()

    imgsz = normalize_imgsz(args.imgsz)
    if args.warmup < 0 or args.repeats <= 0 or args.limit_images <= 0:
        raise SystemExit("warmup must be >=0 and repeats/limit-images must be positive")
    validate_model_contract(args.pt, args.expected_sha256, imgsz)
    validate_model_contract(args.openvino, args.expected_sha256, imgsz)

    files = select_image_files(args.images, args.limit_images)
    decoded = [(path, decode(path)) for path in files]
    decoded = [(path, image) for path, image in decoded if image is not None]
    if not decoded:
        raise SystemExit("no decodable jpg/jpeg/png images")
    image_paths = [path for path, _ in decoded]
    images = [image for _, image in decoded]
    camera = load_camera(args.camera)
    expected_hw = (int(camera["image_height"]), int(camera["image_width"]))
    wrong_resolution = [
        (path, tuple(int(value) for value in image.shape[:2]))
        for path, image in decoded
        if tuple(int(value) for value in image.shape[:2]) != expected_hw
    ]
    if wrong_resolution:
        examples = ", ".join(
            f"{path}:{list(actual_hw)}" for path, actual_hw in wrong_resolution[:3]
        )
        raise SystemExit(
            "recorded frame resolution does not match the calibrated camera: "
            f"expected={list(expected_hw)}, examples={examples}"
        )

    cv2.setNumThreads(4)
    pt = measure(args.pt, images, imgsz, camera, args.warmup, args.repeats, args.device)
    ov = measure(
        args.openvino,
        images,
        imgsz,
        camera,
        args.warmup,
        args.repeats,
        args.device,
    )

    pt_outputs = pt.pop("outputs")
    ov_outputs = ov.pop("outputs")
    ov_mask_shapes = {
        tuple(output["native_mask_shape"])
        for output in ov_outputs
        if output["native_mask_shape"] is not None
    }
    if any(shape != imgsz for shape in ov_mask_shapes):
        raise SystemExit(
            "OpenVINO native mask shape violates the static manifest contract: "
            f"expected={list(imgsz)}, actual={sorted(ov_mask_shapes)}"
        )
    comparisons: list[dict[str, Any]] = []
    for path, left, right in zip(image_paths, pt_outputs, ov_outputs):
        row: dict[str, Any] = {
            "image": str(path),
            "pt_native_mask_shape": left["native_mask_shape"],
            "openvino_native_mask_shape": right["native_mask_shape"],
        }
        for name in ("lane", "mid"):
            left_bev = left["bev"][name]
            right_bev = right["bev"][name]
            row[f"{name}_bev_iou"] = bev_iou(left_bev, right_bev)
            row[f"{name}_changed_cells"] = int(
                np.count_nonzero(left_bev != right_bev)
            )
            row[f"pt_{name}_cells"] = int(np.count_nonzero(left_bev))
            row[f"openvino_{name}_cells"] = int(np.count_nonzero(right_bev))
        comparisons.append(row)

    report = {
        "schema_version": "yolo_backend_calibrated_bev_v2_gpt",
        "imgsz": list(imgsz),
        "expected_model_sha256": args.expected_sha256,
        "camera": str(args.camera),
        "frames": len(images),
        "comparison_frames": len(comparisons),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "pt": pt,
        "openvino": ov,
        "speedup_predict_mean": pt["predict_mean_ms"]
        / max(1.0e-9, ov["predict_mean_ms"]),
        "speedup_yolo_to_bev_mean": pt["yolo_to_bev_mean_ms"]
        / max(1.0e-9, ov["yolo_to_bev_mean_ms"]),
        "calibrated_bev": {
            name: {
                "iou_mean": float(
                    np.mean([row[f"{name}_bev_iou"] for row in comparisons])
                ),
                "iou_min": float(
                    np.min([row[f"{name}_bev_iou"] for row in comparisons])
                ),
                "changed_cells_mean": float(
                    np.mean([row[f"{name}_changed_cells"] for row in comparisons])
                ),
            }
            for name in ("lane", "mid")
        },
        "per_frame": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
