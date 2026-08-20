#!/usr/bin/env python3
"""One-frame offline smoke test for the exact runtime YOLO->BEV conversion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np

from track_drive_cnn_gpt.bev_geometry import (
    BevMaskRemapper,
    class_ids,
    load_camera,
    make_bev_image,
    model_class_names,
    normalize_imgsz,
)


def decode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def class_union(result, class_id):
    if result.masks is None or result.boxes is None:
        return None
    selected = result.boxes.cls == float(class_id)
    if not bool(selected.any()):
        shape = tuple(int(value) for value in result.masks.data.shape[-2:])
        return np.zeros(shape, np.uint8)
    return (
        result.masks.data[selected].amax(dim=0).detach().cpu().numpy() > 0.5
    ).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path)
    parser.add_argument(
        "--reference-geometry-code",
        type=Path,
        help="bev_dataset_gpt.py used to create the training arrays",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=(384, 640),
        metavar=("HEIGHT", "WIDTH"),
    )
    args = parser.parse_args()
    imgsz = normalize_imgsz(args.imgsz)

    from ultralytics import YOLO

    image = decode(args.image)
    if image is None:
        raise SystemExit(f"cannot decode {args.image}")
    camera = load_camera(args.camera)
    model = YOLO(str(args.model), task="segment")
    names = model_class_names(args.model, model)
    lane_id, mid_id = class_ids(names)
    started = time.perf_counter()
    result = model.predict(
        image,
        imgsz=list(imgsz),
        batch=1,
        conf=0.25,
        iou=0.70,
        device="cpu",
        retina_masks=False,
        verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - started) * 1000.0
    lane = class_union(result, lane_id)
    mid = class_union(result, mid_id)
    if lane is None or mid is None:
        raise SystemExit("YOLO returned no segmentation masks")
    remapper = BevMaskRemapper(
        camera,
        image.shape[:2],
        imgsz,
        mask_hw=lane.shape,
        supersample=3,
    )
    lane_grid = remapper.apply(lane)
    mid_grid = remapper.apply(mid)
    bev = make_bev_image(mid_grid, lane_grid)

    npz_output = args.output.with_suffix(".npz")
    np.savez_compressed(npz_output, bev=np.transpose(bev, (2, 0, 1)))

    view = np.zeros((bev.shape[0], bev.shape[1], 3), np.uint8)
    view[bev[..., 1] > 0] = (230, 230, 230)
    view[bev[..., 0] > 0] = (0, 220, 255)
    view = cv2.resize(view, (bev.shape[1] * 5, bev.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".png", view)
    if not encoded_ok:
        raise SystemExit(f"failed to encode {args.output}")
    encoded.tofile(str(args.output))
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise SystemExit(f"failed to save {args.output}")
    report = {
        "image": str(args.image),
        "model": str(args.model),
        "imgsz": list(imgsz),
        "native_mask_shape": list(lane.shape),
        "bev_shape": list(bev.shape),
        "lane_cells": int(lane_grid.sum()),
        "mid_cells": int(mid_grid.sum()),
        "inference_ms": round(inference_ms, 3),
        "output": str(args.output),
        "npz_output": str(npz_output),
    }
    if args.reference_geometry_code is not None:
        spec = importlib.util.spec_from_file_location(
            "training_bev_geometry_reference", args.reference_geometry_code
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot import {args.reference_geometry_code}")
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
        reference_camera = reference_module.load_camera(args.camera)
        reference_remapper = reference_module.BevMaskRemapper(
            reference_camera,
            image.shape[:2],
            max(imgsz),
            mask_hw=lane.shape,
            supersample=3,
        )
        reference_lane = reference_remapper.apply(lane)
        reference_mid = reference_remapper.apply(mid)
        report["geometry_code_exact"] = bool(
            np.array_equal(reference_lane, lane_grid)
            and np.array_equal(reference_mid, mid_grid)
        )
        report["geometry_code_changed_cells"] = int(
            np.count_nonzero(reference_lane != lane_grid)
            + np.count_nonzero(reference_mid != mid_grid)
        )
    if args.reference_npz is not None:
        reference = np.load(args.reference_npz)["bev"]
        actual = np.transpose(bev, (2, 0, 1))
        if reference.shape != actual.shape:
            raise SystemExit(
                f"reference shape {reference.shape} != runtime shape {actual.shape}"
            )
        report["reference_exact"] = bool(np.array_equal(reference, actual))
        report["reference_changed_cells"] = int(np.count_nonzero(reference != actual))
        channel_iou = {}
        for channel, name in ((0, "mid"), (1, "lane"), (2, "lidar")):
            left = reference[channel] > 0
            right = actual[channel] > 0
            channel_iou[name] = float((left & right).sum()) / max(
                1, int((left | right).sum())
            )
        report["reference_iou"] = channel_iou
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
