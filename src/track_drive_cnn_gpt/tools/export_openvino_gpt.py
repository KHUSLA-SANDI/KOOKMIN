#!/usr/bin/env python3
"""Export the one production YOLO segmentation model to OpenVINO FP32.

Run this in the vehicle-compatible Python environment.  INT8 is intentionally
not offered here: it needs representative calibration data and a separate
mask/path accuracy gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from track_drive_cnn_gpt.bev_geometry import (
    OPENVINO_MANIFEST_SCHEMA,
    normalize_imgsz,
    validate_model_contract,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="production YOLO .pt checkpoint")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=(384, 640),
        metavar=("HEIGHT", "WIDTH"),
        help="static OpenVINO input height and width (default: 384 640)",
    )
    args = parser.parse_args()
    imgsz = normalize_imgsz(args.imgsz)
    if any(value % 32 for value in imgsz):
        raise SystemExit(f"imgsz must be stride-32 aligned: {list(imgsz)}")

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    exported = Path(
        model.export(
            format="openvino",
            imgsz=list(imgsz),
            batch=1,
            dynamic=False,
            half=False,
        )
    ).resolve()
    manifest = {
        "schema_version": OPENVINO_MANIFEST_SCHEMA,
        "source": str(model_path),
        "source_sha256": sha256(model_path),
        "imgsz": list(imgsz),
        "input_shape": [1, 3, imgsz[0], imgsz[1]],
        "dynamic": False,
        "batch": 1,
        "precision": "FP32",
        "task": "segment",
        "classes": model.names,
        "exported": str(exported),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = exported / "export_manifest_gpt.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_model_contract(exported, manifest["source_sha256"], imgsz)
    print(exported)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
