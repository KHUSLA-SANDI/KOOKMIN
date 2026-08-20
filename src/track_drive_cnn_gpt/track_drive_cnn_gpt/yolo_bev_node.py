"""Latest-frame-only YOLO segmentation to full-topology BEV ROS 2 node."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from ultralytics import YOLO

from .bev_geometry import (
    DEFAULT_EXPECTED_MODEL_SHA256,
    GRID_H,
    GRID_W,
    BevMaskRemapper,
    class_ids,
    load_camera,
    make_bev_image,
    model_class_names,
    native_lane_mid_unions,
    normalize_imgsz,
    optional_native_mask_hw,
    remap_optional_native_masks,
    validate_model_contract,
)
from .signal_mission import (
    encode_signal_payload,
    extract_signal_confidences,
    signal_class_ids,
)


class _LatestOnlyBuffer:
    """Thread-safe one-item slot that overwrites unprocessed camera messages."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: Any | None = None
        self._closed = False

    def replace(self, item: Any) -> bool | None:
        """Store item; return overwritten status, or ``None`` after close."""

        with self._condition:
            if self._closed:
                return None
            overwritten = self._pending is not None
            self._pending = item
            self._condition.notify()
            return overwritten

    def take(self) -> Any | None:
        with self._condition:
            self._condition.wait_for(lambda: self._closed or self._pending is not None)
            if self._closed:
                return None
            item = self._pending
            self._pending = None
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()


def _sensor_qos_depth_one() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def _image_to_bgr(message: Image) -> np.ndarray:
    """Decode common ROS Image encodings into an owning, contiguous BGR array."""

    height, width, step = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError(f"invalid image dimensions: {height}x{width}, step={step}")

    encoding = str(message.encoding).strip().lower()
    channels_by_encoding = {
        "mono8": 1,
        "8uc1": 1,
        "bgr8": 3,
        "rgb8": 3,
        "8uc3": 3,
        "bgra8": 4,
        "rgba8": 4,
        "8uc4": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported ROS image encoding: {message.encoding!r}")
    channels = channels_by_encoding[encoding]
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(f"image step {step} is shorter than {row_bytes} bytes")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise ValueError(f"image data has {raw.size} bytes; expected at least {required}")
    pixels = raw[:required].reshape(height, step)[:, :row_bytes]

    if channels == 1:
        mono = pixels.reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    image = pixels.reshape(height, width, channels)
    if encoding in {"bgr8", "8uc3"}:
        # The copy owns the ROS message buffer after this callback returns.
        return np.ascontiguousarray(image).copy()
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding in {"bgra8", "8uc4"}:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


_SIGNAL_COLORS = {
    "GREEN": (40, 220, 40),
    "LEFT": (255, 220, 30),
    "RED": (30, 30, 240),
    "YELLOW": (0, 220, 255),
}


def _as_numpy(value: Any) -> np.ndarray:
    """Convert an Ultralytics/Torch value without retaining its device buffer."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def make_signal_preview(
    bgr: np.ndarray,
    result: Any,
    signal_ids: dict[str, int],
    signals: dict[str, float],
    *,
    sequence: int,
    inference_ms: float,
    output_width: int = 960,
) -> np.ndarray:
    """Draw only the four traffic-signal classes from the shared YOLO result."""

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR HxWx3, got {bgr.shape}")
    width = min(max(320, int(output_width)), int(bgr.shape[1]))
    scale = width / float(bgr.shape[1])
    height = max(1, int(round(float(bgr.shape[0]) * scale)))
    view = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

    id_to_name = {int(class_id): name for name, class_id in signal_ids.items()}
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        xyxy = _as_numpy(boxes.xyxy).reshape(-1, 4)
        classes = _as_numpy(boxes.cls).reshape(-1).astype(np.int64)
        confidences = _as_numpy(boxes.conf).reshape(-1)
        count = min(len(xyxy), len(classes), len(confidences))
        for index in range(count):
            name = id_to_name.get(int(classes[index]))
            if name is None:
                continue
            x1, y1, x2, y2 = np.rint(xyxy[index] * scale).astype(np.int32)
            x1 = int(np.clip(x1, 0, width - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            color = _SIGNAL_COLORS[name]
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{name} {float(confidences[index]):.2f}"
            label_y = max(62, y1 - 7)
            cv2.putText(
                view,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.rectangle(view, (0, 0), (width, 54), (10, 10, 10), -1)
    status_parts = []
    for name in ("GREEN", "LEFT", "RED", "YELLOW"):
        confidence = signals.get(name)
        status_parts.append(
            f"{name}:{confidence:.2f}" if confidence is not None else f"{name}:--"
        )
    cv2.putText(
        view,
        "  ".join(status_parts),
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        f"frame={int(sequence)}  inference={float(inference_ms):.1f}ms  q:quit  s:save",
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return view


class YoloBevNode(Node):
    """Run exactly one ``best_0817`` segmentation model on the newest frame."""

    def __init__(self) -> None:
        super().__init__("yolo_bev_node")

        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("bev_topic", "/perception/bev")
        self.declare_parameter("signal_topic", "/perception/signals")
        self.declare_parameter("diag_topic", "/diagnostics/yolo_bev_timing")
        self.declare_parameter("model_path", "/home/xytron/xycar_ws/models/best_0817.pt")
        self.declare_parameter(
            "expected_model_sha256", DEFAULT_EXPECTED_MODEL_SHA256
        )
        self.declare_parameter("camera_yaml", "")
        self.declare_parameter("imgsz", [384, 640])
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("signal_conf", 0.25)
        self.declare_parameter("iou", 0.70)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("cv_threads", 4)
        self.declare_parameter("supersample", 3)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("publish_signal_preview", False)
        self.declare_parameter(
            "signal_preview_topic", "/debug/yolo_signal_preview/compressed"
        )
        self.declare_parameter("signal_preview_width", 960)
        self.declare_parameter("signal_preview_jpeg_quality", 82)
        self.declare_parameter("diag_period_sec", 2.0)
        self.declare_parameter("bev_frame_id", "lidar_frame")

        self._image_topic = str(self.get_parameter("image_topic").value)
        self._bev_topic = str(self.get_parameter("bev_topic").value)
        self._signal_topic = str(self.get_parameter("signal_topic").value)
        self._diag_topic = str(self.get_parameter("diag_topic").value)
        self._model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        self._expected_model_sha256 = str(
            self.get_parameter("expected_model_sha256").value
        )
        camera_yaml = Path(str(self.get_parameter("camera_yaml").value)).expanduser()
        self._imgsz = normalize_imgsz(self.get_parameter("imgsz").value)
        self._conf = float(self.get_parameter("conf").value)
        self._signal_conf = float(self.get_parameter("signal_conf").value)
        if not 0.0 <= self._signal_conf <= 1.0:
            raise ValueError("signal_conf must be within [0, 1]")
        if self._signal_conf < self._conf:
            raise ValueError(
                "signal_conf cannot be lower than conf because Ultralytics "
                "already removes boxes below conf in the shared inference result"
            )
        self._iou = float(self.get_parameter("iou").value)
        self._device = str(self.get_parameter("device").value)
        self._supersample = int(self.get_parameter("supersample").value)
        self._publish_debug = bool(self.get_parameter("publish_debug").value)
        self._publish_signal_preview = bool(
            self.get_parameter("publish_signal_preview").value
        )
        self._signal_preview_topic = str(
            self.get_parameter("signal_preview_topic").value
        )
        self._signal_preview_width = int(
            self.get_parameter("signal_preview_width").value
        )
        self._signal_preview_jpeg_quality = int(
            self.get_parameter("signal_preview_jpeg_quality").value
        )
        if not 10 <= self._signal_preview_jpeg_quality <= 100:
            raise ValueError("signal_preview_jpeg_quality must be within [10, 100]")
        self._diag_period_ns = int(
            max(0.1, float(self.get_parameter("diag_period_sec").value)) * 1.0e9
        )
        self._bev_frame_id = str(self.get_parameter("bev_frame_id").value)
        cv2.setNumThreads(max(1, int(self.get_parameter("cv_threads").value)))

        self._backend = validate_model_contract(
            self._model_path, self._expected_model_sha256, self._imgsz
        )
        if not camera_yaml.is_file():
            raise FileNotFoundError(f"camera_yaml does not exist: {camera_yaml}")
        self._camera = load_camera(camera_yaml)
        self._expected_image_hw = (
            int(self._camera["image_height"]),
            int(self._camera["image_width"]),
        )

        # Ultralytics accepts either the training .pt or an exported OpenVINO
        # directory.  ndarray inputs must remain BGR; Ultralytics performs the
        # BGR-to-RGB conversion internally.
        self._model = YOLO(str(self._model_path), task="segment")
        names = model_class_names(self._model_path, self._model)
        self._lane_id, self._mid_id = class_ids(names)
        # Signals are extracted from this same Results object; no second model
        # or inference pass is permitted.  Fail closed if a lane-only checkpoint
        # is accidentally deployed instead of the six-class best_0817 model.
        self._signal_ids = signal_class_ids(names)

        self._latest = _LatestOnlyBuffer()
        self._stopping = threading.Event()
        self._stats_lock = threading.Lock()
        self._sequence = 0
        self._received = 0
        self._overwritten = 0
        self._published = 0
        self._errors = 0
        self._remapper: BevMaskRemapper | None = None
        self._remapper_key: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._diag_last_ns = time.monotonic_ns()
        self._diag_samples = 0
        self._diag_infer_sum_ms = 0.0
        self._diag_total_sum_ms = 0.0

        sensor_qos = _sensor_qos_depth_one()
        self._bev_publisher = self.create_publisher(Image, self._bev_topic, sensor_qos)
        self._signal_publisher = self.create_publisher(
            String, self._signal_topic, 1
        )
        self._diag_publisher = (
            self.create_publisher(String, self._diag_topic, 10)
            if self._publish_debug
            else None
        )
        self._signal_preview_publisher = (
            self.create_publisher(
                CompressedImage, self._signal_preview_topic, sensor_qos
            )
            if self._publish_signal_preview
            else None
        )
        self._subscription = self.create_subscription(
            Image, self._image_topic, self._on_image, sensor_qos
        )

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="yolo-bev-latest-only",
            daemon=True,
        )
        self._worker.start()

        self._emit_json(
            {
                "event": "yolo_bev_ready",
                "backend": self._backend,
                "model_path": str(self._model_path),
                "lane_id": self._lane_id,
                "mid_id": self._mid_id,
                "signal_ids": self._signal_ids,
                "signal_topic": self._signal_topic,
                "signal_conf": self._signal_conf,
                "signal_preview_topic": (
                    self._signal_preview_topic
                    if self._signal_preview_publisher is not None
                    else None
                ),
                "imgsz": list(self._imgsz),
                "image_qos": "best_effort_keep_last_1",
            },
            force_log=True,
        )

    def _on_image(self, message: Image) -> None:
        receive_stamp = self.get_clock().now().to_msg()
        receive_mono_ns = time.monotonic_ns()
        actual_hw = (int(message.height), int(message.width))
        if actual_hw != self._expected_image_hw:
            # H/new_K were calibrated at one raw resolution.  Silently scaling
            # another resolution would create a plausible but metrically wrong
            # BEV, so reject it instead of publishing a stale/incorrect result.
            with self._stats_lock:
                self._errors += 1
            self._emit_json(
                {
                    "event": "yolo_bev_resolution_rejected",
                    "expected_hw": list(self._expected_image_hw),
                    "actual_hw": list(actual_hw),
                },
                force_log=True,
            )
            return

        # Keep only a reference to the Python ROS message.  rclpy owns this
        # object's data for as long as the reference lives, so overwritten
        # frames cost no 6.22 MB BGR decode/copy.
        with self._stats_lock:
            self._sequence += 1
            self._received += 1
            sequence = self._sequence
        overwritten = self._latest.replace(
            (sequence, message, receive_stamp, receive_mono_ns)
        )
        if overwritten:
            with self._stats_lock:
                self._overwritten += 1

    def _worker_loop(self) -> None:
        while True:
            item = self._latest.take()
            if item is None:
                return
            sequence, message, receive_stamp, receive_mono_ns = item
            try:
                self._process_frame(
                    sequence, message, receive_stamp, receive_mono_ns
                )
            except Exception as exc:
                if self._stopping.is_set():
                    return
                with self._stats_lock:
                    self._errors += 1
                self._emit_json(
                    {
                        "event": "yolo_bev_processing_error",
                        "sequence": sequence,
                        "error": repr(exc),
                    },
                    force_log=True,
                )

    def _process_frame(
        self,
        sequence: int,
        message: Image,
        receive_stamp: Any,
        receive_mono_ns: int,
    ) -> None:
        process_started_ns = time.monotonic_ns()
        queue_ms = (process_started_ns - receive_mono_ns) / 1.0e6

        decode_started_ns = time.monotonic_ns()
        bgr = _image_to_bgr(message)
        decode_ms = (time.monotonic_ns() - decode_started_ns) / 1.0e6
        del message

        inference_started_ns = time.monotonic_ns()
        results = self._model.predict(
            source=bgr,
            imgsz=list(self._imgsz),
            # Ultralytics 8.3.x creates its OpenVINO AutoBackend before it
            # reads metadata.yaml. Its generic predict default is batch=16,
            # which incorrectly selects CUMULATIVE_THROUGHPUT for our static
            # batch-1 segmentation model and corrupts post-processing.
            batch=1,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            retina_masks=False,
            verbose=False,
        )
        inference_ms = (time.monotonic_ns() - inference_started_ns) / 1.0e6
        if not results:
            raise RuntimeError("YOLO returned no Results object")
        result = results[0]

        signals = extract_signal_confidences(
            result,
            self._signal_ids,
            min_confidence=self._signal_conf,
        )
        signal_message = String()
        signal_message.data = encode_signal_payload(sequence, signals)
        self._signal_publisher.publish(signal_message)

        post_started_ns = time.monotonic_ns()
        lane_native, mid_native = native_lane_mid_unions(
            result, self._lane_id, self._mid_id
        )
        union_ms = (time.monotonic_ns() - post_started_ns) / 1.0e6

        remap_started_ns = time.monotonic_ns()
        mask_hw = optional_native_mask_hw(lane_native, mid_native)
        if mask_hw is not None:
            if self._backend == "openvino" and mask_hw != self._imgsz:
                raise RuntimeError(
                    "static OpenVINO mask shape does not match its manifest: "
                    f"mask={list(mask_hw)}, manifest={list(self._imgsz)}"
                )
            orig_hw = (int(bgr.shape[0]), int(bgr.shape[1]))
            key = (orig_hw, mask_hw)
            if self._remapper is None or self._remapper_key != key:
                self._remapper = BevMaskRemapper(
                    self._camera,
                    orig_hw,
                    self._imgsz,
                    mask_hw=mask_hw,
                    supersample=self._supersample,
                )
                self._remapper_key = key
        lane_grid, mid_grid = remap_optional_native_masks(
            lane_native, mid_native, self._remapper if mask_hw is not None else None
        )
        bev = make_bev_image(mid_grid, lane_grid)
        remap_ms = (time.monotonic_ns() - remap_started_ns) / 1.0e6

        if self._stopping.is_set():
            return
        output = Image()
        # This is capture arrival time, not the potentially offset camera header
        # stamp.  No timer/cache exists, so this BEV can be published only once.
        output.header.stamp = receive_stamp
        output.header.frame_id = self._bev_frame_id
        output.height = GRID_H
        output.width = GRID_W
        output.encoding = "8UC3"
        output.is_bigendian = 0
        output.step = GRID_W * 3
        output.data = np.ascontiguousarray(bev).tobytes()
        self._bev_publisher.publish(output)
        with self._stats_lock:
            self._published += 1

        total_ms = (time.monotonic_ns() - receive_mono_ns) / 1.0e6
        self._record_timing(
            {
                "event": "yolo_bev_timing",
                "sequence": sequence,
                "backend": self._backend,
                "decode_ms": round(decode_ms, 3),
                "queue_ms": round(queue_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "union_ms": round(union_ms, 3),
                "remap_ms": round(remap_ms, 3),
                "total_from_receive_ms": round(total_ms, 3),
                "mid_cells": int(np.count_nonzero(mid_grid)),
                "lane_cells": int(np.count_nonzero(lane_grid)),
                "signals": signals,
            },
            inference_ms,
            total_ms,
        )
        # Debug JPEG is intentionally generated after the control BEV has been
        # published and timed.  It uses the same Results object, so enabling the
        # window never runs a second YOLO inference.
        self._publish_signal_preview_frame(
            bgr,
            result,
            signals,
            sequence=sequence,
            inference_ms=inference_ms,
            receive_stamp=receive_stamp,
        )

    def _publish_signal_preview_frame(
        self,
        bgr: np.ndarray,
        result: Any,
        signals: dict[str, float],
        *,
        sequence: int,
        inference_ms: float,
        receive_stamp: Any,
    ) -> None:
        publisher = self._signal_preview_publisher
        if publisher is None or publisher.get_subscription_count() <= 0:
            return
        view = make_signal_preview(
            bgr,
            result,
            self._signal_ids,
            signals,
            sequence=sequence,
            inference_ms=inference_ms,
            output_width=self._signal_preview_width,
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            view,
            [cv2.IMWRITE_JPEG_QUALITY, self._signal_preview_jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode signal preview JPEG")
        message = CompressedImage()
        message.header.stamp = receive_stamp
        message.header.frame_id = "camera"
        message.format = "jpeg"
        message.data = encoded.tobytes()
        publisher.publish(message)

    def _record_timing(
        self, payload: dict[str, Any], inference_ms: float, total_ms: float
    ) -> None:
        if not self._publish_debug:
            return
        now_ns = time.monotonic_ns()
        with self._stats_lock:
            self._diag_samples += 1
            self._diag_infer_sum_ms += inference_ms
            self._diag_total_sum_ms += total_ms
            if now_ns - self._diag_last_ns < self._diag_period_ns:
                return
            samples = max(1, self._diag_samples)
            payload.update(
                {
                    "window_samples": samples,
                    "mean_inference_ms": round(
                        self._diag_infer_sum_ms / samples, 3
                    ),
                    "mean_total_from_receive_ms": round(
                        self._diag_total_sum_ms / samples, 3
                    ),
                    "frames_received": self._received,
                    "frames_overwritten_before_inference": self._overwritten,
                    "frames_published": self._published,
                    "errors": self._errors,
                }
            )
            self._diag_last_ns = now_ns
            self._diag_samples = 0
            self._diag_infer_sum_ms = 0.0
            self._diag_total_sum_ms = 0.0
        self._emit_json(payload)

    def _emit_json(self, payload: dict[str, Any], *, force_log: bool = False) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if force_log or self._publish_debug:
            self.get_logger().info(encoded)
        if self._diag_publisher is not None:
            message = String()
            message.data = encoded
            self._diag_publisher.publish(message)

    def destroy_node(self) -> bool:
        self._stopping.set()
        self._latest.close()
        if self._worker.is_alive() and threading.current_thread() is not self._worker:
            self._worker.join(timeout=3.0)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: YoloBevNode | None = None
    try:
        node = YoloBevNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
