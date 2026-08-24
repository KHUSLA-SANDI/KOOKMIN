"""Latest-frame-only BEV plus conditional traffic-light classification node."""

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
from std_msgs.msg import Bool, String
from ultralytics import YOLO

from .bev_geometry import (
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
    validate_classifier_model_contract,
    validate_model_contract,
)
from .signal_mission import (
    classifier_detections_from_traffic_light,
    crop_normalized_detection,
    encode_signal_payload,
    extract_class_detection,
    extract_classifier_probabilities,
    optional_class_id,
    signal_class_ids,
    TimedTriggerLatch,
)


DETECTOR_SOURCE_SHA256 = (
    "6ead43178acb34a1006c2c9d5b21a73f9d2500a9dcfc8f8ee8229c0c4e7acc59"
)
CLASSIFIER_SOURCE_SHA256 = (
    "2c16877f7fe43231ae08ccbfce6c8a9388d17aeb50a01dc21f6359f925f8e959"
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
    "START_R": (255, 220, 30),
    "traffic_light": (255, 200, 30),
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
    traffic_light_id: int,
    signals: dict[str, float],
    *,
    traffic_detection: dict[str, float] | None = None,
    cone_trigger_id: int | None = None,
    cone_detection: dict[str, float] | None = None,
    sequence: int,
    inference_ms: float,
    output_width: int = 960,
) -> np.ndarray:
    """Draw detector traffic-light/START_R boxes and classifier probabilities."""

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR HxWx3, got {bgr.shape}")
    width = min(max(320, int(output_width)), int(bgr.shape[1]))
    scale = width / float(bgr.shape[1])
    height = max(1, int(round(float(bgr.shape[0]) * scale)))
    view = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

    id_to_name = {int(traffic_light_id): "traffic_light"}
    if cone_trigger_id is not None:
        id_to_name[int(cone_trigger_id)] = "START_R"
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
            source_x1, source_y1, source_x2, source_y2 = np.rint(
                xyxy[index]
            ).astype(np.int32)
            source_cx = int(round(0.5 * (source_x1 + source_x2)))
            source_cy = int(round(0.5 * (source_y1 + source_y2)))
            x1, y1, x2, y2 = np.rint(xyxy[index] * scale).astype(np.int32)
            x1 = int(np.clip(x1, 0, width - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            color = _SIGNAL_COLORS[name]
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = (
                f"{name} {float(confidences[index]):.2f} "
                f"C({source_cx},{source_cy}) "
                f"B({source_x1},{source_y1},{source_x2},{source_y2})"
            )
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
    for name in ("GREEN", "LEFT", "RED", "YELLOW", "START_R"):
        confidence = (
            cone_detection.get("confidence")
            if name == "START_R" and cone_detection is not None
            else signals.get(name)
        )
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
    detector_confidence = (
        None
        if traffic_detection is None
        else float(traffic_detection["confidence"])
    )
    detector_text = (
        f"{detector_confidence:.2f}" if detector_confidence is not None else "--"
    )
    cv2.putText(
        view,
        f"frame={int(sequence)}  detector={float(inference_ms):.1f}ms  "
        f"traffic_det={detector_text}  "
        f"source={int(bgr.shape[1])}x{int(bgr.shape[0])} px  q:quit  s:save",
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return view


class YoloBevNode(Node):
    """Run one detector and conditionally classify its traffic-light crop."""

    def __init__(self) -> None:
        super().__init__("yolo_bev_node")

        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("bev_topic", "/perception/bev")
        self.declare_parameter("signal_topic", "/perception/signals")
        self.declare_parameter("yolo_state_topic", "/debug/yolo_state")
        self.declare_parameter("diag_topic", "/diagnostics/yolo_bev_timing")
        self.declare_parameter(
            "model_path",
            "/home/xytron/xycar_ws/models/traffic_detector_seg_best_openvino_model",
        )
        self.declare_parameter(
            "expected_model_sha256", DETECTOR_SOURCE_SHA256
        )
        self.declare_parameter(
            "classifier_model_path",
            "/home/xytron/xycar_ws/models/yolo11n_cls_best_openvino_model",
        )
        self.declare_parameter(
            "classifier_expected_model_sha256", CLASSIFIER_SOURCE_SHA256
        )
        self.declare_parameter("classifier_imgsz", [320, 320])
        self.declare_parameter("traffic_light_class_name", "traffic_light")
        self.declare_parameter("traffic_light_conf", 0.25)
        self.declare_parameter("traffic_light_crop_padding", 0.12)
        self.declare_parameter("traffic_light_crop_min_px", 8)
        self.declare_parameter("camera_yaml", "")
        self.declare_parameter("imgsz", [384, 640])
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("signal_conf", 0.25)
        self.declare_parameter("cone_mode_topic", "/cone_mode")
        self.declare_parameter(
            "start_r_detected_topic", "/perception/start_r_detected"
        )
        self.declare_parameter(
            "cone_approach_topic", "/perception/cone_approach"
        )
        # Roboflow v3 names the cone-section trigger class START_R.
        self.declare_parameter("cone_trigger_class_name", "START_R")
        self.declare_parameter("cone_trigger_conf", 0.25)
        self.declare_parameter("cone_approach_bottom_y_min", 0.70)
        self.declare_parameter("cone_approach_confirm_frames", 1)
        self.declare_parameter("cone_approach_exit_hold_sec", 1.0)
        self.declare_parameter("cone_enter_bottom_y_min", 0.55)
        self.declare_parameter("cone_enter_confirm_frames", 2)
        self.declare_parameter("cone_exit_hold_sec", 1.0)
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
        self._yolo_state_topic = str(
            self.get_parameter("yolo_state_topic").value
        )
        self._cone_mode_topic = str(self.get_parameter("cone_mode_topic").value)
        self._start_r_detected_topic = str(
            self.get_parameter("start_r_detected_topic").value
        )
        self._cone_approach_topic = str(
            self.get_parameter("cone_approach_topic").value
        )
        self._diag_topic = str(self.get_parameter("diag_topic").value)
        self._model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        self._expected_model_sha256 = str(
            self.get_parameter("expected_model_sha256").value
        )
        self._classifier_model_path = Path(
            str(self.get_parameter("classifier_model_path").value)
        ).expanduser()
        self._classifier_expected_model_sha256 = str(
            self.get_parameter("classifier_expected_model_sha256").value
        )
        camera_yaml = Path(str(self.get_parameter("camera_yaml").value)).expanduser()
        self._imgsz = normalize_imgsz(self.get_parameter("imgsz").value)
        self._classifier_imgsz = normalize_imgsz(
            self.get_parameter("classifier_imgsz").value
        )
        self._conf = float(self.get_parameter("conf").value)
        self._signal_conf = float(self.get_parameter("signal_conf").value)
        self._traffic_light_class_name = str(
            self.get_parameter("traffic_light_class_name").value
        ).strip()
        self._traffic_light_conf = float(
            self.get_parameter("traffic_light_conf").value
        )
        self._traffic_light_crop_padding = float(
            self.get_parameter("traffic_light_crop_padding").value
        )
        self._traffic_light_crop_min_px = int(
            self.get_parameter("traffic_light_crop_min_px").value
        )
        self._cone_trigger_class_name = str(
            self.get_parameter("cone_trigger_class_name").value
        ).strip()
        self._cone_trigger_conf = float(
            self.get_parameter("cone_trigger_conf").value
        )
        self._cone_approach_bottom_y_min = float(
            self.get_parameter("cone_approach_bottom_y_min").value
        )
        self._cone_enter_bottom_y_min = float(
            self.get_parameter("cone_enter_bottom_y_min").value
        )
        if not 0.0 <= self._signal_conf <= 1.0:
            raise ValueError("signal_conf must be within [0, 1]")
        if not self._conf <= self._traffic_light_conf <= 1.0:
            raise ValueError("traffic_light_conf must be within [conf, 1]")
        if not 0.0 <= self._traffic_light_crop_padding <= 1.0:
            raise ValueError("traffic_light_crop_padding must be within [0, 1]")
        if self._traffic_light_crop_min_px < 1:
            raise ValueError("traffic_light_crop_min_px must be positive")
        if not self._conf <= self._cone_trigger_conf <= 1.0:
            raise ValueError("cone_trigger_conf must be within [conf, 1]")
        if not 0.0 <= self._cone_approach_bottom_y_min <= 1.0:
            raise ValueError("cone_approach_bottom_y_min must be within [0, 1]")
        if not 0.0 <= self._cone_enter_bottom_y_min <= 1.0:
            raise ValueError("cone_enter_bottom_y_min must be within [0, 1]")
        if self._cone_approach_bottom_y_min >= self._cone_enter_bottom_y_min:
            raise ValueError(
                "cone_approach_bottom_y_min must be lower than "
                "cone_enter_bottom_y_min"
            )
        self._cone_approach_latch = TimedTriggerLatch(
            enter_confirm_frames=int(
                self.get_parameter("cone_approach_confirm_frames").value
            ),
            exit_hold_sec=float(
                self.get_parameter("cone_approach_exit_hold_sec").value
            ),
        )
        self._cone_mode_latch = TimedTriggerLatch(
            enter_confirm_frames=int(
                self.get_parameter("cone_enter_confirm_frames").value
            ),
            exit_hold_sec=float(self.get_parameter("cone_exit_hold_sec").value),
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
        self._classifier_backend = validate_classifier_model_contract(
            self._classifier_model_path,
            self._classifier_expected_model_sha256,
            self._classifier_imgsz,
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
        self._traffic_light_id = optional_class_id(
            names, self._traffic_light_class_name
        )
        if self._traffic_light_id is None:
            raise ValueError(
                "detector is missing required traffic-light class: "
                f"{self._traffic_light_class_name!r}"
            )
        self._signal_classifier = YOLO(
            str(self._classifier_model_path), task="classify"
        )
        classifier_names = model_class_names(
            self._classifier_model_path, self._signal_classifier
        )
        self._signal_ids = signal_class_ids(classifier_names)
        self._cone_trigger_id = optional_class_id(
            names, self._cone_trigger_class_name
        )
        if self._cone_trigger_id is None:
            self.get_logger().warning(
                "cone trigger class is not in this model; /cone_mode remains false: "
                f"name={self._cone_trigger_class_name!r}"
            )
        self._warm_models()

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
        # This is a small viewer-only JSON stream.  The publisher exists so a
        # dashboard can attach later, but serialization is skipped entirely
        # while there is no subscriber.
        self._yolo_state_publisher = self.create_publisher(
            String, self._yolo_state_topic, 1
        )
        self._cone_mode_publisher = self.create_publisher(
            Bool, self._cone_mode_topic, 1
        )
        self._start_r_detected_publisher = self.create_publisher(
            Bool, self._start_r_detected_topic, 1
        )
        self._cone_approach_publisher = self.create_publisher(
            Bool, self._cone_approach_topic, 1
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
                "classifier_backend": self._classifier_backend,
                "classifier_model_path": str(self._classifier_model_path),
                "lane_id": self._lane_id,
                "mid_id": self._mid_id,
                "traffic_light_id": self._traffic_light_id,
                "signal_ids": self._signal_ids,
                "signal_topic": self._signal_topic,
                "yolo_state_topic": self._yolo_state_topic,
                "signal_conf": self._signal_conf,
                "cone_trigger_class_name": self._cone_trigger_class_name,
                "cone_trigger_id": self._cone_trigger_id,
                "cone_mode_topic": self._cone_mode_topic,
                "cone_approach_topic": self._cone_approach_topic,
                "cone_approach_bottom_y_min": self._cone_approach_bottom_y_min,
                "cone_enter_bottom_y_min": self._cone_enter_bottom_y_min,
                "signal_preview_topic": (
                    self._signal_preview_topic
                    if self._signal_preview_publisher is not None
                    else None
                ),
                "imgsz": list(self._imgsz),
                "classifier_imgsz": list(self._classifier_imgsz),
                "image_qos": "best_effort_keep_last_1",
            },
            force_log=True,
        )

    def _warm_models(self) -> None:
        """Compile both static OpenVINO engines before the first camera frame."""

        detector_dummy = np.zeros((self._imgsz[0], self._imgsz[1], 3), np.uint8)
        classifier_dummy = np.zeros(
            (self._classifier_imgsz[0], self._classifier_imgsz[1], 3), np.uint8
        )
        detector_results = self._model.predict(
            source=detector_dummy,
            imgsz=list(self._imgsz),
            batch=1,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            retina_masks=False,
            verbose=False,
        )
        classifier_results = self._signal_classifier.predict(
            source=classifier_dummy,
            imgsz=list(self._classifier_imgsz),
            batch=1,
            device=self._device,
            verbose=False,
        )
        if not detector_results or not classifier_results:
            raise RuntimeError("detector/classifier warm-up returned no result")

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

        traffic_detection = extract_class_detection(
            result,
            self._traffic_light_id,
            image_hw=(int(bgr.shape[0]), int(bgr.shape[1])),
            min_confidence=self._traffic_light_conf,
        )
        signals: dict[str, float] = {}
        signal_detections: dict[str, dict[str, float]] = {}
        classifier_ms = 0.0
        classifier_ran = False
        if traffic_detection is not None:
            try:
                traffic_crop, _ = crop_normalized_detection(
                    bgr,
                    traffic_detection,
                    padding_ratio=self._traffic_light_crop_padding,
                    min_size_px=self._traffic_light_crop_min_px,
                )
                classifier_started_ns = time.monotonic_ns()
                classifier_results = self._signal_classifier.predict(
                    source=traffic_crop,
                    imgsz=list(self._classifier_imgsz),
                    batch=1,
                    device=self._device,
                    verbose=False,
                )
                classifier_ms = (
                    time.monotonic_ns() - classifier_started_ns
                ) / 1.0e6
                if not classifier_results:
                    raise RuntimeError("traffic classifier returned no result")
                signals = extract_classifier_probabilities(
                    classifier_results[0], self._signal_ids
                )
                signal_detections = classifier_detections_from_traffic_light(
                    signals, traffic_detection
                )
                classifier_ran = True
            except Exception as exc:
                # Lane/mid BEV and START_R must continue even when a bad crop or
                # classifier failure temporarily removes traffic decisions.
                self.get_logger().warning(
                    f"traffic-light classification skipped: {exc}",
                    throttle_duration_sec=1.0,
                )
        cone_detection = extract_class_detection(
            result,
            self._cone_trigger_id,
            image_hw=(int(bgr.shape[0]), int(bgr.shape[1])),
            min_confidence=self._cone_trigger_conf,
        )
        # Raw START_R visibility is intentionally independent of the y2 entry
        # threshold.  CNN mode logic uses it to suppress cone-shaped LiDAR
        # clusters from starting OVERTAKE while approaching the cone section.
        self._start_r_detected_publisher.publish(
            Bool(data=cone_detection is not None)
        )
        cone_approach_ready = bool(
            cone_detection is not None
            and cone_detection["y2_norm"] >= self._cone_approach_bottom_y_min
        )
        now_sec = time.monotonic()
        approach_update = self._cone_approach_latch.observe(
            cone_approach_ready,
            now_sec=now_sec,
        )
        self._cone_approach_publisher.publish(Bool(data=approach_update.active))
        if approach_update.changed:
            self.get_logger().info(
                "START_R approach speed limit changed: %s"
                % ("ACTIVE" if approach_update.active else "released")
            )
        cone_enter_ready = bool(
            cone_detection is not None
            and cone_detection["y2_norm"] >= self._cone_enter_bottom_y_min
        )
        cone_update = self._cone_mode_latch.observe(
            cone_enter_ready,
            now_sec=now_sec,
        )
        self._cone_mode_publisher.publish(Bool(data=cone_update.active))
        if cone_update.changed:
            self.get_logger().info(
                "motion mode trigger changed: %s"
                % ("CONE" if cone_update.active else "NORMAL")
            )

        signal_message = String()
        signal_message.data = encode_signal_payload(
            sequence,
            signals,
            signal_detections,
        )
        self._signal_publisher.publish(signal_message)
        self._publish_yolo_state(
            sequence=sequence,
            image_width=int(bgr.shape[1]),
            image_height=int(bgr.shape[0]),
            traffic_detection=traffic_detection,
            classifier_ran=classifier_ran,
            cone_detection=cone_detection,
            approach_ready=cone_approach_ready,
            approach_latch_active=approach_update.active,
            enter_ready=cone_enter_ready,
            latch_active=cone_update.active,
        )

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
                "detector_ms": round(inference_ms, 3),
                "classifier_ms": round(classifier_ms, 3),
                "classifier_ran": classifier_ran,
                "union_ms": round(union_ms, 3),
                "remap_ms": round(remap_ms, 3),
                "total_from_receive_ms": round(total_ms, 3),
                "mid_cells": int(np.count_nonzero(mid_grid)),
                "lane_cells": int(np.count_nonzero(lane_grid)),
                "signals": signals,
                "traffic_light": traffic_detection,
                "cone_approach": approach_update.active,
                "cone_approach_ready": cone_approach_ready,
                "cone_mode": cone_update.active,
                "cone_trigger_ready": cone_enter_ready,
                "cone_trigger": cone_detection,
            },
            inference_ms + classifier_ms,
            total_ms,
        )
        # Debug JPEG is intentionally generated after the control BEV has been
        # published and timed.  It uses the same Results object, so enabling the
        # window never runs a second YOLO inference.
        self._publish_signal_preview_frame(
            bgr,
            result,
            signals,
            traffic_detection=traffic_detection,
            cone_detection=cone_detection,
            sequence=sequence,
            inference_ms=inference_ms,
            receive_stamp=receive_stamp,
        )

    def _publish_yolo_state(
        self,
        *,
        sequence: int,
        image_width: int,
        image_height: int,
        traffic_detection: dict[str, float] | None,
        classifier_ran: bool,
        cone_detection: dict[str, float] | None,
        approach_ready: bool,
        approach_latch_active: bool,
        enter_ready: bool,
        latch_active: bool,
    ) -> None:
        """Publish viewer-only detector geometry without another inference."""

        publisher = self._yolo_state_publisher
        if publisher.get_subscription_count() <= 0:
            return
        y2_norm = (
            None if cone_detection is None else float(cone_detection["y2_norm"])
        )
        confirm_frames = int(self._cone_mode_latch.enter_confirm_frames)
        confirm_count = (
            confirm_frames
            if latch_active
            else int(self._cone_mode_latch.enter_streak)
        )
        approach_confirm_frames = int(
            self._cone_approach_latch.enter_confirm_frames
        )
        approach_confirm_count = (
            approach_confirm_frames
            if approach_latch_active
            else int(self._cone_approach_latch.enter_streak)
        )
        payload = {
            "schema_version": "yolo_state_v1_gpt",
            "sequence": int(sequence),
            "traffic_light": {
                "detected": traffic_detection is not None,
                "classifier_ran": bool(classifier_ran),
                "confidence": (
                    None
                    if traffic_detection is None
                    else float(traffic_detection["confidence"])
                ),
                "center_x_norm": (
                    None
                    if traffic_detection is None
                    else float(traffic_detection["center_x_norm"])
                ),
                "center_y_norm": (
                    None
                    if traffic_detection is None
                    else float(traffic_detection["center_y_norm"])
                ),
                "width_norm": (
                    None
                    if traffic_detection is None
                    else float(traffic_detection["width_norm"])
                ),
                "height_norm": (
                    None
                    if traffic_detection is None
                    else float(traffic_detection["height_norm"])
                ),
                "center_x_px": (
                    None
                    if traffic_detection is None
                    else float(traffic_detection["center_x_norm"]) * image_width
                ),
                "center_y_px": (
                    None
                    if traffic_detection is None
                    else float(traffic_detection["center_y_norm"]) * image_height
                ),
            },
            "start_r": {
                "class_name": self._cone_trigger_class_name,
                "detected": cone_detection is not None,
                "confidence": (
                    None
                    if cone_detection is None
                    else float(cone_detection["confidence"])
                ),
                "y2_norm": y2_norm,
                "y2_px": None if y2_norm is None else y2_norm * image_height,
                "approach_threshold_norm": float(
                    self._cone_approach_bottom_y_min
                ),
                "approach_threshold_px": (
                    float(self._cone_approach_bottom_y_min) * image_height
                ),
                "approach_ready": bool(approach_ready),
                "approach_confirm_count": approach_confirm_count,
                "approach_confirm_frames": approach_confirm_frames,
                "approach_latch_active": bool(approach_latch_active),
                "threshold_norm": float(self._cone_enter_bottom_y_min),
                "threshold_px": float(self._cone_enter_bottom_y_min) * image_height,
                "enter_ready": bool(enter_ready),
                "confirm_count": confirm_count,
                "confirm_frames": confirm_frames,
                "latch_active": bool(latch_active),
            },
        }
        publisher.publish(
            String(
                data=json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        )

    def _publish_signal_preview_frame(
        self,
        bgr: np.ndarray,
        result: Any,
        signals: dict[str, float],
        *,
        traffic_detection: dict[str, float] | None,
        cone_detection: dict[str, float] | None,
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
            self._traffic_light_id,
            signals,
            traffic_detection=traffic_detection,
            cone_trigger_id=self._cone_trigger_id,
            cone_detection=cone_detection,
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
