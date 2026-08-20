#!/usr/bin/env python3
"""Record end-to-end replay diagnostics and both CNN paths as JSONL."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String

from .replay_io import percentile


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class ReplayRecorderNode(Node):
    """Write machine-readable traces plus a compact latency/rate summary."""

    def __init__(self) -> None:
        super().__init__("replay_recorder_node")
        p = self._parameter
        requested = Path(str(p("output_path", "/tmp/track_drive_replay_gpt.jsonl")))
        overwrite = bool(p("overwrite", False))
        self._record_points = bool(p("record_path_points", True))
        self._log_period_sec = float(p("log_period_sec", 5.0))
        if self._log_period_sec <= 0.0:
            raise ValueError("log_period_sec must be positive")
        self._output_path = self._available_path(requested.expanduser(), overwrite)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary_path = self._output_path.with_name(
            self._output_path.stem + "_summary.json"
        )
        self._stream = self._output_path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._started_wall = time.time()
        self._started_mono = time.monotonic()
        self._counts: dict[str, int] = {}
        self._main_stamps: list[int] = []
        self._shortcut_stamps: list[int] = []
        self._main_valid_stamps: list[int] = []
        self._shortcut_valid_stamps: list[int] = []
        self._selected_stamps: list[int] = []
        self._selected_valid_stamps: list[int] = []
        self._route_intent = "unknown"
        self._route_intent_counts: dict[str, int] = {}
        self._route_intent_messages = 0
        self._route_intent_transitions = 0
        self._yolo_inference_ms: list[float] = []
        self._yolo_total_ms: list[float] = []
        self._cnn_total_ms: list[float] = []
        self._cnn_ok = 0
        self._cnn_reasons: dict[str, int] = {}

        self.create_subscription(PoseArray, "/cnn/path_main", self._on_main, 10)
        self.create_subscription(
            PoseArray, "/cnn/path_shortcut", self._on_shortcut, 10
        )
        self.create_subscription(PoseArray, "/center_path", self._on_selected, 10)
        self.create_subscription(String, "/route_intent", self._on_route_intent, 10)
        self.create_subscription(
            String, "/diagnostics/yolo_bev_timing", self._on_yolo_diag, 10
        )
        self.create_subscription(String, "/debug/cnn_path", self._on_cnn_diag, 10)
        self.create_subscription(String, "/replay/status", self._on_replay_status, 10)
        self._log_timer = self.create_timer(self._log_period_sec, self._log_progress)
        self._write(
            "recorder_started",
            output_path=str(self._output_path),
            summary_path=str(self._summary_path),
        )
        self.get_logger().info(f"Replay trace: {self._output_path}")

    def _parameter(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    @staticmethod
    def _available_path(requested: Path, overwrite: bool) -> Path:
        if overwrite or not requested.exists():
            return requested
        suffix = time.strftime("_%Y%m%d_%H%M%S")
        return requested.with_name(requested.stem + suffix + requested.suffix)

    def _write(self, event: str, **payload) -> None:
        row = {
            "event": event,
            "recorded_wall_time": time.time(),
            "elapsed_sec": round(time.monotonic() - self._started_mono, 6),
            **payload,
        }
        with self._lock:
            self._stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self._counts[event] = self._counts.get(event, 0) + 1

    def _path_payload(self, message: PoseArray) -> dict:
        payload = {
            "stamp_ns": _stamp_ns(message.header),
            "frame_id": message.header.frame_id,
            "point_count": len(message.poses),
        }
        if self._record_points:
            payload["path_xy"] = [
                [round(float(pose.position.x), 6), round(float(pose.position.y), 6)]
                for pose in message.poses
            ]
        return payload

    def _on_main(self, message: PoseArray) -> None:
        stamp = _stamp_ns(message.header)
        self._main_stamps.append(stamp)
        if message.poses:
            self._main_valid_stamps.append(stamp)
        self._write("path_main", **self._path_payload(message))

    def _on_shortcut(self, message: PoseArray) -> None:
        stamp = _stamp_ns(message.header)
        self._shortcut_stamps.append(stamp)
        if message.poses:
            self._shortcut_valid_stamps.append(stamp)
        self._write("path_shortcut", **self._path_payload(message))

    def _on_selected(self, message: PoseArray) -> None:
        stamp = _stamp_ns(message.header)
        self._selected_stamps.append(stamp)
        if message.poses:
            self._selected_valid_stamps.append(stamp)
        self._write(
            "path_selected",
            route_intent=self._route_intent,
            **self._path_payload(message),
        )

    def _on_route_intent(self, message: String) -> None:
        intent = str(message.data).strip().casefold() or "unknown"
        previous = self._route_intent
        changed = intent != previous
        if changed and previous != "unknown":
            self._route_intent_transitions += 1
        self._route_intent = intent
        self._route_intent_messages += 1
        self._route_intent_counts[intent] = self._route_intent_counts.get(intent, 0) + 1
        self._write(
            "route_intent",
            route_intent=intent,
            previous_route_intent=previous,
            changed=changed,
        )

    @staticmethod
    def _decode_payload(message: String) -> dict:
        try:
            decoded = json.loads(message.data)
        except (TypeError, ValueError):
            return {"raw": message.data, "parse_error": True}
        return decoded if isinstance(decoded, dict) else {"raw": decoded}

    def _on_yolo_diag(self, message: String) -> None:
        payload = self._decode_payload(message)
        # Periodic YOLO diagnostics include the mean over every frame in the
        # window.  Prefer that value over the final frame's sample so replay
        # summaries do not accidentally describe only one frame per window.
        for mean_key, sample_key, target in (
            ("mean_inference_ms", "inference_ms", self._yolo_inference_ms),
            (
                "mean_total_from_receive_ms",
                "total_from_receive_ms",
                self._yolo_total_ms,
            ),
        ):
            value = payload.get(mean_key, payload.get(sample_key))
            if isinstance(value, (int, float)):
                target.append(float(value))
        self._write("yolo_diag", payload=payload)

    def _on_cnn_diag(self, message: String) -> None:
        payload = self._decode_payload(message)
        value = payload.get("cnn_total_ms")
        if isinstance(value, (int, float)):
            self._cnn_total_ms.append(float(value))
        if payload.get("ok") is True:
            self._cnn_ok += 1
        else:
            reason = str(payload.get("reason", "unknown"))
            self._cnn_reasons[reason] = self._cnn_reasons.get(reason, 0) + 1
        self._write("cnn_diag", payload=payload)

    def _on_replay_status(self, message: String) -> None:
        self._write("replay_status", payload=self._decode_payload(message))

    @staticmethod
    def _rate(stamps: list[int]) -> float | None:
        usable = sorted(set(stamp for stamp in stamps if stamp > 0))
        if len(usable) < 2 or usable[-1] <= usable[0]:
            return None
        return (len(usable) - 1) * 1.0e9 / (usable[-1] - usable[0])

    @staticmethod
    def _latency(values: list[float]) -> dict:
        if not values:
            return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
        return {
            "samples": len(values),
            "mean_ms": round(sum(values) / len(values), 3),
            "p50_ms": round(percentile(values, 0.50), 3),
            "p95_ms": round(percentile(values, 0.95), 3),
        }

    def _summary(self) -> dict:
        return {
            "schema": "track_drive_replay_summary_v2_gpt",
            "output_path": str(self._output_path),
            "started_wall_time": self._started_wall,
            "duration_sec": round(time.monotonic() - self._started_mono, 3),
            "event_counts": dict(self._counts),
            "valid_path_counts": {
                "main": len(self._main_valid_stamps),
                "shortcut": len(self._shortcut_valid_stamps),
                "selected": len(self._selected_valid_stamps),
            },
            # Message rate includes explicit empty paths used for fail-closed
            # STOP.  Valid-path rate counts only non-empty CNN outputs.
            "main_path_hz": self._rate(self._main_stamps),
            "shortcut_path_hz": self._rate(self._shortcut_stamps),
            "selected_path_hz": self._rate(self._selected_stamps),
            "main_valid_path_hz": self._rate(self._main_valid_stamps),
            "shortcut_valid_path_hz": self._rate(self._shortcut_valid_stamps),
            "selected_valid_path_hz": self._rate(self._selected_valid_stamps),
            "route_intent": {
                "last": self._route_intent,
                "messages": self._route_intent_messages,
                "transitions": self._route_intent_transitions,
                "counts": dict(self._route_intent_counts),
            },
            "yolo_inference": self._latency(self._yolo_inference_ms),
            "yolo_receive_to_bev": self._latency(self._yolo_total_ms),
            "cnn_total": self._latency(self._cnn_total_ms),
            "cnn_ok": self._cnn_ok,
            "cnn_fail_reasons": dict(self._cnn_reasons),
        }

    def _log_progress(self) -> None:
        summary = self._summary()
        self.get_logger().info(
            "replay metrics "
            f"main={summary['event_counts'].get('path_main', 0)} "
            f"selected={summary['event_counts'].get('path_selected', 0)} "
            f"route={summary['route_intent']['last']} "
            f"cnn_ok={summary['cnn_ok']} "
            f"path_hz={summary['main_path_hz']}"
        )

    def destroy_node(self) -> bool:
        if not self._stream.closed:
            summary = self._summary()
            self._write("recorder_stopped", summary=summary)
            self._stream.flush()
            self._summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._stream.close()
            self.get_logger().info(f"Replay summary: {self._summary_path}")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ReplayRecorderNode()
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
