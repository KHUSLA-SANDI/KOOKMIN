#!/usr/bin/env python3
"""주행 영상 녹화 노드 — 키보드 주행 중 실제 주행과 같은 조건으로 프레임 저장.

역할: /image_raw 를 받아 JPEG 로 저장한다. 시작/정지는 teleop 의 r 키가 보내는
      /record_toggle 로 제어한다. 저장할 때 그 순간의 조향/속도 명령도 같이 기록해
      나중에 "이 그림일 때 이렇게 조향했다" 를 되짚을 수 있게 한다.
입력: /image_raw(sensor QoS), /record_toggle(Bool), /teleop_cmd, /xycar_motor
출력: 파일만 — <out_dir>/<타임스탬프>/{000001.jpg..., frames.csv, meta.yaml, camera.yaml}

★ 왜 teleop 안에 넣지 않고 별도 노드인가
   1080p JPEG 인코딩은 10~20ms 이고 디스크가 튀면 수백 ms 까지 간다. teleop 루프는
   20Hz 로 모터 명령을 계속 내보내야 하는데, 거기서 인코딩을 하면 루프가 밀리는
   순간 차가 직전 명령을 그대로 유지한 채 굴러간다. 그래서 분리했다.
   저장 자체도 워커 스레드로 빼서, 구독 콜백이 디스크를 기다리지 않게 했다.

★ 무엇을 저장하는가 — "실제 주행과 같은 조건"
   /image_raw 를 그대로 저장한다(기본). yolo_node 가 받는 것과 완전히 같은 그림이다.
   왜곡보정(undistort)은 하지 않는다 — 보정은 camera.yaml 만 있으면 나중에 언제든
   똑같이 재현되지만, 보정된 것만 저장하면 원본으로 되돌릴 수 없기 때문이다.
   YOLO 학습용으로 보정본이 필요하면 --undistort 를 쓰거나 나중에 일괄 변환하면 된다
   (yolo_node 는 보정된 프레임을 모델에 넣는다 — yolo_node.py 의 _undist.apply).
   재현에 필요하도록 촬영 당시의 camera.yaml 을 폴더에 같이 복사해 둔다.

사용: ros2 run track_drive recorder
      ros2 run track_drive recorder --ros-args -p rec_fps:=30.0 -p undistort:=true
"""

import os
import queue
import shutil
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32MultiArray

from .lib import bev
from .lib import preprocess

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAMERA_YAML = os.path.join(_PKG_ROOT, "config", "camera.yaml")

# 워커가 밀릴 때 메모리가 무한정 늘지 않게 큐를 막는다. 넘치면 버리되 반드시 센다
# (조용한 드랍 금지 — 몇 장 빠졌는지 모르면 데이터셋을 믿을 수 없다).
DEFAULT_QUEUE = 60

# 디스크 여유가 이보다 적으면 시작 자체를 거부한다 (녹화 중 꽉 차면 최악이다)
MIN_FREE_GB = 2.0


class RecorderNode(Node):
    def __init__(self):
        super().__init__("recorder_node")

        self.declare_parameter("out_dir", os.path.expanduser("~/recordings"))
        # 초당 저장 장수. 0 이면 들어오는 모든 프레임 (약 30fps, 초당 약 12MB).
        # 학습용이면 10 이면 충분하다 — 연속 프레임은 거의 같은 그림이라 값이 없다.
        self.declare_parameter("rec_fps", 10.0)
        # q90 = 장당 366KB, 10fps 로 214MB/분. q95(466KB) 대비 20% 작은데
        # 재압축 평균 화소차가 0.69 라 육안 차이가 없다 (실제 카메라 프레임으로 실측).
        self.declare_parameter("jpeg_quality", 90)
        # 왜곡보정을 적용해 저장할지. 기본 false(원본 저장) — 위 docstring 참고.
        self.declare_parameter("undistort", False)
        self.declare_parameter("camera_yaml", DEFAULT_CAMERA_YAML)
        self.declare_parameter("queue_size", DEFAULT_QUEUE)
        # 시작하자마자 녹화할지 (r 키 없이 단독으로 쓸 때)
        self.declare_parameter("start_recording", False)
        # 저장이 이만큼 연속 실패하면 녹화를 멈춘다 (조용히 빈 파일만 쌓이는 것 방지)
        self.declare_parameter("fail_abort", 20)
        # ★ 라이다도 같이 녹화한다. 장애물이 BEV 에서 어떻게 찍히는지 알아야
        #   합성 장애물을 실제와 비슷하게 만들 수 있다 (지금은 근거가 없다).
        self.declare_parameter("record_scan", True)
        # 사진은 빼고 라이다만 — 장애물 모양만 볼 때 가볍게 쓴다
        self.declare_parameter("scan_only", False)

        g = lambda k: self.get_parameter(k).value
        self.out_root = os.path.expanduser(str(g("out_dir")))
        self.rec_fps = float(g("rec_fps"))
        self.jpeg_q = int(g("jpeg_quality"))
        self.do_undist = bool(g("undistort"))
        self.camera_yaml = str(g("camera_yaml"))
        self.qsize = max(4, int(g("queue_size")))
        self.fail_abort = max(1, int(g("fail_abort")))

        self._min_dt = (1.0 / self.rec_fps) if self.rec_fps > 0 else 0.0

        # ── 녹화 상태 ──
        self._rec = False
        self._dir = None
        self._csv = None
        self._idx = 0
        self._dropped = 0
        self._next_save_t = 0.0    # 다음 저장 예정 시각 (마감 누적 방식 — 아래 설명)
        self._t_start = 0.0
        self._bytes = 0
        self._fail = 0             # 저장 실패 누적 (0바이트 사고 감시)
        self._fail_run = 0         # 연속 실패

        # ── 최근 제어 명령 (프레임과 같이 기록) ──
        self._angle = 0.0
        self._speed = 0.0
        self._estop = 0.0
        self._cmd_src = "-"

        # ── 왜곡보정 (필요할 때만 생성) ──
        self._cam = bev.load_camera_config(self.camera_yaml)
        self._undist = None
        self._undist_size = None

        # ── 저장 워커 ──
        self._q = queue.Queue(maxsize=self.qsize)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._writer, daemon=True)
        self._worker.start()

        # ── 통신 ──
        img_qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE)
        self._rec_scan = bool(self.get_parameter("record_scan").value)
        self._scan_only = bool(self.get_parameter("scan_only").value)
        self._scan_f = None
        self._scan_n = 0
        self._scan_last = None
        if not self._scan_only:
            self.create_subscription(Image, "/image_raw", self._on_image, img_qos)
        if self._rec_scan or self._scan_only:
            self.create_subscription(LaserScan, "/scan", self._on_scan, img_qos)
        self.create_subscription(Bool, "/record_toggle", self._on_toggle, 10)
        self.create_subscription(Float32MultiArray, "/teleop_cmd", self._on_teleop, 10)
        self.create_subscription(Float32MultiArray, "/xycar_motor", self._on_motor, 10)

        self.create_timer(2.0, self._tick_status)

        self.get_logger().info(
            f"recorder 준비 — out={self.out_root} rec_fps={self.rec_fps:g} "
            f"jpeg_q={self.jpeg_q} undistort={self.do_undist} "
            f"(teleop 에서 r 키로 시작/정지)")

        if bool(g("start_recording")):
            self._start()

    # ---------------- 제어 명령 수집 ----------------

    def _on_teleop(self, msg):
        d = list(msg.data)
        if len(d) >= 2:
            self._angle, self._speed = float(d[0]), float(d[1])
            self._estop = float(d[2]) if len(d) >= 3 else 0.0
            self._cmd_src = "teleop"

    def _on_motor(self, msg):
        # /xycar_motor 는 기계 단위(트림 적용 후)다. teleop_cmd 가 있으면 그쪽이 우선.
        d = list(msg.data)
        if len(d) >= 2 and self._cmd_src != "teleop":
            self._angle, self._speed = float(d[0]), float(d[1])
            self._cmd_src = "motor"

    # ---------------- 시작 / 정지 ----------------

    def _on_toggle(self, msg):
        want = bool(msg.data)
        if want and not self._rec:
            self._start()
        elif (not want) and self._rec:
            self._stop_rec()

    def _free_gb(self, path):
        try:
            st = os.statvfs(path)
            return st.f_bavail * st.f_frsize / (1024.0 ** 3)
        except OSError:
            return -1.0

    def _start(self):
        try:
            os.makedirs(self.out_root, exist_ok=True)
        except OSError as e:
            self.get_logger().error(f"녹화 폴더를 만들 수 없다: {e}")
            return

        free = self._free_gb(self.out_root)
        if 0 <= free < MIN_FREE_GB:
            self.get_logger().error(
                f"디스크 여유 {free:.1f}GB — {MIN_FREE_GB:.0f}GB 미만이라 녹화를 시작하지 않는다. "
                "공간을 비우고 다시 시도해라.")
            return

        name = time.strftime("%Y%m%d_%H%M%S")
        self._dir = os.path.join(self.out_root, name)
        try:
            os.makedirs(self._dir, exist_ok=True)
            self._csv = open(os.path.join(self._dir, "frames.csv"), "w",
                             encoding="utf-8")
            self._csv.write("idx,file,stamp_sec,wall_sec,angle,speed,estop,cmd_src\n")
            if self._rec_scan or self._scan_only:
                # 한 줄에 스캔 하나. 거리는 cm 정수로 줄여 파일을 가볍게 한다
                self._scan_f = open(os.path.join(self._dir, "scan.jsonl"), "w",
                                    encoding="utf-8")
                self._scan_n = 0
        except OSError as e:
            self.get_logger().error(f"녹화 파일을 열 수 없다: {e}")
            self._dir = None
            return

        # 재현에 필요하므로 촬영 당시 캘리브를 그대로 복사해 둔다
        try:
            if os.path.isfile(self.camera_yaml):
                shutil.copyfile(self.camera_yaml,
                                os.path.join(self._dir, "camera.yaml"))
        except OSError:
            pass

        self._idx = 0
        self._dropped = 0
        self._bytes = 0
        self._fail = 0
        self._fail_run = 0
        self._next_save_t = 0.0
        self._t_start = time.time()
        self._rec = True
        self.get_logger().info(
            f"● 녹화 시작 → {self._dir}  (여유 {free:.1f}GB)")

    def _on_scan(self, msg):
        """라이다 한 스캔. 녹화 중이면 그대로 한 줄 적는다.

        ranges 를 cm 정수로 저장한다 — 1도 간격 500빔이면 한 줄 3KB 쯤이라
        10Hz 로 몇 분 찍어도 부담이 없다. 무한/음수는 0 으로 둔다.
        """
        self._scan_last = msg
        if not self._rec or self._scan_f is None:
            return
        try:
            import json
            r = np.asarray(msg.ranges, np.float32)
            r = np.where(np.isfinite(r) & (r > 0.0), r, 0.0)
            row = {
                "i": self._scan_n,
                "stamp": float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9,
                "wall": round(time.time(), 3),
                "a0": round(float(msg.angle_min), 6),
                "da": round(float(msg.angle_increment), 8),
                "rmin": round(float(msg.range_min), 3),
                "rmax": round(float(msg.range_max), 3),
                "angle": round(float(self._angle), 1),
                "speed": round(float(self._speed), 1),
                "cm": np.round(r * 100.0).astype(np.int32).tolist(),
            }
            self._scan_f.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._scan_n += 1
        except Exception as e:                       # noqa: BLE001
            self.get_logger().warn(f"스캔 저장 실패: {e}", throttle_duration_sec=5.0)

    def _stop_rec(self):
        self._rec = False
        # 큐에 남은 것을 다 쓸 때까지 잠깐 기다린다 (마지막 몇 장이 날아가지 않게)
        t0 = time.time()
        while not self._q.empty() and time.time() - t0 < 5.0:
            time.sleep(0.05)

        dur = max(1e-6, time.time() - self._t_start)
        if self._csv:
            try:
                self._csv.flush()
                self._csv.close()
            except OSError:
                pass
            self._csv = None
        if self._scan_f:
            try:
                self._scan_f.flush()
                self._scan_f.close()
            except OSError:
                pass
            self._scan_f = None

        self._write_meta(dur)
        saved = self._idx - self._fail
        scan_msg = f"  라이다 {self._scan_n}스캔" if self._scan_n else ""
        msg = (f"■ 녹화 정지 — 저장 {saved}장{scan_msg} / {dur:.1f}초 "
               f"({saved / dur:.1f}fps) / {self._bytes / 1048576.0:.0f}MB "
               f"→ {self._dir}")
        if self._dropped:
            msg += f"  ▲ 디스크가 못 따라가 {self._dropped}장 버림"
        if self._fail:
            msg += f"  ★ 저장 실패 {self._fail}장 (파일 안 남김)"
        self.get_logger().info(msg)
        self._dir = None

    def _write_meta(self, dur):
        if not self._dir:
            return
        size = self._cam.get("image_size") or ("?", "?")
        try:
            with open(os.path.join(self._dir, "meta.yaml"), "w",
                      encoding="utf-8") as f:
                f.write("# 녹화 조건 — 나중에 이 영상이 무엇인지 알기 위한 기록\n")
                f.write(f"frames: {self._idx - self._fail}\n")
                f.write(f"attempted: {self._idx}\n")
                f.write(f"dropped: {self._dropped}\n")
                f.write(f"failed: {self._fail}\n")
                f.write(f"duration_sec: {dur:.2f}\n")
                f.write(f"effective_fps: {self._idx / dur:.2f}\n")
                f.write(f"scans: {self._scan_n}\n")
                f.write(f"rec_fps_setting: {self.rec_fps:g}\n")
                f.write(f"jpeg_quality: {self.jpeg_q}\n")
                f.write(f"undistorted: {str(self.do_undist).lower()}\n")
                f.write(f"image_width: {size[0]}\n")
                f.write(f"image_height: {size[1]}\n")
                f.write(f"bytes: {self._bytes}\n")
                f.write("source_topic: /image_raw\n")
                f.write("# camera.yaml 은 촬영 당시 캘리브를 그대로 복사한 것이다.\n")
                f.write("# undistorted: false 면 원본이므로, 보정본이 필요하면\n")
                f.write("# 이 camera.yaml 로 나중에 똑같이 재현할 수 있다.\n")
        except OSError:
            pass

    # ---------------- 프레임 수신 ----------------

    def _on_image(self, msg):
        if not self._rec:
            return
        now = time.time()
        if self._min_dt > 0.0:
            # ★ 마감 누적 방식. "직전 저장 시각 + min_dt" 로 하면 안 된다 —
            #   기준이 매번 실제(=조금 늦은) 도착 시각으로 리셋돼서, 33ms 간격
            #   입력이 100ms 문턱을 아슬아슬하게 못 넘으면 4프레임씩 건너뛴다.
            #   그러면 10fps 설정인데 7.5fps 만 저장된다(실측으로 확인한 버그).
            #   대신 예정 시각을 min_dt 씩 더해 나가면 장기 평균이 설정값에 맞는다.
            #   eps 는 프레임 간격의 절반 — 도착 지터로 한 프레임 밀리는 것을 막는다.
            eps = self._min_dt * 0.5
            if now < self._next_save_t - eps:
                return
            self._next_save_t = max(now, self._next_save_t) + self._min_dt

        bgr, why = preprocess.image_msg_to_bgr(msg)
        if bgr is None:
            self.get_logger().warn(f"프레임 변환 실패: {why}", throttle_duration_sec=5.0)
            return

        if self.do_undist:
            bgr = self._apply_undistort(bgr)

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        item = (self._idx, bgr, stamp, now,
                self._angle, self._speed, self._estop, self._cmd_src)
        try:
            self._q.put_nowait(item)
            self._idx += 1
        except queue.Full:
            # 조용히 버리지 않는다 — 세고, 주기적으로 경고한다
            self._dropped += 1
            self.get_logger().warn(
                f"디스크가 못 따라간다 — 누적 {self._dropped}장 버림. "
                "rec_fps 를 낮추거나 jpeg_quality 를 내려라.",
                throttle_duration_sec=3.0)

    def _apply_undistort(self, bgr):
        h, w = bgr.shape[:2]
        if self._undist is None or self._undist_size != (w, h):
            self._undist_size = (w, h)
            self._undist = bev.Undistorter.from_config(self._cam, (w, h))
            if not self._undist.enabled:
                self.get_logger().warn(
                    "undistort=true 인데 캘리브가 없어 패스스루로 저장된다 "
                    "(camera.yaml 확인). meta.yaml 의 undistorted 표기와 다를 수 있다.")
        return self._undist.apply(bgr)

    # ---------------- 저장 워커 ----------------

    def _writer(self):
        # 1920x1080 JPEG 은 아무리 단색이어도 이 크기 아래로 안 나온다.
        # 이보다 작으면 인코딩이 사실상 실패한 것으로 본다.
        MIN_JPEG = 2000
        enc = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q]
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            idx, bgr, stamp, wall, angle, speed, estop, src = item
            d = self._dir
            if d is None:
                continue
            fn = "%06d.jpg" % idx
            path = os.path.join(d, fn)
            why = ""
            try:
                if bgr is None or bgr.size == 0:
                    why = "빈 프레임"
                else:
                    ok, buf = cv2.imencode(".jpg", bgr, enc)
                    # ★ ok=True 인데 버퍼가 비어 있는 경우가 실제로 있었다
                    #   (2026-08-05 photo2: 170장이 0바이트로 저장됨).
                    #   원인은 못 찾았지만, 최소한 조용히 넘어가지는 않게 한다.
                    if not ok or buf is None or int(buf.size) < MIN_JPEG:
                        why = "인코딩 결과 %d바이트" % (0 if buf is None else int(buf.size))
                    else:
                        with open(path, "wb") as f:
                            f.write(buf.tobytes())
                            f.flush()
                            os.fsync(f.fileno())
                        wrote = os.path.getsize(path)
                        if wrote < MIN_JPEG:
                            why = "기록된 파일 %d바이트" % wrote
            except (OSError, cv2.error) as e:
                why = "%s: %s" % (type(e).__name__, e)

            if why:
                # 실패한 프레임은 CSV 에도 안 남긴다 (있는 척하면 안 된다)
                self._fail += 1
                self._fail_run += 1
                try:
                    if os.path.exists(path) and os.path.getsize(path) < MIN_JPEG:
                        os.remove(path)          # 0바이트 껍데기를 남기지 않는다
                except OSError:
                    pass
                self.get_logger().error(
                    f"★ 저장 실패 {fn} ({why}) — 누적 {self._fail}장, "
                    f"연속 {self._fail_run}장", throttle_duration_sec=1.0)
                if self._fail_run >= self.fail_abort:
                    self.get_logger().error(
                        f"★★ 연속 {self._fail_run}장 실패 — 녹화를 중단한다. "
                        "디스크/메모리를 확인할 것.")
                    self._rec = False
                continue

            self._fail_run = 0
            self._bytes += os.path.getsize(path)
            if self._csv:
                self._csv.write("%d,%s,%.6f,%.6f,%.2f,%.2f,%.0f,%s\n"
                                % (idx, fn, stamp, wall, angle, speed, estop, src))

    # ---------------- 상태 표시 ----------------

    def _tick_status(self):
        if not self._rec:
            return
        dur = max(1e-6, time.time() - self._t_start)
        free = self._free_gb(self._dir or self.out_root)
        saved = self._idx - self._fail
        self.get_logger().info(
            f"● REC {saved}장 {dur:.0f}초 ({saved / dur:.1f}fps) "
            f"{self._bytes / 1048576.0:.0f}MB  큐{self._q.qsize()}/{self.qsize}"
            + (f"  버림{self._dropped}" if self._dropped else "")
            + (f"  ★실패{self._fail}" if self._fail else "")
            + (f"  여유{free:.1f}GB" if free >= 0 else ""))
        if 0 <= free < 1.0:
            self.get_logger().error("디스크 여유 1GB 미만 — 녹화를 자동 정지한다")
            self._stop_rec()

    def shutdown(self):
        if self._rec:
            self._stop_rec()
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C — 녹화 중이었어도 아래 finally 에서 정상 마감한다(meta.yaml 기록).
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
