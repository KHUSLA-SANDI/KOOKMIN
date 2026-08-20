#!/usr/bin/env python3
"""yolo_node — 예선 YOLO 3노드(yolo_lane/yolo_detect/yolo_light) 통합 (실차용).

역할: /image_raw 1회 수신 → 3모델(lane.pt seg / best.pt detect / light.pt) 순차 추론.
      CPU 전용(imgsz416 순차 ≈11.5Hz 실측)이라 백로그 금지 — 처리 중 프레임은 드랍.
입력: /image_raw (rgb8(yuyv2rgb) 640x480 ~29.8Hz, sensor_data QoS depth1)
출력: /detect/lane          PoseArray, 라이다 좌표(x전방/y좌, m), z=클래스ID(8=MID)
      /detect/lane_quality  Float32 0~1 등급형 (점 개수·x스팬 기반)
      /detect/objects       PoseArray, x=bbox중심 u(px), y=bbox하단 v(px), z=클래스ID
      /detect/traffic_light String "GREEN"|"RED"|"NONE" (N프레임 연속 디바운스 후)
예선 대비 변경점:
  - 3노드 → 1노드: 이미지 1회 수신으로 3모델 순차 추론 (콜백 1회 안에서).
  - encoding 하드코딩(rgb8 아니면 무음 드랍) → lib/preprocess.image_msg_to_bgr
    (실패 사유 5초 스로틀 경고 — 재도입 금지 버그 8 대응).
  - config/camera.yaml 캘리브 로드 + Undistorter (차선 처리 전 적용).
    렌즈는 실측 HFOV≈87.2°/VFOV≈64.1° 의 일반 광각(plumb_bob) — 어안 아님.
    왜곡모델/new_K/alpha 는 전부 yaml 을 따른다 (bev.Undistorter.from_config).
  - /detect/road_pixels + /detect/events_raw → /detect/objects 통합.
    road_pixels 의 (u중심, v하단) 인코딩 계승, events_raw 의 x,y=0 방식 폐기
    (신호/표지 클래스도 u,v 를 채워 발행 — 거리 판단은 planner 몫).
  - 신호등: 매 프레임 발행 대신 N프레임 연속 동일 판정 디바운스 후 상태 발행.
"""

import os
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32, String

from ultralytics import YOLO

from .lib import bev, preprocess

# ======================== 상수 ========================

# 예선 방식의 소스상대 경로 (.../track_drive/config)
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_LANE_MODEL = os.path.join(_PKG_ROOT, "config", "lane.pt")
DEFAULT_OBJ_MODEL = os.path.join(_PKG_ROOT, "config", "best.pt")
DEFAULT_LIGHT_MODEL = os.path.join(_PKG_ROOT, "config", "light.pt")
DEFAULT_CAMERA_YAML = os.path.join(_PKG_ROOT, "config", "camera.yaml")

LIDAR_FRAME = "lidar_frame"        # 예선 common.LIDAR_FRAME 과 동일
CAM_FRONT_FRAME = "usb_cam_front"  # 예선 yolo_detect_node 와 동일

# camera.yaml 에 image_width/height 가 없을 때 쓰는 기본 해상도 (usb_cam 실측 640x480)
DEFAULT_IMAGE_SIZE = (640, 480)

# ── 모델별 입력 색순서 정책 (근거: 예선 yolo_light_node.py:64-68 주석) ──
# ultralytics 는 numpy 입력을 BGR 로 가정한다.
#   lane.pt / best.pt: 예선에서 rgb8 배열을 스왑 없이 그대로 넣은 채 학습/검증된
#                      비표준 모델 → 모델이 기대하는 배열 채널순서는 RGB.
#   light.pt: 표준 색순서로 학습 → BGR 그대로 (예선 swap_rb=True 와 등가).
# 본 파이프라인 기준색은 BGR(preprocess.image_msg_to_bgr)이므로
# RGB 를 기대하는 모델만 채널을 되스왑해서 넣는다.
# TODO: 재학습 시 전 모델 표준(BGR)으로 통일하고 이 테이블 제거.
MODEL_WANTS_RGB = {"lane": True, "obj": True, "light": False}

# ── lane.pt (segmentation) 클래스 → 발행 z ──
SEG_MID = 8          # 중앙선 → z=8 (예선 /detect/lane 관습 그대로)
SEG_CHILD_DASH = 2   # 어린이보호 점선 → z=16 (클래스 id 는 lane.pt 재확인 필요)
Z_MID = 8.0
Z_CHILD_DASH = 16.0

# ── best.pt (detect) 클래스 (예선 학습 시점 매핑) ──
#   0=BLACK_CAR  1=CHILD_END  2=CHILD_LANE  3=CHILD_START
#   4=CROSSROAD_OUT  5=GOAL  6=GREEN  7=GREEN_CAR  8=HUMAN
#   9=LANE  10=LEFT  11=MID  12=POLICE  13=RED  14=STOP  15=YELLOW
# 차선 클래스는 lane.pt(seg)가 담당하므로 무시. 나머지는 전부 u,v 채워 발행.
OBJ_IGNORE_CLASS_IDS = {2, 9, 11}  # CHILD_LANE, LANE, MID

# ── light.pt 클래스 ──
LIGHT_GREEN_CLS = 0
LIGHT_LEFT_CLS = 1   # 좌회전 화살표 — 미사용 (LEFT 는 best.pt cls 10 이 담당)
LIGHT_RED_CLS = 2


# ======================== ROS 노드 ========================

class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        # ── 파라미터 ──
        self.declare_parameter("imgsz", 416)          # [SIM=640] CPU 3모델 순차 ≈11.5Hz 실측
        self.declare_parameter("conf_lane", 0.25)     # [SIM]
        self.declare_parameter("conf_obj", 0.25)      # [SIM]
        self.declare_parameter("conf_light", 0.4)     # [SIM]
        self.declare_parameter("light_debounce_n", 3)     # 신호등 연속 동일판정 프레임 수
        self.declare_parameter("enable_lane", True)
        self.declare_parameter("enable_obj", True)
        self.declare_parameter("enable_light", True)
        self.declare_parameter("enable_child_dash", False)  # dash(z=16) 발행 루프만 준비
        self.declare_parameter("device", "cpu")       # 실차 CPU 전용 (CUDA 없음)
        self.declare_parameter("lane_model_path", DEFAULT_LANE_MODEL)
        self.declare_parameter("obj_model_path", DEFAULT_OBJ_MODEL)
        self.declare_parameter("light_model_path", DEFAULT_LIGHT_MODEL)
        self.declare_parameter("camera_yaml", DEFAULT_CAMERA_YAML)
        # undistort 의 getOptimalNewCameraMatrix alpha (plumb_bob 전용, 0=크롭/1=화소보존).
        # ★ H 캘리브(tools/calib_ground_h.py) 때 쓴 alpha 와 반드시 같아야 한다 —
        #   alpha 가 다르면 new_K 가 달라져 H 가 통째로 무효다.
        self.declare_parameter("undistort_alpha", bev.UNDISTORT_ALPHA_DEFAULT)

        self._imgsz = int(self.get_parameter("imgsz").value)
        self._conf_lane = float(self.get_parameter("conf_lane").value)
        self._conf_obj = float(self.get_parameter("conf_obj").value)
        self._conf_light = float(self.get_parameter("conf_light").value)
        self._debounce_n = max(1, int(self.get_parameter("light_debounce_n").value))
        self._device = str(self.get_parameter("device").value)

        # ── 카메라 캘리브 (K/D: 렌즈 내부·왜곡, H: 이미지픽셀→라이다미터) ──
        # K/D 유효성(intrinsics_calibrated)과 H 유효성(h_calibrated)은 별개다.
        # K/D 만 잡힌 중간 상태에서도 undistort 는 켜져야 한다.
        camera_yaml = str(self.get_parameter("camera_yaml").value)
        self._cam = bev.load_camera_config(camera_yaml)
        self._undist_alpha = float(self.get_parameter("undistort_alpha").value)
        # 캘리브 기준 해상도로 미리 리맵 테이블 생성 → 기동 로그에서 상태 확정 가능.
        # 실제 프레임 크기가 다르면 첫 프레임에서 그 크기로 다시 만든다.
        self._undist_size = tuple(self._cam["image_size"] or DEFAULT_IMAGE_SIZE)
        self._undist = bev.Undistorter.from_config(
            self._cam, self._undist_size, alpha=self._undist_alpha)
        # 차체 가림선 — 이 v 아래는 라이다 원통/범퍼라 지면이 아니다. 안 자르면
        # 차체 화소가 BEV 에서 x≈0.1~0.2m 의 가짜 차선점으로 들어온다.
        self._car_mask_v = self._cam["car_mask_v"]
        # ★ 차선을 원본으로 볼지 (기본 켬). 끄면 예전처럼 보정영상을 본다.
        self.declare_parameter("lane_on_raw", True)
        self._lane_on_raw = bool(self.get_parameter("lane_on_raw").value)
        self._new_K_cfg = None
        self._cm_rows_raw = None
        self._log_camera_summary(camera_yaml)

        # ── 모델 로드 (스위치 꺼진 모델은 로드 자체를 생략 — CPU/메모리 절약) ──
        self._model_lane = (self._load_model("lane", str(self.get_parameter("lane_model_path").value))
                            if bool(self.get_parameter("enable_lane").value) else None)
        self._model_obj = (self._load_model("obj", str(self.get_parameter("obj_model_path").value))
                           if bool(self.get_parameter("enable_obj").value) else None)
        self._model_light = (self._load_model("light", str(self.get_parameter("light_model_path").value))
                             if bool(self.get_parameter("enable_light").value) else None)
        # ★ 중앙선 클래스 id 를 모델에서 직접 찾는다.
        #   예전엔 SEG_MID=8 이 박혀 있었는데 그건 lane.pt(12클래스, MID=8) 기준이다.
        #   재학습본은 클래스 수가 달라 id 가 바뀐다 (2026-08 재학습본은 mid=2).
        #   상수를 그대로 두고 모델만 바꾸면 "존재하지 않는 클래스"를 찾게 되어
        #   에러 없이 검출 0개로 조용히 굴러간다 — 그 사고를 막는다.
        self._seg_mid = self._resolve_seg_cls(self._model_lane, "mid", SEG_MID)
        self._seg_dash = self._resolve_seg_cls(
            self._model_lane, "child_lane", SEG_CHILD_DASH)
        self._warmup()

        # ── 통신 ──
        qos = QoSProfile(  # sensor_data 성격 + depth1 (최신 프레임만 유지)
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, "/image_raw", self._on_image, qos)

        self._pub_lane = self.create_publisher(PoseArray, "/detect/lane", 10)
        self._pub_quality = self.create_publisher(Float32, "/detect/lane_quality", 10)
        self._pub_objects = self.create_publisher(PoseArray, "/detect/objects", 10)
        self._pub_light = self.create_publisher(String, "/detect/traffic_light", 10)

        # ── 상태 ──
        self._busy = False              # 처리 중 플래그 (백로그 금지: 처리 중 수신 프레임 드랍)
        self._light_candidate = "NONE"  # 디바운스 후보 판정
        self._light_count = 0           # 후보 연속 프레임 수
        self._light_state = "NONE"      # 확정(발행) 상태
        self._perf_t0 = time.monotonic()
        self._perf_n = 0
        self._perf_total = self._perf_lane = self._perf_obj = self._perf_light = 0.0

        self.get_logger().info(
            f"yolo_node started (lane={'on' if self._model_lane else 'off'} "
            f"obj={'on' if self._model_obj else 'off'} "
            f"light={'on' if self._model_light else 'off'} "
            f"imgsz={self._imgsz} device={self._device})")

    # ---------------- 초기화 보조 ----------------

    def _log_camera_summary(self, camera_yaml):
        """카메라 캘리브 상태를 한 줄로 남긴다 — 현장에서 로그만 보고 판별 가능하게.

        찍는 것: 왜곡모델 / 두 캘리브 플래그 / undistort 실제 활성 여부 / H 출처 /
                 해상도 / note(폴백 사유). 이어서 문제가 있을 때만 경고를 덧붙인다.
        """
        cam = self._cam
        intr = bool(cam["intrinsics_calibrated"])
        h_ok = bool(cam["h_calibrated"])
        w, h = self._undist_size
        if self._undist.enabled:
            nk = self._undist.new_K
            undist = (f"enabled(alpha={self._undist_alpha:.2f}, "
                      f"new_K fx={nk[0, 0]:.2f} fy={nk[1, 1]:.2f} "
                      f"cx={nk[0, 2]:.2f} cy={nk[1, 2]:.2f})")
        else:
            undist = "disabled(passthrough)"
        self.get_logger().info(
            f"[camera] model={cam['model']} "
            f"intrinsics_calibrated={str(intr).lower()} "
            f"h_calibrated={str(h_ok).lower()} "
            f"undistort={undist} "
            f"H={'yaml' if h_ok else 'sim-fallback'} size={w}x{h} "
            f"car_mask_v={'-' if self._car_mask_v is None else f'{self._car_mask_v:.0f}'} "
            f"yaml={camera_yaml} note={cam['note'] or '-'}")

        # 차체를 안 자르면 가짜 근거리 차선점이 계속 들어온다 — 조용히 넘기지 않는다.
        if self._car_mask_v is None:
            self.get_logger().warn(
                "camera.yaml 에 car_mask_v 없음 — 차체(라이다/범퍼) 화소가 "
                "BEV 근거리 가짜 차선점으로 들어올 수 있다. "
                "tools/bev_view.py 에서 m 키로 가림선을 정하고 s 로 저장할 것.")

        # K/D 는 유효한데 리맵 생성이 실패 → 조용히 패스스루로 도는 것을 막는다.
        if intr and not self._undist.enabled:
            self.get_logger().error(
                "Undistorter 초기화 실패 (camera.yaml K/D 확인) — 패스스루로 동작")
        # new_K 대조: H 는 캘리브 당시 new_K 기준이라 지금 값과 다르면 H 가 무효다.
        if cam["new_K"] is not None:
            msg = bev.check_new_K_match(cam["new_K"], self._undist.new_K)
            if msg:
                self.get_logger().warn(f"[camera] {msg}")
            else:
                self.get_logger().info(
                    "[camera] new_K 일치 — H 좌표계가 런타임 undistort 와 같다")
        if not h_ok:
            self.get_logger().warn(
                "[camera] H 미보정(h_calibrated=false) — 예선 시뮬 폴백 H 로 동작. "
                "/detect/lane 좌표는 참고용이다. tools/calib_ground_h.py 로 잡을 것.")

    def _load_model(self, name, path):
        if not os.path.exists(path):
            self.get_logger().error(f"NO MODEL ({name}): {path}")
            raise FileNotFoundError(path)
        self.get_logger().info(f"loading {name} model: {path}")
        model = YOLO(path)
        self.get_logger().info(f"{name}: task={model.task} names={model.names}")
        return model

    def _resolve_seg_cls(self, model, want, fallback):
        """세그 클래스 이름으로 id 를 찾는다 (모델을 바꿔도 안 깨지게).

        재학습할 때마다 클래스 수가 달라져 id 가 바뀐다. id 를 코드에 박아두면
        모델 교체 시 에러 없이 '검출 0개'로 조용히 굴러가서 원인을 못 찾는다.
        이름(mid 등)으로 찾고, 못 찾으면 옛 상수로 떨어지되 반드시 경고를 남긴다.
        """
        if model is None:
            return fallback
        names = dict(getattr(model, "names", {}) or {})
        for cid, nm in names.items():
            if str(nm).strip().lower() == want:
                if int(cid) != int(fallback):
                    self.get_logger().info(
                        "세그 클래스 '%s' = id %d (옛 상수 %d 과 다름 — 모델 기준으로 씀)"
                        % (want, int(cid), int(fallback)))
                return int(cid)
        self.get_logger().warning(
            "모델에 '%s' 클래스가 없다 — 옛 상수 %d 로 진행한다. "
            "id 가 안 맞으면 검출이 0개가 된다. 모델 클래스: %s"
            % (want, int(fallback), names))
        return int(fallback)

    def _warmup(self):
        """CPU 첫 추론이 느려 첫 실프레임 지연 방지용 더미 1회 추론."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for model in (self._model_lane, self._model_obj, self._model_light):
            if model is not None:
                model.predict(dummy, imgsz=self._imgsz, device=self._device, verbose=False)

    # ---------------- 이미지 콜백 ----------------

    def _on_image(self, msg: Image):
        if self._busy:
            return  # 처리 중 도착한 프레임 드랍 → 항상 최신 프레임만 처리
        self._busy = True
        try:
            self._process(msg)
        finally:
            self._busy = False

    def _process(self, msg: Image):
        t0 = time.perf_counter()

        bgr, info = preprocess.image_msg_to_bgr(msg)
        if bgr is None:
            # 무음 드랍 금지 (재도입 금지 버그 8) — 사유 포함 5초 스로틀 경고
            self.get_logger().warn(f"image 변환 실패: {info}",
                                   throttle_duration_sec=5.0)
            return

        # 렌즈 왜곡 보정(undistort) — 리맵 테이블은 기동 때 캘리브 해상도로 이미 만들었다.
        # 실제 프레임 크기가 캘리브 기준과 다르면 그 크기로 다시 만들고 경고한다
        # (해상도가 바뀌면 new_K 가 달라져 H 가 무효 — 조용히 넘어가면 안 된다).
        # 3모델 모두 undistort 된 프레임을 본다 (K/D 미보정 시 no-op 패스스루).
        # → /detect/objects 의 u,v 임계값(planner)은 undistort 프레임 기준으로 튜닝.
        h, w = bgr.shape[:2]
        if (w, h) != self._undist_size:
            self.get_logger().warn(
                f"[camera] 프레임 크기 {w}x{h} 가 캘리브 기준 "
                f"{self._undist_size[0]}x{self._undist_size[1]} 와 다르다 — "
                "리맵 재생성. new_K 가 달라져 H 는 무효일 수 있다.")
            self._undist_size = (w, h)
            self._undist = bev.Undistorter.from_config(
                self._cam, self._undist_size, alpha=self._undist_alpha)
            self._log_camera_summary(
                str(self.get_parameter("camera_yaml").value))
        frame = self._undist.apply(bgr)
        if self._lane_on_raw and self._new_K_cfg is None:
            import numpy as _np
            nk = self._cam.get("new_K")
            self._new_K_cfg = _np.asarray(
                nk if nk is not None else self._undist.new_K, _np.float64
            ).reshape(3, 3)
            self._cm_rows_raw = bev.car_mask_rows_raw(
                self._car_mask_v, self._cam["K"], self._cam["D"],
                self._new_K_cfg, (bgr.shape[1], bgr.shape[0]),
                self._cam.get("h_model"))
            self.get_logger().info(
                "[lane] 원본 프레임으로 추론한다 (검출점만 펴서 지면 변환)")

        t_lane = t_obj = t_light = 0.0
        if self._model_lane is not None:
            t = time.perf_counter()
            self._run_lane(frame, msg.header, raw=bgr)
            t_lane = time.perf_counter() - t
        if self._model_obj is not None:
            t = time.perf_counter()
            self._run_obj(frame, msg.header)
            t_obj = time.perf_counter() - t
        if self._model_light is not None:
            t = time.perf_counter()
            self._run_light(frame)
            t_light = time.perf_counter() - t

        # ── 처리시간 10초 주기 로그 ──
        self._perf_n += 1
        self._perf_total += time.perf_counter() - t0
        self._perf_lane += t_lane
        self._perf_obj += t_obj
        self._perf_light += t_light
        now = time.monotonic()
        elapsed = now - self._perf_t0
        if elapsed >= 10.0:
            n = max(self._perf_n, 1)
            self.get_logger().info(
                f"[perf] {self._perf_n}f/{elapsed:.1f}s "
                f"({self._perf_n / elapsed:.1f}Hz) "
                f"avg={self._perf_total / n * 1e3:.0f}ms "
                f"(lane={self._perf_lane / n * 1e3:.0f} "
                f"obj={self._perf_obj / n * 1e3:.0f} "
                f"light={self._perf_light / n * 1e3:.0f}) "
                f"light_state={self._light_state}")
            self._perf_t0 = now
            self._perf_n = 0
            self._perf_total = self._perf_lane = self._perf_obj = self._perf_light = 0.0

    # ---------------- 차선 (lane.pt seg) ----------------

    def _run_lane(self, frame, header, raw=None):
        """★ 차선만 **원본**을 본다.

        모델을 원본으로 학습시켰고(녹화 원본과 학습 사진 md5 동일), 사진을 펴서
        넣으면 out_fx 때문에 좌우가 잘리거나(원본의 65%) 물체가 작아져 먼 점선을
        놓친다. 검출된 점만 나중에 펴서 지면으로 보낸다 — 결과는 수학적으로 같다.
        물체/신호등 모델은 u,v 임계값이 보정영상 기준으로 맞춰져 있어 그대로 둔다.
        """
        use_raw = raw is not None and self._lane_on_raw
        frame = raw if use_raw else frame
        img = frame[:, :, ::-1] if MODEL_WANTS_RGB["lane"] else frame
        results = self._model_lane.predict(
            np.ascontiguousarray(img), conf=self._conf_lane, imgsz=self._imgsz,
            device=self._device, verbose=False, retina_masks=False)
        r = results[0] if results else None

        out = PoseArray()
        out.header.stamp = header.stamp
        out.header.frame_id = LIDAR_FRAME

        # 발행 대상 (seg 클래스, z) — dash 는 스위치 켤 때만 (기본 False, 루프만 준비)
        targets = [(self._seg_mid, Z_MID)]
        if bool(self.get_parameter("enable_child_dash").value):
            targets.append((self._seg_dash, Z_CHILD_DASH))

        H_img, W_img = frame.shape[:2]
        xs_mid = np.array([], dtype=np.float32)
        ys_mid = np.array([], dtype=np.float32)

        if r is not None and r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            masks = r.masks.data.cpu().numpy()  # (N, Hm, Wm) — 모델 해상도

            for seg_cls, z_id in targets:
                sel = np.where(cls == seg_cls)[0]
                if sel.size == 0:
                    continue
                # 클래스별 인스턴스 마스크 합성 → 원본 크기 → 이진화
                binmask = np.zeros((H_img, W_img), dtype=np.uint8)
                for i in sel:
                    m = masks[i]
                    if m.shape != (H_img, W_img):
                        m = cv2.resize(m, (W_img, H_img),
                                       interpolation=cv2.INTER_NEAREST)
                    binmask[m > 0.5] = 255
                # 차체(라이다 원통/범퍼) 영역 제거 — BEV 워프 전에 잘라야 한다.
                # 캘리브 기준 높이를 넘겨 프레임 크기가 달라도 비율 환산되게 한다.
                if use_raw:
                    bev.apply_car_mask_raw(binmask, self._cm_rows_raw)
                    xs, ys = bev.raw_mask_to_bev_points(
                        binmask, self._cam["H"], self._cam["K"], self._cam["D"],
                        self._new_K_cfg, self._cam.get("h_model"))
                else:
                    bev.apply_car_mask(binmask, self._car_mask_v,
                                       calib_h=self._undist_size[1])
                    xs, ys = bev.mask_to_bev_points(binmask, self._cam["H"])
                for x, y in zip(xs, ys):
                    p = Pose()
                    p.position.x = float(x)
                    p.position.y = float(y)
                    p.position.z = z_id
                    out.poses.append(p)
                if seg_cls == self._seg_mid:
                    xs_mid, ys_mid = xs, ys

        self._pub_lane.publish(out)
        # 품질은 주행 기준선인 MID 점만으로 등급 산정 (점 개수 × x스팬)
        self._pub_quality.publish(
            Float32(data=float(bev.lane_quality(xs_mid, ys_mid))))

    # ---------------- 객체 (best.pt detect) ----------------

    def _run_obj(self, frame, header):
        img = frame[:, :, ::-1] if MODEL_WANTS_RGB["obj"] else frame
        results = self._model_obj.predict(
            np.ascontiguousarray(img), conf=self._conf_obj, imgsz=self._imgsz,
            device=self._device, verbose=False)
        r = results[0] if results else None

        out = PoseArray()
        out.header.stamp = header.stamp
        out.header.frame_id = CAM_FRONT_FRAME

        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()  # (N,4): x1,y1,x2,y2
            classes = r.boxes.cls.cpu().numpy().astype(int)
            # 차체 가림선 (없으면 None → 아래 판정 전부 통과)
            mask_row = bev.car_mask_row(self._car_mask_v, frame.shape[0],
                                        calib_h=self._undist_size[1])
            for bbox, cls_id in zip(boxes, classes):
                cls_id = int(cls_id)
                if cls_id in OBJ_IGNORE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = bbox
                if mask_row is not None:
                    if y1 >= mask_row:
                        continue          # 박스 전체가 차체 위 → 오검출, 버린다
                    # 접지점만 차체에 걸친 경우는 버리지 않고 가림선까지 당긴다.
                    # (범퍼 코앞의 진짜 장애물을 버리면 안 된다 — 가장 가까운
                    #  거리로 읽히게 보수적으로 클램프)
                    y2 = min(float(y2), float(mask_row))
                p = Pose()
                p.position.x = float((x1 + x2) * 0.5)  # u 중심
                p.position.y = float(y2)               # v 하단 (클수록 가까움)
                p.position.z = float(cls_id)
                out.poses.append(p)

        self._pub_objects.publish(out)

    # ---------------- 신호등 (light.pt) ----------------

    def _run_light(self, frame):
        img = frame[:, :, ::-1] if MODEL_WANTS_RGB["light"] else frame
        results = self._model_light.predict(
            np.ascontiguousarray(img), conf=self._conf_light, imgsz=self._imgsz,
            device=self._device, verbose=False)
        r = results[0] if results else None

        # 예선 계승: green/red 최고 conf 비교, 동률이면 GREEN 우선
        raw = "NONE"
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            green_conf = float(confs[cls == LIGHT_GREEN_CLS].max()) \
                if (cls == LIGHT_GREEN_CLS).any() else 0.0
            red_conf = float(confs[cls == LIGHT_RED_CLS].max()) \
                if (cls == LIGHT_RED_CLS).any() else 0.0
            if green_conf > 0.0 and green_conf >= red_conf:
                raw = "GREEN"
            elif red_conf > 0.0:
                raw = "RED"

        # N프레임 연속 동일 판정일 때만 확정 상태 갱신 (깜빡임/오검출 억제)
        if raw == self._light_candidate:
            self._light_count += 1
        else:
            self._light_candidate = raw
            self._light_count = 1
        if self._light_count >= self._debounce_n:
            self._light_state = raw

        # 확정 상태를 매 프레임 발행 (1회 라치 금지 — 재도입 금지 버그 4 대응)
        self._pub_light.publish(String(data=self._light_state))


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
