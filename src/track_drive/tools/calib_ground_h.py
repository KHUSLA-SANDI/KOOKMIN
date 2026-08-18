#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""바닥 호모그래피 H 캘리브 — undistort 이미지 픽셀 → 차량좌표(m) (독립 CLI, ROS 불필요).

역할: camera.yaml 의 model/K/D 로 undistort 한 이미지 위에서 바닥 기준점을 클릭하고,
      각 점의 실측 차량좌표(x전방/y좌, m)를 콘솔로 받아 cv2.findHomography 로
      H(픽셀→차량좌표)를 계산. 재투영 오차 출력 후 camera.yaml 의 H 갱신.
입력: --image 바닥 기준점이 보이는 사진, --camera camera.yaml (model/K/D 필요),
      비대화 모드는 --points CSV (한 줄에 u,v,x,y — u,v 는 undistort 픽셀).
출력: 콘솔에 H/점별 오차/RMS(m). --save 시 camera.yaml 의 H 갱신 + calibrated 재계산.
예선 대비 변경점: 신규 도구 (예선 H 는 시뮬 tf_static 기반 자동 산출이었음).
      undistort 는 lib/bev.py 의 Undistorter 를 그대로 불러 쓴다(모델 분기 포함) —
      주행 노드와 undistort 가 1픽셀이라도 다르면 여기서 잡은 H 가 어긋난다.

사용법:
  # 대화 모드 — 클릭 → 콘솔에 "x y" 입력 반복, q 로 종료 후 계산
  python3 calib_ground_h.py --image ground.jpg --camera ../config/camera.yaml --save
  # 격자(타일 줄눈) 모드 — 클릭 → 콘솔에 격자번호 "i j" 입력 (아래 참고)
  python3 calib_ground_h.py --image calib_raw.jpg --camera ../config/camera.yaml \
      --grid --pitch 0.453333 --enhance --points tile_pts.csv --append --save
  # A4 사각형 모드 — 네 꼭짓점만 클릭하면 4개 대응점 자동 생성 (아래 참고)
  python3 calib_ground_h.py --image ground.jpg --camera ../config/camera.yaml \
      --rect --rect-size 0.210 0.297 --rect-origin 1.00 0.00 \
      --points a4_pts.csv --append --save
  # 비대화 모드 — 측정값 CSV 재사용 (u,v,x,y / # 주석 허용)
  python3 calib_ground_h.py --image ground.jpg --camera ../config/camera.yaml \
      --points pts.csv --save
  # 누적 CSV 위에 개별 점을 클릭으로 더 얹기 (--click 이 비대화 모드를 덮어씀)
  python3 calib_ground_h.py --image ground.jpg --camera ../config/camera.yaml \
      --click --points pts.csv --append --save
  --save [경로] : 경로 생략 시 --camera 파일에 저장.

--grid 모드 (바닥 타일 격자를 자로 쓰는 법 — 줄자 실측 0회):
  복도 테라조처럼 줄눈이 규칙적인 바닥이면, 줄눈 교차점 자체가 정확한 격자자다.
  교차점을 클릭하고 콘솔에 "격자번호" 두 개(i j)만 넣으면 지면좌표를 자동 계산한다:
      x = origin_x + i * pitch,   y = origin_y + j * pitch
      i = 전방 칸수(+ 앞),  j = 좌측 칸수(+ 왼쪽)     ※ 반칸(0.5) 같은 소수도 허용
  - --pitch 는 타일 한 칸 실측 피치(m). 여러 칸을 한 번에 재서 나누면 오차가 준다.
    예) 6칸을 2.72m 로 실측 → 2.72/6 = 0.453333
  - 원점(i=0,j=0) 은 기본적으로 "카메라 렌즈의 지면 수직투영점 A'" 이다.
    차를 세울 때 A' 가 줄눈 교차점에 오도록 맞추면 모든 교차점이 (i·p, j·p) 가 된다.
    A' 가 교차점에서 벗어났다면 그 어긋난 양을 --origin-x/--origin-y 로 넣는다.
  - 차체에 가려 안 보이는 i=0 줄눈은 못 찍는다. 화면 최근접 가로 줄눈이 보통 i=1.
  - 격자는 전/후·좌/우로 넓게 찍을 것. 한 줄에 몰리면 H 가 퇴화한다.

--enhance (표시용 선명화):
  테라조/무광 타일은 줄눈 대비가 약하다. CLAHE(clip 3.0, 8x8) + 언샤프를 걸면
  원거리 줄눈이 또렷해진다. ★ 이 강화는 undistort 이후 "표시 단계" 에만 적용되고
  이미지 크기·화소 격자를 바꾸지 않는다 — 같은 줄눈은 강화 전/후에 같은 (u,v) 에
  있으므로 클릭 좌표를 그대로 H 계산에 쓴다. 기하(H)는 --enhance 유무와 무관하다.

확대 클릭 보조 (--grid 모드):
  원거리 줄눈은 몇 픽셀 간격이라 맨눈 클릭이 어렵다.
  - 마우스 휠 또는 [+]/[-] : 표시 배율 1x → 2x → 4x (커서 지점을 중심으로 확대).
    창 크기는 그대로 두고 뷰포트를 잘라 확대하므로, 창 픽셀 → 원본 픽셀 역변환이
    정확하다. 4x 에서는 창 1픽셀 = 원본 0.25픽셀 → 서브픽셀 조준이 된다.
  - 커서 주변 60x60 을 4배 확대한 매그니파이어를 화면 구석에 항상 띄운다.
  - 우클릭(또는 [c]) : 그 지점으로 뷰포트 중심 이동(패닝).

A4 용지를 마커로 쓰는 법 (바닥에 격자가 없을 때 — 줄자 실측을 최소화):
  A4(210x297mm) 를 바닥에 놓고 네 꼭짓점을 클릭 → 각 꼭짓점의 실측 (x,y) 입력.
  여러 장을 전방 0.5~2.5m 에 흩어 놓을 것.
  - 용지의 긴 변(297mm)을 전방(x축)과 나란히, 짧은 변(210mm)을 좌우(y축)와 나란히.
    반대로 눕혔다면 --rect-size 0.297 0.210 으로 뒤집어 준다.
  - 용지가 들뜨면 안 된다(테이프로 고정). H 는 z=0 평면 가정이다.
  - 전방 한 곳에 몰아 놓으면 원거리 오차가 커진다 — 좌/우/근/원으로 흩을 것.

--rect 모드 (편의 모드):
  사각형 네 꼭짓점을 아래 순서로 클릭하면, 실제 크기(--rect-size)와 근단 중앙의
  지면좌표(--rect-origin)로부터 4개 대응점을 자동 생성한다. 꼭짓점마다 줄자를
  대지 않아도 된다.
    클릭 순서: 1)근단 좌  2)근단 우  3)원단 우  4)원단 좌   ("근단" = 차에 가까운 변)
  사각형 하나를 끝낼 때마다 콘솔이 그 용지의 근단 중앙 좌표를 묻는다(첫 장은
  --rect-origin 이 기본값). 한 이미지 안에서 여러 장을 이어서 찍을 수 있다.
  --points 파일과 --append 를 같이 주면 새 대응점을 CSV 에 누적해, 다른 이미지로
  다시 돌려도 이전 점들과 합쳐 H 를 계산한다.

원거리 점은 --rect 로 못 잡는다 (실측 기하):
  카메라가 지면에서 ~0.2m 높이에 거의 수평으로 달려 있어, 바닥 한 점의 화면 행 v 는
  전방거리 x 에 대해 v ≈ cy + fy*h/x 로 급격히 포화한다. A4 를 2.5m 앞에 놓으면
  화면에서 세로 3px / 면적 86px² 밖에 안 돼 MIN_RECT_AREA_PX 에 걸려 거부된다.
  → A4 사각형은 0.4~1.2m 에만 쓰고, 1.5m 밖은 --click 으로 개별 점(테이프 십자 등)을
    찍어 같은 CSV 에 --append 하는 게 맞다.

좌표/규약:
  - 차량좌표: x 전방+, y 좌+ (단위 m). 예) 앞 1.2m 오른쪽 0.3m → "1.2 -0.3"
  - ★ H 의 출력 "프레임" 을 --frame 으로 명시하고 camera.yaml 의 h_frame 에 기록한다.
      --frame a_prime (기본) : 원점 = A' = 카메라 렌즈의 지면 수직투영점.
                               --grid 로 A' 를 줄눈 교차점에 맞춰 세웠을 때가 이 경우다.
      --frame lidar          : 원점 = 라이다. 런타임(lib/bev)이 기대하는 프레임.
    A' 기준으로 잡은 점을 라이다 기준으로 옮기려면 라이다가 A' 보다 앞선 거리만큼
    x 를 빼면 된다: --frame lidar --lidar-offset-x 0.117 (저장 직전 대응점 x 에서 뺌).
    ※ h_frame 이 lidar_frame 이 아니면 런타임 좌표가 그 오프셋만큼 통째로 어긋난다 —
      일부러 A' 프레임으로 저장할 때만 그렇게 두고, 나중에 같은 CSV 로 다시 저장할 것.
  - H 는 undistort 된 이미지 픽셀 기준이고, 그 픽셀 좌표계는 new_K 가 정한다.
    fisheye 는 new_K=K 지만 plumb_bob 은 new_K != K 다
    (cv2.getOptimalNewCameraMatrix(K, D, size, alpha), 기본 alpha 는 lib/bev.py 값).
    그래서 H 를 저장할 때 그때 쓴 new_K 와 alpha 도 camera.yaml 에 같이 적는다 —
    런타임이 bev.check_new_K_match() 로 좌표계 일치를 검증한다.
  - K/D 가 미캘리브(단위행렬)면 undistort 를 건너뛰고 경고 —
    calib_camera.py 또는 import_camera_info.py 를 먼저.
  - 왜곡 모델(plumb_bob/fisheye)·K/D·alpha·이미지 크기 중 하나라도 바뀌면 new_K 가
    달라져 기존 H 는 무효다. 그 경우 이 도구를 반드시 다시 돌릴 것.
  - 점 4개 이상 필요. 한 직선 위 점들만 찍으면 H 퇴화 — 전후/좌우로 퍼뜨릴 것.
"""

import argparse
import datetime
import importlib.util
import inspect
import os
import re
import sys

import cv2
import numpy as np
import yaml

WIN = "calib_ground_h"
MIN_POINTS = 4          # findHomography 최소 점 수
RMS_WARN_M = 0.05       # 재투영 RMS 가 이보다 크면 경고 (5cm)
MODELS = ("plumb_bob", "fisheye")
A4_W_M, A4_L_M = 0.210, 0.297   # A4 짧은 변(좌우) / 긴 변(전후)
MIN_RECT_AREA_PX = 200.0        # 클릭한 사각형이 이보다 작으면 오클릭 취급
ALPHA_FALLBACK = 0.0            # lib/bev.py 를 못 읽을 때의 alpha 기본값

# --grid: 바닥 타일 격자
GRID_PITCH_DEFAULT = 0.453333   # 본선 복도 테라조 실측 2.72m / 6칸

# --enhance: 표시용 선명화 (기하 불변 — 픽셀값만 바꾼다)
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)
UNSHARP_SIGMA = 3.0             # GaussianBlur sigma
UNSHARP_W_SRC = 1.6             # addWeighted: 원본 가중치
UNSHARP_W_BLUR = -0.6           # addWeighted: 블러 가중치

# 확대 클릭 보조
ZOOM_LEVELS = (1, 2, 4)         # 표시 배율 (휠 / [+][-])
MAG_SRC_PX = 60                 # 매그니파이어가 확대할 원본 영역 한 변(px)
MAG_ZOOM = 4                    # 매그니파이어 배율 → 240x240 패널
HINT_BAR_H = 26                 # 상단 힌트 바 높이(px)

# H 출력 프레임 이름 (--frame → camera.yaml 의 h_frame)
FRAME_NAMES = {"a_prime": "A_prime", "lidar": "lidar_frame"}

COLOR_NEW = (0, 0, 255)         # 이번 실행에서 찍은 점 (BGR 빨강)
COLOR_OLD = (160, 160, 160)     # CSV 에서 읽어온 기존 점 (회색)
COLOR_HL = (0, 255, 255)        # 입력 대기 중인 점 (노랑)


# ======================== camera.yaml 로드/병합 저장 (calib_camera.py 와 동일 규약) ========================

def _load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _flag(data, key):
    """부분 캘리브 플래그. 구버전 파일(플래그 없음)은 calibrated 값으로 간주."""
    if key in data:
        return bool(data[key])
    return bool(data.get("calibrated", False))


def _save_camera_yaml(path, data, tool_name):
    """기존 키 보존 + 지정 순서로 기록. (주의: 기존 파일의 주석은 사라짐)"""
    order = ["calibrated", "intrinsics_calibrated", "h_calibrated", "model",
             "image_width", "image_height", "K", "D", "new_K", "H", "h_frame"]
    out = {k: data[k] for k in order if k in data}
    for k, v in data.items():
        if k not in out:
            out[k] = v
    header = (
        "# 실차 카메라 캘리브 파일 — track_drive.lib.bev.load_camera_config() 가 읽는다.\n"
        "# %s 가 %s 에 갱신.\n"
        "# calibrated: K/D(내부)와 H(바닥) 둘 다 완료 시에만 true (부분 플래그는 아래 참고)\n"
        "# model: plumb_bob(일반/광각, D 5개) | fisheye(등거리 어안, D 4개)\n"
        "# K/D: 내부 파라미터.  new_K: undistort 결과 픽셀 좌표계를 정의하는 행렬.\n"
        "# H: undistort(new_K 기준) 이미지 픽셀 → 지면 미터 (x전방/y좌).\n"
        "#    new_K 가 바뀌면 H 는 통째로 무효 — bev.check_new_K_match() 가 대조한다.\n"
        "# h_frame: H 출력 좌표의 원점. lidar_frame = 라이다 기준(런타임이 기대하는 값) /\n"
        "#    A_prime = 카메라 렌즈의 지면 수직투영점 기준. A_prime 이면 라이다-A' 거리만큼\n"
        "#    x 가 어긋나므로, tools/calib_ground_h.py 를 같은 대응점 CSV 로 다시 돌려\n"
        "#    --frame lidar --lidar-offset-x <A'→라이다 거리> 로 저장해야 한다.\n"
        % (tool_name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None,
                       allow_unicode=True)
    print("저장: %s" % path)


def _load_camera(path):
    """camera.yaml 에서 model/K/D 를 읽는다 (calibrated 값과 무관하게 원본 사용)."""
    data = _load_yaml(path)
    if not data:
        sys.exit("camera.yaml 읽기 실패: %s" % path)
    try:
        K = np.asarray(data["K"], np.float64).reshape(3, 3)
        D = np.asarray(data["D"], np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError):
        sys.exit("camera.yaml 에 K/D 없음/형식 오류: %s" % path)

    model = str(data.get("model", "")).strip().lower()
    if model not in MODELS:
        # 구버전 파일(model 키 없음): D 길이로 추정 (fisheye 4개 / plumb_bob 5개 이상)
        model = "fisheye" if D.size == 4 else "plumb_bob"
        print("[안내] camera.yaml 에 model 키 없음 — D 계수 %d개를 보고 '%s' 로 추정."
              % (D.size, model))
        print("        calib_camera.py / import_camera_info.py 로 model 을 명시할 것.")
    return K, D, model


# ======================== undistort (lib/bev.py 규약과 반드시 일치) ========================

_BEV_MEMO = []   # _import_bev 1회 로드 캐시


def _import_bev():
    """lib/bev.py 를 파일 경로로 로드한다 (tools/ 는 ROS 패키지 밖이라 일반 import 불가).

    bev 는 rclpy 미의존(numpy/cv2/yaml 만)이라 CLI 도구에서 그대로 쓸 수 있다.
    """
    if _BEV_MEMO:
        return _BEV_MEMO[0]
    mod = None
    path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "track_drive", "lib", "bev.py"))
    if os.path.isfile(path):
        try:
            spec = importlib.util.spec_from_file_location("_track_drive_bev", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:  # 로드 실패해도 도구는 폴백으로 계속 굴러가야 한다
            print("[안내] lib/bev.py 로드 실패 (%s)" % e)
            mod = None
    _BEV_MEMO.append(mod)
    return mod


def _default_alpha():
    """lib/bev.UNDISTORT_ALPHA_DEFAULT 를 따라간다 (주행 노드와 new_K 를 맞추기 위함)."""
    bev = _import_bev()
    try:
        return float(getattr(bev, "UNDISTORT_ALPHA_DEFAULT", ALPHA_FALLBACK))
    except (TypeError, ValueError):
        return ALPHA_FALLBACK


class _FallbackUndistorter:
    """lib/bev.Undistorter 를 못 쓸 때의 동일 규약 대체본.

    new_K 규약도 그대로 따라간다: fisheye → new_K=K,
    plumb_bob → cv2.getOptimalNewCameraMatrix(K, D, size, alpha).
    """

    def __init__(self, K, D, size, model, alpha):
        self.enabled = False
        self.model = model
        self.alpha = float(alpha)
        self.new_K = np.asarray(K, np.float64)
        try:
            K_m = np.asarray(K, np.float64).reshape(3, 3)
            D_m = np.asarray(D, np.float64).reshape(-1)
            wh = (int(size[0]), int(size[1]))
            if model == "fisheye":
                D_m = np.concatenate([D_m, np.zeros(max(0, 4 - D_m.size))])[:4]
                nk = K_m.copy()
                self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                    K_m, D_m.reshape(4, 1), np.eye(3), nk, wh, cv2.CV_16SC2)
            else:
                D_m = np.concatenate([D_m, np.zeros(max(0, 5 - D_m.size))])
                nk, _roi = cv2.getOptimalNewCameraMatrix(K_m, D_m, wh, self.alpha)
                nk = np.asarray(nk, np.float64).reshape(3, 3)
                self._map1, self._map2 = cv2.initUndistortRectifyMap(
                    K_m, D_m, np.eye(3), nk, wh, cv2.CV_16SC2)
            self.new_K = nk
            self.enabled = True
        except (cv2.error, TypeError, ValueError, IndexError):
            self.enabled = False

    def apply(self, bgr):
        if not self.enabled or bgr is None:
            return bgr
        return cv2.remap(bgr, self._map1, self._map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def _make_undistorter(K, D, model, size, alpha):
    """가능하면 lib/bev.Undistorter 를 그대로 쓴다 (주행 노드와 동일 픽셀 보장)."""
    bev = _import_bev()
    if bev is not None and hasattr(bev, "Undistorter"):
        try:
            params = inspect.signature(bev.Undistorter.__init__).parameters
        except (TypeError, ValueError):
            params = {}
        if "model" in params:
            kw = {"calibrated": True, "model": model}
            if "alpha" in params:
                kw["alpha"] = alpha
            return bev.Undistorter(K, D, size, **kw), "lib/bev.py"
        if model == "fisheye":
            return bev.Undistorter(K, D, size, calibrated=True), "lib/bev.py (구버전)"
        print("[경고] lib/bev.py 의 Undistorter 가 모델 분기를 지원하지 않는다(fisheye 전용).")
        print("        model=%s 는 로컬 폴백으로 처리한다 — lib/bev.py 를 먼저 갱신할 것."
              % model)
        print("        (주행 노드가 다른 undistort 를 쓰면 여기서 잡은 H 가 어긋난다)")
    return _FallbackUndistorter(K, D, size, model, alpha), "로컬 폴백"


def _undistort(img, K, D, model, alpha):
    """undistort 적용. 반환 (이미지, undistorter) — undistorter 가 None 이면 원본 픽셀 기준."""
    if np.allclose(K, np.eye(3)) or not np.any(D):
        print("[경고] K/D 미캘리브(단위행렬/0) — undistort 생략, 원본(raw) 픽셀 기준 H 가 됨.")
        print("        calib_camera.py 또는 import_camera_info.py 를 먼저 돌릴 것.")
        return img, None
    h, w = img.shape[:2]
    und, src = _make_undistorter(K, D, model, (w, h), alpha)
    if not getattr(und, "enabled", False):
        print("[경고] undistort 맵 생성 실패 (model=%s, K/D 형식 확인) — "
              "원본 픽셀 기준 H 가 됨." % model)
        return img, None
    nk = np.asarray(und.new_K, np.float64).reshape(3, 3)
    print("undistort: model=%s, 구현=%s, D %d개, alpha=%.2f"
          % (model, src, np.size(D), alpha))
    print("  new_K fx,fy,cx,cy = %.3f %.3f %.3f %.3f  (H 는 이 좌표계 기준 — "
          "camera.yaml 의 new_K 로 저장된다)"
          % (nk[0, 0], nk[1, 1], nk[0, 2], nk[1, 2]))
    if model != "fisheye" and not np.allclose(nk, np.asarray(K, np.float64).reshape(3, 3)):
        print("  ※ plumb_bob 은 new_K != K 다 (getOptimalNewCameraMatrix(alpha=%.2f)). "
              "alpha 가 달라지면 H 가 무효." % alpha)
    return und.apply(img), und


# ======================== 표시용 선명화 (--enhance) ========================

def _enhance_for_display(bgr):
    """표시용 대비 강화 — CLAHE + 언샤프. ★ 기하(좌표계)는 절대 바뀌지 않는다.

    테라조/무광 타일은 줄눈 대비가 약해 원거리 교차점이 눈에 안 잡힌다.
    여기서 하는 일은 undistort '이후' 이미지의 화소 "값" 만 바꾸는 것이다:
      - 크기(shape) 변경·리사이즈·크롭·워핑·시프트 없음
      - 따라서 같은 줄눈은 강화 전/후에 정확히 같은 (u,v) 에 있고,
        클릭 좌표를 그대로 H 계산에 넣는다 (--enhance 는 H 에 영향이 없다).
    밝기(L) 채널에만 CLAHE 를 걸어 색 왜곡을 피한다.
    """
    if bgr is None:
        return bgr
    try:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        if bgr.ndim == 2:
            out = clahe.apply(bgr)
        else:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(out, (0, 0), UNSHARP_SIGMA)
        sharp = cv2.addWeighted(out, UNSHARP_W_SRC, blur, UNSHARP_W_BLUR, 0.0)
    except cv2.error as e:
        print("[안내] --enhance 실패 (%s) — 원본 그대로 표시한다." % e)
        return bgr
    assert sharp.shape == bgr.shape   # 기하 불변 보증 (리사이즈/크롭 금지)
    return sharp


# ======================== 점 수집: CSV ========================

def _load_points_csv(path):
    """CSV: 한 줄에 u,v,x,y (쉼표/공백 구분, # 주석·빈 줄 허용).

    # 는 줄 중간에서도 주석으로 친다 — --grid 가 각 줄 끝에 격자번호를
    "# grid 2,1" 로 남겨 두기 때문(사람이 나중에 대조할 수 있게).
    """
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.split("#", 1)[0].strip()
            if not s:
                continue
            vals = re.split(r"[,\s]+", s)
            if len(vals) != 4:
                sys.exit("%s %d행: u,v,x,y 4개 값 필요: %r" % (path, ln, s))
            try:
                u, v, x, y = (float(t) for t in vals)
            except ValueError:
                sys.exit("%s %d행: 숫자 변환 실패: %r" % (path, ln, s))
            pts.append([u, v, x, y])
    return pts


def _append_points_csv(path, pts, note, labels=None):
    """새로 만든 대응점을 CSV 에 이어 쓴다 (--append). 다음 실행에서 --points 로 재사용.

    x,y 는 "수집한 그대로의" 프레임 값이다 (--lidar-offset-x 는 저장 직전 H 계산에만
    적용하고 CSV 에는 반영하지 않는다). 그래야 같은 CSV 를 프레임만 바꿔 재사용할 수
    있다 — 어느 프레임인지는 아래 note 줄에 적힌다.
    labels 를 주면 각 줄 끝에 "# grid i,j" 주석을 달아 사람이 대조할 수 있게 한다.
    """
    new_file = not os.path.isfile(path)
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# calib_ground_h.py 누적 대응점 — u,v,x,y\n")
            f.write("# u,v = undistort 이미지 픽셀 / x,y = 지면 m (x전방+, y좌+)\n")
            f.write("# x,y 의 원점 프레임은 아래 각 블록의 주석(frame=...)을 볼 것\n")
        f.write("# %s  (%s)\n"
                % (note, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        for n, (u, v, x, y) in enumerate(pts):
            tail = ""
            if labels and n < len(labels):
                tail = "  # grid %s,%s" % (_fmt_ij(labels[n][0]),
                                           _fmt_ij(labels[n][1]))
            f.write("%.2f,%.2f,%.4f,%.4f%s\n" % (u, v, x, y, tail))
    print("누적 저장: %s (+%d점)" % (path, len(pts)))


# ======================== 점 수집: 대화 모드 ========================

def _draw_points(base, pts, hint=None):
    disp = base.copy()
    for i, (u, v, x, y) in enumerate(pts):
        c = (int(round(u)), int(round(v)))
        cv2.drawMarker(disp, c, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(disp, "%d(%.2f,%.2f)" % (i + 1, x, y),
                    (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
    if hint is None:
        hint = "click ground point | [u] undo  [q] done (%d pts)" % len(pts)
    cv2.putText(disp, hint, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA)
    return disp


def _collect_interactive(base, base_pts=()):
    """클릭 → 콘솔에 실측 차량좌표 입력. q/ESC 종료, u 마지막 점 취소.

    base_pts (CSV 에서 이미 읽은 점들) 는 배경에 회색으로 깔아 참고용으로만 보여준다.
    [u] 취소는 이번 실행에서 새로 찍은 점에만 적용된다.
    """
    if base_pts:
        # 기존 점을 배경에 한 번만 그려 둔다 (새 점과 헷갈리지 않게 회색 + 작은 마커)
        base = base.copy()
        for u, v, x, y in base_pts:
            c = (int(round(u)), int(round(v)))
            cv2.drawMarker(base, c, (160, 160, 160), cv2.MARKER_TILTED_CROSS, 10, 1)
        print("[안내] 기존 CSV %d점은 회색으로 표시된다 (취소 불가 — 참고용)."
              % len(base_pts))
    print("\n[안내] 이미지에서 바닥 기준점 클릭 → 콘솔에 실측 'x y' (m) 입력.")
    print("  x=전방+, y=좌+ (lidar_frame). 예) 앞1.2m 오른쪽0.3m → 1.2 -0.3")
    print("  키: [u] 마지막 점 취소 / [q]·[ESC] 입력 종료 (%d점 이상)" % MIN_POINTS)
    print("  팁: 한 직선 위 점만 찍으면 H 퇴화 — 전후·좌우로 퍼뜨릴 것.")
    print("  팁: 꼭짓점마다 줄자 대기 번거로우면 --rect (A4 사각형) 모드를 쓸 것.\n")

    pts = []      # 확정 [u, v, x, y]
    pending = []  # 클릭됐지만 좌표 미입력

    def on_mouse(event, mx, my, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((mx, my))

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        cv2.imshow(WIN, _draw_points(base, pts))
        key = cv2.waitKey(30) & 0xFF
        while pending:
            u, v = pending.pop(0)
            tmp = _draw_points(base, pts)
            cv2.drawMarker(tmp, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(WIN, tmp)
            cv2.waitKey(1)  # 콘솔 입력 동안 창 갱신 멈춤 — 정상
            s = input("점%d 픽셀(%d,%d) 의 차량좌표 x y (m, 빈 입력=이 점 취소): "
                      % (len(pts) + 1, u, v)).strip()
            if not s:
                print("  → 취소")
                continue
            vals = re.split(r"[,\s]+", s)
            try:
                x, y = float(vals[0]), float(vals[1])
            except (ValueError, IndexError):
                print("  → 형식 오류 (예: 1.2 -0.3) — 이 점 취소")
                continue
            pts.append([float(u), float(v), x, y])
        if key == ord("u") and pts:
            rm = pts.pop()
            print("취소: 점%d (%.0f,%.0f)→(%.2f,%.2f)" % (len(pts) + 1, *rm))
        elif key in (ord("q"), 27):
            break
    return pts


# ======================== 점 수집: --rect (A4 사각형) 모드 ========================

def _rect_corner_xy(origin, w_m, l_m):
    """근단 중앙 (x0,y0) + 사각형 크기 → 클릭 순서대로의 지면좌표 4개.

    클릭 순서 = 1)근단 좌  2)근단 우  3)원단 우  4)원단 좌.
    y 는 좌+ 이므로 '좌' 가 +w/2, '우' 가 -w/2. '원단' 은 x 가 +l_m.
    """
    x0, y0 = origin
    half = w_m / 2.0
    return [(x0, y0 + half), (x0, y0 - half),
            (x0 + l_m, y0 - half), (x0 + l_m, y0 + half)]


def _ask_origin(default):
    """이번 사각형의 근단 중앙 지면좌표를 콘솔에서 받는다. None = 이 사각형 취소."""
    if default is not None:
        s = input("  이 사각형 근단(가까운) 변 중앙의 차량좌표 x y "
                  "[엔터=%.3f %.3f, c=취소]: " % default).strip()
        if not s:
            return default
    else:
        s = input("  이 사각형 근단(가까운) 변 중앙의 차량좌표 x y [c=취소]: ").strip()
    if s.lower() == "c":
        return None
    vals = re.split(r"[,\s]+", s)
    try:
        return (float(vals[0]), float(vals[1]))
    except (ValueError, IndexError):
        print("  → 형식 오류 (예: 1.20 -0.30) — 이 사각형 취소")
        return None


def _draw_rect_progress(base, pts, cur, n_rect):
    disp = _draw_points(
        base, pts,
        hint="rect: click 4 corners (near-L, near-R, far-R, far-L) | "
             "[u] undo  [q] done (%d rect, %d pts)" % (n_rect, len(pts)))
    labels = ("nL", "nR", "fR", "fL")
    for i, (u, v) in enumerate(cur):
        cv2.circle(disp, (u, v), 5, (0, 255, 255), -1)
        cv2.putText(disp, labels[i], (u + 7, v - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if i > 0:
            cv2.line(disp, cur[i - 1], (u, v), (0, 255, 255), 1, cv2.LINE_AA)
    return disp


def _collect_rects(base, rect_size, rect_origin, base_pts):
    """사각형(A4) 네 꼭짓점 클릭 → 대응점 4개 자동 생성. 여러 장 반복 가능."""
    w_m, l_m = rect_size
    print("\n[안내] --rect 모드: 사각형 네 꼭짓점을 아래 순서로 클릭.")
    print("  1) 근단 좌   2) 근단 우   3) 원단 우   4) 원단 좌   (근단 = 차에 가까운 변)")
    print("  사각형 크기: 좌우 %.3fm x 전후 %.3fm (--rect-size)" % (w_m, l_m))
    print("  4점을 찍을 때마다 콘솔이 그 사각형의 근단 중앙 좌표를 묻는다.")
    print("  키: [u] 되돌리기(현재 클릭 → 직전 사각형) / [q]·[ESC] 종료")
    print("  팁: A4 는 전방 0.5~2.5m 에 좌/우/근/원으로 흩어 놓을 것.\n")

    new_pts = []    # 이번 실행에서 생성한 [u,v,x,y]
    cur = []        # 현재 사각형의 클릭 (u,v)
    clicks = []     # 마우스 큐
    next_origin = rect_origin

    def on_mouse(event, mx, my, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((mx, my))

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        shown = base_pts + new_pts
        cv2.imshow(WIN, _draw_rect_progress(base, shown, cur, len(new_pts) // 4))
        key = cv2.waitKey(30) & 0xFF
        while clicks:
            cur.append(clicks.pop(0))
            if len(cur) < 4:
                continue
            cv2.imshow(WIN, _draw_rect_progress(
                base, base_pts + new_pts, cur, len(new_pts) // 4))
            cv2.waitKey(1)  # 콘솔 입력 동안 창 갱신 멈춤 — 정상
            quad = np.asarray(cur, np.float32).reshape(-1, 1, 2)
            area = abs(cv2.contourArea(quad))
            if area < MIN_RECT_AREA_PX:
                print("  → 사각형 면적 %.0f px² 로 너무 작음/퇴화 — 이 사각형 취소" % area)
                cur = []
                continue
            origin = _ask_origin(next_origin)
            if origin is None:
                print("  → 이 사각형 취소")
                cur = []
                continue
            next_origin = None  # --rect-origin 기본값은 첫 사각형에만 적용
            xy = _rect_corner_xy(origin, w_m, l_m)
            for (u, v), (x, y) in zip(cur, xy):
                new_pts.append([float(u), float(v), x, y])
            print("  → 사각형 %d 추가: 근단중앙(%.3f,%.3f), 이번 실행 누적 %d점"
                  % (len(new_pts) // 4, origin[0], origin[1], len(new_pts)))
            cur = []
        if key == ord("u"):
            if cur:
                cur.pop()
            elif new_pts:
                del new_pts[-4:]
                print("취소: 마지막 사각형 (이번 실행 누적 %d점)" % len(new_pts))
        elif key in (ord("q"), 27):
            break
    return new_pts


# ======================== 확대 클릭 보조 (뷰포트 줌 + 매그니파이어) ========================

class _ZoomViewer:
    """1x/2x/4x 확대 뷰 + 커서 매그니파이어를 갖춘 클릭 수집 창.

    ★ 정확도 규약: 창 크기는 원본과 같게 두고, "원본의 일부를 잘라 확대한" 그림을
      우리가 직접 그린다. 그래서 창 픽셀 → 원본 픽셀 역변환이 정확한 상수배다:
          u = x0 + mx / zoom,   v = y0 + my / zoom     (x0,y0 = 뷰포트 좌상단, 정수)
      4x 에서는 창 1픽셀 = 원본 0.25픽셀이라 서브픽셀 조준이 된다.
      (OpenCV 의 WINDOW_NORMAL 리사이즈에 맡기면 마우스 좌표계가 백엔드마다 달라
       이 역변환이 깨진다 — 그래서 직접 자른다.)
    """

    def __init__(self, base, win=WIN):
        self.base = base
        self.win = win
        self.h, self.w = base.shape[:2]
        self.zi = 0                                  # ZOOM_LEVELS 인덱스
        self.cx, self.cy = self.w / 2.0, self.h / 2.0   # 뷰포트 중심(원본 좌표)
        self.cur = (self.w / 2.0, self.h / 2.0)         # 커서(원본 좌표, 서브픽셀)
        self.clicks = []                             # 마우스 큐 [(u,v), ...]
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)

    # ---- 좌표 변환 ----
    @property
    def zoom(self):
        return ZOOM_LEVELS[self.zi]

    def viewport(self):
        """현재 뷰포트 (x0, y0, vw, vh) — 원본 픽셀 정수 좌표."""
        z = self.zoom
        vw = min(self.w, -(-self.w // z))   # ceil(w/z)
        vh = min(self.h, -(-self.h // z))
        x0 = max(0, min(int(round(self.cx - vw / 2.0)), self.w - vw))
        y0 = max(0, min(int(round(self.cy - vh / 2.0)), self.h - vh))
        return x0, y0, vw, vh

    def win_to_img(self, mx, my):
        x0, y0, _vw, _vh = self.viewport()
        z = float(self.zoom)
        return (x0 + mx / z, y0 + my / z)

    def img_to_win(self, u, v):
        x0, y0, _vw, _vh = self.viewport()
        z = float(self.zoom)
        return ((u - x0) * z, (v - y0) * z)

    # ---- 입력 ----
    def set_zoom(self, step, at=None):
        """배율 변경. at(원본좌표) 을 중심으로 확대해 보던 지점을 놓치지 않게 한다."""
        zi = max(0, min(len(ZOOM_LEVELS) - 1, self.zi + step))
        if zi == self.zi:
            return
        self.zi = zi
        if at is not None:
            self.cx, self.cy = float(at[0]), float(at[1])

    def _on_mouse(self, event, mx, my, flags, _param):
        u, v = self.win_to_img(mx, my)
        self.cur = (u, v)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((u, v))
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self.cx, self.cy = u, v          # 우클릭 = 이 지점으로 패닝
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", -1):
            # 휠은 백엔드(GTK/Qt/Win32)에 따라 없을 수 있다 → [+]/[-] 키가 정식 경로
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except (AttributeError, cv2.error):
                delta = 0
            if delta:
                self.set_zoom(1 if delta > 0 else -1, at=(u, v))

    def handle_key(self, key):
        """줌/패닝 키. 처리했으면 True (호출부의 다른 키 처리를 건너뛰게)."""
        if key in (ord("+"), ord("="), ord("]")):
            self.set_zoom(+1, at=self.cur)
            return True
        if key in (ord("-"), ord("_"), ord("[")):
            self.set_zoom(-1, at=self.cur)
            return True
        if key == ord("c"):
            self.cx, self.cy = self.cur
            return True
        return False

    # ---- 그리기 ----
    def _magnifier(self, view):
        """커서 주변 MAG_SRC_PX 정사각을 MAG_ZOOM 배로 확대해 화면 구석에 붙인다.

        확대 원본은 마커를 그리지 않은 깨끗한 이미지다 — 십자 마커가 줄눈을 덮으면
        조준이 안 되기 때문. 패널 한가운데가 현재 커서 위치(서브픽셀)다.
        """
        pan = MAG_SRC_PX * MAG_ZOOM
        if view.shape[0] < pan + HINT_BAR_H + 8 or view.shape[1] < pan + 16:
            return
        try:
            patch = cv2.getRectSubPix(self.base, (MAG_SRC_PX, MAG_SRC_PX),
                                      (float(self.cur[0]), float(self.cur[1])))
            mag = cv2.resize(patch, (pan, pan), interpolation=cv2.INTER_NEAREST)
        except cv2.error:
            return
        px, py = 8, HINT_BAR_H + 4
        cwx, cwy = self.img_to_win(*self.cur)
        if cwx < px + pan + 24 and cwy < py + pan + 24:
            px = view.shape[1] - pan - 8       # 커서가 겹치면 반대쪽 구석으로
        view[py:py + pan, px:px + pan] = mag
        cv2.rectangle(view, (px - 1, py - 1), (px + pan, py + pan), COLOR_HL, 1)
        mx, my = px + pan // 2, py + pan // 2
        cv2.line(view, (mx - 16, my), (mx - 3, my), COLOR_NEW, 1)
        cv2.line(view, (mx + 3, my), (mx + 16, my), COLOR_NEW, 1)
        cv2.line(view, (mx, my - 16), (mx, my - 3), COLOR_NEW, 1)
        cv2.line(view, (mx, my + 3), (mx, my + 16), COLOR_NEW, 1)
        half = MAG_ZOOM // 2
        cv2.rectangle(view, (mx - half, my - half), (mx + half, my + half),
                      (0, 255, 0), 1)          # 초록 사각 = 원본 1픽셀 크기
        cv2.putText(view, "x%d" % MAG_ZOOM, (px + 4, py + pan - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HL, 1, cv2.LINE_AA)

    def render(self, marks, hint):
        """marks = [(u, v, text, color), ...] (원본좌표) 를 얹은 현재 뷰 이미지."""
        x0, y0, vw, vh = self.viewport()
        z = self.zoom
        crop = self.base[y0:y0 + vh, x0:x0 + vw]
        view = cv2.resize(crop, (vw * z, vh * z), interpolation=(
            cv2.INTER_NEAREST if z > 1 else cv2.INTER_LINEAR))
        if view.shape[0] != self.h or view.shape[1] != self.w:
            view = view[:self.h, :self.w].copy()
        for u, v, text, color in marks:
            wx, wy = (u - x0) * z, (v - y0) * z
            if not (-20 <= wx <= self.w + 20 and -20 <= wy <= self.h + 20):
                continue
            c = (int(round(wx)), int(round(wy)))
            cv2.drawMarker(view, c, color, cv2.MARKER_CROSS, 14, 1)
            if text:
                cv2.putText(view, text, (c[0] + 7, c[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        self._magnifier(view)
        cv2.rectangle(view, (0, 0), (self.w, HINT_BAR_H), (0, 0, 0), -1)
        # 힌트 + 현재 배율/커서(서브픽셀). 창 폭을 넘으면 글자를 줄여 잘리지 않게 한다
        text = "%s | %dx u%.2f v%.2f" % (hint, z, self.cur[0], self.cur[1])
        fs = 0.44
        for cand in (0.44, 0.40, 0.36, 0.32):
            fs = cand
            (tw, _th), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            if tw <= self.w - 10:
                break
        cv2.putText(view, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 255, 0), 1, cv2.LINE_AA)
        return view

    def show(self, marks, hint):
        cv2.imshow(self.win, self.render(marks, hint))


# ======================== 점 수집: --grid (바닥 타일 격자) ========================

def _fmt_ij(v):
    """격자번호 표시 — 2.0 은 '2', 0.5 는 '0.5' 로."""
    return "%g" % float(v)


def _parse_grid_ij(s):
    """'2 1' / '2,1' / '1.5 -0.5' → (i, j). 형식이 틀리면 None."""
    vals = [t for t in re.split(r"[,\s]+", s.strip()) if t]
    if len(vals) != 2:
        return None
    try:
        return (float(vals[0]), float(vals[1]))
    except ValueError:
        return None


def _ask_grid_ij(n, u, v, used):
    """격자번호를 콘솔에서 받는다 (틀리면 다시 묻는다). None = 이 점 취소."""
    while True:
        try:
            s = input("점%d 픽셀(%.2f,%.2f) 의 격자 (i j) 입력 "
                      "[i=전방칸+, j=좌칸+, 엔터=이 점 취소]: " % (n, u, v))
        except (EOFError, KeyboardInterrupt):
            print("\n  → 입력 중단 — 이 점 취소")
            return None
        s = s.strip()
        if not s:
            print("  → 취소")
            return None
        ij = _parse_grid_ij(s)
        if ij is None:
            print("  → 형식 오류: 숫자 2개를 넣을 것 (예: '2 1', 반칸은 '2 0.5') — 다시")
            continue
        if ij in used:
            try:
                c = input("  → (%s,%s) 는 이미 찍은 격자다. 그래도 추가? [y/N]: "
                          % (_fmt_ij(ij[0]), _fmt_ij(ij[1]))).strip().lower()
            except (EOFError, KeyboardInterrupt):
                c = ""
            if c != "y":
                print("  → 다시 입력 (다른 교차점을 원하면 엔터로 취소하고 재클릭)")
                continue
        return ij


def _collect_grid(base, pitch, origin, base_pts):
    """줄눈 교차점 클릭 → 격자번호 (i,j) 입력 → 지면좌표 자동 계산.

    지면좌표 = (origin_x + i*pitch, origin_y + j*pitch).
    반환 (new_pts=[[u,v,x,y],...], labels=[(i,j),...]).
    """
    ox, oy = origin
    print("\n[안내] --grid 모드: 타일 줄눈 교차점 클릭 → 콘솔에 격자번호 'i j' 입력.")
    print("  피치 p = %.6f m,  원점(i=0,j=0) = (%.4f, %.4f) m" % (pitch, ox, oy))
    print("  지면좌표 = (%.4f + i*p,  %.4f + j*p)   i=전방 칸수(+앞), j=좌 칸수(+왼쪽)"
          % (ox, oy))
    print("  예) 바로 앞 첫 가로줄눈 위 정면 교차점 → '1 0' = (%.4f, %.4f) m"
          % (ox + pitch, oy))
    print("  반칸(0.5) 같은 소수도 된다. 잘못 넣으면 다시 물어본다.")
    print("  키: [u] 마지막 점 취소 / [q]·[ESC] 종료 (%d점 이상)" % MIN_POINTS)
    print("      [+]/[-]·휠 = 배율 1x/2x/4x (커서 중심) / [c]·우클릭 = 그 지점으로 이동")
    print("  팁: 원거리 줄눈은 4x 로 키우고 좌상단 매그니파이어(x4)로 조준할 것.")
    print("  팁: 한 줄에 몰아 찍으면 H 가 퇴화한다 — 전후·좌우로 넓게.\n")
    if base_pts:
        print("[안내] 기존 CSV %d점은 회색으로 표시된다 (취소 불가 — 참고용)."
              % len(base_pts))

    view = _ZoomViewer(base)
    new_pts = []      # [u, v, x, y]
    labels = []       # [(i, j)]

    def marks():
        """지금까지의 점: 기존 CSV=회색(라벨 없음), 이번 실행=빨강 (i,j) 라벨."""
        out = [(p[0], p[1], "", COLOR_OLD) for p in base_pts]
        out += [(p[0], p[1], "(%s,%s)" % (_fmt_ij(ij[0]), _fmt_ij(ij[1])),
                 COLOR_NEW) for p, ij in zip(new_pts, labels)]
        return out

    while True:
        view.show(marks(), "grid: click joint + type 'i j' | [u]ndo [q]uit "
                  "[+/-]zoom [c]enter (%dpts)" % (len(base_pts) + len(new_pts)))
        key = cv2.waitKey(30) & 0xFF
        while view.clicks:
            u, v = view.clicks.pop(0)
            # 입력 대기 중인 점을 노랗게 강조 (콘솔 입력 동안 창 갱신은 멈춘다 — 정상)
            view.show(marks() + [(u, v, "?", COLOR_HL)],
                      "grid: type 'i j' in console")
            cv2.waitKey(1)
            ij = _ask_grid_ij(len(new_pts) + 1, u, v, labels)
            if ij is None:
                continue
            x, y = ox + ij[0] * pitch, oy + ij[1] * pitch
            new_pts.append([float(u), float(v), float(x), float(y)])
            labels.append(ij)
            print("  → (i=%s, j=%s) = (%.4f, %.4f) m   [이번 실행 %d점]"
                  % (_fmt_ij(ij[0]), _fmt_ij(ij[1]), x, y, len(new_pts)))
        if key == ord("u") and new_pts:
            rm, rij = new_pts.pop(), labels.pop()
            print("취소: 점%d 격자(%s,%s) 픽셀(%.2f,%.2f)"
                  % (len(new_pts) + 1, _fmt_ij(rij[0]), _fmt_ij(rij[1]),
                     rm[0], rm[1]))
        elif key in (ord("q"), 27):
            break
        else:
            view.handle_key(key)
    return new_pts, labels


# ======================== H 계산/검증 ========================

def _shift_frame_x(pts, offset_x):
    """대응점 x 에서 offset 을 빼 원점을 앞으로 옮긴다 (A' 프레임 → 라이다 프레임).

    라이다가 A'(카메라 지면투영점) 보다 offset_x 만큼 앞에 있으면, 같은 바닥점의
    라이다 기준 x 는 (A' 기준 x - offset_x) 다. 픽셀 좌표(u,v)는 그대로다.
    H 는 이렇게 옮긴 점들로 계산해야 출력 프레임이 실제로 바뀐다.
    """
    if not offset_x:
        return [list(p) for p in pts]
    off = float(offset_x)
    return [[u, v, x - off, y] for u, v, x, y in pts]


def _compute_h(pts):
    arr = np.asarray(pts, np.float64)
    uv, xy = arr[:, :2], arr[:, 2:]
    H, _ = cv2.findHomography(uv, xy, 0)  # 점 수 적음 — 전점 최소자승
    if H is None:
        sys.exit("H 계산 실패 — 점 배치가 퇴화(일직선?)했거나 좌표 오입력")
    proj = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - xy, axis=1)
    print("\n===== 재투영 오차 (픽셀 → H → 차량좌표, 단위 m) =====")
    print(" 점   픽셀(u,v)        실측(x,y)          H투영(x,y)         오차")
    for i in range(len(err)):
        print(" %2d  (%6.1f,%6.1f)  (%6.3f,%6.3f)  (%6.3f,%6.3f)  %.3f"
              % (i + 1, uv[i, 0], uv[i, 1], xy[i, 0], xy[i, 1],
                 proj[i, 0], proj[i, 1], err[i]))
    rms = float(np.sqrt(np.mean(err ** 2)))
    print("RMS %.3f m / 최대 %.3f m" % (rms, float(err.max())))
    if rms > RMS_WARN_M:
        print("[경고] RMS %.0fcm > %.0fcm — 실측/클릭 재확인 권장"
              % (rms * 100, RMS_WARN_M * 100))
    np.set_printoptions(precision=8, suppress=False)
    print("H =\n%s" % H)
    return H, rms


def _show_verify(base, pts, H):
    """실측 좌표를 H 역변환으로 되투영(초록 원) — 클릭점(빨강 십자)과 겹치면 정상."""
    disp = _draw_points(base, pts)
    Hinv = np.linalg.inv(H)
    xy = np.asarray(pts, np.float64)[:, 2:]
    back = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    for u, v in back:
        cv2.circle(disp, (int(round(u)), int(round(v))), 7, (0, 255, 0), 2)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(disp, "green=measured reprojected | any key = quit", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(WIN, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="바닥 H 캘리브 — 픽셀→차량좌표 (자세한 사용법은 파일 헤더)")
    ap.add_argument("--image", required=True, help="바닥 기준점 사진")
    ap.add_argument("--camera", required=True, help="camera.yaml (model/K/D 로 undistort)")
    ap.add_argument("--points", default=None,
                    help="u,v,x,y CSV. 단독이면 비대화 모드 입력, "
                         "--grid/--rect/--click 과 같이 주면 누적 파일"
                         "(있으면 읽어서 합침)")
    ap.add_argument("--grid", action="store_true",
                    help="바닥 타일 격자 모드 — 줄눈 교차점을 클릭하고 격자번호 "
                         "'i j' 를 입력하면 지면좌표를 자동 계산 (자세한 설명은 헤더)")
    ap.add_argument("--pitch", type=float, default=GRID_PITCH_DEFAULT,
                    help="--grid 타일 한 칸 피치 m (기본 %.6f = 본선 복도 실측 "
                         "2.72m/6칸)" % GRID_PITCH_DEFAULT)
    ap.add_argument("--origin-x", type=float, default=0.0,
                    help="--grid 격자 원점(i=0) 의 x m (기본 0 = A' 가 줄눈 교차점)")
    ap.add_argument("--origin-y", type=float, default=0.0,
                    help="--grid 격자 원점(j=0) 의 y m (기본 0)")
    ap.add_argument("--enhance", action="store_true",
                    help="표시용 CLAHE+언샤프 (줄눈 대비 강화). 표시 전용이라 "
                         "클릭 좌표·H 에는 영향이 없다")
    ap.add_argument("--frame", choices=sorted(FRAME_NAMES), default="a_prime",
                    help="H 출력 좌표 프레임을 camera.yaml 의 h_frame 에 기록. "
                         "a_prime=카메라 지면투영점 기준(기본), "
                         "lidar=라이다 기준(런타임이 기대하는 프레임)")
    ap.add_argument("--lidar-offset-x", type=float, default=0.0,
                    help="저장 직전 모든 대응점의 x 에서 뺄 값 m "
                         "(A' → 라이다 평행이동. 라이다가 A' 보다 앞이면 양수). 기본 0")
    ap.add_argument("--rect", action="store_true",
                    help="A4 등 직사각형 4점 클릭 모드 (대응점 4개 자동 생성)")
    ap.add_argument("--click", action="store_true",
                    help="--points 를 줘도 개별 점 클릭 수집을 강제 (기본은 비대화). "
                         "누적 CSV 위에 원거리 점 등을 더 얹을 때 --append 와 같이 쓴다")
    ap.add_argument("--rect-size", nargs=2, type=float,
                    default=[A4_W_M, A4_L_M], metavar=("W", "L"),
                    help="사각형 실측 크기 m: 좌우 폭 W, 전후 길이 L "
                         "(기본 %.3f %.3f = A4 세로 배치)" % (A4_W_M, A4_L_M))
    ap.add_argument("--rect-origin", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="첫 사각형 근단(가까운) 변 중앙의 차량좌표 m "
                         "(생략 시 콘솔에서 입력)")
    ap.add_argument("--append", action="store_true",
                    help="이번에 만든 대응점을 --points CSV 에 이어 쓴다 (여러 장 누적)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="plumb_bob undistort 의 getOptimalNewCameraMatrix alpha "
                         "(0=크롭, 1=화소보존). 생략 시 lib/bev.py 기본값을 따라감 — "
                         "주행 노드와 반드시 같아야 H 가 유효하다")
    ap.add_argument("--save", nargs="?", const="@camera", default=None,
                    help="H 저장. 경로 생략 시 --camera 파일에 병합")
    args = ap.parse_args()

    if args.append and not args.points:
        sys.exit("--append 는 --points <csv> 와 같이 써야 한다 (이어 쓸 파일 필요)")
    modes = [n for n, on in (("--grid", args.grid), ("--rect", args.rect),
                             ("--click", args.click)) if on]
    if len(modes) > 1:
        sys.exit("%s 는 같이 쓸 수 없다 — 한 번에 한 수집 방식만. 하나를 "
                 "--append 로 돌린 뒤, 같은 CSV 에 다음 방식을 --append 로 얹을 것"
                 % " / ".join(modes))
    if any(v <= 0.0 for v in args.rect_size):
        sys.exit("--rect-size 는 양수 2개여야 함: %s" % args.rect_size)
    if args.grid and not args.pitch > 0.0:
        sys.exit("--pitch 는 양수여야 함 (타일 한 칸 m): %s" % args.pitch)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit("이미지 읽기 실패: %s" % args.image)
    K, D, model = _load_camera(args.camera)
    alpha = _default_alpha() if args.alpha is None else float(args.alpha)
    und, undistorter = _undistort(img, K, D, model, alpha)

    # 표시용 이미지 — 강화는 undistort '이후' 표시 단계에만. 기하(u,v)는 그대로다.
    disp = und
    if args.enhance:
        disp = _enhance_for_display(und)
        print("--enhance: 표시용 CLAHE(clip %.1f, %dx%d) + 언샤프(sigma %.1f) 적용 "
              "— 픽셀 격자/좌표는 불변이라 H 에는 영향 없음"
              % (CLAHE_CLIP, CLAHE_TILE[0], CLAHE_TILE[1], UNSHARP_SIGMA))

    base_pts = []
    if args.points and os.path.isfile(args.points):
        base_pts = _load_points_csv(args.points)
        print("CSV 로드: %d점 (%s)" % (len(base_pts), args.points))
    elif args.points and not (args.grid or args.rect or args.click):
        # 수집 모드면 CSV 를 새로 만들 수 있다. 비대화 모드는 불가.
        sys.exit("--points 파일 없음: %s" % args.points)

    labels = None
    if args.grid:
        new_pts, labels = _collect_grid(
            disp, float(args.pitch), (float(args.origin_x), float(args.origin_y)),
            base_pts)
    elif args.rect:
        origin = tuple(args.rect_origin) if args.rect_origin else None
        new_pts = _collect_rects(disp, tuple(args.rect_size), origin, base_pts)
    elif args.points and not args.click:
        new_pts = []      # 비대화 모드 — CSV 만 사용
    else:
        new_pts = _collect_interactive(disp, base_pts)

    pts = base_pts + new_pts
    frame = FRAME_NAMES[args.frame]
    if new_pts and args.append:
        if args.grid:
            mode = "grid p=%.6f, origin %.4f/%.4f" % (
                args.pitch, args.origin_x, args.origin_y)
        elif args.rect:
            mode = "rect %.3fx%.3f" % (args.rect_size[0], args.rect_size[1])
        else:
            mode = "click"
        note = "%s (%s, frame=%s)" % (os.path.basename(args.image), mode, frame)
        _append_points_csv(args.points, new_pts, note, labels)

    if len(pts) < MIN_POINTS:
        sys.exit("점 %d개 — 최소 %d개 필요" % (len(pts), MIN_POINTS))
    print("H 계산에 쓰는 대응점: %d개 (기존 %d + 신규 %d)"
          % (len(pts), len(base_pts), len(new_pts)))

    # 프레임 변환 — 저장/계산에 쓰는 점은 여기서 옮긴 값이다 (CSV 는 원본 프레임 유지).
    off = float(args.lidar_offset_x)
    if off:
        print("프레임 변환: 모든 대응점의 x 에서 %.4f m 를 뺀다 → h_frame=%s"
              % (off, frame))
    if frame == "lidar_frame" and not off:
        print("[경고] --frame lidar 인데 --lidar-offset-x 가 0 이다 — 수집 원점이 "
              "이미 라이다가 아니면 x 가 통째로 어긋난다.")
    if frame == "A_prime" and off:
        print("[경고] --lidar-offset-x 를 줬는데 --frame 은 a_prime 이다 — "
              "라이다 기준으로 옮길 거면 --frame lidar 도 같이 줄 것.")
    pts = _shift_frame_x(pts, off)

    H, rms = _compute_h(pts)

    if new_pts:
        _show_verify(disp, pts, H)

    if args.save is not None:
        save_path = args.camera if args.save == "@camera" else args.save
        data = _load_yaml(save_path)
        data["H"] = [[float(v) for v in row] for row in H]
        data["h_calibrated"] = True
        data["h_rms_m"] = float(round(rms, 4))
        data["h_model"] = model  # 이 H 가 어느 왜곡 모델의 undistort 기준인지 기록
        # ★ H 의 출력 원점이 어디인지 — 이게 없으면 11.7cm 급 오프셋을 나중에 못 잡는다
        data["h_frame"] = frame
        data["h_lidar_offset_x"] = off
        if args.grid:
            data["h_grid_pitch_m"] = float(args.pitch)
            data["h_grid_origin"] = [float(args.origin_x), float(args.origin_y)]
        data["h_points"] = [[float(v) for v in p] for p in pts]  # 기록용 (u,v,x,y)
        if undistorter is not None:
            # ★ H 는 undistort 픽셀 기준 — 그때 쓴 new_K 를 같이 남겨야 런타임이
            #   bev.check_new_K_match() 로 좌표계 일치를 검증할 수 있다.
            nk = np.asarray(undistorter.new_K, np.float64).reshape(3, 3)
            data["new_K"] = [[float(v) for v in row] for row in nk]
            data["h_alpha"] = float(alpha)
            print("new_K 기록: fx,fy,cx,cy = %.3f %.3f %.3f %.3f (alpha=%.2f)"
                  % (nk[0, 0], nk[1, 1], nk[0, 2], nk[1, 2], alpha))
        else:
            # undistort 를 안 탔으면 H 는 원본 픽셀 기준 — 낡은 new_K 를 남기면
            # 런타임이 잘못된 좌표계를 '일치'로 오판한다.
            if data.pop("new_K", None) is not None:
                print("[안내] undistort 미적용이라 기존 new_K 항목을 제거했다.")
            data.pop("h_alpha", None)
        intr = _flag(data, "intrinsics_calibrated")
        data["calibrated"] = bool(intr)  # K/D 까지 있어야 true
        _save_camera_yaml(save_path, data, "tools/calib_ground_h.py")
        print("h_frame 기록: %s" % frame)
        if frame != "lidar_frame":
            print("[경고] 이 H 의 출력 원점은 %s 다 (라이다가 아니다)." % frame)
            print("        런타임(lib/bev)은 H 를 lidar_frame 으로 간주하므로, "
                  "A' 와 라이다 사이 거리만큼")
            print("        모든 x 가 어긋난 채로 쓰인다 (본선 실차 기준 약 0.117m).")
            print("        라이다 기준으로 바꾸려면 같은 대응점 CSV 로 다시 돌릴 것:")
            print("          --points <csv> --frame lidar --lidar-offset-x 0.117 --save")
        if not intr:
            print("[경고] K/D 미캘리브 상태 — calibrated 는 false 유지. "
                  "calib_camera.py(또는 import_camera_info.py) 후 이 도구를 다시 "
                  "돌릴 것 (undistort 기준이 달라짐).")
        else:
            print("다음 단계: verify_lidar_cam.py 로 라이다-카메라 정합 육안 확인")


if __name__ == "__main__":
    main()
