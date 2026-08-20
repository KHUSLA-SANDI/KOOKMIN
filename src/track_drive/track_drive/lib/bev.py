"""BEV 호모그래피 + 렌즈 왜곡 보정 + 차선점 추출 (순수 파이썬, rclpy 금지).

역할: 카메라 캘리브(config/camera.yaml) 로드, 렌즈 왜곡 보정(plumb_bob/fisheye),
      이미지 마스크 → BEV warp → lidar 좌표 차선 중심점 추출, 품질 점수.
입력: camera.yaml (model/K/D/new_K/H + intrinsics_calibrated/h_calibrated),
      BGR 이미지, 차선 마스크(uint8, 이미지 픽셀 좌표).
출력: (xs, ys) lidar_frame 미터 (x전방/y좌), lane_quality 0~1.
예선 대비 변경점: lane_detect_node 의 BEV 상수·_mask_to_centerline_xy 를 lib 로
      승격(원문 유지). H 는 yaml 에서 로드하고 미캘리브 시 예선 시뮬 폴백 H 사용.
      Undistorter 신설 — intrinsics_calibrated=False 면 no-op 패스스루.
캘리브 플래그는 2개로 분리돼 있다 (부분 캘리브가 실제로 자주 생긴다):
      intrinsics_calibrated = K/D 유효 → undistort 실행 여부를 결정,
      h_calibrated          = H 유효    → 지면 좌표 신뢰 여부를 결정.
      구버전 파일의 calibrated 단일 키는 "둘 다"의 의미로 해석한다(하위호환).
본선 실측 반영: 벤더 PDF 는 "170도 어안" 이라 했으나 실측 HFOV≈87°/VFOV≈64° 의
      일반 광각 렌즈였다. 87° 렌즈에 등거리(fisheye) 모델을 쓰면 과파라미터화로
      CHECK_COND 실패·가장자리 악화가 난다. 그래서 기본 왜곡모델을 plumb_bob 으로
      두고, fisheye 경로는 지우지 않고 model 필드로 고르는 하위호환으로 남긴다
      (카메라 교체나 진짜 어안 사용 대비).
"""

import os

import cv2
import numpy as np

try:
    import yaml
except ImportError:  # PyYAML 없는 환경(순수 유닛테스트)에서도 import 는 되게
    yaml = None


# ======================== BEV 상수 (예선 common.py 와 동일 유지) ========================

# ★ 2026-08-12 실차 실측으로 교체. 예선 시뮬 값(RES 0.05 / SUB 3 / BIN 0.18)은
#   굵은 시뮬 차선 기준이라, 실차의 얇은 노란테이프에서는 마스크가 14,451화소나
#   되는데도 BEV 점이 1개만 나왔다 (경로 생성 최소 6개 → 영구 실패).
#   같은 프레임에서 RES 0.02 / SUB 1 / BIN 0.05 로 바꾸면 20개 안팎이 나온다.
#   워프 비용은 0.14ms → 0.82ms 로 무시할 수준(전체 주기 128ms).
BEV_RES = 0.02                      # 실측 (was 0.05 [SIM])
BEV_X_MIN, BEV_X_MAX = -2.0, 12.0   # [SIM] 전방 범위 m
BEV_Y_HALF = 5.0                    # [SIM] 좌우 반폭 m
BEV_W = int(round((2 * BEV_Y_HALF) / BEV_RES))          # 500
BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))   # 700

# lidar (x전방/y좌, m) → BEV 픽셀 (u,v) 그리드 행렬
M_GRID = np.array([
    [ 0.0,           -1.0 / BEV_RES, BEV_Y_HALF / BEV_RES],
    [-1.0 / BEV_RES,  0.0,           BEV_X_MAX  / BEV_RES],
    [ 0.0,            0.0,           1.0],
], dtype=np.float64)

# 예선 시뮬 캘리브 호모그래피 (이미지 픽셀 → lidar 미터). 실차 캘리브 전 폴백.
H_PIX2LIDAR_SIM_FALLBACK = np.array([  # [SIM] common.H_PIX2LIDAR_FALLBACK 복사본
    [ 0.00000000000000e+00,  4.63055528548851e-03, -2.73689017508590e+00],
    [ 4.18097842466549e-03,  1.87301533010043e-04, -1.37819064484720e+00],
    [ 0.00000000000000e+00, -2.40130177039888e-03,  5.16378846754945e-01],
], dtype=np.float64)

LANE_PIX_SUBSAMPLE = 1          # 실측 (was 3 [SIM]) — 얇은 테이프는 3칸 건너뛰면 사라진다
LANE_CENTERLINE_BIN_M = 0.05    # 실측 (was 0.18 [SIM]) — 점이 1.6m 안에만 있어 촘촘히 떠야 한다

# lane_quality 등급 기준 (본선 스펙: n/20 × x스팬/3.0)
LANE_QUALITY_GOOD_POINTS = 20.0
LANE_QUALITY_GOOD_SPAN_M = 3.0


# ======================== 왜곡 모델 ========================

# plumb_bob = 브라운-콘라디(일반 광각). D = [k1, k2, p1, p2, k3] (5개)
MODEL_PLUMB_BOB = "plumb_bob"
# fisheye = 등거리 어안. D = [k1, k2, k3, k4] (4개)
MODEL_FISHEYE = "fisheye"
VALID_MODELS = (MODEL_PLUMB_BOB, MODEL_FISHEYE)
# 실차 Xycar 카메라가 실측 HFOV≈87° 일반 광각이므로 기본값은 plumb_bob
DEFAULT_MODEL = MODEL_PLUMB_BOB

# getOptimalNewCameraMatrix alpha 기본값.
# 0.0 = 검은 여백 없이 크롭, 1.0 = 원본 화소 전부 보존(가장자리에 검은 여백)
UNDISTORT_ALPHA_DEFAULT = 0.0

# new_K 일치 판정 허용 오차 (픽셀). 이보다 크게 벌어지면 H 가 무효.
NEW_K_MATCH_TOL = 1e-3


def resolve_distortion_model(model, D=None):
    """model 문자열 정규화. 미지정/미상이면 D 길이로 추론한다 (5→plumb_bob, 4→fisheye).

    반환: (model, note) — note 는 추론이 일어났을 때만 채워지는 로그용 한국어 문자열
          (정상적으로 model 이 명시돼 있으면 빈 문자열).
    """
    name = str(model).strip().lower() if model is not None else ""
    if name in VALID_MODELS:
        return name, ""

    try:
        n = int(np.asarray(D, dtype=np.float64).reshape(-1).size) if D is not None else 0
    except (TypeError, ValueError):
        n = 0

    if n == 4:
        guessed = MODEL_FISHEYE
    elif n >= 5:
        guessed = MODEL_PLUMB_BOB
    else:
        guessed = DEFAULT_MODEL

    if name:
        note = (f"camera.yaml model='{model}' 은 알 수 없는 값 — "
                f"D 길이({n})로 '{guessed}' 추론. model 은 "
                f"{'/'.join(VALID_MODELS)} 중 하나여야 한다.")
    else:
        note = (f"camera.yaml 에 model 필드 없음 — D 길이({n})로 '{guessed}' 자동 추론. "
                f"model 을 명시할 것 ({'/'.join(VALID_MODELS)}).")
    return guessed, note


def _coerce_D(D, n_min):
    """D 를 1차원 float64 로 펴고 최소 길이 n_min 을 0 패딩으로 보장."""
    d = np.asarray(D, dtype=np.float64).reshape(-1)
    if d.size < n_min:
        d = np.concatenate([d, np.zeros(n_min - d.size, dtype=np.float64)])
    return d


def _as_3x3_or_none(M):
    """3x3 float64 로 변환. 실패하거나 None 이면 None."""
    if M is None:
        return None
    try:
        return np.asarray(M, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None


def _as_size_or_none(data):
    """yaml 의 image_width/image_height → (w, h). 없거나 이상하면 None.

    new_K 는 이 크기 기준으로 계산되므로, 런타임 프레임 크기가 다르면 H 가 무효다.
    """
    try:
        w = int(data["image_width"])
        h = int(data["image_height"])
    except (KeyError, TypeError, ValueError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def _as_car_mask_v(data, image_size):
    """yaml 의 car_mask_v → 차체 가림선 v (보정영상 픽셀). 없거나 이상하면 None.

    이 v 아래(=화면 더 아래)는 차체(라이다 원통·범퍼·바퀴)라 지면이 아니다.
    그대로 두면 차체 화소가 BEV 에서 x≈0.1~0.2m 의 가짜 차선점으로 변환된다.
    tools/bev_view.py 에서 m 키로 클릭해 정한 값 — 보정영상 좌표계 기준이라
    프레임 크기가 캘리브 기준과 다르면 비율로 환산해 써야 한다.
    """
    try:
        v = float(data["car_mask_v"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    # 캘리브 기준 높이를 넘는 값은 사실상 "안 자름" — None 과 같으니 버린다.
    if image_size and v >= image_size[1]:
        return None
    return v


# ======================== 캘리브 로드 ========================

def resolve_calib_flags(data):
    """yaml dict → (intrinsics_calibrated, h_calibrated). 하위호환 규칙 포함.

    - 두 플래그가 각각 있으면 그대로 쓴다.
    - 없으면 구버전 단일 키 calibrated 값을 기본값으로 쓴다
      (구버전 calibrated:true = K/D·H 둘 다 유효라는 뜻이었다).
    """
    data = data if isinstance(data, dict) else {}
    legacy = bool(data.get("calibrated", False))
    intr = bool(data["intrinsics_calibrated"]) \
        if "intrinsics_calibrated" in data else legacy
    h_ok = bool(data["h_calibrated"]) if "h_calibrated" in data else legacy
    return intr, h_ok


def load_camera_config(path):
    """camera.yaml → dict(K, D, H, model, new_K, image_size, note,
                          intrinsics_calibrated, h_calibrated, calibrated).

    ★ 플래그 분리: K/D 의 유효성(intrinsics_calibrated)과 H 의 유효성(h_calibrated)은
      따로 간다. 실차에서는 "K/D 는 임포트했지만 지면 H 는 아직" 이 정상 중간 상태라,
      예전처럼 하나의 calibrated 로 묶으면 실측 K/D 까지 폴백으로 버려져
      undistort 가 통째로 죽는다.
        intrinsics_calibrated=True  → yaml 의 K/D 를 그대로 쓴다 (undistort 실행).
        intrinsics_calibrated=False → K=단위행렬 / D=0 폴백 (undistort 패스스루).
        h_calibrated=True  → yaml 의 H(+new_K) 사용.
        h_calibrated=False → 예선 시뮬 폴백 H 사용, new_K 는 무시(H 와 짝이므로).
      구버전 파일의 단일 키 calibrated 는 두 플래그의 기본값으로 해석한다.
      반환 dict 의 "calibrated" 는 하위호환용 요약 = (intrinsics and h).

    model: "plumb_bob"(일반 광각·기본) | "fisheye"(등거리 어안).
           yaml 에 없거나 값이 이상하면 D 길이로 추론(5→plumb_bob, 4→fisheye)하고
           그 사실을 note(로그용 문자열)에 남긴다.
    D:     model 에 맞춰 길이 정규화 — fisheye 는 4개, plumb_bob 은 최소 5개.
    new_K: undistort 결과 픽셀 좌표계를 정의하는 카메라 행렬. H 가 이 좌표계 기준이라
           캘리브 때 쓴 값을 그대로 저장/로드해야 H 가 유효하다. yaml 에 없으면 None
           이고, 이때 Undistorter 가 모델 규약대로 계산한다
           (fisheye → new_K=K, plumb_bob → getOptimalNewCameraMatrix(alpha)).
    image_size: yaml 의 image_width/height → (w, h). 캘리브 기준 해상도이며
           new_K/H 가 이 크기 기준이다. 없으면 None.
    note:  호출측(노드)이 그대로 찍으면 되는 한국어 경고/안내. 없으면 빈 문자열.
           폴백이 일어난 사유(K/D 미보정, H 미보정 등)가 여기 쌓인다.

    파일 없음 / 파싱 실패면 전부 폴백 + 두 플래그 False 반환 —
    호출측(노드)이 반드시 경고 로그를 남길 것.
    """
    fallback = {
        "K": np.eye(3, dtype=np.float64),
        "D": np.zeros(5, dtype=np.float64),   # 기본 모델(plumb_bob) 길이
        "H": H_PIX2LIDAR_SIM_FALLBACK.copy(),
        "calibrated": False,
        "intrinsics_calibrated": False,
        "h_calibrated": False,
        "model": DEFAULT_MODEL,
        "new_K": None,
        "image_size": None,
        "car_mask_v": None,
        "note": "",
    }

    if yaml is None or not path or not os.path.isfile(path):
        return fallback

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return fallback
        intr, h_ok = resolve_calib_flags(data)
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        D_raw = np.asarray(data["D"], dtype=np.float64).reshape(-1)
        H = np.asarray(data["H"], dtype=np.float64).reshape(3, 3)
        # model 은 D 를 자르기 전의 원본 길이로 추론해야 한다 (5 vs 4 구분)
        model, note = resolve_distortion_model(data.get("model"), D_raw)
        D = _coerce_D(D_raw, 4)[:4] if model == MODEL_FISHEYE else _coerce_D(D_raw, 5)
        new_K = _as_3x3_or_none(data.get("new_K"))
        image_size = _as_size_or_none(data)
        car_mask_v = _as_car_mask_v(data, image_size)
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError):
        return fallback

    notes = [note] if note else []

    if not intr:
        # K/D 를 신뢰할 수 없다 → 폴백으로 치환해 undistort 가 패스스루가 되게 한다.
        K = fallback["K"].copy()
        D = np.zeros(4 if model == MODEL_FISHEYE else 5, dtype=np.float64)
        notes.append(
            "intrinsics_calibrated=false — K/D 를 단위행렬/0 폴백으로 대체했다 "
            "(undistort 패스스루). tools/import_camera_info.py 또는 "
            "calib_camera.py 로 K/D 를 먼저 채울 것.")

    if not h_ok:
        # 미캘리브 H 는 신뢰 금지 — 검증된 예선 시뮬 폴백 H 를 쓴다.
        H = H_PIX2LIDAR_SIM_FALLBACK.copy()
        reason = ("h_calibrated=false — H 를 예선 시뮬 폴백값으로 대체했다. "
                  "실차 지면 좌표(BEV/차선)는 부정확하다 — "
                  "tools/calib_ground_h.py 로 H 를 잡을 것.")
        if new_K is not None:
            # new_K 는 H 와 짝(같은 undistort 픽셀 좌표계)이라 H 없이는 의미가 없다.
            reason += " yaml 의 new_K 는 H 와 짝이므로 함께 무시한다."
            new_K = None
        if intr:
            # undistort 는 켜지는데 H 는 원본픽셀 기준 시뮬값 → 좌표계가 어긋난다.
            reason += (" ※ K/D 는 유효해 undistort 는 켜지지만 폴백 H 는 "
                       "undistort 이전 픽셀 기준이라 좌표계가 어긋난 상태다.")
        notes.append(reason)
    elif not intr:
        notes.append("h_calibrated=true 인데 K/D 는 미보정 — H 가 원본(raw) 픽셀 "
                     "기준으로 잡힌 것인지 확인할 것.")

    return {"K": K, "D": D, "H": H,
            "calibrated": bool(intr and h_ok),   # 하위호환 요약 키
            "intrinsics_calibrated": bool(intr),
            "h_calibrated": bool(h_ok),
            "model": model, "new_K": new_K, "image_size": image_size,
            "car_mask_v": car_mask_v,
            "note": " / ".join(notes)}


# ======================== 렌즈 왜곡 보정 (plumb_bob / fisheye) ========================

class Undistorter:
    """렌즈 왜곡 보정. 리맵 테이블 1회 생성 후 캐시 (프레임마다 재계산 금지).

    model="plumb_bob": 일반 광각(브라운-콘라디). D=[k1,k2,p1,p2,k3],
        cv2.getOptimalNewCameraMatrix(alpha) + cv2.initUndistortRectifyMap.
        실차 Xycar 카메라(실측 HFOV≈87°/VFOV≈64°)가 여기 해당 — 기본값.
    model="fisheye": 등거리 어안. D=[k1,k2,k3,k4], cv2.fisheye.* (기존 경로 그대로).
        87° 급 렌즈에 쓰면 과파라미터화로 가장자리가 오히려 나빠지니 진짜 어안일 때만.
    model=None: D 길이로 자동 추론 (5→plumb_bob, 4→fisheye). 추론 사유는
        self.model_note 에 남는다.

    calibrated=False (= K/D 미보정, camera.yaml 의 intrinsics_calibrated=false) 면
    no-op 패스스루. 초기화 실패 시에도 패스스루로 떨어지므로 호출측은 enabled 를
    확인해 경고 로그를 남길 것.
    ※ 이 인자는 K/D 유효성만 뜻한다 — H 유효성(h_calibrated)과 무관하다.
      from_config() 을 쓰면 intrinsics_calibrated 가 자동으로 전달된다.

    ★ new_K 주의: H 는 "undistort 된 픽셀 → 지면좌표" 라서, undistort 에 쓴 new_K 가
      바뀌면 H 가 통째로 무효가 된다 (plumb_bob 은 alpha·이미지크기에 따라 new_K 가
      달라진다). 캘리브 때 쓴 new_K 를 camera.yaml 의 new_K 에 적어 넘기고,
      런타임 값과 check_new_K_match() 로 대조할 것.
    """

    def __init__(self, K, D, size, calibrated=True,
                 model=None, alpha=UNDISTORT_ALPHA_DEFAULT, new_K=None):
        # size = (width, height) 예: (640, 480)
        self.enabled = False
        self.alpha = float(alpha)
        self.roi = None                 # plumb_bob 에서 유효 화소 영역 (x, y, w, h)
        self.model, self.model_note = resolve_distortion_model(model, D)
        # 패스스루/초기화 실패 시엔 원본 픽셀 좌표 그대로이므로 new_K = K
        try:
            self.new_K = None if K is None else np.asarray(K, dtype=np.float64)
        except (TypeError, ValueError):
            self.new_K = None
        if not calibrated or K is None or D is None:
            return

        try:
            K_m = np.asarray(K, dtype=np.float64).reshape(3, 3)
            w, h = int(size[0]), int(size[1])
            if self.model == MODEL_FISHEYE:
                D_m = _coerce_D(D, 4)[:4]
                # 어안 규약: new_K = K 유지 → yaml 의 H(undistort 이미지 기준)와 좌표 일치.
                # (yaml 에 new_K 가 명시돼 있으면 그쪽을 우선 — 캘리브 당시 값이 정답)
                nk = K_m.copy() if new_K is None else np.asarray(
                    new_K, dtype=np.float64).reshape(3, 3)
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K_m, D_m.reshape(4, 1), np.eye(3), nk, (w, h), cv2.CV_16SC2)
            else:
                D_m = _coerce_D(D, 5)
                if new_K is None:
                    # alpha=0.0 → 검은 여백 없이 크롭, 1.0 → 원본 화소 전부 보존
                    nk, roi = cv2.getOptimalNewCameraMatrix(
                        K_m, D_m, (w, h), self.alpha)
                    nk = np.asarray(nk, dtype=np.float64).reshape(3, 3)
                    self.roi = tuple(int(v) for v in roi)
                else:
                    nk = np.asarray(new_K, dtype=np.float64).reshape(3, 3)
                map1, map2 = cv2.initUndistortRectifyMap(
                    K_m, D_m, np.eye(3), nk, (w, h), cv2.CV_16SC2)
            # 맵 생성이 끝난 뒤에만 상태 확정 (실패 시 패스스루 상태 유지)
            self._map1, self._map2 = map1, map2
            self.new_K = nk
            self.enabled = True
        except (cv2.error, TypeError, ValueError, IndexError):
            # 잘못된 K/D/size — 패스스루 유지 (호출측이 enabled 확인해 경고 로그)
            self.enabled = False

    @classmethod
    def from_config(cls, cam, size, alpha=UNDISTORT_ALPHA_DEFAULT):
        """load_camera_config() 결과 dict 로 바로 생성 (model/new_K 자동 전달).

        undistort 가능 여부는 K/D 유효성 = intrinsics_calibrated 로 판단한다.
        (H 미보정(h_calibrated=false)이어도 K/D 가 실측이면 undistort 는 켠다.)
        구버전 dict 처럼 intrinsics_calibrated 키가 없으면 calibrated 로 대체.
        """
        cam = cam or {}
        intr = cam.get("intrinsics_calibrated", cam.get("calibrated", False))
        return cls(cam.get("K"), cam.get("D"), size,
                   calibrated=bool(intr),
                   model=cam.get("model"), alpha=alpha, new_K=cam.get("new_K"))

    def apply(self, bgr):
        if not self.enabled or bgr is None:
            return bgr
        return cv2.remap(bgr, self._map1, self._map2,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def check_new_K_match(new_K_calib, new_K_runtime, tol=NEW_K_MATCH_TOL):
    """캘리브 때 쓴 new_K 와 런타임 new_K 를 대조해 H 유효성을 확인한다.

    H 는 undistort 된 픽셀 좌표계 기준이라 new_K 가 달라지면 H 가 통째로 무효다.
    보통 new_K_calib=cam["new_K"], new_K_runtime=undist.new_K 로 부른다.

    반환: 문제 없으면 None, 문제가 있으면 로그용 한국어 경고 문자열.
      - 둘 다 없음: 비교 대상 자체가 없음(미캘리브 등) → None
      - 한쪽만 없음: 기록 누락 → 경고 (좌표계 일치를 검증할 수 없다)
      - 최대 절대차 > tol: 불일치 → 경고 (H 재캘리브 필요)
    """
    a = _as_3x3_or_none(new_K_calib)
    b = _as_3x3_or_none(new_K_runtime)

    if a is None and b is None:
        return None
    if a is None:
        return ("camera.yaml 에 new_K 가 없다 — 런타임이 자체 계산한 new_K 로 "
                "undistort 하므로 H(undistort 픽셀 → 지면) 좌표계가 맞는지 검증할 수 "
                "없다. H 캘리브에 쓴 new_K 를 camera.yaml 에 기록할 것.")
    if b is None:
        return ("런타임 new_K 를 얻지 못했다 (Undistorter 초기화 실패?) — "
                "camera.yaml 의 new_K 와 대조 불가.")

    diff = float(np.max(np.abs(a - b)))
    if diff > float(tol):
        return (f"new_K 불일치 (최대 절대차 {diff:.4f} > 허용 {float(tol):g}) — "
                "H 는 캘리브 당시 new_K 기준이라 지금 좌표계에서는 무효다. "
                "왜곡모델/alpha/이미지 크기를 캘리브 때와 같게 맞추거나 H 를 재캘리브할 것.")
    return None


# ======================== 마스크 → lidar 좌표 차선점 ========================

def _lane_mask_to_lidar_xy(mask):
    """BEV 마스크 픽셀 → lidar_frame (x전방, y좌) 미터. 예선 원문 유지."""
    if mask is None or not mask.any():
        return np.array([]), np.array([])
    vs, us = np.where(mask > 0)
    vs = vs[::LANE_PIX_SUBSAMPLE]
    us = us[::LANE_PIX_SUBSAMPLE]
    right_m = us.astype(np.float32) * BEV_RES - BEV_Y_HALF
    forward_m = BEV_X_MAX - vs.astype(np.float32) * BEV_RES
    return forward_m, -right_m


def _mask_to_centerline_xy(mask):
    """YOLO seg 출력처럼 전방 x-bin마다 차선 중심점 하나만 만든다.

    원 YOLO 노드는 각 mask row에서 u 중심점만 발행했다. OpenCV 마스크 픽셀을
    그대로 넘기면 차선 두께/노이즈가 2차식 fit에 들어가 직선 와리가리가 생긴다.
    (예선 lane_detect_node._mask_to_centerline_xy 원문 이식)
    """
    xs, ys = _lane_mask_to_lidar_xy(mask)
    if xs.size == 0:
        return xs, ys

    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size == 0:
        return xs, ys

    bins = np.floor(xs / LANE_CENTERLINE_BIN_M).astype(np.int32)
    out_x = []
    out_y = []
    for bid in np.unique(bins):
        mask_bin = bins == bid
        if int(np.count_nonzero(mask_bin)) < 2:
            continue
        out_x.append(float(np.median(xs[mask_bin])))
        out_y.append(float(np.median(ys[mask_bin])))

    if not out_x:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    out_x = np.asarray(out_x, dtype=np.float32)
    out_y = np.asarray(out_y, dtype=np.float32)
    order = np.argsort(out_x)
    return out_x[order], out_y[order]


# ================== 원본 픽셀에서 바로 지면으로 (사진을 펴지 않는다) ==================
# ★ 왜 이 경로가 필요한가
#   예전에는 사진 전체를 undistort 한 뒤 YOLO 에 넣었다. 그런데
#     · YOLO 는 **원본**으로 학습했다 (녹화 원본 md5 와 학습 사진이 동일)
#     · undistort 결과를 같은 1920x1080 에 담으려면 out_fx 로 확대/축소해야 하는데,
#       확대하면 좌우가 잘리고(원본의 65%만 남았다), 축소하면 물체가 작아져
#       먼 점선을 YOLO 가 못 잡는다 (실측: 검출거리 1.40m → 1.07m)
#   사진 전체를 펼 이유가 없다. **검출된 점 몇백 개만** 펴서 H 에 넣으면
#   결과는 수학적으로 같고, 화각도 해상도도 하나도 안 잃는다.


def undistort_pixels(us, vs, K, D, new_K, model=None):
    """원본 픽셀 → 보정영상 픽셀. 사진이 아니라 **점만** 편다."""
    us = np.asarray(us, np.float64).ravel()
    vs = np.asarray(vs, np.float64).ravel()
    if us.size == 0:
        return np.empty((0, 2), np.float64)
    pts = np.stack([us, vs], 1).reshape(-1, 1, 2)
    K = np.asarray(K, np.float64).reshape(3, 3)
    nk = np.asarray(new_K, np.float64).reshape(3, 3)
    mdl, _ = resolve_distortion_model(model, D)
    if mdl == MODEL_FISHEYE:
        d = _coerce_D(D, 4)[:4].reshape(4, 1)
        out = cv2.fisheye.undistortPoints(pts, K, d, np.eye(3), nk)
    else:
        out = cv2.undistortPoints(pts, K, _coerce_D(D, 5), None, np.eye(3), nk)
    return out.reshape(-1, 2)


def pixels_to_ground(us, vs, H):
    """보정영상 픽셀 → 지면 (x 전방, y 좌) 미터."""
    us = np.asarray(us, np.float64).ravel()
    vs = np.asarray(vs, np.float64).ravel()
    if us.size == 0:
        return np.array([]), np.array([])
    q = np.asarray(H, np.float64) @ np.stack([us, vs, np.ones_like(us)], 0)
    w = q[2]
    ok = np.abs(w) > 1e-9
    w = np.where(ok, w, 1.0)
    x = np.where(ok, q[0] / w, np.nan)
    y = np.where(ok, q[1] / w, np.nan)
    return x, y


def centerline_from_ground(xs, ys, bin_m=LANE_CENTERLINE_BIN_M):
    """지면 점 무리 → x 칸마다 중앙값 하나. (_mask_to_centerline_xy 와 같은 규칙)"""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        return np.array([], np.float32), np.array([], np.float32)
    bins = np.floor(xs / bin_m).astype(np.int64)
    ox, oy = [], []
    for bid in np.unique(bins):
        m = bins == bid
        if int(np.count_nonzero(m)) < 2:
            continue
        ox.append(float(np.median(xs[m])))
        oy.append(float(np.median(ys[m])))
    if not ox:
        return np.array([], np.float32), np.array([], np.float32)
    ox = np.asarray(ox, np.float32)
    oy = np.asarray(oy, np.float32)
    o = np.argsort(ox)
    return ox[o], oy[o]


def raw_mask_to_bev_points(mask, H, K, D, new_K, model=None,
                           subsample=LANE_PIX_SUBSAMPLE):
    """**원본(왜곡 있는) 마스크** → 지면 차선 중심점 (xs, ys).

    mask 는 YOLO 가 원본 프레임에서 낸 이진 마스크다. 사진을 펴지 않았으므로
    화각도 해상도도 그대로다.
    """
    if mask is None or mask.size == 0:
        return np.array([], np.float32), np.array([], np.float32)
    vs, us = np.where(mask > 0)
    if vs.size == 0:
        return np.array([], np.float32), np.array([], np.float32)
    if subsample > 1:
        vs = vs[::subsample]
        us = us[::subsample]
    uv = undistort_pixels(us, vs, K, D, new_K, model)
    x, y = pixels_to_ground(uv[:, 0], uv[:, 1], H)
    # ★ 반드시 잘라야 한다. 지평선 근처 화소는 지면에서 수백 m 로 튄다
    #   (옛 경로는 BEV 캔버스가 대신 잘라 줬다 — 실측 최대 186m 가 나왔다).
    ok = (np.isfinite(x) & np.isfinite(y)
          & (x >= BEV_X_MIN) & (x <= BEV_X_MAX) & (np.abs(y) <= BEV_Y_HALF))
    return centerline_from_ground(x[ok], y[ok])


def car_mask_rows_raw(car_mask_v, K, D, new_K, size, model=None):
    """보정영상의 차체 가림선(v) 이 **원본에서는 어느 행**인가 — 열마다 다르다.

    보정영상의 가로선은 원본에서 휘어 있으므로 한 줄이 아니라 곡선이 된다.
    반환: 길이 W 의 배열. 그 열에서 이 행 이후는 차체다.
    """
    if car_mask_v is None:
        return None
    W, Hh = int(size[0]), int(size[1])
    K = np.asarray(K, np.float64).reshape(3, 3)
    nk = np.asarray(new_K, np.float64).reshape(3, 3)
    d = _coerce_D(D, 5)
    us = np.arange(W, dtype=np.float64)
    vs = np.full(W, float(car_mask_v), np.float64)
    xn = (us - nk[0, 2]) / nk[0, 0]
    yn = (vs - nk[1, 2]) / nk[1, 1]
    r2 = xn * xn + yn * yn
    k1, k2, p1, p2, k3 = (list(np.asarray(d).ravel()) + [0.0] * 5)[:5]
    rad = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    v_raw = K[1, 1] * yd + K[1, 2]
    return np.clip(v_raw, 0, Hh)


def apply_car_mask_raw(mask, rows):
    """원본 마스크에서 차체 영역(열마다 다른 가림선 아래)을 0 으로."""
    if mask is None or mask.size == 0 or rows is None:
        return mask
    Hh, W = mask.shape[:2]
    r = np.asarray(rows, np.float64)
    if r.size != W:
        r = np.interp(np.linspace(0, 1, W), np.linspace(0, 1, r.size), r)
    keep = np.arange(Hh, dtype=np.float64)[:, None] < r[None, :]
    mask[~keep] = 0
    return mask


def car_mask_row(car_mask_v, frame_h, calib_h=None):
    """차체 가림선 v → 현재 프레임 높이 기준의 행 인덱스. 자를 게 없으면 None.

    car_mask_v 는 캘리브 기준 해상도(calib_h)의 보정영상 좌표다. 런타임 프레임이
    다른 높이로 들어오면 비율로 환산해야 한다 (환산 안 하면 640x480 프레임에서
    v=951 이 화면 밖이라 마스킹이 통째로 무효가 된다).
    """
    if car_mask_v is None or not frame_h:
        return None
    v = float(car_mask_v)
    if calib_h and calib_h > 0 and int(calib_h) != int(frame_h):
        v *= float(frame_h) / float(calib_h)
    row = int(round(v))
    if row <= 0:
        return 0           # 전부 차체 (사실상 오설정이지만 정직하게 반영)
    if row >= int(frame_h):
        return None        # 화면 밖 → 자를 게 없다
    return row


def apply_car_mask(mask, car_mask_v, calib_h=None):
    """마스크의 차체 영역(가림선 아래 전부)을 0 으로. 원본을 제자리 수정하고 반환.

    호출측이 매번 슬라이스 계산을 반복하지 않도록 lib 로 올렸다.
    car_mask_v 가 None 이면 아무것도 안 한다 (미설정 = 안 자름).
    """
    if mask is None or mask.size == 0:
        return mask
    row = car_mask_row(car_mask_v, mask.shape[0], calib_h)
    if row is not None:
        mask[row:, :] = 0
    return mask


def mask_to_bev_points(mask, H):
    """이미지 픽셀 마스크 → BEV warp → lidar 좌표 차선 중심점 (xs, ys).

    mask: uint8 이진 마스크 (undistort 된 이미지와 같은 픽셀 좌표계).
    H: 이미지 픽셀 → lidar 미터 호모그래피 (load_camera_config 의 "H").
    """
    if mask is None or mask.size == 0:
        return (np.array([], dtype=np.float32),
                np.array([], dtype=np.float32))

    H_pix2bev = M_GRID @ np.asarray(H, dtype=np.float64)
    bev_mask = cv2.warpPerspective(
        mask.astype(np.uint8), H_pix2bev, (BEV_W, BEV_H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _mask_to_centerline_xy(bev_mask)


# ======================== 차선 품질 (등급형 0~1) ========================

def lane_quality(xs, ys):
    """중심점 개수 × 전방 x스팬 기반 등급 점수 (본선 /detect/lane_quality 규격)."""
    xs = np.asarray(xs, dtype=np.float64)
    n = int(xs.size)
    if n == 0:
        return 0.0
    span_x = float(np.max(xs) - np.min(xs)) if n > 1 else 0.0
    count_score = float(np.clip(n / LANE_QUALITY_GOOD_POINTS, 0.0, 1.0))
    span_score = float(np.clip(span_x / LANE_QUALITY_GOOD_SPAN_M, 0.0, 1.0))
    return count_score * span_score
