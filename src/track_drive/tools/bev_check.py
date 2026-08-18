#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BEV 타일 격자 검증 — "폈을 때 타일이 실제 크기 정사각형인가" 육안 확인 (독립 CLI).

역할: camera.yaml 의 H(undistort 픽셀 → 지면 m)로 역방향 매핑 BEV 를 만들고,
      그 위에 실측 타일 피치 격자선(노랑 점선)과 0.5 m 눈금(초록)을 그린다.
입력: --image 바닥 사진(왜곡 있는 raw), --camera camera.yaml (model/K/D/H/new_K).
출력: BEV 오버레이 이미지(--save 없으면 창 표시) + 콘솔 정량 지표.
예선 대비 변경점: 신규 도구 (시뮬은 tf_static 으로 지면 좌표가 보장돼 검증이 불필요했음).

★ 최종 판정 기준: 노란 점선이 사진 속 줄눈 위에 겹치는가.
  겹치면 H 캘리브 성공, 어긋나면 calib_ground_h.py 재캘리브.
  격자선을 실선이 아니라 점선으로 그리는 이유가 이것이다 — 실선은 판정 대상인
  줄눈을 덮어 버려서 "겹쳤는지" 를 눈으로 확인할 수가 없다.

콘솔 수치의 의미 (오해 주의):
  BEV 는 지면 미터를 --ppm 배 한 선형 축소판이라, BEV 상의 타일 한 칸 크기는
  구성상 항상 pitch × ppm 이다. 따라서 [2] 의 픽셀 크기 오차는 "타일이 정사각형이냐"
  가 아니라 H 의 수치 건전성(퇴화·불량조건 여부)만 본다. 타일 크기 판정은
  어디까지나 육안(노란 점선 vs 줄눈)이다. 수치가 0 이어도 H 가 엉터리일 수 있다.

undistort 는 lib/bev.py 의 load_camera_config + Undistorter.from_config 에 위임한다 —
  주행 노드와 1픽셀이라도 다르게 펴면 멀쩡한 H 를 "실패" 로 오판하게 된다.
  그래서 lib/bev.py 를 못 읽으면 자체 폴백 없이 종료한다(틀린 검증보다 낫다).

BEV 축 규약: 세로 = x(전방), 위쪽이 먼 쪽 / 가로 = y(좌+), 왼쪽이 +y / --ppm 픽셀/미터.

타일 좌표 규약 (본선 실측):
  원점 = A' (카메라 렌즈의 지면 수직투영점). 차를 세울 때 A' 가 줄눈 교차점에 오도록
  맞췄으므로 모든 줄눈 교차점 = (i·p, j·p), i = 전방 칸수(+), j = 왼쪽 칸수(+).
  p = 2.72 m / 6칸 = 0.453333 m (복도 테라조 타일 실측).
  i=0 줄눈은 차체에 가려 안 보이고, 화면에서 가장 가까운 가로 줄눈이 i=1 이다.
  ※ 이 H 는 A' 기준 프레임이다. 라이다 원점은 A' 보다 11.7 cm 앞이므로 라이다
    프레임으로 옮기려면 나중에 x 를 -0.117 m 평행이동해야 한다. 어느 프레임인지는
    camera.yaml 의 h_frame 키에 기록되며, 이 도구가 읽어서 콘솔에 찍어 준다.

화면에 찍는 글자는 ASCII 만 쓴다 — cv2.putText 는 한글 폰트를 못 그린다(□ 로 나온다).
한국어 설명은 전부 콘솔로 낸다.

사용법:
  # 기본 (전방 0~3 m, 좌우 ±1.5 m, 200 px/m)
  python3 bev_check.py --image ~/calib_raw.jpg --camera ../config/camera.yaml \
      --save /tmp/bev.jpg
  # 근거리만 크게 보기 + 줄눈 대비 강화 (테라조 바닥)
  python3 bev_check.py --image ~/calib_raw.jpg --camera ../config/camera.yaml \
      --range-x 0.3 1.8 --range-y -1.0 1.0 --ppm 300 --enhance --save /tmp/bev.jpg
  # 창으로 보기 (아무 키나 누르면 종료). 헤드리스면 --save 를 쓸 것
  python3 bev_check.py --image ~/calib_raw.jpg --camera ../config/camera.yaml
"""

import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np

try:
    import yaml
except ImportError:      # h_frame 등 부가 정보만 못 읽을 뿐, 검증 자체는 가능
    yaml = None

WIN = "bev_check"

PITCH_DEFAULT = 0.453333    # 복도 테라조 타일 피치 실측 = 2.72 m / 6칸
RULER_M = 0.5               # 초록 눈금 간격 (m)
PPM_DEFAULT = 200.0         # BEV 해상도 기본값 (픽셀/미터)
MAX_BEV_PX = 4000           # BEV 한 변 상한 — --ppm 오타(2000 등)로 메모리 터지는 것 방어
EPS_W = 1e-9                # 동차좌표 w 유효 판정 임계
PIX_CLIP = 1e5              # 투영 픽셀 폭주 방어 (float32 맵으로 넘기기 전 컷)
RES_WARN_PX = 2.0           # 타일 한 칸이 원본에서 이보다 얇으면 육안 판정 불가
FONT = cv2.FONT_HERSHEY_SIMPLEX
DASH_ON, DASH_OFF = 9, 6    # 점선 패턴 (px)
HUD_SCALE, HUD_PAD, HUD_LH = 0.42, 6, 16    # 좌상단 정보 박스 (크기를 미리 재서 라벨을 피한다)

# 색 (BGR)
C_TILE = (0, 255, 255)      # 타일 격자선 = 노랑 (판정 주인공)
C_RULER = (90, 220, 90)     # 0.5 m 눈금 = 초록
C_AXIS = (255, 170, 60)     # y=0 중심선 = 하늘색
C_ORIGIN = (255, 120, 255)  # 원점 A' = 분홍
C_TEXT = (255, 255, 255)


# ======================== lib/bev.py 로드 (주행 노드와 동일 경로) ========================

def _import_bev():
    """lib/bev.py 를 파일 경로로 로드 (tools/ 는 ROS 패키지 밖이라 일반 import 불가).

    bev 는 rclpy 미의존(numpy/cv2/yaml 만)이라 CLI 도구에서 그대로 쓸 수 있다.
    ★ 다른 도구들과 달리 여기엔 폴백 undistort 를 두지 않는다. 이 도구의 존재 이유가
      "주행 노드가 보는 것과 같은 그림" 을 확인하는 것이라, 조금이라도 다른 경로로
      펴면 검증 자체가 거짓말이 된다. 못 읽으면 그냥 죽는 게 맞다.
    """
    path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "track_drive", "lib", "bev.py"))
    if not os.path.isfile(path):
        sys.exit("lib/bev.py 없음: %s\n"
                 "  이 도구는 주행 노드와 동일한 undistort 를 써야 의미가 있어 "
                 "폴백을 두지 않는다." % path)
    try:
        spec = importlib.util.spec_from_file_location("_track_drive_bev", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:      # noqa: BLE001 - 원인 그대로 보여주고 종료
        sys.exit("lib/bev.py 로드 실패: %s (%s)" % (path, e))
    return mod


def _import_enhance():
    """make_enhanced.enhance_floor 를 같은 tools/ 에서 가져온다 (--enhance 전용).

    CLAHE/언샤프 파라미터를 두 도구에 복붙하면 언젠가 갈라져서, 클릭할 때 본 그림과
    BEV 로 확인하는 그림이 달라진다. 그래서 정의는 make_enhanced.py 한 곳에만 둔다.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "make_enhanced.py")
    if not os.path.isfile(path):
        sys.exit("--enhance 에는 tools/make_enhanced.py 가 필요하다 (없음: %s)" % path)
    try:
        spec = importlib.util.spec_from_file_location("_tools_make_enhanced", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.enhance_floor
    except Exception as e:      # noqa: BLE001
        sys.exit("make_enhanced.py 로드 실패: %s (%s)" % (path, e))


def _imwrite_params(path):
    """확장자에 맞는 인코딩 파라미터.

    JPEG 전용 옵션을 png 저장에 넘기면 OpenCV 가 "unsupported key" 경고를 뱉는다 —
    현장에서 에러로 오해하기 딱 좋아서 확장자를 보고 고른다.
    """
    if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg"):
        return [cv2.IMWRITE_JPEG_QUALITY, 95]
    return []


def _yaml_extra(path):
    """camera.yaml 에서 lib/bev 가 안 돌려주는 부가 기록(h_frame 등)만 따로 읽는다."""
    if yaml is None or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


# ======================== undistort (주행 노드와 동일) ========================

def _undistort(bev, cam, img, alpha):
    """lib/bev 의 Undistorter 로 편다. 반환 (이미지, new_K 또는 None)."""
    h, w = img.shape[:2]
    cal_size = cam.get("image_size")
    if cal_size is not None and tuple(cal_size) != (w, h):
        print("[경고] 이미지 크기 %dx%d 가 캘리브 기준 %dx%d 와 다르다 — "
              "new_K/H 는 그 크기 기준이라 좌표계가 통째로 어긋난다."
              % (w, h, cal_size[0], cal_size[1]))
    und = bev.Undistorter.from_config(cam, (w, h), alpha=alpha)
    if not und.enabled:
        print("[경고] undistort 비활성(K/D 미보정 또는 맵 생성 실패) — "
              "원본(raw) 픽셀을 그대로 쓴다. H 가 undistort 픽셀 기준이면 어긋난다.")
        return img, None
    warn = bev.check_new_K_match(cam.get("new_K"), und.new_K)
    if warn:
        print("[경고] %s" % warn)
    nk = np.asarray(und.new_K, np.float64).reshape(3, 3)
    print("undistort: model=%s, alpha=%.2f, new_K fx,fy,cx,cy = %.3f %.3f %.3f %.3f"
          % (und.model, alpha, nk[0, 0], nk[1, 1], nk[0, 2], nk[1, 2]))
    return und.apply(img), nk


# ======================== 지면 <-> 픽셀 / BEV 매핑 ========================

def _inv(H):
    try:
        return np.linalg.inv(np.asarray(H, np.float64).reshape(3, 3))
    except (np.linalg.LinAlgError, ValueError):
        sys.exit("H 의 역행렬이 없다(특이행렬) — camera.yaml 의 H 를 확인할 것. "
                 "한 직선 위 점들만으로 캘리브하면 이렇게 퇴화한다.")


def _ref_sign(H, Hinv, size):
    """지면 → 픽셀 동차좌표 w 의 '유효한 쪽' 부호.

    지평선 너머(카메라 뒤쪽) 지면점은 w 부호가 뒤집힌 채 멀쩡한 픽셀 좌표를 뱉는다.
    그대로 두면 하늘/뒤쪽이 BEV 앞쪽에 유령처럼 찍힌다. H 의 전체 스케일 부호는
    findHomography 가 임의로 정하므로 상수로 못 박을 수 없어, 확실히 바닥인
    화면 하단 중앙 픽셀이 가리키는 지면점을 기준으로 부호를 정한다.
    """
    w_img, h_img = size
    p = np.asarray(H, np.float64) @ np.array([w_img * 0.5, h_img * 0.75, 1.0])
    if abs(p[2]) > EPS_W:
        g = p[:2] / p[2]
    else:
        g = np.array([1.0, 0.0])        # 퇴화 시 전방 1 m 로 대체
    w = float((Hinv @ np.array([g[0], g[1], 1.0]))[2])
    return 1.0 if w >= 0.0 else -1.0


def ground_to_pixel(Hinv, ref_sign, x, y):
    """지면(m) → 원본(undistort) 픽셀. 지평선 반대편이면 None."""
    w = Hinv[2, 0] * x + Hinv[2, 1] * y + Hinv[2, 2]
    if w * ref_sign <= EPS_W:
        return None
    u = (Hinv[0, 0] * x + Hinv[0, 1] * y + Hinv[0, 2]) / w
    v = (Hinv[1, 0] * x + Hinv[1, 1] * y + Hinv[1, 2]) / w
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return float(u), float(v)


def pixel_to_ground(H, u, v):
    """픽셀 → 지면(m). w≈0(지평선)이면 None."""
    w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    if abs(w) < EPS_W:
        return None
    x = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / w
    y = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / w
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return float(x), float(y)


def bev_pixel_of(x, y, x_max, y_max, ppm):
    """지면(m) → BEV 픽셀 (col, row). build_bev_maps 의 정확한 역함수.

    세로: 위쪽이 먼 쪽(+x), 가로: 왼쪽이 +y. 픽셀 중심 기준이라 -0.5 가 붙는다.
    """
    return (y_max - y) * ppm - 0.5, (x_max - x) * ppm - 0.5


def build_bev_maps(Hinv, ref_sign, x_range, y_range, ppm):
    """역방향 매핑 맵 생성: BEV 픽셀 → 지면좌표 → H⁻¹ → 원본 픽셀.

    정방향(원본→BEV warp)이 아니라 역방향인 이유: BEV 격자를 우리가 미터로 직접
    정의해야 "1 px = 1/ppm m" 가 보장되고, 그래야 그 위에 그린 미터 격자선이
    실제 줄눈과 같은 자를 쓴 게 된다.
    반환: (map_x, map_y, W, Hh) — 유효하지 않은 화소는 -1 로 채워 remap 이 검게 둔다.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    W = int(round((y_max - y_min) * ppm))
    Hh = int(round((x_max - x_min) * ppm))
    if W < 2 or Hh < 2:
        sys.exit("BEV 크기가 너무 작다 (%dx%d) — --range-x/--range-y/--ppm 확인" % (W, Hh))
    if W > MAX_BEV_PX or Hh > MAX_BEV_PX:
        sys.exit("BEV 크기 %dx%d 가 상한 %d 를 넘는다 — --ppm 을 줄일 것"
                 % (W, Hh, MAX_BEV_PX))

    xs = x_max - (np.arange(Hh, dtype=np.float64) + 0.5) / ppm   # 행: 위가 먼 쪽
    ys = y_max - (np.arange(W, dtype=np.float64) + 0.5) / ppm    # 열: 왼쪽이 +y
    YY, XX = np.meshgrid(ys, xs)                                  # (Hh, W)

    w = Hinv[2, 0] * XX + Hinv[2, 1] * YY + Hinv[2, 2]
    ok = (w * ref_sign) > EPS_W
    w_safe = np.where(ok, w, 1.0)                                 # 0 나눗셈 방어
    u = (Hinv[0, 0] * XX + Hinv[0, 1] * YY + Hinv[0, 2]) / w_safe
    v = (Hinv[1, 0] * XX + Hinv[1, 1] * YY + Hinv[1, 2]) / w_safe

    bad = (~ok | ~np.isfinite(u) | ~np.isfinite(v)
           | (np.abs(u) > PIX_CLIP) | (np.abs(v) > PIX_CLIP))
    u = np.where(bad, -1.0, u)
    v = np.where(bad, -1.0, v)
    return u.astype(np.float32), v.astype(np.float32), W, Hh


# ======================== 오버레이 ========================

def _multiples(lo, hi, step, tol=1e-9):
    """[lo, hi] 안의 step 배수 목록 → [(k, k*step), ...] (k 는 정수 인덱스)."""
    k0 = int(np.ceil(lo / step - tol))
    k1 = int(np.floor(hi / step + tol))
    return [(k, k * step) for k in range(k0, k1 + 1)]


def _dash_row(img, row, color, thick=1):
    """가로 점선 (밑에 깔린 줄눈이 틈으로 보이도록)."""
    h, w = img.shape[:2]
    r = int(round(row))
    if not (0 <= r < h):
        return
    c = 0
    while c < w:
        cv2.line(img, (c, r), (min(c + DASH_ON, w) - 1, r), color, thick)
        c += DASH_ON + DASH_OFF


def _dash_col(img, col, color, thick=1):
    """세로 점선."""
    h, w = img.shape[:2]
    c = int(round(col))
    if not (0 <= c < w):
        return
    r = 0
    while r < h:
        cv2.line(img, (c, r), (c, min(r + DASH_ON, h) - 1), color, thick)
        r += DASH_ON + DASH_OFF


def _solid_row(img, row, color, thick=1):
    h, w = img.shape[:2]
    r = int(round(row))
    if 0 <= r < h:
        cv2.line(img, (0, r), (w - 1, r), color, thick)


def _solid_col(img, col, color, thick=1):
    h, w = img.shape[:2]
    c = int(round(col))
    if 0 <= c < w:
        cv2.line(img, (c, 0), (c, h - 1), color, thick)


def _text(img, s, org, color, scale=0.4):
    """검은 테두리 + 본색 — 밝은 바닥/어두운 여백 어디서든 읽히게."""
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def draw_overlay(img, x_range, y_range, ppm, pitch, hud_wh=(0, 0)):
    """0.5 m 눈금(초록 실선) → 타일 격자(노랑 점선) 순서로 그린다.

    순서가 중요하다: 판정 주인공인 타일선이 눈금선에 덮이면 안 된다.
    hud_wh: 나중에 좌상단에 깔릴 HUD 박스 크기 — 그 밑에 라벨이 숨지 않게 비켜 쓴다.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    h, w = img.shape[:2]
    hud_w, hud_h = int(hud_wh[0]), int(hud_wh[1])

    # --- 0.5 m 눈금 (초록 실선 + 라벨) ---
    for _k, gx in _multiples(x_min, x_max, RULER_M):
        _, row = bev_pixel_of(gx, 0.0, x_max, y_max, ppm)
        _solid_row(img, row, C_RULER)
        # HUD 밴드에 걸리는 라벨은 박스 오른쪽으로 밀어 쓴다
        lx = 3 if int(round(row)) > hud_h + 4 else hud_w + 6
        _text(img, "x=%.1f" % gx, (lx, int(round(row)) - 4), C_RULER)
    for _k, gy in _multiples(y_min, y_max, RULER_M):
        col, _ = bev_pixel_of(0.0, gy, x_max, y_max, ppm)
        _solid_col(img, col, C_RULER)
        ly = 12 if int(round(col)) > hud_w else hud_h + 14
        _text(img, "y=%.1f" % gy, (int(round(col)) + 3, ly), C_RULER)

    # --- y=0 중심선 (하늘색 실선) — 차가 똑바로 섰는지 눈으로 잡는 기준 ---
    if y_min <= 0.0 <= y_max:
        col, _ = bev_pixel_of(0.0, 0.0, x_max, y_max, ppm)
        _solid_col(img, col, C_AXIS)

    # --- 타일 격자 (노랑 점선 + i/j 인덱스) ---
    for i, gx in _multiples(x_min, x_max, pitch):
        _, row = bev_pixel_of(gx, 0.0, x_max, y_max, ppm)
        _dash_row(img, row, C_TILE)
        _text(img, "i=%d" % i, (w - 42, int(round(row)) - 4), C_TILE)
    for j, gy in _multiples(y_min, y_max, pitch):
        col, _ = bev_pixel_of(0.0, gy, x_max, y_max, ppm)
        _dash_col(img, col, C_TILE)
        _text(img, "j=%d" % j, (int(round(col)) + 3, h - 6), C_TILE)

    # --- 원점 A' (차체에 가려 안 보이지만 좌표계 기준점) ---
    if x_min <= 0.0 <= x_max and y_min <= 0.0 <= y_max:
        col, row = bev_pixel_of(0.0, 0.0, x_max, y_max, ppm)
        c, r = int(round(col)), int(round(row))
        cv2.circle(img, (c, r), 5, C_ORIGIN, 1, cv2.LINE_AA)
        _text(img, "A'(0,0)", (c + 7, r + 4), C_ORIGIN)


def hud_size(lines):
    """draw_hud 가 차지할 (폭, 높이) 를 미리 잰다 (라벨 겹침 회피용)."""
    if not lines:
        return 0, 0
    tw = max(cv2.getTextSize(t, FONT, HUD_SCALE, 1)[0][0] for t in lines)
    return tw + HUD_PAD * 2, HUD_LH * len(lines) + HUD_PAD


def draw_hud(img, lines):
    """좌상단 반투명 정보 박스 (ASCII 만 — cv2 는 한글 폰트가 없다)."""
    if not lines:
        return
    bw, bh = hud_size(lines)
    bw = min(img.shape[1], bw)
    bh = min(img.shape[0], bh)
    box = img[0:bh, 0:bw]
    cv2.addWeighted(box, 0.30, np.zeros_like(box), 0.0, 0.0, dst=box)
    for i, t in enumerate(lines):
        _text(img, t, (HUD_PAD, HUD_PAD + HUD_LH * i + 10), C_TEXT, HUD_SCALE)


def draw_banner(img, text):
    """하단 빨간 경고 띠 — 저장된 jpg 만 나중에 다시 볼 때 오판 방지용."""
    h, w = img.shape[:2]
    y0 = max(0, h - 24)
    cv2.rectangle(img, (0, y0), (w, h), (0, 0, 150), -1)
    _text(img, text, (6, h - 8), (255, 255, 255), 0.45)


# ======================== 정량 지표 ========================

def _roundtrip(H, Hinv, ref_sign, x, y):
    """지면 → H⁻¹ → 원본픽셀 → H → 지면. 반환 (uv, xy2). 불가하면 None 포함."""
    uv = ground_to_pixel(Hinv, ref_sign, x, y)
    if uv is None:
        return None, None
    return uv, pixel_to_ground(H, uv[0], uv[1])


def _inside(uv, size, pad=0.0):
    w, h = size
    return (uv is not None and -pad <= uv[0] < w + pad and -pad <= uv[1] < h + pad)


def report_roundtrip(H, Hinv, ref_sign, tiles_x, tiles_y, size, pitch):
    """[1] 격자점 왕복 오차 (mm). H 의 수치 건전성 + 각 점이 화면 안인지."""
    print("")
    print("[1] 타일 격자점 왕복 오차 — 지면 → H⁻¹ → 원본픽셀 → H → 지면 (mm)")
    print("    첫 칸 네 꼭짓점 (p=%.6f m):" % pitch)
    for gx, gy in ((0.0, 0.0), (pitch, 0.0), (0.0, pitch), (pitch, pitch)):
        uv, xy2 = _roundtrip(H, Hinv, ref_sign, gx, gy)
        if uv is None:
            print("      (%6.3f, %6.3f) → 지평선 반대편 — 투영 불가" % (gx, gy))
            continue
        where = "화면안" if _inside(uv, size) else "화면밖"
        if xy2 is None:
            print("      (%6.3f, %6.3f) → px(%8.2f,%8.2f) %s  복원 불가(w≈0)"
                  % (gx, gy, uv[0], uv[1], where))
            continue
        err = np.hypot(xy2[0] - gx, xy2[1] - gy) * 1000.0
        print("      (%6.3f, %6.3f) → px(%8.2f,%8.2f) %s  오차 %7.3f mm"
              % (gx, gy, uv[0], uv[1], where, err))

    errs, worst, n_vis = [], (0.0, None), 0
    for i, gx in tiles_x:
        for j, gy in tiles_y:
            uv, xy2 = _roundtrip(H, Hinv, ref_sign, gx, gy)
            if uv is None or xy2 is None:
                continue
            e = np.hypot(xy2[0] - gx, xy2[1] - gy) * 1000.0
            errs.append(e)
            if e > worst[0]:
                worst = (e, (i, j))
            if _inside(uv, size):
                n_vis += 1
    if errs:
        arr = np.asarray(errs)
        print("    전체 %d점: RMS %.4f mm / 최대 %.4f mm (i=%d, j=%d)"
              % (arr.size, float(np.sqrt(np.mean(arr ** 2))), worst[0],
                 worst[1][0], worst[1][1]))
        print("    그중 원본 화면 안에 들어오는 점: %d개 %s"
              % (n_vis, "" if n_vis >= 4 else "← 너무 적다. --range 를 좁힐 것"))
    else:
        print("    [경고] 왕복 가능한 격자점이 하나도 없다 — H 가 완전히 깨졌거나 "
              "--range 가 화면 밖이다.")
    print("    ※ 이 값은 H 의 수치 건전성만 본다. 0 에 가까워도 H 가 실제 지면과 "
          "맞는다는 뜻은 아니다.")


def report_bev_scale(H, Hinv, ref_sign, tiles_x, tiles_y, pitch, ppm, x_max, y_max):
    """[2] BEV 상 타일 한 칸의 픽셀 크기 vs 이상값(pitch × ppm)."""
    ideal = pitch * ppm
    print("")
    print("[2] BEV 타일 한 칸 픽셀 크기 (이상값 = pitch × ppm = %.2f px)" % ideal)

    def _bev_rt(gx, gy):
        """왕복시킨 지면점을 BEV 픽셀로. 파이프라인 전체를 태운 값."""
        _uv, xy2 = _roundtrip(H, Hinv, ref_sign, gx, gy)
        if xy2 is None:
            return None
        return bev_pixel_of(xy2[0], xy2[1], x_max, y_max, ppm)

    for axis, name in ((0, "x(전방)"), (1, "y(좌우)")):
        d = []
        for i, gx in tiles_x:
            for j, gy in tiles_y:
                p0 = _bev_rt(gx, gy)
                gx2, gy2 = (gx + pitch, gy) if axis == 0 else (gx, gy + pitch)
                p1 = _bev_rt(gx2, gy2)
                if p0 is None or p1 is None:
                    continue
                d.append(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if not d:
            print("    %s 방향: 측정 가능한 인접 쌍 없음" % name)
            continue
        arr = np.asarray(d)
        dev = arr - ideal
        k = int(np.argmax(np.abs(dev)))
        print("    %s 방향 %3d쌍: 평균 %8.3f px (편차 %+.4f px, %+.4f %%) / "
              "최대편차 %+.4f px"
              % (name, arr.size, float(arr.mean()), float(dev.mean()),
                 float(dev.mean() / ideal * 100.0), float(dev[k])))
    print("    ※ BEV 는 지면 미터의 선형 축소판이라 이 값은 구성상 정확하다 "
          "(0 이 정상).")
    print("      0 이 아니면 H 가 불량조건(퇴화)이라는 뜻이지, 타일이 삐뚤다는 "
          "뜻이 아니다.")
    print("      → 타일이 실제 크기 정사각형인지는 노란 점선과 줄눈의 겹침(육안)으로 "
          "판정할 것.")


def report_resolution(H, Hinv, ref_sign, tiles_x, pitch, size):
    """[3] 원본 해상도로 본 '판정 가능 구간'.

    카메라가 지면 ~0.2 m 높이에 거의 수평이라 바닥 한 점의 화면 행 v 는
    v ≈ cy + fy·h/x 로 급격히 포화한다. 먼 타일은 원본에서 몇 px 밖에 안 돼
    BEV 로 아무리 확대해도 보간으로 뭉갠 그림일 뿐이다 — 그 구간의 "겹침" 판정은
    의미가 없으니 미리 잘라서 알려 준다.
    """
    print("")
    print("[3] 원본 해상도로 본 판정 가능 구간 (타일 한 칸이 원본에서 몇 px 인가)")
    print("      i    x[m]   원본 v[px]   세로[px]   가로[px]   판정")
    prev_v = None
    for i, gx in tiles_x:
        uv = ground_to_pixel(Hinv, ref_sign, gx, 0.0)
        if uv is None:
            print("     %2d  %6.3f        —           —          —      지평선 반대편"
                  % (i, gx))
            prev_v = None
            continue
        uv_y = ground_to_pixel(Hinv, ref_sign, gx, pitch)
        dv = abs(uv[1] - prev_v) if prev_v is not None else None
        du = abs(uv_y[0] - uv[0]) if uv_y is not None else None
        prev_v = uv[1]
        if not _inside(uv, size):
            verdict = "화면밖"
        elif dv is not None and dv < RES_WARN_PX:
            verdict = "★ %.0fpx 미만 — 육안 판정 불가" % RES_WARN_PX
        else:
            verdict = "ok"
        print("     %2d  %6.3f   %9.2f   %8s   %8s   %s"
              % (i, gx, uv[1],
                 "—" if dv is None else "%.1f" % dv,
                 "—" if du is None else "%.1f" % du,
                 verdict))
    print("    ※ 세로[px] = 직전 줄눈과의 원본 행 간격, 가로[px] = y 방향 한 칸의 "
          "원본 열 간격.")


# ======================== main ========================

def _parse_args():
    ap = argparse.ArgumentParser(
        description="BEV 타일 격자 검증 (자세한 규약·판정법은 파일 헤더 참고)")
    ap.add_argument("--image", required=True, help="바닥 사진 (왜곡 있는 raw)")
    ap.add_argument("--camera", required=True, help="camera.yaml (model/K/D/H/new_K)")
    ap.add_argument("--pitch", type=float, default=PITCH_DEFAULT,
                    help="타일 피치 m (기본 %.6f = 2.72/6 실측)" % PITCH_DEFAULT)
    ap.add_argument("--range-x", nargs=2, type=float, default=[0.0, 3.0],
                    metavar=("XMIN", "XMAX"), help="BEV 전방 범위 m")
    ap.add_argument("--range-y", nargs=2, type=float, default=[-1.5, 1.5],
                    metavar=("YMIN", "YMAX"), help="BEV 좌우 범위 m (좌가 +)")
    ap.add_argument("--ppm", type=float, default=PPM_DEFAULT,
                    help="BEV 해상도 픽셀/미터 (기본 %g)" % PPM_DEFAULT)
    ap.add_argument("--alpha", type=float, default=None,
                    help="plumb_bob undistort 의 alpha. 생략 시 lib/bev.py 기본값 "
                         "(주행 노드·H 캘리브와 같아야 좌표계가 맞는다)")
    ap.add_argument("--enhance", action="store_true",
                    help="CLAHE+언샤프로 줄눈 대비 강화 (make_enhanced.py 와 동일 처리)")
    ap.add_argument("--save", default=None,
                    help="BEV 저장 경로 (.jpg/.png). 생략하면 창으로 표시 — "
                         "헤드리스(DISPLAY 없음)면 저장 필수")
    args = ap.parse_args()

    if args.pitch <= 0.0:
        sys.exit("--pitch 는 양수여야 한다: %g" % args.pitch)
    if args.ppm <= 0.0:
        sys.exit("--ppm 은 양수여야 한다: %g" % args.ppm)
    if args.range_x[1] <= args.range_x[0]:
        sys.exit("--range-x 는 XMIN < XMAX 여야 한다: %s" % args.range_x)
    if args.range_y[1] <= args.range_y[0]:
        sys.exit("--range-y 는 YMIN < YMAX 여야 한다: %s" % args.range_y)
    return args


def main():
    args = _parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit("이미지 읽기 실패: %s" % args.image)
    # ★ 파일이 없어도 load_camera_config 는 조용히 폴백 dict 를 돌려준다. 이 도구에서는
    #   그게 제일 위험하다 — --camera 경로 오타를 눈치 못 채고 시뮬 폴백 H 로 그린
    #   BEV 를 보며 "캘리브가 틀렸네" 라고 오판하게 된다. 여기서 딱 끊는다.
    if not os.path.isfile(args.camera):
        sys.exit("camera.yaml 이 없다: %s (경로 확인 — 보통 ../config/camera.yaml)"
                 % args.camera)

    bev = _import_bev()
    cam = bev.load_camera_config(args.camera)
    extra = _yaml_extra(args.camera)
    alpha = (bev.UNDISTORT_ALPHA_DEFAULT if args.alpha is None else float(args.alpha))

    print("=== BEV 타일 격자 검증 (bev_check.py) ===")
    print("이미지 : %s (%dx%d)" % (args.image, img.shape[1], img.shape[0]))
    print("카메라 : %s" % args.camera)
    if cam.get("note"):
        print("[camera.yaml] %s" % cam["note"])

    h_ok = bool(cam.get("h_calibrated", False))
    if not h_ok:
        print("")
        print("!!!=========================================================!!!")
        print("!!! [경고] h_calibrated = false                             !!!")
        print("!!! camera.yaml 에 실차 H 가 없어 예선 시뮬 폴백 H 로 편다.  !!!")
        print("!!! 아래 BEV 는 엉망으로 나오는 게 정상이며 캘리브 판정에    !!!")
        print("!!! 쓸 수 없다. tools/calib_ground_h.py 로 H 를 먼저 잡을 것.!!!")
        print("!!!=========================================================!!!")
        print("")
    frame = extra.get("h_frame")
    if frame:
        print("H 프레임: %s" % frame)
    else:
        print("H 프레임: camera.yaml 에 h_frame 미기록 — 이 도구는 원점을 그대로 "
              "(0,0) 으로 그린다.")
        print("          A' 기준 H 라면 라이다 프레임과 x 로 0.117 m 차이가 난다는 "
              "점을 기억할 것.")
    if extra.get("h_rms_m") is not None:
        print("H 캘리브 재투영 RMS: %.4f m (calib_ground_h.py 기록)"
              % float(extra["h_rms_m"]))

    und_img, _nk = _undistort(bev, cam, img, alpha)
    if args.enhance:
        und_img = _import_enhance()(und_img)
        print("--enhance: CLAHE+언샤프 적용 (기하 불변)")

    H = np.asarray(cam["H"], np.float64).reshape(3, 3)
    Hinv = _inv(H)
    size = (und_img.shape[1], und_img.shape[0])
    ref_sign = _ref_sign(H, Hinv, size)

    x_range = (float(args.range_x[0]), float(args.range_x[1]))
    y_range = (float(args.range_y[0]), float(args.range_y[1]))
    ppm = float(args.ppm)
    pitch = float(args.pitch)

    map_x, map_y, W, Hh = build_bev_maps(Hinv, ref_sign, x_range, y_range, ppm)
    disp = cv2.remap(und_img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # "실제 사진 화소가 들어온 비율" — 지평선 반대편(맵 -1)뿐 아니라 사진 밖으로
    # 나간 화소도 검게 남으므로, 원본 이미지 범위 안에 떨어진 것만 세야 정직하다.
    inside = ((map_x >= 0.0) & (map_x < size[0])
              & (map_y >= 0.0) & (map_y < size[1]))
    filled = float(np.count_nonzero(inside)) / float(W * Hh) * 100.0
    print("BEV : x %.2f~%.2f m, y %.2f~%.2f m, %g px/m → %dx%d px "
          "(사진이 실제로 덮는 영역 %.1f%%)"
          % (x_range[0], x_range[1], y_range[0], y_range[1], ppm, W, Hh, filled))
    print("타일: 피치 %.6f m → BEV 한 칸 %.2f px" % (pitch, pitch * ppm))
    if filled < 5.0:
        print("[경고] 사진이 덮는 영역이 거의 없다 — H 가 깨졌거나 --range 가 시야 밖이다.")
    elif filled < 40.0:
        print("[안내] BEV 의 절반 이상이 사진 밖이다 — --range-x/--range-y 를 좁히면 "
              "판정할 구간을 더 크게 볼 수 있다.")

    tiles_x = _multiples(x_range[0], x_range[1], pitch)
    tiles_y = _multiples(y_range[0], y_range[1], pitch)
    report_roundtrip(H, Hinv, ref_sign, tiles_x, tiles_y, size, pitch)
    report_bev_scale(H, Hinv, ref_sign, tiles_x, tiles_y, pitch, ppm,
                     x_range[1], y_range[1])
    report_resolution(H, Hinv, ref_sign, tiles_x, pitch, size)

    hud_lines = [
        "BEV  x %.2f-%.2f m  y %.2f-%.2f m  %.0f px/m" % (
            x_range[0], x_range[1], y_range[0], y_range[1], ppm),
        "tile pitch %.4f m = %.1f px (YELLOW dashed)" % (pitch, pitch * ppm),
        "GREEN solid = %.1f m ruler   CYAN = y0 axis" % RULER_M,
        "up = far (+x)   left = +y   H: %s" % (
            "calibrated" if h_ok else "SIM FALLBACK"),
        "OK if yellow dashes sit on the grout lines",
    ]
    draw_overlay(disp, x_range, y_range, ppm, pitch, hud_wh=hud_size(hud_lines))
    draw_hud(disp, hud_lines)
    if not h_ok:
        draw_banner(disp, "WARNING: h_calibrated=false - SIM fallback H, NOT a valid check")

    print("")
    print("판정: 노란 점선이 사진 속 줄눈 위에 겹치면 성공. "
          "가까운 쪽(i=1,2)부터 맞는지 보고, 먼 쪽이 벌어지면 원거리 대응점을 "
          "보강해 calib_ground_h.py 를 다시 돌릴 것.")

    if args.save:
        out_dir = os.path.dirname(os.path.abspath(args.save))
        if not os.path.isdir(out_dir):
            sys.exit("저장 경로의 디렉터리가 없다: %s" % out_dir)
        if not cv2.imwrite(args.save, disp, _imwrite_params(args.save)):
            sys.exit("저장 실패: %s (확장자 확인)" % args.save)
        print("저장: %s" % args.save)
        return

    # ★ DISPLAY 만 본다 (WAYLAND_DISPLAY 는 인정하지 않는다).
    #   opencv-python 의 HighGUI 는 Qt xcb 플러그인만 들어 있어서 X11 이 필요하다.
    #   WSLg 처럼 WAYLAND_DISPLAY 만 살아 있는 환경에서 imshow 를 부르면 Qt 가
    #   abort() 로 프로세스를 죽인다 — 파이썬 예외가 아니라서 try/except 로 못 잡는다.
    #   그래서 부르기 전에 반드시 여기서 걸러야 한다. (Wayland 세션도 XWayland 가
    #   DISPLAY 를 채워 주므로 이 검사로 충분하다.)
    if not os.environ.get("DISPLAY"):
        sys.exit("표시할 X11 디스플레이가 없다 (DISPLAY 미설정) — "
                 "--save /tmp/bev.jpg 처럼 파일로 저장할 것.\n"
                 "  (ssh 면 -X, WSL 이면 WSLg/X 서버 확인)")
    try:
        cv2.imshow(WIN, disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        sys.exit("창 표시 실패 (%s) — --save 로 파일 저장할 것." % e)


if __name__ == "__main__":
    main()
