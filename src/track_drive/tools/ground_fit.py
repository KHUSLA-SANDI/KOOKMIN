#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""지면 H 를 물리값 4개로 맞춘다 — 왜곡보정 없이, 원본 픽셀 그대로.

왜 이렇게 하나
  · 행렬 9개는 서로 얽혀 있어 손으로 맞추기 어렵다. 카메라 자세 4개
    (높이·내림각·좌우틀어짐·기울기)면 H 가 결정되고, 각각 뜻이 명확하다.
  · 왜곡보정까지 동시에 건드리면 변수가 너무 많아 어디가 원인인지 알 수 없다.
    그래서 여기서는 **왜곡보정을 아예 쓰지 않고** 원본 픽셀에 H 를 건다
    (camera.yaml 의 intrinsics_calibrated=false 경로 — Undistorter 가 통과만 한다).
    살짝 휘는 건 감수하고, H 를 먼저 확정한 뒤 곡률을 다룰지 정한다.

조절값
  카메라 자세 4개 : 높이(m) / 내림각(deg) / 좌우틀어짐(deg) / 기울기(deg)
  격자 위치 2개   : 가로줄 시작 x(m) / 중앙 세로줄 y(m)   ← 차를 어디 세웠나
  초점거리 fx     : 실측 화각 기반 추정값. 위 6개로 안 맞을 때만 건드린다

판정
  · 왼쪽 화면의 초록선이 바닥 테이프 위에 얹히면 맞다.
  · 오른쪽 BEV 에서 두 점을 클릭하면 실제 거리(mm) 가 나온다 — 45cm 자로 검증.

성능
  BEV 역매핑 격자는 파라미터가 바뀔 때만 다시 만든다(캐시). 표시는 축소본으로
  하고 H 는 원본 픽셀 기준을 유지한다 — 안 그러면 1920x1080 에서 창이 버벅인다.

실행
  python3 tools/ground_fit.py --topic /image_raw
  python3 tools/ground_fit.py --image dev/shots/xxx.jpg
  화면이 크면:  --view-h 320 --ppm 110
"""
import argparse
import os
import shutil
import sys
import threading
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.realpath(__file__))
_PKG = os.path.dirname(_HERE)
sys.path.insert(0, _PKG)

try:
    import yaml
except ImportError:
    yaml = None


def _load_text():
    import importlib.util
    p = os.path.join(_HERE, "tuner.py")
    if not os.path.isfile(p):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_tt", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_tt"] = m
        old, sys.argv = sys.argv, ["tuner.py"]
        try:
            spec.loader.exec_module(m)
        finally:
            sys.argv = old
        return getattr(m, "Text", None)
    except Exception:
        return None


_T = _load_text()
TXT = _T() if _T is not None else None


def draw_text(img, items):
    if TXT is not None:
        return TXT.draw(img, items)
    for t, x, y, s, c in items:
        cv2.putText(img, t, (int(x), int(y + s)), cv2.FONT_HERSHEY_SIMPLEX,
                    s / 30.0, c, 1, cv2.LINE_AA)
    return img


def build_H(fx, fy, cx, cy, height, pitch_deg, yaw_deg, roll_deg,
            cam_x=0.0, cam_y=0.0):
    """카메라 자세 → H (이미지 픽셀 → 지면 미터, x전방/y좌).

    cam_x/cam_y = 지면 원점에서 카메라까지의 오프셋.
    출력을 라이다 기준으로 하려면 cam_x = -(라이다가 카메라보다 앞선 거리).
    """
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)
    th, ps, rl = np.radians([pitch_deg, yaw_deg, roll_deg])
    R0 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    Rp = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Ry = np.array([[np.cos(ps), -np.sin(ps), 0], [np.sin(ps), np.cos(ps), 0], [0, 0, 1]])
    Rr = np.array([[np.cos(rl), -np.sin(rl), 0], [np.sin(rl), np.cos(rl), 0], [0, 0, 1]])
    R = Rr @ Rp @ R0 @ Ry
    t = -R @ np.array([cam_x, cam_y, height], np.float64)
    return np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], t]))


PARAMS = [
    ("내림각", "카메라가 아래를 보는 각 deg ★. 크게=먼 곳이 화면 위로", 0.2),
    ("높이", "카메라 지면높이 m ★. 크게=BEV 가 통째로 커짐", 0.005),
    ("좌우틀어짐", "yaw deg. 격자가 좌우로 회전", 0.2),
    ("기울기", "roll deg. 화면이 갸우뚱한 정도", 0.2),
    ("격자_앞", "★라이다 중심에서★ 첫 가로줄까지 거리 m (카메라 렌즈 아님)", 0.01),
    ("격자_옆", "라이다 기준 중앙 세로줄의 좌우 위치 m (0=정면)", 0.01),
    ("fx", "★가로 축척. 작게=지면 좌우가 넓게 계산됨 (격자 좌우가 안 맞을 때)", 4.0),
    ("fy", "세로 축척. 전방거리는 높이/내림각으로 맞추는 게 나으니 보통 안 건드림", 4.0),
    ("출력화각", "보정영상을 얼마나 넓게 담을지. 작게=더 넓게 보임. BEV 좌표엔 영향 없음", 10.0),
    ("왜곡중심X", "★보정 켤 때만. 좌우 오차가 다르면 여기 — 왜곡의 대칭 중심", 3.0),
    ("왜곡중심Y", "보정 켤 때만. 위아래 오차가 다르면 여기", 3.0),
    ("왜곡k1", "보정 켤 때만. 보정 세기. 더 음수 = 더 세게 편다", 0.004),
]
KEYS = [p[0] for p in PARAMS]

# 해상도별 정합배율 실측 (그 모드 1px = 1080p 몇 px). fx 초기값 환산에 쓴다.
# 이 카메라는 저해상도 모드가 센서를 '잘라낸' 것이라 화각이 달라, 폭 비례는 틀린다.
SCALE = {(1920, 1080): 1.0, (1280, 720): 1.0003, (1024, 768): 1.3981,
         (800, 600): 1.1948, (640, 480): 2.0007}
FX_1080 = 438.783367 * 2.001        # 640x480 캘리브 fx 를 1080p 기준으로 환산


class Src:
    def __init__(self, a):
        self._lock = threading.Lock()
        self._f = None
        if a.image:
            f = cv2.imread(a.image)
            if f is None:
                print("사진을 못 읽었다:", a.image); sys.exit(1)
            self._f = f
            self.kind = "static"
        else:
            self.kind = "ros"
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            from track_drive.lib import preprocess as PP
            rclpy.init()
            nd = Node("ground_fit")

            def cb(msg):
                bgr, _i = PP.image_msg_to_bgr(msg)
                if bgr is not None:
                    with self._lock:
                        self._f = bgr
            nd.create_subscription(Image, a.topic, cb, qos_profile_sensor_data)
            self._nd, self._rc = nd, rclpy
            threading.Thread(target=lambda: rclpy.spin(nd), daemon=True).start()
            print("구독 %s — 첫 프레임 대기…" % a.topic)

    def read(self):
        with self._lock:
            return None if self._f is None else self._f.copy()

    def close(self):
        if self.kind == "ros":
            try:
                self._nd.destroy_node()
                if self._rc.ok():
                    self._rc.shutdown()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="지면 H 물리값 튜너 (왜곡보정 없이)")
    ap.add_argument("--camera", default=os.path.join(_PKG, "config", "camera.yaml"))
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--image", default=None)
    ap.add_argument("--pitch-m", type=float, default=0.453333, help="타일 격자 피치 m")
    ap.add_argument("--nx", type=int, default=6, help="가로줄 개수")
    ap.add_argument("--ny", type=int, default=5, help="세로줄 개수")
    ap.add_argument("--lidar-offset", type=float, default=0.10,
                    help="라이다가 카메라보다 앞선 수평거리 m")
    ap.add_argument("--range-x", type=float, nargs=2, default=[-0.35, 2.4],
                    help="BEV 전방범위. 음수까지 넣어야 원점 뒤쪽도 보인다")
    ap.add_argument("--range-y", type=float, nargs=2, default=[-1.1, 1.1])
    ap.add_argument("--ppm", type=float, default=0.0,
                    help="BEV 픽셀/미터. 0 이면 표시 높이에서 자동 역산(권장)")
    ap.add_argument("--view-h", type=int, default=380, help="화면 패널 높이 px")
    ap.add_argument("--undistort", action="store_true",
                    help="왜곡보정 켜고 시작 (실행 중 [u] 로 전환 가능)")
    ap.add_argument("--fps", type=float, default=6.0,
                    help="화면 갱신 상한 (원격 데스크톱 부하 줄이기. 낮출수록 가볍다)")
    args = ap.parse_args()

    # ---- 왜곡보정용 K/D (camera.yaml 에서) ----
    KD = {"K": None, "D": None, "map": None, "size": None}
    if yaml is not None and os.path.isfile(args.camera):
        try:
            _y = yaml.safe_load(open(args.camera, encoding="utf-8")) or {}
            if _y.get("intrinsics_calibrated"):
                KD["K"] = np.asarray(_y["K"], np.float64).reshape(3, 3)
                KD["D"] = np.ravel(np.asarray(_y["D"], np.float64))[:5]
                print("왜곡보정 K/D 로드: fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
                      % (KD["K"][0, 0], KD["K"][1, 1], KD["K"][0, 2], KD["K"][1, 2]))
                print("  [u] 키로 왜곡보정 켜기/끄기 — 좌우가 안 맞으면 켜볼 것")
        except Exception as e:
            print("K/D 로드 실패: %s" % e)
    undist_on = bool(args.undistort) and KD["K"] is not None

    # ---- 저장해 둔 값(h_phys) 복원 — 없으면 기본 추측값 ----
    LOADED = {}
    LOADED_UND = None
    if yaml is not None and os.path.isfile(args.camera):
        try:
            _y2 = yaml.safe_load(open(args.camera, encoding="utf-8")) or {}
            hp = _y2.get("h_phys") or {}
            m_ = {"pitch_deg": "내림각", "height": "높이", "yaw_deg": "좌우틀어짐",
                  "roll_deg": "기울기", "grid_x0": "격자_앞", "grid_y0": "격자_옆",
                  "fx": "fx", "fy": "fy", "dist_cx": "왜곡중심X",
                  "dist_cy": "왜곡중심Y", "dist_k1": "왜곡k1",
                  "out_fx": "출력화각"}
            for a_, b_ in m_.items():
                if a_ in hp:
                    LOADED[b_] = float(hp[a_])
            if "undistort" in hp:
                LOADED_UND = bool(hp["undistort"])
            elif _y2.get("h_source"):
                # 옛 파일 — 문구로 추론 (왜곡보정 미사용 이라고 적혀 있으면 False)
                LOADED_UND = ("미사용" not in str(_y2["h_source"]))
            if LOADED:
                print("저장돼 있던 값 복원: " +
                      "  ".join("%s=%.4g" % (k, v) for k, v in LOADED.items()))
                if LOADED_UND is not None:
                    print("  그때 왜곡보정: %s" % ("켬" if LOADED_UND else "끔"))
        except Exception as e:
            print("h_phys 로드 실패: %s" % e)
    if LOADED_UND is not None and not args.undistort:
        undist_on = bool(LOADED_UND) and KD["K"] is not None

    src = Src(args)
    P = args.pitch_m
    val = {"내림각": 12.0, "높이": 0.18, "좌우틀어짐": 0.0, "기울기": 0.0,
           "격자_앞": 0.45, "격자_옆": 0.0, "fx": 0.0, "fy": 0.0,
           "왜곡중심X": 0.0, "왜곡중심Y": 0.0, "왜곡k1": 0.0}
    if KD["K"] is not None:            # camera.yaml 의 K/D 를 시작값으로
        val["왜곡중심X"] = float(KD["K"][0, 2])
        val["왜곡중심Y"] = float(KD["K"][1, 2])
        val["왜곡k1"] = float(KD["D"][0])
        val["fx"] = float(KD["K"][0, 0])
        val["fy"] = float(KD["K"][1, 1])
        val["출력화각"] = float(KD["K"][0, 0])
    val.update(LOADED)          # 저장돼 있던 값이 있으면 그것부터 시작
    init = None
    sel, stepmul = 0, 1.0
    picks = []
    msg, msg_t = "", 0.0
    st = {"boff": (0, 0), "bs": 1.0, "W": 0, "H": 0}
    win = "ground fit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    x0b, x1b = args.range_x
    y0b, y1b = args.range_y
    # ★ BEV 를 표시 크기로 **직접** 렌더한다. 작게 만들어 확대하면 영상이 번지고,
    #   격자선도 뭉갠다. ppm 은 표시 높이에서 역산 (--ppm 을 주면 그 값을 쓴다).
    ppm = args.ppm if args.ppm > 0 else (args.view_h / max(1e-6, x1b - x0b))
    Wb = max(8, int(round((y1b - y0b) * ppm)))
    Hb = max(8, int(round((x1b - x0b) * ppm)))
    st["W"], st["H"] = Wb, Hb
    print("BEV %dx%d px (%.0f px/m, 전방 %.1f~%.1fm, 좌우 %.1f~%.1fm)"
          % (Wb, Hb, ppm, x0b, x1b, y0b, y1b))
    # ★ 지면 격자는 상수 — 한 번만 만든다 (매 프레임 만들면 창이 버벅인다)
    _ys = y1b - np.arange(Wb) / ppm
    _xs = x1b - np.arange(Hb) / ppm
    _XX, _YY = np.meshgrid(_xs, _ys, indexing="ij")
    GRID = np.stack([_XX, _YY, np.ones_like(_XX)], -1).reshape(-1, 3).T

    cache = {"key": None, "map": None}

    boot = {"on": False, "pts": []}      # 왼쪽 화면 클릭으로 대략 맞추기

    def on_mouse(ev, x, y, flags, _):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = st["boff"]
        st["dirty"] = True          # 클릭했으면 화면을 한 번 다시 그린다
        if boot["on"] and x < ox:                    # 왼쪽(원본) 화면 클릭
            boot["pts"].append((x / st["ls"], y / st["ls"]))   # 원본 픽셀로 환산
            return
        s = st["bs"]
        c, r = (x - ox) / s, (y - oy) / s
        if 0 <= c < Wb and 0 <= r < Hb:
            picks.append((x1b - r / ppm, y1b - c / ppm))
            if len(picks) > 2:
                del picks[0]
    cv2.setMouseCallback(win, on_mouse)

    def rough_fit(pts, fx, Wi, Hi_, lidar_off):
        """가로 테이프 3줄을 화면에서 클릭한 좌표로 내림각·높이·격자_앞 을 푼다.

        연속한 가로줄이므로 지면 x 는 gx0, gx0+P, gx0+2P 여야 한다.
        미지수 3개(내림각·높이·gx0)에 식 3개 — 좌우틀어짐/기울기는 0 으로 두고 푼다.
        """
        try:
            from scipy.optimize import least_squares
        except ImportError:
            return None, "scipy 없음"
        uv = np.asarray(pts, np.float64)
        order = np.argsort(-uv[:, 1])                # 화면 아래(가까운 것)부터
        uv = uv[order]

        def res(q):
            pitch, hgt, gx0 = q
            if hgt <= 0.02 or not (-30 < pitch < 80):
                return np.full(len(uv), 1e3)
            Hm_ = build_H(fx, fx, Wi / 2.0, Hi_ / 2.0, hgt, pitch, 0.0, 0.0,
                          cam_x=-lidar_off)
            out = []
            for i, (u, v) in enumerate(uv):
                p = Hm_ @ np.array([u, v, 1.0])
                if abs(p[2]) < 1e-12:
                    return np.full(len(uv), 1e3)
                out.append(p[0] / p[2] - (gx0 + i * P))
            return np.asarray(out)

        best, bc = None, np.inf
        for p0 in ([12.0, 0.18, 0.45], [25.0, 0.18, 0.3], [8.0, 0.22, 0.6],
                   [35.0, 0.15, 0.25], [18.0, 0.20, 0.5]):
            try:
                s_ = least_squares(res, p0, bounds=([-20, 0.03, -1.0],
                                                    [75, 0.6, 3.0]), max_nfev=3000)
            except Exception:
                continue
            if s_.cost < bc:
                bc, best = s_.cost, s_.x
        if best is None:
            return None, "풀이 실패"
        return best, "잔차 %.1f mm" % (1000 * np.sqrt(2 * bc / len(uv)))

    print(__doc__.split("실행")[0])
    # ★ 기본이 '정지'다. 카메라도 장면도 안 움직이므로 실시간일 이유가 없고,
    #   원격 데스크톱으로 매 프레임 화면을 보내는 것이 진짜 병목이었다.
    #   [스페이스] 로 새 프레임을 한 장 받아온다.
    held = None
    dirty = True
    last_draw = 0.0
    min_dt = 1.0 / max(1.0, args.fps)
    while True:
        if held is None:
            held = src.read()
            if held is not None:
                dirty = True
        frame = held
        if frame is None:
            b = np.full((150, 620, 3), 25, np.uint8)
            draw_text(b, [("이미지 대기중…", 20, 55, 18, (200, 200, 200))])
            cv2.imshow(win, b)
            if (cv2.waitKeyEx(100) & 0xFF) in (27, ord('q')):
                break
            continue
        Himg, Wimg = frame.shape[:2]
        if val["fx"] == 0.0:      # 저장값이 없을 때만 추정
            s = SCALE.get((Wimg, Himg))
            note = ("실측 정합배율 %.4f" % s) if s else "미측정 해상도 — 폭 비례 근사"
            if s is None:
                s = 1920.0 / max(1, Wimg)
            val["fx"] = FX_1080 / s
            if val["fy"] == 0.0:
                val["fy"] = val["fx"]
            init = dict(val)
            print("초기 fx %.1f (%dx%d, %s) — 안 맞으면 마지막 항목에서 조정"
                  % (val["fx"], Wimg, Himg, note))

        # ---- 왜곡보정 (켜져 있으면) — 리맵 테이블은 1회만 만든다 ----
        if undist_on:
            # 주점/강도는 조절값을 쓴다 (좌우 오차 차이를 잡는 손잡이)
            # ★ 조절되는 fx/fy 는 **왜곡모델 K** 에 들어가야 한다.
            #   출력 new_K 에만 넣으면 뒤에서 inv(new_K) 로 상쇄돼 아무 효과가 없다.
            #   픽셀->각도 변환을 정하는 것이 이 fx 이고, 그게 곧 지면 축척이다.
            Kd = KD["K"].copy()
            Kd[0, 0] = float(val["fx"])
            Kd[1, 1] = float(val["fy"])
            Kd[0, 2] = float(val["왜곡중심X"])
            Kd[1, 2] = float(val["왜곡중심Y"])
            Dc = np.array(KD["D"], np.float64)
            Dc[0] = float(val["왜곡k1"])
            # 출력 new_K — 화면 틀만 정한다 (지면 좌표에는 상쇄돼 영향 없음).
            # 작게 잡으면 보정된 화각을 더 많이 담는다(1920x1080 을 고른 이유).
            ofx = float(val["출력화각"]) if val["출력화각"] > 1.0 else float(Kd[0, 0])
            Kc = np.array([[ofx, 0.0, Wimg / 2.0],
                           [0.0, ofx * (Kd[1, 1] / max(1e-9, Kd[0, 0])), Himg / 2.0],
                           [0.0, 0.0, 1.0]], np.float64)
            ukey = (round(Kd[0, 0], 3), round(Kd[1, 1], 3),
                    round(Kd[0, 2], 3), round(Kd[1, 2], 3), round(Dc[0], 6),
                    round(Kc[0, 0], 3), round(Kc[1, 1], 3), Wimg, Himg)
            if KD.get("ukey") != ukey:
                KD["map"] = cv2.initUndistortRectifyMap(
                    Kd, Dc, np.eye(3), Kc, (Wimg, Himg), cv2.CV_16SC2)
                KD["ukey"] = ukey
                KD["size"] = (Wimg, Himg)
                cache["key"] = None   # 영상이 바뀌었으니 BEV 맵도 다시
            frame = cv2.remap(frame, KD["map"][0], KD["map"][1],
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # 보정영상의 내부파라미터 = 출력 new_K
            pfx, pfy = float(Kc[0, 0]), float(Kc[1, 1])
            pcx, pcy = float(Kc[0, 2]), float(Kc[1, 2])
            KD["Kc"], KD["Dc"], KD["Kd"] = Kc, Dc, Kd
        else:
            pfx = val["fx"] if val["fx"] > 0 else 1.0
            pfy = val["fy"] if val["fy"] > 0 else pfx
            pcx, pcy = Wimg / 2.0, Himg / 2.0
        fx = pfx
        Hm = build_H(pfx, pfy, pcx, pcy,
                     val["높이"], val["내림각"], val["좌우틀어짐"], val["기울기"],
                     cam_x=-args.lidar_offset, cam_y=0.0)

        # ---- 표시용 축소본 (H 는 원본 픽셀 기준을 유지) ----
        sL = args.view_h / float(Himg)
        left = cv2.resize(frame, (int(Wimg * sL), args.view_h),
                          interpolation=cv2.INTER_AREA)
        LW, LH = left.shape[1], left.shape[0]
        st["ls"] = sL
        for i, (pu, pv) in enumerate(boot["pts"]):
            cv2.drawMarker(left, (int(pu * sL), int(pv * sL)), (0, 140, 255),
                           cv2.MARKER_TILTED_CROSS, 16, 2)
            draw_text(left, [("%d" % (i + 1), int(pu * sL) + 8,
                              int(pv * sL) - 16, 14, (0, 140, 255))])

        try:
            Hi = np.linalg.inv(Hm)
            wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2]) or 1.0
        except np.linalg.LinAlgError:
            Hi, wref = None, 1.0

        def line_img(lg, col, th):
            """지면 직선 → 화면 직선. 축소본 좌표로 바로 환산해 그린다."""
            li = Hm.T @ lg                     # 원본 픽셀 기준 직선
            a, b, c = li[0], li[1], li[2] * sL   # u=U/s 대입 → (a, b, c*s)
            if abs(a) < 1e-12 and abs(b) < 1e-12:
                return
            if abs(b) >= abs(a):
                pts = [(0, -c / b), (LW - 1, (-c - a * (LW - 1)) / b)]
            else:
                pts = [(-c / a, 0), ((-c - b * (LH - 1)) / a, LH - 1)]
            if all(-8000 < q[0] < 8000 and -8000 < q[1] < 8000 for q in pts):
                cv2.line(left, (int(pts[0][0]), int(pts[0][1])),
                         (int(pts[1][0]), int(pts[1][1])), col, th, cv2.LINE_AA)

        gx0, gy0 = val["격자_앞"], val["격자_옆"]
        for k in range(args.nx):
            line_img(np.array([1.0, 0.0, -(gx0 + k * P)]), (0, 255, 0), 2)
        m = args.ny // 2
        for j in range(-m, args.ny - m):
            line_img(np.array([0.0, 1.0, -(gy0 + j * P)]),
                     (255, 170, 0) if j == 0 else (0, 255, 0), 3 if j == 0 else 2)

        # ---- 원점(라이다 지면투영점)을 화면에 찍는다 ----
        # 라이다는 지면 위에 떠 있으므로 '원점' 은 그 **바로 아래 바닥** 이고,
        # 보통 차체에 가려 화면 밖(아래)이다. 그래서 밖이면 아래쪽 화살표로 알린다.
        near_x = None
        if Hi is not None:
            po = Hi @ np.array([0.0, 0.0, 1.0])
            if abs(po[2]) > 1e-12 and np.sign(po[2]) == wref:
                ou, ov = po[0] / po[2] * sL, po[1] / po[2] * sL
                if 0 <= ou < LW and 0 <= ov < LH:
                    cv2.circle(left, (int(ou), int(ov)), 7, (255, 60, 0), -1)
                    cv2.circle(left, (int(ou), int(ov)), 11, (255, 255, 255), 2)
                    draw_text(left, [("원점(라이다 지면점)", int(ou) + 14,
                                      int(ov) - 8, 13, (255, 120, 0))])
                elif 0 <= ou < LW:
                    xa = int(np.clip(ou, 12, LW - 12))
                    ya = LH - 6 if ov >= LH else 6
                    cv2.arrowedLine(left, (xa, ya - 26 if ov >= LH else ya + 26),
                                    (xa, ya), (255, 60, 0), 3, tipLength=0.4)
                    draw_text(left, [("원점 화면 밖 (%+.0fpx)" % (ov - LH),
                                      max(2, xa - 60), ya - 46 if ov >= LH else ya + 30,
                                      12, (255, 120, 0))])
            # 화면 맨 아랫줄 중앙이 지면 어디에 닿는가 = 실제로 보이는 가장 가까운 지면
            pb = Hm @ np.array([Wimg / 2.0, Himg - 1.0, 1.0])
            if abs(pb[2]) > 1e-12:
                near_x = float(pb[0] / pb[2])

        # ---- BEV: 맵은 파라미터가 바뀔 때만 다시 만든다 ----
        key = (round(pfx, 6), round(pfy, 6), round(pcx, 4), round(pcy, 4),
               round(val["높이"], 6), round(val["내림각"], 6),
               round(val["좌우틀어짐"], 6), round(val["기울기"], 6),
               Wimg, Himg, undist_on)
        if cache["key"] != key:
            if Hi is None:
                cache["map"] = None
            else:
                Q = Hi @ GRID
                w = Q[2]
                bad = (np.abs(w) < 1e-12) | (np.sign(w) != wref)
                w = np.where(bad, 1.0, w)
                mu = (Q[0] / w).reshape(Hb, Wb)   # 원본에서 샘플 (화질 유지)
                mv = (Q[1] / w).reshape(Hb, Wb)
                mu[bad.reshape(Hb, Wb)] = -1
                mv[bad.reshape(Hb, Wb)] = -1
                cache["map"] = (mu.astype(np.float32), mv.astype(np.float32))
            cache["key"] = key
        if cache["map"] is not None:
            bev = cv2.remap(frame, cache["map"][0], cache["map"][1],
                            cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        else:
            bev = np.zeros((Hb, Wb, 3), np.uint8)

        # BEV 는 이미 표시 크기로 렌더됐다 — 리사이즈하지 않는다(번짐 방지).
        B = bev
        st["boff"], st["bs"] = (LW, 0), 1.0

        def bx_(y):
            return int(round((y1b - y) * ppm))

        def by_(x):
            return int(round((x1b - x) * ppm))

        for k in range(args.nx):
            x = gx0 + k * P
            if x0b <= x <= x1b:
                cv2.line(B, (0, by_(x)), (B.shape[1] - 1, by_(x)), (0, 255, 255), 1)
        for j in range(-m, args.ny - m):
            y = gy0 + j * P
            if y0b <= y <= y1b:
                cv2.line(B, (bx_(y), 0), (bx_(y), B.shape[0] - 1),
                         (255, 170, 0) if j == 0 else (0, 255, 255), 1)
        # 원점(라이다 지면투영점) — 모든 거리의 기준. 점으로도 찍는다.
        if x0b <= 0.0 <= x1b:
            yy0 = by_(0.0)
            cv2.line(B, (0, yy0), (B.shape[1] - 1, yy0), (0, 0, 255), 1)
            if y0b <= 0.0 <= y1b:
                cv2.circle(B, (bx_(0.0), yy0), 6, (255, 60, 0), -1)
                cv2.circle(B, (bx_(0.0), yy0), 9, (255, 255, 255), 1)
            draw_text(B, [("x=0 라이다 중심(원점)", 4, max(0, yy0 - 15), 12,
                           (0, 0, 255))])
        if x0b <= -args.lidar_offset <= x1b:
            yy_ = by_(-args.lidar_offset)
            cv2.line(B, (0, yy_), (B.shape[1] - 1, yy_), (255, 0, 255), 1)
            draw_text(B, [("카메라 렌즈", 4, max(0, yy_ + 2), 11, (255, 0, 255))])
        # 실제로 카메라가 볼 수 있는 가장 가까운 지면 — 이보다 아래는 지면이 아니다
        if near_x is not None and x0b <= near_x <= x1b:
            yn = by_(near_x)
            for cx_ in range(0, B.shape[1], 12):
                cv2.line(B, (cx_, yn), (min(cx_ + 6, B.shape[1] - 1), yn),
                         (0, 200, 255), 2)
            draw_text(B, [("여기부터 실제 지면 (x=%.2fm) — 아래는 차체" % near_x,
                           4, max(0, yn - 15), 12, (0, 200, 255))])
        dist_txt = ""
        for gx, gy in picks:
            cv2.drawMarker(B, (bx_(gy), by_(gx)), (0, 0, 255), cv2.MARKER_CROSS, 13, 2)
        if len(picks) == 2:
            (ax, ay), (bx2, by2) = picks
            cv2.line(B, (bx_(ay), by_(ax)), (bx_(by2), by_(bx2)), (0, 0, 255), 2)
            dist_txt = "잰 거리 %.0f mm" % (1000 * np.hypot(bx2 - ax, by2 - ay))

        if B.shape[0] != left.shape[0]:
            pad_h = max(B.shape[0], left.shape[0])
            def _pad(im):
                if im.shape[0] == pad_h:
                    return im
                return cv2.copyMakeBorder(im, 0, pad_h - im.shape[0], 0, 0,
                                          cv2.BORDER_CONSTANT, value=(24, 24, 24))
            top = cv2.hconcat([_pad(left), _pad(B)])
        else:
            top = cv2.hconcat([left, B])

        lh = 17
        pad = np.full((36 + lh * len(PARAMS) + 46, top.shape[1], 3), 24, np.uint8)
        it = [("초록선이 테이프 위에 얹히면 맞다  |  왜곡보정 %s [u]  |  "
               "정지프레임[스페이스]  |  격자_앞은 라이다 중심 기준"
               % ("켬" if undist_on else "끔"),
               8, 4, 13, (140, 255, 160) if undist_on else (200, 220, 255))]
        for i, (n, d, s) in enumerate(PARAMS):
            y = 24 + i * lh
            on = (i == sel)
            if on:
                cv2.rectangle(pad, (4, y - 1), (pad.shape[1] - 4, y + lh - 4),
                              (58, 58, 58), -1)
            col = (255, 255, 120) if on else (185, 185, 185)
            it += [("▶" if on else " ", 6, y, 12, col), (n, 20, y, 12, col),
                   ("%.4g" % val[n], 112, y, 12, col),
                   ("±%.3g" % (s * stepmul), 180, y, 11, (135, 135, 135)),
                   (d, 250, y, 11, (245, 245, 165) if on else (140, 140, 140))]
        y = 32 + lh * len(PARAMS)
        it.append((dist_txt + "   (45cm 자=450, 타일 한 칸=453)" if dist_txt
                   else "BEV 에서 두 점 클릭 → 실제 거리", 8, y, 14,
                   (120, 255, 160) if dist_txt else (150, 150, 150)))
        if boot["on"]:
            it.append(("★ 대략맞추기: 왼쪽 화면에서 **가로 테이프 3줄**을 "
                       "가까운 것부터 차례로 클릭 (%d/3).  ESC 취소"
                       % len(boot["pts"]), 8, y + 18, 13, (0, 200, 255)))
        else:
            it.append(("↑↓(w x) 선택  ←→(a d) 값  [ ] 스텝  0 되돌리기  R 전체  "
                       "★b 대략맞추기  u 왜곡보정  스페이스 새프레임  c 지움  s 저장  q 종료",
                       8, y + 18, 11, (150, 150, 150)))
        if msg and time.time() - msg_t < 4:
            it.append((msg, 8, y + 32, 12, (120, 255, 255)))
        draw_text(pad, it)
        cv2.imshow(win, cv2.vconcat([top, pad]))
        last_draw = time.time()
        dirty = False

        # 키를 안 누르면 화면을 다시 그리지 않는다 (원격 전송량 절감)
        # 키/클릭이 없으면 화면을 다시 그리지 않는다 (원격 전송량 절감)
        while True:
            k = cv2.waitKeyEx(40)
            if k != -1:
                break
            if st.get("dirty") and (time.time() - last_draw) >= min_dt:
                st["dirty"] = False
                break
        if k == -1:
            continue
        ch = k & 0xFF
        UP = k in (65362, 2490368) or ch == ord('w')
        DOWN = k in (65364, 2621440) or ch == ord('x')
        LEFT = k in (65361, 2424832) or ch == ord('a')
        RIGHT = k in (65363, 2555904) or ch == ord('d')
        name = KEYS[sel]
        if boot["on"]:
            if ch == 27:
                boot["on"], boot["pts"] = False, []
                msg, msg_t = "대략맞추기 취소", time.time()
            elif len(boot["pts"]) >= 3:
                r_, note = rough_fit(boot["pts"][:3], val["fx"], Wimg, Himg,
                                     args.lidar_offset)
                if r_ is None:
                    msg, msg_t = "대략맞추기 실패: %s" % note, time.time()
                else:
                    val["내림각"], val["높이"], val["격자_앞"] = (
                        float(r_[0]), float(r_[1]), float(r_[2]))
                    msg, msg_t = ("대략맞춤 — 내림각 %.1f도 / 높이 %.3fm / "
                                  "격자_앞 %.3fm  (%s)"
                                  % (r_[0], r_[1], r_[2], note)), time.time()
                    print("  " + msg)
                boot["on"], boot["pts"] = False, []
            continue
        if ch in (27, ord('q')):
            break
        elif ch == ord('b'):
            boot["on"], boot["pts"] = True, []
            msg, msg_t = "왼쪽 화면에서 가로 테이프 3줄을 가까운 것부터 클릭", time.time()
        elif UP:
            sel = (sel - 1) % len(PARAMS)
        elif DOWN:
            sel = (sel + 1) % len(PARAMS)
        elif LEFT:
            val[name] -= PARAMS[sel][2] * stepmul
        elif RIGHT:
            val[name] += PARAMS[sel][2] * stepmul
        elif ch == ord('['):
            stepmul = max(1e-3, stepmul / 10)
            msg, msg_t = "스텝 x%.4g" % stepmul, time.time()
        elif ch == ord(']'):
            stepmul = min(1e3, stepmul * 10)
            msg, msg_t = "스텝 x%.4g" % stepmul, time.time()
        elif ch == ord('0') and init:
            val[name] = init[name]
        elif ch == ord('R') and init:
            val.update(init)
            msg, msg_t = "전부 초기값", time.time()
        elif ch == ord('u'):
            if KD["K"] is None:
                msg, msg_t = ("camera.yaml 에 K/D 가 없다", time.time())
            else:
                undist_on = not undist_on
                cache["key"] = None
                msg, msg_t = (("왜곡보정 %s" % ("켬 — fx/cx 는 K 값 사용"
                              if undist_on else "끔")), time.time())
        elif ch == ord(' '):
            held = None                 # 다음 루프에서 새 프레임 한 장 받아온다
            msg, msg_t = "새 프레임 받음", time.time()
        elif ch == ord('c'):
            picks.clear()
        elif ch == ord('s'):
            if yaml is None:
                msg, msg_t = "PyYAML 없음", time.time()
            else:
                p = args.camera
                try:
                    y_ = yaml.safe_load(open(p, encoding="utf-8")) or {}
                except Exception:
                    y_ = {}
                stamp = time.strftime("%m%d-%H%M%S")
                try:
                    if not os.path.exists(p + ".orig"):
                        shutil.copyfile(p, p + ".orig")
                    shutil.copyfile(p, "%s.bak.%s" % (p, stamp))
                except OSError:
                    pass
                y_["H"] = [[float(v) for v in r] for r in Hm]
                y_["h_calibrated"] = True
                y_["intrinsics_calibrated"] = bool(undist_on)
                y_["calibrated"] = bool(undist_on)
                if undist_on:
                    _Kc = KD.get("Kc", KD["K"])     # 출력 new_K
                    _Kd = KD.get("Kd", KD["K"])     # 왜곡모델 K
                    _Dc = KD.get("Dc", KD["D"])
                    y_["new_K"] = [[float(v) for v in r] for r in _Kc]
                    y_["K"] = [[float(v) for v in r] for r in _Kd]
                    y_["D"] = [float(v) for v in _Dc]
                else:
                    y_.pop("new_K", None)
                y_["image_width"], y_["image_height"] = int(Wimg), int(Himg)
                y_["h_frame"] = "lidar_frame"
                y_["h_lidar_offset_x"] = float(args.lidar_offset)
                y_["h_grid_pitch_m"] = float(P)
                y_["h_phys"] = {"undistort": bool(undist_on),
                                "height": float(val["높이"]),
                                "pitch_deg": float(val["내림각"]),
                                "yaw_deg": float(val["좌우틀어짐"]),
                                "roll_deg": float(val["기울기"]),
                                "fx": float(val["fx"]),
                                "grid_x0": float(val["격자_앞"]),
                                "grid_y0": float(val["격자_옆"]),
                                "fy": float(val["fy"]),
                                "out_fx": float(val["출력화각"]),
                                "dist_cx": float(val["왜곡중심X"]),
                                "dist_cy": float(val["왜곡중심Y"]),
                                "dist_k1": float(val["왜곡k1"])}
                y_["h_source"] = ("tools/ground_fit.py 물리값 수동 튜닝 (%s)"
                                  % ("왜곡보정 사용, new_K=K 기준 보정픽셀"
                                     if undist_on else "왜곡보정 미사용, 원본 픽셀"))
                for k_ in ("h_rms_m", "h_max_m", "h_points", "h_points_raw",
                           "h_points_new_K", "h_points_n"):
                    y_.pop(k_, None)
                with open(p, "w", encoding="utf-8") as f:
                    f.write("# tools/ground_fit.py 로 맞춘 값 (%s)\n"
                            "# intrinsics_calibrated=false — 왜곡보정을 쓰지 않고\n"
                            "#   H 를 원본(왜곡 있는) 픽셀에 직접 건다.\n"
                            "# H: 원본 픽셀 -> 지면 미터 (x전방/y좌), 원점 = 라이다\n"
                            % stamp)
                    yaml.safe_dump(y_, f, allow_unicode=True, sort_keys=True)
                msg, msg_t = "저장 완료 (백업 .bak.%s)" % stamp, time.time()
                print("  " + msg)
                print("  H =\n%s" % np.array2string(Hm, precision=10))

    cv2.destroyAllWindows()
    src.close()


if __name__ == "__main__":
    main()
