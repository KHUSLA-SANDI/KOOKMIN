#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실시간 BEV 뷰어 — 좌표 읽기 · 거리 재기 · 차체 잘라내기.

좌표 규약 (ROS, 코드 전체가 이걸 쓴다)
  원점 = 라이다 회전중심을 지면에 수직으로 내린 점
  x = 전방 (+),  y = 좌측 (+)          ← 화면에서는 위쪽이 +x, 왼쪽이 +y
  단위 m. camera.yaml 의 H 가 이 좌표를 내놓는다.

쓰는 법
  · 바닥에 물건을 놓고 BEV 위에 마우스를 올리면 그 지점의 (x, y) 가 실시간으로 뜬다.
  · 두 점을 클릭하면 그 사이 실제 거리(mm). 45cm 자를 놓고 재면 축척 검증이 된다.
  · [m] 을 누르고 왼쪽(카메라) 화면을 클릭하면 그 아래를 '차체' 로 잘라낸다.
    바퀴·라이다 픽셀이 차선으로 잡히면 H 를 통과해 가짜 지면점이 되므로 걸러야 한다.
    [s] 로 camera.yaml 에 car_mask_v 로 저장된다.

실행
  python3 tools/bev_view.py --topic /image_raw
  화면이 크면  --view-h 320
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
from track_drive.lib import bev as B                      # noqa: E402

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
            nd = Node("bev_view")

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
    ap = argparse.ArgumentParser(description="실시간 BEV 뷰어 (좌표·거리·차체잘라내기)")
    ap.add_argument("--camera", default=os.path.join(_PKG, "config", "camera.yaml"))
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--image", default=None)
    ap.add_argument("--range-x", type=float, nargs=2, default=[-0.2, 2.4],
                    help="전방 범위 m (x)")
    ap.add_argument("--range-y", type=float, nargs=2, default=[-1.2, 1.2],
                    help="좌우 범위 m (y, 좌가 +)")
    ap.add_argument("--pitch-m", type=float, default=0.453333, help="타일 피치 m")
    ap.add_argument("--view-h", type=int, default=420, help="화면 패널 높이 px")
    ap.add_argument("--fps", type=float, default=8.0, help="화면 갱신 상한")
    args = ap.parse_args()

    cam = B.load_camera_config(args.camera)
    print("camera.yaml: intrinsics=%s  h=%s  size=%s"
          % (cam["intrinsics_calibrated"], cam["h_calibrated"], cam["image_size"]))
    if cam.get("note"):
        print("  note:", cam["note"])
    if not cam["h_calibrated"]:
        print("  ▲ H 미보정 — 좌표를 믿지 말 것")
    H = np.asarray(cam["H"], np.float64)
    raw_y = {}
    if yaml is not None and os.path.isfile(args.camera):
        try:
            raw_y = yaml.safe_load(open(args.camera, encoding="utf-8")) or {}
        except Exception:
            raw_y = {}
    mask_v = raw_y.get("car_mask_v")
    mask_v = float(mask_v) if mask_v is not None else None

    src = Src(args)
    undist = None
    x0, x1 = args.range_x
    y0, y1 = args.range_y
    ppm = args.view_h / max(1e-6, x1 - x0)
    Wb = max(8, int(round((y1 - y0) * ppm)))
    Hb = max(8, int(round((x1 - x0) * ppm)))
    ys = y1 - np.arange(Wb) / ppm          # 왼쪽 열이 +y (좌측)
    xs = x1 - np.arange(Hb) / ppm          # 위쪽 행이 +x (전방)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    GRID = np.stack([XX, YY, np.ones_like(XX)], -1).reshape(-1, 3).T
    Hi = np.linalg.inv(H)
    Q = Hi @ GRID
    wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2]) or 1.0
    w = Q[2]
    bad = (np.abs(w) < 1e-12) | (np.sign(w) != wref)
    w = np.where(bad, 1.0, w)
    MU = (Q[0] / w).reshape(Hb, Wb).astype(np.float32)
    MV = (Q[1] / w).reshape(Hb, Wb).astype(np.float32)
    MU[bad.reshape(Hb, Wb)] = -1
    MV[bad.reshape(Hb, Wb)] = -1
    print("BEV %dx%d px (%.0f px/m) — 전방 %.1f~%.1f m, 좌우 %.1f~%.1f m"
          % (Wb, Hb, ppm, x0, x1, y0, y1))

    def gx(y):                              # 지면 y -> BEV 열
        return int(round((y1 - y) * ppm))

    def gy(x):                              # 지면 x -> BEV 행
        return int(round((x1 - x) * ppm))

    st = {"boff": (0, 0), "ls": 1.0, "hover": None, "dirty": True}
    picks = []
    mask_mode = False
    show_grid = True
    msg, msg_t = "", 0.0
    win = "BEV viewer"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, mx, my, flags, _):
        ox = st["boff"][0]
        if ev == cv2.EVENT_MOUSEMOVE:
            if mx >= ox and 0 <= my < Hb and (mx - ox) < Wb:
                st["hover"] = (x1 - my / ppm, y1 - (mx - ox) / ppm)
            else:
                st["hover"] = None
            return
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        st["dirty"] = True
        if mask_mode and mx < ox:
            st["newmask"] = my / st["ls"]           # 원본 픽셀 v
            return
        if mx >= ox and 0 <= my < Hb and (mx - ox) < Wb:
            picks.append((x1 - my / ppm, y1 - (mx - ox) / ppm))
            if len(picks) > 2:
                del picks[0]
    cv2.setMouseCallback(win, on_mouse)

    print(__doc__.split("실행")[0])
    last = 0.0
    while True:
        frame = src.read()
        if frame is None:
            b = np.full((150, 620, 3), 25, np.uint8)
            draw_text(b, [("이미지 대기중…", 20, 55, 18, (200, 200, 200))])
            cv2.imshow(win, b)
            if (cv2.waitKeyEx(100) & 0xFF) in (27, ord('q')):
                break
            continue
        Himg, Wimg = frame.shape[:2]
        if undist is None:
            undist = B.Undistorter.from_config(cam, (Wimg, Himg))
            print("undistort=%s" % undist.enabled)
            wmsg = B.check_new_K_match(cam.get("new_K"), undist.new_K)
            if wmsg:
                print("  ▲", wmsg)
        if "newmask" in st:
            mask_v = float(st.pop("newmask"))
            mask_mode = False
            msg, msg_t = "차체 경계 v=%.0f (그 아래 무시). [s] 로 저장" % mask_v, time.time()

        und = undist.apply(frame)
        bev = cv2.remap(und, MU, MV, cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        # 차체 영역을 BEV 에서도 어둡게 (그 픽셀은 지면이 아니다)
        if mask_v is not None:
            cut = np.zeros((Himg, Wimg), np.uint8)
            cut[int(np.clip(mask_v, 0, Himg - 1)):, :] = 255
            cm = cv2.remap(cut, MU, MV, cv2.INTER_NEAREST, borderValue=0)
            bev[cm > 127] = (bev[cm > 127] * 0.35).astype(np.uint8)

        if show_grid:
            P = args.pitch_m
            k = int(np.ceil(x0 / P))
            while k * P <= x1:
                r = gy(k * P)
                for c in range(0, Wb, 14):
                    cv2.line(bev, (c, r), (min(c + 7, Wb - 1), r), (0, 200, 200), 1)
                k += 1
            k = int(np.ceil(y0 / P))
            while k * P <= y1:
                c = gx(k * P)
                for r in range(0, Hb, 14):
                    cv2.line(bev, (c, r), (c, min(r + 7, Hb - 1)), (0, 200, 200), 1)
                k += 1
            it = []
            d = np.ceil(x0 / 0.5) * 0.5
            while d <= x1:
                r = gy(d)
                cv2.line(bev, (0, r), (Wb - 1, r), (0, 170, 0), 1)
                it.append(("x=%+.1f" % d, 3, max(0, r - 15), 12, (0, 235, 0)))
                d += 0.5
            d = np.ceil(y0 / 0.5) * 0.5
            while d <= y1:
                c = gx(d)
                cv2.line(bev, (c, 0), (c, Hb - 1), (0, 170, 0), 1)
                it.append(("y=%+.1f" % d, min(c + 3, Wb - 46), Hb - 18, 12, (0, 235, 0)))
                d += 0.5
            draw_text(bev, it)
        # 원점 + 축
        if x0 <= 0 <= x1:
            cv2.line(bev, (0, gy(0)), (Wb - 1, gy(0)), (0, 0, 255), 1)
        if y0 <= 0 <= y1:
            cv2.line(bev, (gx(0), 0), (gx(0), Hb - 1), (0, 0, 255), 1)
        if x0 <= 0 <= x1 and y0 <= 0 <= y1:
            cv2.circle(bev, (gx(0), gy(0)), 6, (255, 60, 0), -1)
            cv2.circle(bev, (gx(0), gy(0)), 9, (255, 255, 255), 1)
            draw_text(bev, [("원점(라이다)", min(gx(0) + 12, Wb - 90),
                             max(0, gy(0) - 16), 12, (255, 120, 0))])
        # +x / +y 화살표
        cv2.arrowedLine(bev, (Wb - 40, Hb - 30), (Wb - 40, Hb - 80),
                        (255, 255, 255), 2, tipLength=0.3)
        cv2.arrowedLine(bev, (Wb - 40, Hb - 30), (Wb - 95, Hb - 30),
                        (255, 255, 255), 2, tipLength=0.3)
        draw_text(bev, [("+x 전방", Wb - 78, Hb - 100, 12, (255, 255, 255)),
                        ("+y 좌", Wb - 145, Hb - 26, 12, (255, 255, 255))])

        for px_, py_ in picks:
            cv2.drawMarker(bev, (gx(py_), gy(px_)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        dist_txt = ""
        if len(picks) == 2:
            (ax, ay), (bx, by) = picks
            cv2.line(bev, (gx(ay), gy(ax)), (gx(by), gy(bx)), (0, 0, 255), 2)
            dist_txt = ("잰 거리 %.0f mm   (Δx %+.0f, Δy %+.0f)"
                        % (1000 * np.hypot(bx - ax, by - ay),
                           1000 * (bx - ax), 1000 * (by - ay)))

        sL = args.view_h / float(Himg)
        left = cv2.resize(und, (int(Wimg * sL), args.view_h), interpolation=cv2.INTER_AREA)
        st["ls"] = sL
        if mask_v is not None:
            yv = int(mask_v * sL)
            if 0 <= yv < left.shape[0]:
                ov = left.copy()
                cv2.rectangle(ov, (0, yv), (left.shape[1] - 1, left.shape[0] - 1),
                              (0, 0, 200), -1)
                left = cv2.addWeighted(left, 0.72, ov, 0.28, 0)
                cv2.line(left, (0, yv), (left.shape[1] - 1, yv), (0, 80, 255), 2)
                draw_text(left, [("차체 — 무시 (v=%.0f)" % mask_v, 6, yv + 4, 13,
                                  (120, 180, 255))])
        if mask_mode:
            draw_text(left, [("차체 경계로 쓸 높이를 클릭", 8, 8, 16, (0, 200, 255))])
        st["boff"] = (left.shape[1], 0)
        if bev.shape[0] != left.shape[0]:
            ph = max(bev.shape[0], left.shape[0])
            def _p(im):
                return im if im.shape[0] == ph else cv2.copyMakeBorder(
                    im, 0, ph - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            top = cv2.hconcat([_p(left), _p(bev)])
        else:
            top = cv2.hconcat([left, bev])

        pad = np.full((96, top.shape[1], 3), 24, np.uint8)
        hv = st["hover"]
        it = [("좌표: 원점=라이다 지면투영점,  x=전방(+),  y=좌측(+)   [ROS 규약]",
               8, 5, 14, (200, 220, 255)),
              ("마우스 위치: %s"
               % ("x=%+.3f m   y=%+.3f m" % hv if hv else "BEV 위에 올려보세요"),
               8, 26, 17, (140, 255, 160) if hv else (140, 140, 140))]
        if dist_txt:
            it.append((dist_txt + "   (45cm 자=450, 타일 한 칸=453)",
                       8, 50, 15, (120, 255, 160)))
        else:
            it.append(("BEV 에서 두 점 클릭 → 실제 거리", 8, 50, 13, (150, 150, 150)))
        it.append(("[m] 차체경계 지정   [s] 저장   [c] 점 지움   [g] 격자   "
                   "[f] 화면저장   [q] 종료", 8, 72, 12, (150, 150, 150)))
        if msg and time.time() - msg_t < 4:
            it.append((msg, 8, 86, 12, (120, 255, 255)))
        draw_text(pad, it)
        cv2.imshow(win, cv2.vconcat([top, pad]))

        dt = max(0.001, 1.0 / max(1.0, args.fps) - (time.time() - last))
        k = cv2.waitKeyEx(max(1, int(dt * 1000)))
        last = time.time()
        if k == -1:
            continue
        ch = k & 0xFF
        if ch in (27, ord('q')):
            break
        elif ch == ord('m'):
            mask_mode = not mask_mode
            msg, msg_t = ("왼쪽 화면에서 차체 경계 높이를 클릭" if mask_mode
                          else "차체경계 지정 취소"), time.time()
        elif ch == ord('c'):
            picks.clear()
        elif ch == ord('g'):
            show_grid = not show_grid
        elif ch == ord('f'):
            t = time.strftime("%m%d_%H%M%S")
            os.makedirs(os.path.expanduser("~/bev_view"), exist_ok=True)
            p_ = os.path.expanduser("~/bev_view/bev_%s.jpg" % t)
            cv2.imwrite(p_, cv2.vconcat([top, pad]))
            msg, msg_t = "저장: %s" % p_, time.time()
            print("  " + msg)
        elif ch == ord('s'):
            if yaml is None or mask_v is None:
                msg, msg_t = "저장할 차체경계가 없다 ([m] 로 먼저 지정)", time.time()
            else:
                p_ = args.camera
                try:
                    y_ = yaml.safe_load(open(p_, encoding="utf-8")) or {}
                except Exception:
                    y_ = {}
                stamp = time.strftime("%m%d-%H%M%S")
                try:
                    shutil.copyfile(p_, "%s.bak.%s" % (p_, stamp))
                except OSError:
                    pass
                y_["car_mask_v"] = float(mask_v)
                y_["car_mask_note"] = ("이 v 아래는 차체(바퀴·라이다)라 지면이 아니다. "
                                       "차선 마스크에서 제외할 것 — 안 그러면 H 를 "
                                       "통과해 가짜 지면점이 된다.")
                with open(p_, "w", encoding="utf-8") as f:
                    f.write("# car_mask_v 추가 (%s)\n" % stamp)
                    yaml.safe_dump(y_, f, allow_unicode=True, sort_keys=True)
                msg, msg_t = ("car_mask_v=%.0f 저장 (백업 .bak.%s)"
                              % (mask_v, stamp)), time.time()
                print("  " + msg)

    cv2.destroyAllWindows()
    src.close()


if __name__ == "__main__":
    main()
