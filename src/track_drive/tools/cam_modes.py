#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""카메라 해상도/포맷 고르기 + 조준 도우미 (ROS 없이 /dev/video 직접).

왜 필요한가
  이 카메라는 해상도마다 **화각이 다르다** (640x480 은 센서를 잘라낸 것).
  1920x1080 이 상하좌우 모두 제일 넓고, 특히 위쪽이 더 보여 신호등에 유리하다.
  화각은 캡처 모드가 정하고 크기는 나중에 줄이면 되므로, 먼저 모드를 골라야 한다.

기능
  · 숫자키로 해상도 즉시 전환, [m] 으로 MJPG/YUYV 전환, 실측 fps 표시
  · [b] 640x480 화각 상자 — 지금 모드에서 그 화각이 어디까지인지 표시(참고용)
  · 조준 도우미: 화면 아래의 라이다 원통을 찾아 **중심에서 몇 % 벗어났는지** 실시간 표시
    → 카메라를 돌리면서 이 값이 0 이 되게 맞추면 좌우 정렬이 끝난다
  · [r] ROI 드래그 — 잘라 쓸 영역을 마우스로 지정하면 크기/비율을 알려준다
  · [f] 저장

주의: ROS 카메라 노드가 /dev/video0 을 잡고 있으면 안 된다. 먼저 내릴 것.

실행 (차 PC):
  python3 tools/cam_modes.py
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.realpath(__file__))
_PKG = os.path.dirname(_HERE)
sys.path.insert(0, _PKG)


def _load_text():
    import importlib.util
    p = os.path.join(_HERE, "tuner.py")
    if not os.path.isfile(p):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_tuner_text", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_tuner_text"] = m
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


# (가로, 세로, 화각가로%, 화각세로%) — % 는 1920x1080(센서 전체) 대비 실측값.
# 이 카메라는 저해상도 모드가 줄인 것이 아니라 **잘라낸 것**이라 화각이 다르다.
# 1280x720 은 정합 배율이 1.000 = 1080p 픽셀을 1:1 로 가운데만 오려낸 것이었다.
MODES = [(1920, 1080, 100.0, 100.0),   # 1
         (1280, 720, 66.7, 66.7),      # 2
         (1280, 1024, None, None),     # 3  (아직 미측정)
         (1024, 768, 74.6, 99.4),      # 4
         (800, 600, 49.8, 66.4),       # 5
         (640, 480, 66.7, 88.9),       # 6  (지금 주행에 쓰는 모드)
         (320, 240, None, None)]       # 7


def open_cam(dev, w, h, fourcc):
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        return None
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def find_lidar(gray):
    """화면 아래쪽 중앙의 가장 큰 어두운 덩어리(라이다 원통) 중심 u 를 찾는다."""
    H, W = gray.shape[:2]
    band = gray[int(H * 0.70):, :]
    thr = max(50, int(np.percentile(band, 3)) + 25)
    m = (band < thr).astype(np.uint8)
    k = max(3, int(W / 200) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    best, bi = 0, -1
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < (W * H) * 0.002:
            continue
        if abs(ce[i][0] - W / 2) < W * 0.32 and x > 3 and x + w < W - 3 and a > best:
            best, bi = a, i
    if bi < 0:
        return None
    return float(ce[bi][0]), int(st[bi][1] + int(H * 0.70)), int(best)


def main():
    ap = argparse.ArgumentParser(description="카메라 해상도 고르기 + 조준")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/cam_modes"))
    ap.add_argument("--maxw", type=int, default=1100, help="화면 표시 최대 폭")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mi = 0
    fourcc = "MJPG"
    cap = open_cam(args.device, MODES[mi][0], MODES[mi][1], fourcc=fourcc)
    if cap is None:
        print("카메라를 못 열었다. ROS 카메라 노드가 /dev/video%d 를 잡고 있는지 확인:"
              % args.device)
        print('  PAT="usb_cam""_node_exe"; pkill -f "$PAT"')
        sys.exit(1)

    win = "camera modes"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    show_box = False
    show_grid = False
    roi = {"drag": None, "rect": None}
    st = {"s": 1.0}
    times = []
    msg, msg_t = "", 0.0
    frozen = None

    def on_mouse(ev, x, y, flags, _):
        s = st["s"]
        px, py = x / s, y / s
        if ev == cv2.EVENT_LBUTTONDOWN:
            roi["drag"] = (px, py)
            roi["rect"] = None
        elif ev == cv2.EVENT_MOUSEMOVE and roi["drag"] is not None:
            a = roi["drag"]
            roi["rect"] = (min(a[0], px), min(a[1], py), abs(px - a[0]), abs(py - a[1]))
        elif ev == cv2.EVENT_LBUTTONUP and roi["drag"] is not None:
            a = roi["drag"]
            roi["rect"] = (min(a[0], px), min(a[1], py), abs(px - a[0]), abs(py - a[1]))
            roi["drag"] = None
    cv2.setMouseCallback(win, on_mouse)

    print(__doc__.split("실행")[0])
    print("숫자키 1~%d 로 해상도 전환:" % len(MODES))
    for i, mo in enumerate(MODES):
        fv = ("화각 가로 %5.1f%% / 세로 %5.1f%%" % (mo[2], mo[3])
              if mo[2] is not None else "화각 미측정")
        print("   %d) %4dx%-4d  %s%s" % (i + 1, mo[0], mo[1], fv,
                                         "   ← 센서 전체" if mo[2] == 100.0 else ""))

    while True:
        if frozen is None:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            times.append(time.time())
            if len(times) > 30:
                del times[0]
        else:
            frame = frozen
        H, W = frame.shape[:2]
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 5 else 0.0

        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 중심 십자
        cv2.line(vis, (W // 2, 0), (W // 2, H - 1), (90, 90, 90), 1)
        cv2.line(vis, (0, H // 2), (W - 1, H // 2), (90, 90, 90), 1)
        if show_grid:
            # 상하 각도 잡을 때 재현 가능하게 — 세로 10% 간격 눈금에 % 표시
            gi = []
            for p in range(10, 100, 10):
                y = H * p // 100
                col = (110, 110, 110) if p != 50 else (0, 165, 255)
                cv2.line(vis, (0, y), (W - 1, y), col, 1)
                gi.append(("%d%%" % p, 6, max(0, y - int(W / 55)),
                           max(11, int(W / 70)), col))
            for p in (25, 50, 75):
                x = W * p // 100
                cv2.line(vis, (x, 0), (x, H - 1), (110, 110, 110), 1)
            draw_text(vis, gi)
        if show_box and MODES[mi][2]:
            # 지금 모드 화면 안에서 640x480 모드의 화각이 차지하는 비율
            # = (640 모드 화각) / (지금 모드 화각).  실측값 66.7% / 88.9%
            bw = int(W * min(1.0, 66.7 / MODES[mi][2]))
            bh = int(H * min(1.0, 88.9 / MODES[mi][3]))
            x0, y0 = (W - bw) // 2, (H - bh) // 2
            cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (0, 200, 255), 2)
            draw_text(vis, [("640x480 화각 (대략)", x0 + 6, y0 + 6,
                             max(12, int(W / 60)), (0, 200, 255))])

        # 조준: 라이다 중심
        aim = find_lidar(gray)
        aim_txt = "라이다 원통을 못 찾음 (화면 아래에 보이게)"
        aim_col = (150, 150, 150)
        if aim is not None:
            lu, ltop, larea = aim
            off = lu - W / 2.0
            pct = 200.0 * off / W
            cv2.drawMarker(vis, (int(lu), min(H - 5, ltop + 12)), (0, 0, 255),
                           cv2.MARKER_TRIANGLE_UP, max(14, W // 45), 3)
            aim_txt = ("라이다 중심 %+.0f px  (화면폭의 %+.1f%%)   %s"
                       % (off, pct,
                          "정렬 좋음" if abs(pct) < 1.0 else
                          ("카메라를 %s 로 조금 더" % ("오른쪽" if off > 0 else "왼쪽"))))
            aim_col = (140, 255, 160) if abs(pct) < 1.0 else (0, 200, 255)

        if roi["rect"]:
            x, y, w, h = [int(v) for v in roi["rect"]]
            if w > 4 and h > 4:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 120, 0), 2)

        s = min(1.0, args.maxw / float(W))
        st["s"] = s
        disp = cv2.resize(vis, (int(W * s), int(H * s))) if s < 1.0 else vis

        pad = np.full((104, disp.shape[1], 3), 24, np.uint8)
        fv = MODES[mi]
        fvtxt = ("화각 가로 %.0f%% / 세로 %.0f%%" % (fv[2], fv[3])
                 if fv[2] is not None else "화각 미측정")
        it = [("%d) %dx%d [%s]  %.1f fps   %s%s"
               % (mi + 1, W, H, fourcc, fps, fvtxt,
                  "   [정지]" if frozen is not None else ""),
               8, 5, 16,
               (140, 255, 160) if (fv[2] or 0) >= 99 else (0, 200, 255)),
              (aim_txt, 8, 28, 15, aim_col)]
        if roi["rect"]:
            x, y, w, h = [int(v) for v in roi["rect"]]
            it.append(("잘라낸 영역 %dx%d (비율 %.3f) — 원본의 가로 %.0f%% / 세로 %.0f%%"
                       % (w, h, w / max(1, h), 100 * w / W, 100 * h / H),
                       8, 50, 14, (255, 170, 80)))
        else:
            it.append(("마우스로 끌면 잘라 쓸 영역을 재볼 수 있다", 8, 50, 13, (140, 140, 140)))
        it.append(("숫자키 1~%d 해상도   [m] MJPG/YUYV   [b] 640x480 화각상자   "
                   "[g] 격자   [스페이스] 정지   [f] 저장   [q] 종료" % len(MODES),
                   8, 72, 12, (150, 150, 150)))
        if msg and time.time() - msg_t < 3.5:
            it.append((msg, 8, 88, 13, (120, 255, 255)))
        draw_text(pad, it)
        cv2.imshow(win, cv2.vconcat([disp, pad]))

        k = cv2.waitKeyEx(20)
        if k == -1:
            continue
        ch = k & 0xFF
        if ch in (27, ord('q')):
            break
        elif ord('1') <= ch <= ord('0') + len(MODES):
            mi = ch - ord('1')
            cap.release()
            cap = open_cam(args.device, MODES[mi][0], MODES[mi][1], fourcc=fourcc)
            times.clear(); frozen = None
            msg, msg_t = ("해상도 %dx%d 로 전환"
                          % (MODES[mi][0], MODES[mi][1])), time.time()
        elif ch == ord('m'):
            fourcc = "YUYV" if fourcc == "MJPG" else "MJPG"
            cap.release()
            cap = open_cam(args.device, MODES[mi][0], MODES[mi][1], fourcc=fourcc)
            times.clear(); frozen = None
            msg, msg_t = "포맷 %s 로 전환" % fourcc, time.time()
        elif ch == ord('b'):
            show_box = not show_box
        elif ch == ord('g'):
            show_grid = not show_grid
        elif ch == ord(' '):
            frozen = None if frozen is not None else frame.copy()
        elif ch == ord('f'):
            t = time.strftime("%m%d_%H%M%S")
            p = os.path.join(args.out, "cam_%dx%d_%s.jpg" % (W, H, t))
            cv2.imwrite(p, frame)
            msg, msg_t = "저장: %s" % p, time.time()
            print("  " + msg)

    cap.release()
    cv2.destroyAllWindows()
    print("\n선택한 모드: %dx%d [%s]" % (MODES[mi][0], MODES[mi][1], fourcc))
    print("ROS 카메라를 이 모드로 띄우려면:")
    print("  ros2 run usb_cam usb_cam_node_exe --ros-args -r __node:=xycar_cam \\")
    print("    -p video_device:=/dev/video0 -p pixel_format:=%s \\"
          % ("mjpeg2rgb" if fourcc == "MJPG" else "yuyv2rgb"))
    print("    -p image_width:=%d -p image_height:=%d -p framerate:=30.0"
          % MODES[mi])


if __name__ == "__main__":
    main()
