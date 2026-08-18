#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원격 BEV 스트림 뷰어 — 노트북에서 차의 BEV 를 실시간으로 본다.

WSL2 는 NAT 뒤에 있어 ROS2 DDS 가 차까지 닿지 않는다. 그래서 네트워크 설정을 바꾸는 대신
SSH 파이프 하나로 JPEG 프레임을 흘려보낸다. (설정 불필요, 대역폭 ~1MB/s)

  [차]  --server : /dev/video0 캡처 → undistort → 지면 H → BEV → JPEG → stdout
  [노트북] 기본  : ssh 로 위를 실행하고 stdout 을 읽어 창에 표시

실행 (노트북 WSL 대화형 터미널에서):
    cd ~/xycar_ws/본선/track_drive/tools
    python3 bev_stream.py

키 : [g] 격자 on/off  [e] 대비강화  [s] 저장  [q] 종료
※ 서버쪽 config 를 쓰므로 차의 camera.yaml 이 최신인지 먼저 확인할 것.
"""
import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.realpath(__file__))

# ─────────────────────────── 서버 (차에서 실행) ───────────────────────────
SERVER_CODE = r'''
import sys, os, time, struct
import numpy as np, cv2
PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

CFG = PKG + "/config/camera.yaml"
PITCH = float(os.environ.get("BEV_PITCH", "0.453333"))
X0, X1 = float(os.environ.get("BEV_X0", "0.2")), float(os.environ.get("BEV_X1", "2.5"))
Y0, Y1 = float(os.environ.get("BEV_Y0", "-1.2")), float(os.environ.get("BEV_Y1", "1.2"))
PPM = int(os.environ.get("BEV_PPM", "200"))
QUAL = int(os.environ.get("BEV_Q", "70"))

cam = B.load_camera_config(CFG)
H = np.array(cam["H"], np.float64)
Hi = np.linalg.inv(H)
wref = np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])

W = int(round((Y1 - Y0) * PPM)); Hh = int(round((X1 - X0) * PPM))
xs = X1 - np.arange(Hh) / PPM
ys = Y1 - np.arange(W) / PPM
XX, YY = np.meshgrid(xs, ys, indexing="ij")
G = np.stack([XX, YY, np.ones_like(XX)], -1).reshape(-1, 3).T
Q = Hi @ G
w = Q[2]; bad = (np.abs(w) < 1e-12) | (np.sign(w) != wref); w = np.where(bad, 1.0, w)
MU = (Q[0] / w).reshape(Hh, W).astype(np.float32)
MV = (Q[1] / w).reshape(Hh, W).astype(np.float32)
MU[bad.reshape(Hh, W)] = -1; MV[bad.reshape(Hh, W)] = -1

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(12): cap.read()
und = None
meta = ("%s|%s|%.4f|%.4f|%.4f|%.4f|%.4f|%d" %
        (cam.get("h_frame","?"), cam.get("h_rms_m","?"), PITCH, X0, X1, Y0, Y1, PPM))
sys.stdout.buffer.write(b"BEVM" + struct.pack("<I", len(meta)) + meta.encode())
sys.stdout.buffer.flush()
while True:
    ok, raw = cap.read()
    if not ok: time.sleep(0.05); continue
    if und is None:
        und = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
    u = und.apply(raw)
    b = cv2.remap(u, MU, MV, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    panel = cv2.hconcat([cv2.resize(u, (426, 320)), cv2.resize(b, (int(320*W/Hh), 320))])
    ok2, enc = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, QUAL])
    if not ok2: continue
    sys.stdout.buffer.write(b"BEVF" + struct.pack("<I", len(enc)) + enc.tobytes())
    sys.stdout.buffer.flush()
'''


def run_server():
    exec(compile(SERVER_CODE, "<server>", "exec"), {"__name__": "__main__"})


# ─────────────────────────── 클라이언트 (노트북) ───────────────────────────
def read_exact(f, n):
    buf = b""
    while len(buf) < n:
        c = f.read(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def run_client(a):
    env = ("BEV_PITCH=%s BEV_X0=%s BEV_X1=%s BEV_Y0=%s BEV_Y1=%s BEV_PPM=%s BEV_Q=%s"
           % (a.pitch, a.range_x[0], a.range_x[1], a.range_y[0], a.range_y[1], a.ppm, a.quality))
    remote = "%s python3 %s --server" % (env, a.remote_path)
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
           "-o", "ServerAliveInterval=5", a.host, remote]
    print("차에 접속해 스트림 시작...  (%s)" % a.host)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    hdr = read_exact(proc.stdout, 8)
    if not hdr or hdr[:4] != b"BEVM":
        err = proc.stderr.read(2000).decode("utf-8", "replace")
        print("스트림 시작 실패:\n", err); return 1
    mlen = struct.unpack("<I", hdr[4:])[0]
    meta = read_exact(proc.stdout, mlen).decode()
    hframe, rms, pitch, x0, x1, y0, y1, ppm = meta.split("|")
    pitch = float(pitch); x0 = float(x0); x1 = float(x1); y0 = float(y0); y1 = float(y1); ppm = int(ppm)
    print("h_frame=%s  h_rms_m=%s  BEV %.1f~%.1f m 전방 / %.1f~%.1f m 좌우" %
          (hframe, rms, x0, x1, y0, y1))
    print("키: [g]격자 [e]대비 [s]저장 [q]종료")

    show_grid, n, t0 = True, 0, time.time()
    try:
        while True:
            h = read_exact(proc.stdout, 8)
            if not h or h[:4] != b"BEVF":
                print("스트림 종료"); break
            ln = struct.unpack("<I", h[4:])[0]
            data = read_exact(proc.stdout, ln)
            if data is None:
                break
            panel = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if panel is None:
                continue
            n += 1
            fps = n / max(1e-6, time.time() - t0)

            if show_grid:
                # BEV 패널은 오른쪽 절반. 표시 크기에 맞춰 격자 좌표 환산
                bw = panel.shape[1] - 426
                sx = panel.shape[0] / ((x1 - x0) * ppm)
                sy = bw / ((y1 - y0) * ppm)
                k = int(np.ceil(x0 / pitch))
                while k * pitch <= x1:
                    r = int(round((x1 - k * pitch) * ppm * sx))
                    for cx in range(426, panel.shape[1], 14):
                        cv2.line(panel, (cx, r), (min(cx + 7, panel.shape[1] - 1), r),
                                 (0, 255, 255), 1)
                    k += 1
                k = int(np.ceil(y0 / pitch))
                while k * pitch <= y1:
                    c = 426 + int(round((y1 - k * pitch) * ppm * sy))
                    for cy in range(0, panel.shape[0], 14):
                        cv2.line(panel, (c, cy), (c, min(cy + 7, panel.shape[0] - 1)),
                                 (0, 255, 255), 1)
                    k += 1
                d = x0 - (x0 % 0.5) + 0.5
                while d <= x1:
                    r = int(round((x1 - d) * ppm * sx))
                    cv2.line(panel, (426, r), (panel.shape[1] - 1, r), (0, 200, 0), 1)
                    cv2.putText(panel, "%.1fm" % d, (430, r - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                    d += 0.5

            cv2.rectangle(panel, (0, 0), (panel.shape[1], 16), (0, 0, 0), -1)
            cv2.putText(panel, "UNDIST | BEV  %.1ffps  frame=%s rms=%s  tiles must be SQUARE"
                        % (fps, hframe, rms), (4, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
            cv2.imshow("BEV stream", panel)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('g'):
                show_grid = not show_grid
            elif k == ord('s'):
                fn = os.path.expanduser("~/bev_%d.jpg" % int(time.time()))
                cv2.imwrite(fn, panel); print("저장:", fn)
    finally:
        proc.terminate()
        cv2.destroyAllWindows()
    return 0


def main():
    ap = argparse.ArgumentParser(description="원격 BEV 스트림 뷰어")
    ap.add_argument("--server", action="store_true", help="(차에서 실행되는 내부용)")
    ap.add_argument("--host", default="xytron@10.42.0.1")
    ap.add_argument("--remote-path",
                    default="/home/xytron/xycar_ws/src/track_drive/tools/bev_stream.py")
    ap.add_argument("--pitch", type=float, default=0.453333)
    ap.add_argument("--range-x", type=float, nargs=2, default=[0.2, 2.5])
    ap.add_argument("--range-y", type=float, nargs=2, default=[-1.2, 1.2])
    ap.add_argument("--ppm", type=int, default=200)
    ap.add_argument("--quality", type=int, default=70)
    a = ap.parse_args()
    sys.exit(run_server() if a.server else run_client(a))


if __name__ == "__main__":
    main()
