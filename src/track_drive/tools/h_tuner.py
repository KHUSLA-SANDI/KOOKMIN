#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H 수동 튜너 — 물리 파라미터 슬라이더로 BEV 를 실시간 조정.

H 의 9개 숫자를 직접 만지는 대신, 카메라의 물리적 자세를 조절하면
H 가 자동으로 계산된다. 각 슬라이더의 의미가 직관적이라 손으로 찾기 쉽다.

  height  카메라 높이(m)      ↑ 하면 전체적으로 멀어 보임
  pitch   아래로 기울기(도)    ↑ 하면 가까운 쪽이 늘어남
  yaw     좌우 회전(도)        BEV 가 좌우로 회전 (테이프가 비스듬하면 조정)
  roll    좌우 기울어짐(도)     한쪽이 올라가면 조정
  cam_x   카메라 전후위치(m)    원점(라이다 투영점) 대비. 음수 = 라이다보다 뒤
  cam_y   카메라 좌우위치(m)    양수 = 왼쪽

판정 : BEV 에서 타일이 정사각형이고 노란 격자가 테이프/줄눈과 겹치면 정답.
키   : [s] camera.yaml 에 저장   [r] 초기값 복귀   [p] 현재 H 출력   [q] 종료

실행 (차 화면 터미널에서):
    cd ~/xycar_ws/src/track_drive
    python3 tools/h_tuner.py --image dev/shots/tape2_raw.jpg
"""
import argparse
import os
import sys

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from track_drive.lib import bev as B  # noqa: E402

PITCH_DEFAULT = 0.453333
WIN = "H tuner"


def build_H(K, height, pitch_deg, yaw_deg, roll_deg, cam_x, cam_y):
    """물리 자세 → H (undist 픽셀 → 지면 x전방/y좌).

    월드: x 전방, y 좌, z 위 (원점 = 라이다 지면투영점)
    카메라: X 우, Y 하, Z 전방
    """
    th = np.radians(pitch_deg)      # 아래로 기울기(+)
    ps = np.radians(yaw_deg)        # 좌회전(+)
    rl = np.radians(roll_deg)

    # 월드→카메라 기본 회전 (수평 정면)
    R0 = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]])
    # 카메라 X축(우) 둘레 피치
    Rp = np.array([[1, 0, 0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th), np.cos(th)]])
    # 월드 z축 둘레 yaw
    Ry = np.array([[np.cos(ps), -np.sin(ps), 0],
                   [np.sin(ps), np.cos(ps), 0],
                   [0, 0, 1]])
    # 카메라 Z축(광축) 둘레 roll
    Rr = np.array([[np.cos(rl), -np.sin(rl), 0],
                   [np.sin(rl), np.cos(rl), 0],
                   [0, 0, 1]])
    R = Rr @ Rp @ R0 @ Ry
    C = np.array([cam_x, cam_y, height])       # 월드에서의 카메라 위치
    t = -R @ C
    # 지면(z=0) → 이미지
    Hg2i = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return np.linalg.inv(Hg2i)                 # 이미지 → 지면


def fit_physical(H_target, K, x0):
    """현재 H 에 가장 가까운 물리 파라미터 찾기 (슬라이더 초기값용)"""
    from scipy.optimize import least_squares
    us = np.linspace(40, 600, 12)
    vs = np.linspace(270, 460, 12)
    U, V = np.meshgrid(us, vs)
    P = np.stack([U.ravel(), V.ravel(), np.ones(U.size)])

    def gp(Hm):
        q = Hm @ P
        return (q[:2] / q[2]).ravel()

    tgt = gp(H_target)

    def res(p):
        try:
            return gp(build_H(K, *p)) - tgt
        except Exception:
            return np.full_like(tgt, 1e3)

    sol = least_squares(res, x0, method="lm", max_nfev=3000)
    return sol.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(_HERE, "..", "dev", "shots", "tape2_raw.jpg"))
    ap.add_argument("--camera", default=os.path.join(_HERE, "..", "config", "camera.yaml"))
    ap.add_argument("--pitch-m", type=float, default=PITCH_DEFAULT, help="타일 피치 m")
    ap.add_argument("--ppm", type=int, default=170)
    ap.add_argument("--range-x", type=float, nargs=2, default=[0.2, 2.6])
    ap.add_argument("--range-y", type=float, nargs=2, default=[-1.3, 1.3])
    a = ap.parse_args()

    cam = B.load_camera_config(a.camera)
    raw = cv2.imread(a.image)
    if raw is None:
        print("이미지 없음:", a.image); return
    Hh, Ww = raw.shape[:2]
    und = B.Undistorter.from_config(cam, (Ww, Hh))
    img = und.apply(raw)
    # 대비강화 (테이프/줄눈 잘 보이게)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, ac, bc = cv2.split(lab)
    img = cv2.cvtColor(cv2.merge([cv2.createCLAHE(3.0, (8, 8)).apply(l), ac, bc]),
                       cv2.COLOR_LAB2BGR)

    K = np.array(cam.get("new_K") if cam.get("new_K") is not None else cam["K"], np.float64)
    H_cur = np.array(cam["H"], np.float64)

    # 현재 H 에 해당하는 물리값 추정 (슬라이더 초기값)
    print("현재 H 로부터 물리 파라미터 추정 중...")
    try:
        p0 = fit_physical(H_cur, K, np.array([0.28, 4.0, 0.0, 0.0, -0.10, 0.0]))
    except Exception as e:
        print("추정 실패(%s) — 기본값 사용" % e)
        p0 = np.array([0.28, 4.0, 0.0, 0.0, -0.10, 0.0])
    print("  height=%.3f m  pitch=%.2f°  yaw=%.2f°  roll=%.2f°  cam_x=%.3f  cam_y=%.3f"
          % tuple(p0))

    # 슬라이더는 정수만 → 스케일 지정
    SPEC = [("height mm", 0, 800, lambda v: v / 1000.0, lambda x: int(x * 1000)),
            ("pitch 0.1deg", 0, 400, lambda v: v / 10.0, lambda x: int(x * 10)),
            ("yaw+20 0.1deg", 0, 400, lambda v: v / 10.0 - 20, lambda x: int((x + 20) * 10)),
            ("roll+20 0.1deg", 0, 400, lambda v: v / 10.0 - 20, lambda x: int((x + 20) * 10)),
            ("camx+500 mm", 0, 1000, lambda v: (v - 500) / 1000.0, lambda x: int(x * 1000 + 500)),
            ("camy+500 mm", 0, 1000, lambda v: (v - 500) / 1000.0, lambda x: int(x * 1000 + 500))]

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1180, 640)
    for (name, lo, hi, _, to_int), val in zip(SPEC, p0):
        cv2.createTrackbar(name, WIN, int(np.clip(to_int(val), lo, hi)), hi, lambda x: None)

    x0, x1 = a.range_x
    y0, y1 = a.range_y
    ppm = a.ppm
    BW = int((y1 - y0) * ppm); BH = int((x1 - x0) * ppm)
    xs = x1 - np.arange(BH) / ppm
    ys = y1 - np.arange(BW) / ppm
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    G = np.stack([XX, YY, np.ones_like(XX)], -1).reshape(-1, 3).T

    print("\n슬라이더를 돌려 BEV 에서 타일이 정사각형이 되게 맞추세요.")
    print("[s] 저장  [r] 초기복귀  [p] H 출력  [q] 종료")

    while True:
        vals = [f(cv2.getTrackbarPos(n, WIN)) for (n, _, _, f, _) in SPEC]
        Hm = build_H(K, *vals)
        Hi = np.linalg.inv(Hm)
        Q = Hi @ G
        w = Q[2]
        bad = np.abs(w) < 1e-9
        w = np.where(bad, 1.0, w)
        mu = (Q[0] / w).reshape(BH, BW).astype(np.float32)
        mv = (Q[1] / w).reshape(BH, BW).astype(np.float32)
        # 카메라 뒤쪽(음수 깊이) 제거
        front = (np.sign(w) == np.sign((Hi @ np.array([1.0, 0.0, 1.0]))[2])).reshape(BH, BW)
        mu[~front] = -1; mv[~front] = -1
        bevimg = cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # 타일 격자(노랑 점선) + 0.5m 눈금(초록)
        p = a.pitch_m
        k = int(np.ceil(x0 / p))
        while k * p <= x1:
            r = int((x1 - k * p) * ppm)
            for cx in range(0, BW, 12):
                cv2.line(bevimg, (cx, r), (min(cx + 6, BW - 1), r), (0, 255, 255), 1)
            k += 1
        k = int(np.ceil(y0 / p))
        while k * p <= y1:
            c = int((y1 - k * p) * ppm)
            for cy in range(0, BH, 12):
                cv2.line(bevimg, (c, cy), (c, min(cy + 6, BH - 1)), (0, 255, 255), 1)
            k += 1
        d = x0 - (x0 % 0.5) + 0.5
        while d <= x1:
            r = int((x1 - d) * ppm)
            cv2.line(bevimg, (0, r), (BW - 1, r), (0, 190, 0), 1)
            cv2.putText(bevimg, "x=%.1f" % d, (3, r - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 255, 0), 1)
            d += 0.5
        cv2.line(bevimg, (int((y1 - 0) * ppm), 0), (int((y1 - 0) * ppm), BH - 1), (255, 0, 255), 1)

        sh = 560
        left = cv2.resize(img, (int(img.shape[1] * sh / img.shape[0]), sh))
        rightw = int(bevimg.shape[1] * sh / bevimg.shape[0])
        right = cv2.resize(bevimg, (rightw, sh))
        panel = cv2.hconcat([left, right])
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, "h=%.3fm pitch=%.1f yaw=%.1f roll=%.1f camx=%.3f camy=%.3f"
                    % tuple(vals), (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 150), 1)
        cv2.putText(panel, "UNDIST | BEV  (yellow grid = %.4f m tiles; make tiles SQUARE)"
                    % p, (5, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)
        cv2.imshow(WIN, panel)

        kk = cv2.waitKey(30) & 0xFF
        if kk == ord('q'):
            break
        elif kk == ord('p'):
            print("\nH =\n", np.array2string(Hm / Hm[2, 2], precision=8))
            print("물리값: h=%.4f pitch=%.3f yaw=%.3f roll=%.3f camx=%.4f camy=%.4f" % tuple(vals))
        elif kk == ord('r'):
            for (n, lo, hi, _, to_int), val in zip(SPEC, p0):
                cv2.setTrackbarPos(n, WIN, int(np.clip(to_int(val), lo, hi)))
        elif kk == ord('s'):
            import yaml
            with open(a.camera, encoding="utf-8") as f:
                y = yaml.safe_load(f)
            Hn = Hm / Hm[2, 2]
            y["H"] = [[float(t) for t in r] for r in Hn]
            y["h_frame"] = "lidar_frame"
            y["h_source"] = ("tools/h_tuner.py 수동 (h=%.4f pitch=%.3f yaw=%.3f "
                             "roll=%.3f camx=%.4f camy=%.4f)" % tuple(vals))
            y["h_calibrated"] = True
            y["calibrated"] = bool(y.get("intrinsics_calibrated", True))
            y.pop("h_rms_m", None)
            with open(a.camera, "w", encoding="utf-8") as f:
                f.write("# 자동생성 — tools/h_tuner.py (물리 파라미터 수동 튜닝)\n")
                yaml.safe_dump(y, f, allow_unicode=True, sort_keys=True)
            print("저장 완료:", a.camera)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
