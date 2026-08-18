#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[deprecated: calib_camera.py 사용] 170도 어안 카메라 내부 파라미터(K/D) 체커보드 캘리브.

⚠ deprecated — tools/calib_camera.py 로 대체됐다. 이 파일은 참고용으로만 남긴다.
   실차 카메라는 실측 HFOV≈87°/VFOV≈64° 의 일반 광각 렌즈였고(벤더 PDF 의 "170도
   어안"은 오기), 87° 렌즈에 등거리(fisheye) 모델을 쓰면 과파라미터화로 CHECK_COND
   실패/가장자리 악화가 난다. 아래로 갈아탈 것:
     python3 calib_camera.py --dir calib_imgs --board 9x6 --square 0.025 \
         --model plumb_bob --save ../config/camera.yaml
   진짜 어안 렌즈라면 calib_camera.py --model fisheye 가 이 파일과 동일 로직이다.
   또한 이 파일은 camera.yaml 에 model 키를 쓰지 않으므로 단독 사용 시
   lib/bev.py 의 모델 분기와 어긋날 수 있다.

역할: capture_frames.py 로 모은 체커보드 이미지에서 cv2.fisheye.calibrate 로
      K(내부행렬)/D(왜곡계수 k1~k4)를 구해 콘솔 출력 + camera.yaml 병합 저장.
입력: --dir 이미지 폴더, --board 내부코너 수(예 9x6), --square 한 칸 실측(m).
출력: 콘솔에 K/D/RMS(px). --save 경로를 주면 기존 yaml 을 읽어 K/D 만 갱신.
      calibrated 는 바닥 H 까지 있어야 true (calib_ground_h.py 가 마무리).
예선 대비 변경점: 신규 도구 (시뮬 카메라는 무왜곡이라 K/D 불필요했음).

사용법:
  python3 calib_fisheye.py --dir calib_imgs --board 9x6 --square 0.025 \
      --save ../config/camera.yaml
  --show 를 주면 이미지별 코너 검출 결과를 띄워 확인 (아무 키 = 다음).

어안 캘리브 흔한 실패 방어:
  - 코너 검출 실패 이미지: 자동 제외하고 목록 출력 (가장자리 왜곡이 심한 프레임).
  - cv2.fisheye.CALIB_CHECK_COND 의 'Ill-conditioned' 에러: 에러 메시지의
    문제 이미지 번호를 파싱해 한 장씩 제외하며 재시도. 남은 장수가 너무
    적어지면 CHECK_COND 를 빼고 마지막 1회 시도 후 결과에 경고 표시.
"""

import argparse
import datetime
import glob
import os
import re
import sys

import cv2
import numpy as np
import yaml

MIN_IMAGES = 6        # 이보다 적으면 캘리브 중단 (신뢰 불가)
RECOMMEND_IMAGES = 15 # 이보다 적으면 경고만
SUBPIX_WIN = (5, 5)   # cornerSubPix 탐색 윈도우 (어안 가장자리 미끄러짐 방지용 소형)


# ======================== camera.yaml 병합 저장 (calib_ground_h.py 와 동일 규약) ========================

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
    order = ["calibrated", "intrinsics_calibrated", "h_calibrated",
             "image_width", "image_height", "K", "D", "H"]
    out = {k: data[k] for k in order if k in data}
    for k, v in data.items():
        if k not in out:
            out[k] = v
    header = (
        "# 실차 카메라 캘리브 파일 — track_drive.lib.bev.load_camera_config() 가 읽는다.\n"
        "# %s 가 %s 에 갱신.\n"
        "# calibrated: K/D(어안)와 H(바닥) 둘 다 완료 시에만 true (부분 플래그는 아래 참고)\n"
        "# K/D: fisheye 내부 파라미터.  H: undistort(new_K=K) 이미지 픽셀 → lidar 미터 (x전방/y좌)\n"
        % (tool_name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None,
                       allow_unicode=True)
    print("저장: %s" % path)


# ======================== 코너 검출 ========================

def _detect_corners(paths, board, show):
    """이미지들에서 체커보드 코너 검출. 실패/크기불일치 이미지는 제외."""
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    size = None            # (w, h) — 첫 유효 이미지 기준
    imgpoints, names = [], []
    n_fail = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("[제외] 읽기 실패: %s" % os.path.basename(p))
            n_fail += 1
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            print("[제외] 해상도 불일치 %dx%d (기준 %dx%d): %s"
                  % (w, h, size[0], size[1], os.path.basename(p)))
            n_fail += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board, find_flags)
        if not found:
            # 어안 가장자리에서 흔한 실패 — 조용히 버리지 말고 알려준다
            print("[제외] 코너 검출 실패: %s" % os.path.basename(p))
            n_fail += 1
            continue
        corners = cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1),
                                   subpix_criteria)
        imgpoints.append(corners.reshape(1, -1, 2).astype(np.float64))
        names.append(os.path.basename(p))

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, board, corners, found)
            cv2.imshow("calib_fisheye", vis)
            cv2.waitKey(0)
    if show:
        cv2.destroyAllWindows()
    return size, imgpoints, names, n_fail


# ======================== fisheye 캘리브 (ill-conditioned 방어 루프) ========================

def _calibrate(objp, imgpoints, names, size):
    """CALIB_CHECK_COND 에러 시 문제 이미지를 하나씩 제외하며 재시도."""
    base_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                  | cv2.fisheye.CALIB_FIX_SKEW)
    flags = base_flags | cv2.fisheye.CALIB_CHECK_COND
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    while True:
        objpoints = [objp] * len(imgpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                objpoints, imgpoints, size, K, D, None, None, flags, criteria)
            return rms, K, D, names, bool(flags & cv2.fisheye.CALIB_CHECK_COND)
        except cv2.error as e:
            m = re.search(r"input array (\d+)", str(e))
            if m is not None and len(imgpoints) > MIN_IMAGES:
                i = int(m.group(1))
                if 0 <= i < len(imgpoints):
                    print("[제외] ill-conditioned (왜곡 극단/기울기 과다): %s" % names[i])
                    imgpoints.pop(i)
                    names.pop(i)
                    continue
            if flags & cv2.fisheye.CALIB_CHECK_COND:
                print("[경고] CHECK_COND 실패 반복 — CHECK_COND 없이 마지막 시도 "
                      "(결과 신뢰도 낮음, 재촬영 권장)")
                flags = base_flags
                continue
            raise


def main():
    ap = argparse.ArgumentParser(
        description="170도 어안 체커보드 캘리브 (자세한 사용법은 파일 헤더)")
    ap.add_argument("--dir", required=True, help="체커보드 이미지 폴더")
    ap.add_argument("--board", default="9x6", help="내부 코너 수 가로x세로 (기본 9x6)")
    ap.add_argument("--square", type=float, default=0.025, help="한 칸 실측 길이 m (기본 0.025)")
    ap.add_argument("--save", default=None, help="camera.yaml 경로 (기존 파일에 K/D 병합)")
    ap.add_argument("--show", action="store_true", help="이미지별 코너 검출 결과 표시")
    args = ap.parse_args()

    try:
        nx, ny = (int(v) for v in args.board.lower().split("x"))
    except ValueError:
        sys.exit("--board 형식 오류 (예: 9x6): %r" % args.board)
    board = (nx, ny)

    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in glob.glob(os.path.join(args.dir, ext)))
    if not paths:
        sys.exit("이미지 없음: %s (jpg/png/bmp)" % args.dir)

    size, imgpoints, names, n_fail = _detect_corners(paths, board, args.show)
    n_found = len(imgpoints)
    print("코너 검출: %d장 성공 / %d장 제외 / 전체 %d장" % (n_found, n_fail, len(paths)))
    if n_found < MIN_IMAGES:
        sys.exit("검출 성공 이미지가 %d장 미만 — 중앙 2/3 위주로 다시 촬영할 것" % MIN_IMAGES)
    if n_found < RECOMMEND_IMAGES:
        print("[경고] %d장뿐 — %d장 이상 권장 (정확도 저하 가능)" % (n_found, RECOMMEND_IMAGES))

    # 체커보드 3D 좌표 (z=0 평면, 한 칸 = square m)
    objp = np.zeros((1, nx * ny, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * args.square

    rms, K, D, used_names, cond_ok = _calibrate(objp, imgpoints, names, size)

    print("\n===== 결과 =====")
    print("사용 이미지: %d장 (해상도 %dx%d)" % (len(used_names), size[0], size[1]))
    print("RMS 재투영 오차: %.4f px%s" % (rms, "" if cond_ok else "  [CHECK_COND 미적용]"))
    np.set_printoptions(precision=6, suppress=True)
    print("K =\n%s" % K)
    print("D = %s" % D.ravel())
    if rms > 1.0:
        print("[경고] RMS %.2fpx > 1.0px — 촬영 품질 확인 권장" % rms)

    if args.save:
        data = _load_yaml(args.save)
        data["K"] = [[float(v) for v in row] for row in K]
        data["D"] = [float(v) for v in D.ravel()]
        data["image_width"], data["image_height"] = int(size[0]), int(size[1])
        data["intrinsics_calibrated"] = True
        data["intrinsics_rms_px"] = float(round(rms, 4))
        h_done = _flag(data, "h_calibrated")
        data["calibrated"] = bool(h_done)  # H 까지 있어야 true
        _save_camera_yaml(args.save, data, "tools/calib_fisheye.py")
        if not h_done:
            print("다음 단계: calib_ground_h.py 로 바닥 H 캘리브 → calibrated:true 가 됨")


if __name__ == "__main__":
    main()
