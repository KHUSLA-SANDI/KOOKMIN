#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""카메라 내부 파라미터(K/D) 체커보드 캘리브 — plumb_bob / fisheye 모델 선택 (독립 CLI, ROS 불필요).

역할: capture_frames.py 로 모은 체커보드 이미지에서 K(내부행렬)/D(왜곡계수)를 구해
      콘솔 출력 + camera.yaml 병합 저장. 왜곡 모델을 --model 로 고른다.
입력: --dir 이미지 폴더, --board 내부코너 수(예 9x6), --square 한 칸 실측(m),
      --model plumb_bob(기본) | fisheye.
출력: 콘솔에 model/K/D/RMS(px)/추정 FOV. --save 경로를 주면 기존 yaml 을 읽어
      model/K/D 만 갱신하고 H 는 보존. calibrated 는 바닥 H 까지 있어야 true.
예선 대비 변경점: calib_fisheye.py 를 대체(모델 선택 추가). 실측 결과 실차 카메라는
      HFOV≈87°/VFOV≈64° 의 일반 광각 렌즈였다(벤더 PDF 의 "170도 어안"은 오기).
      87° 렌즈에 등거리(fisheye) 모델을 쓰면 과파라미터화로 CHECK_COND 실패나
      가장자리 악화가 나므로 plumb_bob 이 기본값이다.

사용법:
  # 실차 카메라(일반 광각) — 기본 모델
  python3 calib_camera.py --dir calib_imgs --board 9x6 --square 0.025 \
      --save ../config/camera.yaml
  # 진짜 어안 렌즈로 교체했을 때
  python3 calib_camera.py --dir calib_imgs --board 9x6 --square 0.025 \
      --model fisheye --save ../config/camera.yaml
  --show 를 주면 이미지별 코너 검출 결과를 띄워 확인 (아무 키 = 다음).
  --k3 를 주면 plumb_bob 의 k3 도 추정 (기본은 ROS camera_calibration 과 동일하게 k3=0 고정).

모델 선택 기준:
  - plumb_bob  : 일반/광각 렌즈(HFOV 대략 ~120도 이하). D = [k1 k2 p1 p2 k3] 5개.
                 usb_cam 이 /camera_info 로 발행하는 모델과 동일.
  - fisheye    : 등거리 어안(HFOV 대략 150도 이상). D = [k1 k2 k3 k4] 4개.
  결과의 추정 HFOV 가 모델과 안 맞으면 경고를 띄운다 (모델 재선택 신호).

캘리브 흔한 실패 방어:
  - 코너 검출 실패 이미지: 자동 제외하고 목록 + 사용 장수 출력.
  - fisheye 의 CALIB_CHECK_COND 'Ill-conditioned' 에러: 에러 메시지의 문제 이미지
    번호를 파싱해 한 장씩 제외하며 재시도. 남은 장수가 너무 적어지면 CHECK_COND 를
    빼고 마지막 1회 시도 후 결과에 경고 표시.
"""

import argparse
import datetime
import glob
import math
import os
import re
import sys

import cv2
import numpy as np
import yaml

MIN_IMAGES = 6        # 이보다 적으면 캘리브 중단 (신뢰 불가)
RECOMMEND_IMAGES = 15 # 이보다 적으면 경고만
SUBPIX_WIN = (5, 5)   # cornerSubPix 탐색 윈도우 (가장자리 미끄러짐 방지용 소형)
WIN = "calib_camera"

MODELS = ("plumb_bob", "fisheye")
FOV_PLUMB_WARN_DEG = 120.0   # plumb_bob 인데 이보다 넓으면 어안 의심
FOV_FISHEYE_WARN_DEG = 100.0 # fisheye 인데 이보다 좁으면 일반 광각 의심


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
    order = ["calibrated", "intrinsics_calibrated", "h_calibrated", "model",
             "image_width", "image_height", "K", "D", "new_K", "H"]
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
        "# H: undistort(new_K 기준) 이미지 픽셀 → lidar 미터 (x전방/y좌).\n"
        "#    new_K 가 바뀌면 H 는 통째로 무효 — bev.check_new_K_match() 가 대조한다.\n"
        % (tool_name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None,
                       allow_unicode=True)
    print("저장: %s" % path)


# ======================== 코너 검출 (모델 공통) ========================

def _detect_corners(paths, board, show):
    """이미지들에서 체커보드 코너 검출. 실패/크기불일치 이미지는 제외.

    반환 corners 는 cornerSubPix 원형 (N,1,2) float32 — 모델별로 필요한 모양은
    _calibrate_* 안에서 변환한다.
    """
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    size = None            # (w, h) — 첫 유효 이미지 기준
    corners_list, names = [], []
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
            # 가장자리(왜곡 큰 영역)에서 흔한 실패 — 조용히 버리지 말고 알려준다
            print("[제외] 코너 검출 실패: %s" % os.path.basename(p))
            n_fail += 1
            continue
        corners = cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1),
                                   subpix_criteria)
        corners_list.append(corners)
        names.append(os.path.basename(p))

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, board, corners, found)
            cv2.imshow(WIN, vis)
            cv2.waitKey(0)
    if show:
        cv2.destroyAllWindows()
    return size, corners_list, names, n_fail


# ======================== plumb_bob 캘리브 ========================

def _calibrate_plumb_bob(objp, corners_list, names, size, free_k3):
    """cv2.calibrateCamera — D = [k1 k2 p1 p2 k3] (5개).

    k3 는 기본 고정(0). 광각~일반 렌즈에서 k3 는 대개 불필요하고 자유롭게 두면
    가장자리에서 발산하기 쉽다 (ROS camera_calibration 기본값과 동일한 선택).
    """
    flags = 0 if free_k3 else cv2.CALIB_FIX_K3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    objpoints = [objp.astype(np.float32)] * len(corners_list)
    imgpoints = [c.astype(np.float32) for c in corners_list]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None, flags=flags, criteria=criteria)

    per_img = []
    for i in range(len(imgpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = np.linalg.norm(
            proj.reshape(-1, 2) - imgpoints[i].reshape(-1, 2), axis=1)
        per_img.append(float(np.sqrt(np.mean(err ** 2))))
    return float(rms), K, np.asarray(D, np.float64).reshape(-1), names, True, per_img


# ======================== fisheye 캘리브 (ill-conditioned 방어 루프) ========================

def _calibrate_fisheye(objp, corners_list, names, size):
    """cv2.fisheye.calibrate — D = [k1 k2 k3 k4] (4개).

    CALIB_CHECK_COND 에러 시 문제 이미지를 하나씩 제외하며 재시도.
    (calib_fisheye.py 원문 로직 유지)
    """
    base_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                  | cv2.fisheye.CALIB_FIX_SKEW)
    flags = base_flags | cv2.fisheye.CALIB_CHECK_COND
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    objp_fe = objp.reshape(1, -1, 3).astype(np.float64)
    imgpoints = [c.reshape(1, -1, 2).astype(np.float64) for c in corners_list]
    names = list(names)

    while True:
        objpoints = [objp_fe] * len(imgpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, imgpoints, size, K, D, None, None, flags, criteria)
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
                      "(결과 신뢰도 낮음, 재촬영 또는 --model plumb_bob 권장)")
                flags = base_flags
                continue
            # CHECK_COND 를 빼고도 실패 — 트레이스백 대신 원인/대안을 알려준다
            sys.exit(
                "fisheye 캘리브 실패 (남은 %d장): %s\n"
                "  원인 1) 렌즈가 등거리 어안이 아님 — 일반/광각 렌즈에 등거리 모델을\n"
                "          쓰면 과파라미터화로 이렇게 터진다. --model plumb_bob 로 재시도.\n"
                "  원인 2) 보드 기울기가 과하거나 화면 가장자리에 몰림 — 중앙 2/3 위주로\n"
                "          기울기를 줄여 재촬영."
                % (len(imgpoints), str(e).strip().splitlines()[-1]))

        per_img = []
        for i in range(len(imgpoints)):
            proj, _ = cv2.fisheye.projectPoints(objp_fe, rvecs[i], tvecs[i], K, D)
            err = np.linalg.norm(
                proj.reshape(-1, 2) - imgpoints[i].reshape(-1, 2), axis=1)
            per_img.append(float(np.sqrt(np.mean(err ** 2))))
        cond_ok = bool(flags & cv2.fisheye.CALIB_CHECK_COND)
        return (float(rms), K, np.asarray(D, np.float64).reshape(-1),
                names, cond_ok, per_img)


# ======================== 결과 점검 ========================

def _fisheye_theta(r_d, D):
    """등거리 모델 r_d = θ(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸) 를 θ 에 대해 이분법으로 푼다."""
    k = (list(np.asarray(D, np.float64).reshape(-1)) + [0.0] * 4)[:4]

    def f(t):
        t2 = t * t
        return t * (1.0 + k[0] * t2 + k[1] * t2 ** 2
                    + k[2] * t2 ** 3 + k[3] * t2 ** 4) - r_d

    lo, hi = 0.0, math.pi
    if f(hi) < 0.0:
        return None  # 모델 범위 밖 (계수가 비정상)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _lens_fov_deg(K, D, size, model):
    """실제 렌즈 화각(도) — 가장자리 픽셀의 입사각으로 추정 (왜곡 반영).

    K 만으로 계산하면 '왜곡 제거 후' 화각이라 줄자로 재는 값보다 작게 나온다
    (실차 예: K 기준 72도 / 왜곡 반영 87도 — 후자가 실측과 맞는 값).
    """
    w, h = int(size[0]), int(size[1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx <= 0.0 or fy <= 0.0 or w <= 1 or h <= 1:
        return None, None
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pix = np.array([[0.0, cy], [w - 1.0, cy], [cx, 0.0], [cx, h - 1.0]],
                   np.float64).reshape(-1, 1, 2)
    try:
        if model == "fisheye":
            Dn = np.asarray(D, np.float64).reshape(-1)[:4]
            half = []
            for u, v in pix.reshape(-1, 2):
                r_d = math.hypot((u - cx) / fx, (v - cy) / fy)
                t = _fisheye_theta(r_d, Dn)
                if t is None:
                    return None, None
                half.append(t)
        else:
            Dn = np.asarray(D, np.float64).reshape(1, -1)
            und = cv2.undistortPoints(pix, K, Dn).reshape(-1, 2)
            if not np.all(np.isfinite(und)):
                return None, None
            half = [math.atan(abs(und[0, 0])), math.atan(abs(und[1, 0])),
                    math.atan(abs(und[2, 1])), math.atan(abs(und[3, 1]))]
    except cv2.error:
        return None, None
    return (math.degrees(half[0] + half[1]), math.degrees(half[2] + half[3]))


def _pinhole_fov_deg(K, size):
    """K 가 뜻하는 핀홀 화각(도) — 왜곡을 뺀 기준선.

    실제 undistort 결과 화각은 new_K(=plumb_bob 이면 getOptimalNewCameraMatrix(alpha))
    에 따라 달라지므로 이 값은 참고용이다. 실제 값은 calib_ground_h.py 가 찍어 준다.
    """
    w, h = float(size[0]), float(size[1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        return None, None
    return (math.degrees(2.0 * math.atan(w / (2.0 * fx))),
            math.degrees(2.0 * math.atan(h / (2.0 * fy))))


def _report(model, size, used_names, rms, K, D, cond_ok, per_img):
    print("\n===== 결과 =====")
    print("모델: %s" % model)
    print("사용 이미지: %d장 (해상도 %dx%d)" % (len(used_names), size[0], size[1]))
    print("RMS 재투영 오차: %.4f px%s"
          % (rms, "" if cond_ok else "  [CHECK_COND 미적용]"))
    np.set_printoptions(precision=6, suppress=True)
    print("K =\n%s" % K)
    print("D = %s   (%d개)" % (D, D.size))

    rfx, rfy = _pinhole_fov_deg(K, size)
    if rfx is not None:
        print("K 기준 핀홀 화각: HFOV %.1f도 / VFOV %.1f도 "
              "(참고 — 실제 undistort 화각은 new_K/alpha 에 따라 달라짐)" % (rfx, rfy))
    fovx, fovy = _lens_fov_deg(K, D, size, model)
    if fovx is not None:
        print("추정 렌즈 화각(왜곡 반영): HFOV %.1f도 / VFOV %.1f도 "
              "— 줄자 실측값과 비교할 것" % (fovx, fovy))
        if model == "plumb_bob" and fovx > FOV_PLUMB_WARN_DEG:
            print("[경고] plumb_bob 인데 HFOV %.0f도 — 어안일 가능성. "
                  "--model fisheye 로도 돌려 RMS 를 비교할 것." % fovx)
        elif model == "fisheye" and fovx < FOV_FISHEYE_WARN_DEG:
            print("[경고] fisheye 인데 HFOV %.0f도 — 일반 광각 렌즈일 가능성. "
                  "--model plumb_bob 권장 (등거리 모델은 과파라미터화로 불안정)." % fovx)

    if per_img:
        worst = sorted(range(len(per_img)), key=lambda i: per_img[i], reverse=True)[:3]
        print("이미지별 오차 최악 3장: %s"
              % ", ".join("%s %.3fpx" % (used_names[i], per_img[i]) for i in worst))
    if rms > 1.0:
        print("[경고] RMS %.2fpx > 1.0px — 촬영 품질/보드 규격(--board,--square) 확인 권장" % rms)


# ======================== main ========================

def main():
    ap = argparse.ArgumentParser(
        description="체커보드 내부 파라미터 캘리브 — plumb_bob/fisheye "
                    "(자세한 사용법은 파일 헤더)")
    ap.add_argument("--dir", required=True, help="체커보드 이미지 폴더")
    ap.add_argument("--model", choices=MODELS, default="plumb_bob",
                    help="왜곡 모델 (기본 plumb_bob: 일반/광각 렌즈)")
    ap.add_argument("--board", default="9x6", help="내부 코너 수 가로x세로 (기본 9x6)")
    ap.add_argument("--square", type=float, default=0.025,
                    help="한 칸 실측 길이 m (기본 0.025)")
    ap.add_argument("--save", default=None, help="camera.yaml 경로 (기존 파일에 model/K/D 병합)")
    ap.add_argument("--show", action="store_true", help="이미지별 코너 검출 결과 표시")
    ap.add_argument("--k3", action="store_true",
                    help="plumb_bob 의 k3 도 추정 (기본: k3=0 고정)")
    args = ap.parse_args()

    try:
        nx, ny = (int(v) for v in args.board.lower().split("x"))
    except ValueError:
        sys.exit("--board 형식 오류 (예: 9x6): %r" % args.board)
    if nx < 2 or ny < 2:
        sys.exit("--board 내부 코너 수가 너무 작음: %r" % args.board)
    if args.square <= 0.0:
        sys.exit("--square 는 양수여야 함: %r" % args.square)
    board = (nx, ny)

    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in glob.glob(os.path.join(args.dir, ext)))
    if not paths:
        sys.exit("이미지 없음: %s (jpg/png/bmp)" % args.dir)

    size, corners_list, names, n_fail = _detect_corners(paths, board, args.show)
    n_found = len(corners_list)
    print("코너 검출: %d장 성공 / %d장 제외 / 전체 %d장" % (n_found, n_fail, len(paths)))
    if n_found < MIN_IMAGES:
        sys.exit("검출 성공 이미지가 %d장 미만 — 중앙 2/3 위주로 다시 촬영할 것" % MIN_IMAGES)
    if n_found < RECOMMEND_IMAGES:
        print("[경고] %d장뿐 — %d장 이상 권장 (정확도 저하 가능)" % (n_found, RECOMMEND_IMAGES))

    # 체커보드 3D 좌표 (z=0 평면, 한 칸 = square m)
    objp = np.zeros((nx * ny, 3), np.float64)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * args.square

    if args.model == "fisheye":
        rms, K, D, used_names, cond_ok, per_img = _calibrate_fisheye(
            objp, corners_list, names, size)
    else:
        if args.k3:
            print("[안내] k3 자유 추정 — 가장자리 발산 시 --k3 를 빼고 재시도할 것")
        rms, K, D, used_names, cond_ok, per_img = _calibrate_plumb_bob(
            objp, corners_list, names, size, args.k3)

    _report(args.model, size, used_names, rms, K, D, cond_ok, per_img)

    if args.save:
        data = _load_yaml(args.save)
        old_model = str(data.get("model", "")).strip().lower()
        data["model"] = args.model
        data["K"] = [[float(v) for v in row] for row in K]
        data["D"] = [float(v) for v in D]
        data["image_width"], data["image_height"] = int(size[0]), int(size[1])
        data["intrinsics_calibrated"] = True
        data["intrinsics_rms_px"] = float(round(rms, 4))
        data["intrinsics_source"] = "tools/calib_camera.py (%s, %d장)" % (
            args.model, len(used_names))
        # new_K 는 K/D/model/alpha 에서 파생되는 값이라 K/D 를 갈아끼우면 낡는다.
        # 그대로 두면 런타임이 낡은 좌표계를 '일치'로 오판하므로 지운다 —
        # calib_ground_h.py 가 H 와 함께 새 new_K 를 기록한다. (H 값 자체는 보존)
        had_new_K = data.pop("new_K", None) is not None
        data.pop("h_alpha", None)
        h_done = _flag(data, "h_calibrated")
        data["calibrated"] = bool(h_done)  # H 까지 있어야 true (H 값 자체는 보존)
        _save_camera_yaml(args.save, data, "tools/calib_camera.py")
        if had_new_K:
            print("[안내] 기존 new_K 항목 제거 (새 K/D 기준으로 다시 잡아야 함).")
        if old_model and old_model != args.model:
            print("[주의] 왜곡 모델 %s → %s 로 변경됨. undistort 결과가 달라지므로 "
                  "기존 H 는 무효다." % (old_model, args.model))
        if h_done:
            print("[주의] K/D 가 바뀌면 undistort 픽셀이 달라져 기존 H 는 무효다. "
                  "H 값은 보존했지만 calib_ground_h.py 로 반드시 다시 잡을 것 "
                  "(calibrated 는 true 로 남아 있으니 주행 전 확인).")
        else:
            print("다음 단계: calib_ground_h.py 로 바닥 H 캘리브 → calibrated:true 가 됨")


if __name__ == "__main__":
    main()
