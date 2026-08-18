#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라이다-카메라 정합 육안 검증 — 라이다 점을 H 역변환으로 이미지에 투영 (독립 CLI).

역할: camera.yaml 의 H(픽셀→차량좌표) 역행렬로 라이다 스캔 점과 바닥 격자를
      undistort 이미지 위에 오버레이. 벽/콘 라이다 점이 영상 속 실물 발치와
      겹치면 캘리브 정상. 어긋나면 calib_ground_h.py 재캘리브.
입력: --image 사진, --scan npz(ranges/angle_min/angle_inc), --camera camera.yaml.
출력: 오버레이 창 표시(--no-show 로 생략), --save 시 jpg 저장.
예선 대비 변경점: 신규 도구 (시뮬은 좌표 정합이 tf 로 보장돼 검증 불필요했음).
본선 수정: undistort 를 lib/bev.py 의 load_camera_config + Undistorter 로 위임.
      예전에는 cv2.fisheye.* 를 new_K=K 로 하드코딩해서, plumb_bob 카메라
      (실차 실측 = 일반 광각)에서는 왜곡모델도 new_K 도 둘 다 틀렸다.
      그러면 H 가 정확해도 오버레이가 어긋나 "캘리브 실패" 로 오판하게 된다.

사용법:
  python3 verify_lidar_cam.py --image ground.jpg --scan scan.npz \
      --camera ../config/camera.yaml --save overlay.jpg
  라이다가 뒤집혀 장착된 의심이 들면 --yaw-deg 180 으로 비교해 볼 것.

--scan npz 만들기 (차 위에서 1회 실행 — /scan 은 BEST_EFFORT 라 sensor_data QoS 필수):
  python3 - <<'EOF'
  import numpy as np, rclpy
  from sensor_msgs.msg import LaserScan
  from rclpy.qos import qos_profile_sensor_data
  rclpy.init(); node = rclpy.create_node("scan_dump")
  def cb(m):
      np.savez("scan.npz", ranges=np.asarray(m.ranges, np.float32),
               angle_min=np.float64(m.angle_min), angle_inc=np.float64(m.angle_increment))
      print("scan.npz 저장 (%d빔)" % len(m.ranges)); rclpy.shutdown()
  node.create_subscription(LaserScan, "/scan", cb, qos_profile_sensor_data)
  rclpy.spin(node)
  EOF
  ※ --image 사진은 같은 순간의 것일수록 좋다 (capture_frames.py --topic 병용).
"""

import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np
import yaml

WIN = "verify_lidar_cam"
FWD_MIN_M = 0.05       # 이보다 전방(x)이 작은 점은 카메라 뒤/발밑 — 투영 제외
GRID_X = np.arange(0.5, 4.01, 0.5)     # 전방 등거리선 (m)
GRID_Y = (-1.0, -0.5, 0.0, 0.5, 1.0)   # 좌우 등간격선 (m)
PIX_CLIP = 4000        # 투영 픽셀이 이 범위를 벗어나면 그리기 제외 (수치 폭주 방지)


# ======================== camera.yaml 로드 / undistort ========================
# ★ 주행 노드(lib/bev.py)와 완전히 같은 undistort 를 써야 한다. H 는 undistort 된
#   픽셀 좌표계(new_K) 기준이라, 여기서 1픽셀이라도 다르게 펴면 오버레이가 어긋나
#   멀쩡한 캘리브를 "실패" 로 오판하게 된다.

_BEV_MEMO = []


def _import_bev():
    """lib/bev.py 를 파일 경로로 로드 (tools/ 는 ROS 패키지 밖이라 일반 import 불가).

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
        except Exception as e:      # 로드 실패해도 도구는 굴러가야 한다
            print("[안내] lib/bev.py 로드 실패 (%s) — 로컬 폴백 undistort 사용" % e)
            mod = None
    _BEV_MEMO.append(mod)
    return mod


def _load_camera(path, alpha_override=None):
    """camera.yaml → (H, undistort 함수, 안내문). 주행 노드와 동일 경로로 로드."""
    bev = _import_bev()
    if bev is None:
        return _load_camera_fallback(path, alpha_override)

    cam = bev.load_camera_config(path)
    if cam.get("note"):
        print("[camera.yaml] %s" % cam["note"])
    if not cam.get("h_calibrated", False):
        print("[경고] h_calibrated:false — H 가 실차 캘리브값이 아니다(예선 시뮬 폴백). "
              "오버레이는 당연히 어긋난다. tools/calib_ground_h.py 로 H 를 먼저 잡을 것.")
    if not cam.get("intrinsics_calibrated", False):
        print("[경고] intrinsics_calibrated:false — K/D 미보정이라 undistort 가 "
              "패스스루다. tools/import_camera_info.py 또는 calib_camera.py 를 먼저.")

    alpha = (bev.UNDISTORT_ALPHA_DEFAULT if alpha_override is None
             else float(alpha_override))

    def undistort(img):
        h, w = img.shape[:2]
        size = (w, h)
        cal_size = cam.get("image_size")
        if cal_size is not None and tuple(cal_size) != size:
            print("[경고] 이미지 크기 %dx%d 가 캘리브 기준 %dx%d 와 다르다 — "
                  "new_K/H 가 그 크기 기준이라 좌표계가 어긋난다."
                  % (w, h, cal_size[0], cal_size[1]))
        und = bev.Undistorter.from_config(cam, size, alpha=alpha)
        if not und.enabled:
            print("[경고] undistort 비활성(K/D 미보정 또는 맵 생성 실패) — "
                  "원본 픽셀에 그대로 투영한다.")
            return img
        warn = bev.check_new_K_match(cam.get("new_K"), und.new_K)
        if warn:
            print("[경고] %s" % warn)
        nk = np.asarray(und.new_K, np.float64).reshape(3, 3)
        print("undistort: model=%s, alpha=%.2f, new_K fx,fy,cx,cy = "
              "%.3f %.3f %.3f %.3f"
              % (und.model, alpha, nk[0, 0], nk[1, 1], nk[0, 2], nk[1, 2]))
        return und.apply(img)

    return np.asarray(cam["H"], np.float64).reshape(3, 3), undistort


def _load_camera_fallback(path, alpha_override):
    """lib/bev.py 를 못 읽을 때 — 같은 규약을 직접 구현 (모델 분기 포함)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        sys.exit("camera.yaml 읽기 실패: %s (%s)" % (path, e))
    if not isinstance(data, dict):
        sys.exit("camera.yaml 형식 오류: %s" % path)
    try:
        K = np.asarray(data["K"], np.float64).reshape(3, 3)
        D = np.asarray(data["D"], np.float64).reshape(-1)
        H = np.asarray(data["H"], np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError):
        sys.exit("camera.yaml 에 K/D/H 없음/형식 오류: %s" % path)

    model = str(data.get("model", "")).strip().lower()
    if model not in ("plumb_bob", "fisheye"):
        model = "fisheye" if D.size == 4 else "plumb_bob"
    yaml_new_K = data.get("new_K")
    alpha = 0.0 if alpha_override is None else float(alpha_override)

    def undistort(img):
        if np.allclose(K, np.eye(3)) or not np.any(D):
            print("[경고] K/D 미캘리브 — undistort 생략, 원본 픽셀에 투영.")
            return img
        h, w = img.shape[:2]
        try:
            if model == "fisheye":
                Dm = np.concatenate([D, np.zeros(max(0, 4 - D.size))])[:4]
                nk = (K.copy() if yaml_new_K is None
                      else np.asarray(yaml_new_K, np.float64).reshape(3, 3))
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                    K, Dm.reshape(4, 1), np.eye(3), nk, (w, h), cv2.CV_16SC2)
            else:
                Dm = np.concatenate([D, np.zeros(max(0, 5 - D.size))])
                if yaml_new_K is None:
                    nk, _ = cv2.getOptimalNewCameraMatrix(K, Dm, (w, h), alpha)
                    nk = np.asarray(nk, np.float64).reshape(3, 3)
                else:
                    nk = np.asarray(yaml_new_K, np.float64).reshape(3, 3)
                m1, m2 = cv2.initUndistortRectifyMap(
                    K, Dm, np.eye(3), nk, (w, h), cv2.CV_16SC2)
        except (cv2.error, TypeError, ValueError):
            print("[경고] undistort 맵 생성 실패 — 원본 픽셀에 투영.")
            return img
        print("undistort(폴백): model=%s, alpha=%.2f, new_K fx,fy,cx,cy = "
              "%.3f %.3f %.3f %.3f"
              % (model, alpha, nk[0, 0], nk[1, 1], nk[0, 2], nk[1, 2]))
        return cv2.remap(img, m1, m2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    return H, undistort


# ======================== 차량좌표 → 픽셀 투영 ========================

def _project(Hinv, xy, ref_sign):
    """(N,2) 차량좌표(m) → (N,2) 픽셀. 지평선 반대편(부호 뒤집힘) 점은 NaN."""
    n = len(xy)
    p = (Hinv @ np.column_stack([xy, np.ones(n)]).T).T
    uv = np.full((n, 2), np.nan)
    ok = (p[:, 2] * ref_sign) > 1e-9
    uv[ok] = p[ok, :2] / p[ok, 2:3]
    return uv


def _ref_sign(Hinv):
    """전방 1m 기준점으로 유효한 동차좌표 w 부호를 정한다 (H 스케일 부호 임의성 대응)."""
    w = float((Hinv @ np.array([1.0, 0.0, 1.0]))[2])
    return 1.0 if w >= 0.0 else -1.0


def _draw_polyline(img, uv, color):
    """NaN/화면 밖 폭주 점을 건너뛰며 연속 구간만 선으로 잇는다."""
    h, w = img.shape[:2]
    ok = np.isfinite(uv).all(axis=1) & (np.abs(uv) < PIX_CLIP).all(axis=1)
    for i in range(len(uv) - 1):
        if ok[i] and ok[i + 1]:
            p0 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
            p1 = (int(round(uv[i + 1, 0])), int(round(uv[i + 1, 1])))
            cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)


def _draw_grid(img, Hinv, ref_sign):
    """바닥 격자: 전방 등거리선(연두, 라벨) + 좌우 등간격선(회색)."""
    for gy in GRID_Y:
        xs = np.linspace(0.3, 4.0, 81)
        uv = _project(Hinv, np.column_stack([xs, np.full_like(xs, gy)]), ref_sign)
        _draw_polyline(img, uv, (150, 150, 150))
    for gx in GRID_X:
        ys = np.linspace(-2.0, 2.0, 81)
        uv = _project(Hinv, np.column_stack([np.full_like(ys, gx), ys]), ref_sign)
        _draw_polyline(img, uv, (90, 200, 90))
        lab = _project(Hinv, np.array([[gx, 0.0]]), ref_sign)[0]
        if np.isfinite(lab).all() and 0 <= lab[0] < img.shape[1] and 0 <= lab[1] < img.shape[0]:
            cv2.putText(img, "%.1fm" % gx, (int(lab[0]) + 4, int(lab[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 230, 90), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description="라이다-카메라 정합 육안 검증 (자세한 사용법·npz 스니펫은 파일 헤더)")
    ap.add_argument("--image", required=True, help="카메라 사진")
    ap.add_argument("--scan", required=True, help="npz (ranges/angle_min/angle_inc)")
    ap.add_argument("--camera", required=True, help="camera.yaml (K/D/H)")
    ap.add_argument("--save", default=None, help="오버레이 jpg 저장 경로")
    ap.add_argument("--min-range", type=float, default=0.1, help="라이다 최소 거리 m")
    ap.add_argument("--max-range", type=float, default=5.0, help="라이다 최대 거리 m")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="스캔 회전 보정 테스트용 (라이다 장착 방향 의심 시 180 등)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="plumb_bob undistort 의 getOptimalNewCameraMatrix alpha. "
                         "생략 시 lib/bev.py 기본값 — H 캘리브 때와 같아야 한다")
    ap.add_argument("--no-grid", action="store_true", help="바닥 격자 생략")
    ap.add_argument("--no-show", action="store_true", help="창 표시 생략 (저장만)")
    args = ap.parse_args()

    # ---- 스캔 로드: lidar_frame x전방/y좌 (CCW+) — /detect/lane 과 동일 관습
    try:
        d = np.load(args.scan)
        ranges = np.asarray(d["ranges"], np.float64).ravel()
        angle_min = float(d["angle_min"])
        angle_inc = float(d["angle_inc"])
    except (OSError, KeyError, ValueError) as e:
        sys.exit("npz 읽기 실패 (ranges/angle_min/angle_inc 필요): %s" % e)
    angles = angle_min + np.arange(ranges.size) * angle_inc + np.deg2rad(args.yaw_deg)
    good = np.isfinite(ranges) & (ranges > args.min_range) & (ranges < args.max_range)
    x = ranges[good] * np.cos(angles[good])
    y = ranges[good] * np.sin(angles[good])
    r = ranges[good]
    front = x > FWD_MIN_M   # 카메라 뒤쪽 점 제거
    x, y, r = x[front], y[front], r[front]
    print("스캔: 전체 %d빔 → 유효 %d점 → 전방 %d점 (%.2f~%.2fm)"
          % (ranges.size, int(good.sum()), len(x), args.min_range, args.max_range))

    # ---- 이미지/캘리브
    img = cv2.imread(args.image)
    if img is None:
        sys.exit("이미지 읽기 실패: %s" % args.image)
    H, undistort = _load_camera(args.camera, args.alpha)
    disp = undistort(img).copy()
    Hinv = np.linalg.inv(H)
    sgn = _ref_sign(Hinv)

    if not args.no_grid:
        _draw_grid(disp, Hinv, sgn)

    # ---- 라이다 점 투영 (가까움=빨강 → 멂=파랑, JET)
    uv = _project(Hinv, np.column_stack([x, y]), sgn)
    t = np.clip((r - args.min_range) / max(args.max_range - args.min_range, 1e-6), 0, 1)
    colors = cv2.applyColorMap(((1.0 - t) * 255).astype(np.uint8).reshape(-1, 1),
                               cv2.COLORMAP_JET).reshape(-1, 3)
    h_img, w_img = disp.shape[:2]
    n_in = 0
    for i in range(len(uv)):
        if not np.isfinite(uv[i]).all():
            continue
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= u < w_img and 0 <= v < h_img:
            cv2.circle(disp, (u, v), 3, tuple(int(c) for c in colors[i]), -1)
            n_in += 1
    print("이미지 안에 투영된 라이다 점: %d / %d" % (n_in, len(uv)))
    if n_in == 0:
        print("[경고] 투영점 0개 — H 부호/스케일 또는 라이다 장착 방향(--yaw-deg) 확인")

    cv2.rectangle(disp, (0, 0), (w_img, 26), (0, 0, 0), -1)
    cv2.putText(disp, "lidar near=red far=blue (%.1f~%.1fm) | grid 0.5m | any key = quit"
                % (args.min_range, args.max_range), (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if args.save:
        cv2.imwrite(args.save, disp, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("저장: %s" % args.save)
    if not args.no_show:
        cv2.imshow(WIN, disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
