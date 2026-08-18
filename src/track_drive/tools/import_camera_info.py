#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""기존 ROS camera_info 캘리브(K/D)를 우리 camera.yaml 로 가져오기 (독립 CLI).

역할: 체커보드 재촬영 없이, 차에 이미 있는 유효한 캘리브를 그대로 채택한다.
      ROS camera_info yaml(camera_matrix/distortion_coefficients/distortion_model)을
      파싱해 우리 camera.yaml 형식(model/K/D)으로 변환·저장.
입력: --file <ros camera_info yaml> 또는 --topic <CameraInfo 토픽> 중 하나.
출력: 콘솔에 model/K/D/추정 FOV. --save 를 주면 camera.yaml 에 병합(없으면 dry-run).
예선 대비 변경점: 신규 도구. 실차 usb_cam 은 이미 plumb_bob 캘리브를
      /home/xytron/.ros/camera_info/default_cam.yaml 로 갖고 있고 /camera_info 로
      발행한다 — 이 값으로 undistort 하면 실제 프레임의 직선이 곧게 펴지는 것을 확인함.

사용법:
  # 1) 파일에서 (차 위에서 실행하거나 파일을 복사해 와서)
  python3 import_camera_info.py --file ~/.ros/camera_info/default_cam.yaml
  python3 import_camera_info.py --file ~/.ros/camera_info/default_cam.yaml \
      --save ../config/camera.yaml
  # 2) 토픽에서 (ROS2 환경 소싱 필수, 카메라 노드 구동 중)
  python3 import_camera_info.py --topic /camera_info --save ../config/camera.yaml

distortion_model 매핑:
  plumb_bob / rational_polynomial → model: plumb_bob   (D 5개 또는 8개 그대로)
  equidistant / fisheye          → model: fisheye      (D 4개)
  그 외는 에러 — 확실하면 --force-model 로 지정.

주의:
  - H(바닥 호모그래피)와 calibrated 플래그는 건드리지 않는다. 왜곡 모델/계수가
    바뀌면 undistort 결과가 달라져 기존 H 는 무효다 → calib_ground_h.py 로 H 재캘리브.
    (calibrated 가 true 로 남아 있으면 경고를 띄운다 — 재캘리브 전 주행 금지)
  - new_K 항목은 지운다. new_K 는 K/D/model/alpha 에서 파생되는 값이라 K/D 를
    갈아끼우면 낡고, 낡은 채로 두면 런타임이 잘못된 좌표계를 '일치'로 오판한다.
    calib_ground_h.py 가 H 와 함께 새 new_K 를 기록한다.
  - camera_info 의 rectification_matrix / projection_matrix 는 무시한다.
    우리 undistort 의 new_K 는 lib/bev.py 규약(fisheye: K / plumb_bob:
    getOptimalNewCameraMatrix(alpha))으로 따로 정해지므로, ROS 가 계산한
    P(새 카메라 행렬)를 그대로 쓰면 H 좌표계가 어긋난다.
"""

import argparse
import datetime
import math
import os
import sys
import time

import cv2
import numpy as np
import yaml

# ROS distortion_model → 우리 model
MODEL_MAP = {
    "plumb_bob": "plumb_bob",
    "rational_polynomial": "plumb_bob",
    "equidistant": "fisheye",
    "fisheye": "fisheye",          # 비표준이지만 실사용에서 종종 보임
}
MODELS = ("plumb_bob", "fisheye")
FOV_PLUMB_WARN_DEG = 120.0
FOV_FISHEYE_WARN_DEG = 100.0


# ======================== camera.yaml 병합 저장 (calib_camera.py 와 동일 규약) ========================

def _load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


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


# ======================== 소스 1: ROS camera_info yaml 파일 ========================

def _matrix_data(node, key, want, path):
    """{rows, cols, data:[...]} 블록에서 data 를 뽑는다. 리스트 직접 지정도 허용."""
    if isinstance(node, dict):
        vals = node.get("data")
    else:
        vals = node
    if not isinstance(vals, (list, tuple)):
        sys.exit("%s: '%s' 형식 오류 (data 리스트 없음)" % (path, key))
    try:
        vals = [float(v) for v in vals]
    except (TypeError, ValueError):
        sys.exit("%s: '%s' 에 숫자가 아닌 값이 있음" % (path, key))
    if want is not None and len(vals) != want:
        sys.exit("%s: '%s' 는 %d개여야 하는데 %d개" % (path, key, want, len(vals)))
    return vals


def _read_from_file(path):
    """ROS camera_info yaml → 정규화 dict.

    camera_info_manager 형식(camera_matrix/distortion_coefficients)을 우선 보고,
    ROS2 메시지를 그대로 덤프한 형식(k/d)도 받아준다.
    """
    if not os.path.isfile(path):
        sys.exit("파일 없음: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        sys.exit("yaml 읽기 실패: %s (%s)" % (path, e))
    if not isinstance(data, dict):
        sys.exit("yaml 최상위가 매핑이 아님: %s" % path)

    if "camera_matrix" in data:
        k_vals = _matrix_data(data["camera_matrix"], "camera_matrix", 9, path)
    elif "k" in data:
        k_vals = _matrix_data(data["k"], "k", 9, path)
    else:
        sys.exit("%s: camera_matrix(또는 k) 없음 — ROS camera_info yaml 이 맞는지 확인" % path)

    if "distortion_coefficients" in data:
        d_vals = _matrix_data(data["distortion_coefficients"],
                              "distortion_coefficients", None, path)
    elif "d" in data:
        d_vals = _matrix_data(data["d"], "d", None, path)
    else:
        sys.exit("%s: distortion_coefficients(또는 d) 없음" % path)

    ros_model = str(data.get("distortion_model", "")).strip().lower()
    w = int(data.get("image_width", 0) or 0)
    h = int(data.get("image_height", 0) or 0)
    return {
        "K": np.asarray(k_vals, np.float64).reshape(3, 3),
        "D": np.asarray(d_vals, np.float64).reshape(-1),
        "ros_model": ros_model,
        "width": w,
        "height": h,
        "name": str(data.get("camera_name", "") or ""),
        "has_rect_p": ("rectification_matrix" in data or "projection_matrix" in data),
        "src": "file:%s" % path,
    }


# ======================== 소스 2: /camera_info 토픽 1회 수신 ========================

def _read_from_topic(topic, timeout_s):
    """rclpy 로 CameraInfo 를 1회 수신. rclpy 미설치/미소싱이면 명확히 에러."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from sensor_msgs.msg import CameraInfo
    except ImportError as e:
        sys.exit(
            "rclpy/sensor_msgs import 실패: %s\n"
            "  → ROS2 환경이 소싱되지 않았거나 이 머신에 rclpy 가 없다.\n"
            "     차 위에서:  source /opt/ros/$ROS_DISTRO/setup.bash 후 재실행\n"
            "     노트북이면: --file 로 camera_info yaml 을 직접 읽을 것\n"
            "       (예: scp xytron@10.42.0.1:~/.ros/camera_info/default_cam.yaml .)"
            % e)

    # BEST_EFFORT 구독자는 RELIABLE/BEST_EFFORT 발행자 양쪽과 호환된다.
    qos = QoSProfile(depth=1,
                     history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)

    box = {}
    rclpy.init(args=None)
    node = None
    try:
        node = Node("import_camera_info")
        node.create_subscription(
            CameraInfo, topic, lambda m: box.setdefault("msg", m), qos)
        print("구독 대기: %s (최대 %.1fs)" % (topic, timeout_s))
        t_end = time.time() + timeout_s
        while "msg" not in box and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

    if "msg" not in box:
        sys.exit("%s 에서 %.1fs 내 메시지 없음 — 토픽명/카메라 노드 구동 확인 "
                 "(ros2 topic list | grep camera_info)" % (topic, timeout_s))
    m = box["msg"]
    return {
        "K": np.asarray(list(m.k), np.float64).reshape(3, 3),
        "D": np.asarray(list(m.d), np.float64).reshape(-1),
        "ros_model": str(m.distortion_model).strip().lower(),
        "width": int(m.width),
        "height": int(m.height),
        "name": "",
        "has_rect_p": True,
        "src": "topic:%s" % topic,
    }


# ======================== 변환/점검 ========================

def _map_model(ros_model, force_model):
    if force_model:
        print("[안내] --force-model %s — distortion_model %r 무시"
              % (force_model, ros_model or "(없음)"))
        return force_model
    if not ros_model:
        sys.exit("distortion_model 이 비어 있음 — 확실하면 --force-model 로 지정 "
                 "(plumb_bob | fisheye)")
    model = MODEL_MAP.get(ros_model)
    if model is None:
        sys.exit("지원하지 않는 distortion_model: %r\n"
                 "  지원: %s\n"
                 "  → 확실하면 --force-model plumb_bob|fisheye"
                 % (ros_model, ", ".join(sorted(MODEL_MAP))))
    return model


def _fit_d(D, model, ros_model):
    """모델별 D 길이 정리. plumb_bob 은 5개(또는 rational 8개) 유지, fisheye 는 4개."""
    D = np.asarray(D, np.float64).reshape(-1)
    if model == "fisheye":
        if D.size > 4:
            extra = D[4:]
            if np.any(np.abs(extra) > 1e-12):
                print("[경고] fisheye 는 계수 4개인데 %d개가 왔고 5번째부터 0이 아님(%s) "
                      "— 앞 4개만 쓴다. 원본 모델이 정말 등거리인지 확인할 것."
                      % (D.size, np.array2string(extra, precision=6)))
            D = D[:4]
        elif D.size < 4:
            print("[안내] 계수 %d개 → 뒤를 0으로 채워 4개로 맞춤" % D.size)
            D = np.concatenate([D, np.zeros(4 - D.size)])
        return D
    # plumb_bob
    if D.size in (5, 8, 12, 14):
        if D.size > 5:
            print("[안내] 계수 %d개(%s) — cv2 는 그대로 지원하지만 lib/bev.py 가 "
                  "D 를 자르지 않는지 확인할 것." % (D.size, ros_model or "?"))
        return D
    if D.size < 5:
        print("[안내] 계수 %d개 → 뒤를 0으로 채워 5개(k1 k2 p1 p2 k3)로 맞춤" % D.size)
        return np.concatenate([D, np.zeros(5 - D.size)])
    print("[경고] 계수 %d개는 표준 길이가 아님 — 앞 5개만 쓴다" % D.size)
    return D[:5]


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


def _lens_fov_deg(K, D, w, h, model):
    """실제 렌즈 화각(도) — 가장자리 픽셀의 입사각으로 추정 (왜곡 반영).

    K 만으로 계산하면 '왜곡 제거 후' 화각이라 줄자로 재는 값보다 작게 나온다
    (실차 예: K 기준 72도 / 왜곡 반영 87도 — 후자가 실측과 맞는 값).
    """
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
                t = _fisheye_theta(math.hypot((u - cx) / fx, (v - cy) / fy), Dn)
                if t is None:
                    return None, None
                half.append(t)
        else:
            und = cv2.undistortPoints(
                pix, K, np.asarray(D, np.float64).reshape(1, -1)).reshape(-1, 2)
            if not np.all(np.isfinite(und)):
                return None, None
            half = [math.atan(abs(und[0, 0])), math.atan(abs(und[1, 0])),
                    math.atan(abs(und[2, 1])), math.atan(abs(und[3, 1]))]
    except cv2.error:
        return None, None
    return (math.degrees(half[0] + half[1]), math.degrees(half[2] + half[3]))


def _pinhole_fov_deg(K, w, h):
    """K 가 뜻하는 핀홀 화각(도) — 왜곡을 뺀 기준선.

    실제 undistort 결과 화각은 new_K(=plumb_bob 이면 getOptimalNewCameraMatrix(alpha))
    에 따라 달라지므로 이 값은 참고용이다. 실제 값은 calib_ground_h.py 가 찍어 준다.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx <= 0.0 or fy <= 0.0 or w <= 0 or h <= 0:
        return None, None
    return (math.degrees(2.0 * math.atan(w / (2.0 * fx))),
            math.degrees(2.0 * math.atan(h / (2.0 * fy))))


def main():
    ap = argparse.ArgumentParser(
        description="ROS camera_info(K/D) → camera.yaml 가져오기 "
                    "(자세한 사용법은 파일 헤더)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", default=None,
                     help="ROS camera_info yaml 경로 (예: ~/.ros/camera_info/default_cam.yaml)")
    src.add_argument("--topic", default=None,
                     help="CameraInfo 토픽 (rclpy 로 1회 수신, 예: /camera_info)")
    ap.add_argument("--save", default=None,
                    help="camera.yaml 경로 (생략 시 dry-run: 화면 출력만)")
    ap.add_argument("--force-model", choices=MODELS, default=None,
                    help="distortion_model 무시하고 강제 지정")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="--topic 수신 대기 초 (기본 10)")
    args = ap.parse_args()

    if args.file:
        info = _read_from_file(os.path.expanduser(args.file))
    else:
        info = _read_from_topic(args.topic, args.timeout)

    model = _map_model(info["ros_model"], args.force_model)
    K = info["K"]
    D = _fit_d(info["D"], model, info["ros_model"])
    w, h = info["width"], info["height"]

    print("\n===== camera_info =====")
    print("소스: %s%s" % (info["src"], ("  (camera_name=%s)" % info["name"])
                          if info["name"] else ""))
    print("distortion_model: %s  →  우리 model: %s"
          % (info["ros_model"] or "(없음)", model))
    print("해상도: %dx%d" % (w, h) if w and h else "해상도: (없음)")
    np.set_printoptions(precision=6, suppress=True)
    print("K =\n%s" % K)
    print("D = %s   (%d개)" % (D, D.size))

    rfx, rfy = _pinhole_fov_deg(K, w, h)
    if rfx is not None:
        print("K 기준 핀홀 화각: HFOV %.1f도 / VFOV %.1f도 "
              "(참고 — 실제 undistort 화각은 new_K/alpha 에 따라 달라짐)" % (rfx, rfy))
    fovx, fovy = _lens_fov_deg(K, D, w, h, model)
    if fovx is not None:
        print("추정 렌즈 화각(왜곡 반영): HFOV %.1f도 / VFOV %.1f도 "
              "— 줄자 실측값과 비교할 것" % (fovx, fovy))
        if model == "plumb_bob" and fovx > FOV_PLUMB_WARN_DEG:
            print("[경고] plumb_bob 인데 HFOV %.0f도 — 원본이 어안일 가능성 확인" % fovx)
        elif model == "fisheye" and fovx < FOV_FISHEYE_WARN_DEG:
            print("[경고] fisheye 인데 HFOV %.0f도 — plumb_bob 이 맞는지 확인" % fovx)
    if info["has_rect_p"]:
        print("[안내] rectification_matrix / projection_matrix 는 무시했다 — "
              "우리 undistort 의 new_K 는 lib/bev.py 규약대로 따로 정해진다")
        print("        (fisheye: new_K=K / plumb_bob: getOptimalNewCameraMatrix(alpha)). "
              "그 값은 calib_ground_h.py 가 H 와 함께 new_K 로 기록한다.")

    if not args.save:
        print("\n[dry-run] --save 를 주지 않아 파일에 쓰지 않았다.")
        print("  저장하려면: --save ../config/camera.yaml")
        return

    data = _load_yaml(args.save)
    old_model = str(data.get("model", "")).strip().lower()
    data["model"] = model
    data["K"] = [[float(v) for v in row] for row in K]
    data["D"] = [float(v) for v in D]
    if w and h:
        data["image_width"], data["image_height"] = int(w), int(h)
    data["intrinsics_calibrated"] = True
    data["intrinsics_source"] = "tools/import_camera_info.py (%s)" % info["src"]
    data.pop("intrinsics_rms_px", None)  # 남의 캘리브 — 우리 RMS 아님
    # new_K 는 K/D/model/alpha 에서 파생되는 값이라 K/D 를 갈아끼우면 낡는다.
    # 그대로 두면 런타임이 낡은 좌표계를 '일치'로 오판하므로 지운다 —
    # calib_ground_h.py 가 H 와 함께 새 new_K 를 기록한다.
    had_new_K = data.pop("new_K", None) is not None
    data.pop("h_alpha", None)
    _save_camera_yaml(args.save, data, "tools/import_camera_info.py")

    print("\n갱신한 키: model, K, D, image_width/height, intrinsics_calibrated")
    if had_new_K:
        print("제거한 키: new_K (새 K/D 기준으로 다시 잡아야 함)")
    print("건드리지 않은 키: H (%s), h_calibrated (%s), calibrated (%s)"
          % ("있음" if "H" in data else "없음",
             data.get("h_calibrated", "미설정"),
             data.get("calibrated", "미설정")))
    if old_model and old_model != model:
        print("[주의] 왜곡 모델 %s → %s 로 바뀌었다." % (old_model, model))
    print("[주의] 왜곡 모델/계수가 바뀌면 undistort 결과가 달라져 기존 H 는 무효다.")
    print("        calib_ground_h.py 로 H 를 다시 잡고 calibrated 를 true 로 만들 것.")
    if bool(data.get("calibrated", False)):
        print("[경고] calibrated 가 true 인 채로 남아 있다 (지시대로 건드리지 않음). "
              "H 재캘리브 전까지 주행에 쓰지 말 것.")


if __name__ == "__main__":
    main()
