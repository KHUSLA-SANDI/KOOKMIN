"""이미지 msg → BGR 변환 (순수 파이썬, rclpy 금지).

역할: sensor_msgs/Image 를 OpenCV BGR ndarray 로 변환. msg 는 덕타이핑
      (encoding/height/width/step/data 속성만 사용) — rosbag 재생·유닛테스트 가능.
입력: Image 형태 객체 (실차 /image_raw 는 rgb8(yuyv2rgb) 640x480).
출력: (bgr ndarray | None, info 문자열). 실패 시 반드시 사유를 담아 반환
      — 무음 드랍 금지(재도입 금지 버그 8). 호출측이 info 를 로그로 남긴다.
예선 대비 변경점: lane_detect_node._image_to_rgb 를 lib 로 승격.
      RGB → BGR 기준으로 통일(OpenCV 관례), yuv422(uyvy)/yuy2 지원 추가,
      미지원 인코딩·reshape 실패 시 사유 문자열 반환.
"""

import cv2
import numpy as np


# 인코딩별 픽셀당 바이트 수 (step 패딩 제거용)
_BYTES_PER_PIXEL = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
    "mono8": 1,
    # yuv422 계열은 2바이트/픽셀 (u-y-v-y 또는 y-u-y-v)
    "yuv422": 2,        # ROS 표준명: UYVY 배열
    "uyvy": 2,
    "yuv422_yuy2": 2,   # ROS 표준명: YUY2(=YUYV) 배열
    "yuy2": 2,
    "yuyv": 2,
}

# yuv422 계열 → cv2 변환 코드
_YUV_CVT = {
    "yuv422": cv2.COLOR_YUV2BGR_UYVY,
    "uyvy": cv2.COLOR_YUV2BGR_UYVY,
    "yuv422_yuy2": cv2.COLOR_YUV2BGR_YUY2,
    "yuy2": cv2.COLOR_YUV2BGR_YUY2,
    "yuyv": cv2.COLOR_YUV2BGR_YUY2,
}


def image_msg_to_bgr(msg):
    """sensor_msgs/Image → (bgr | None, info).

    bgr: HxWx3 uint8 (BGR). 실패 시 None.
    info: 성공 시 "ok:<인코딩>", 실패 시 사유 문자열 (호출측이 로그).
    """
    enc = str(msg.encoding).lower()
    bpp = _BYTES_PER_PIXEL.get(enc)
    if bpp is None:
        return None, f"unsupported encoding: {msg.encoding!r}"

    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)
    if h <= 0 or w <= 0:
        return None, f"bad size: {w}x{h}"

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    row_bytes = w * bpp
    try:
        if step >= row_bytes and raw.size >= h * step:
            # step 패딩 제거: 행 단위로 자른 뒤 유효 바이트만 사용
            arr = raw[:h * step].reshape(h, step)[:, :row_bytes]
        elif raw.size == h * row_bytes:
            arr = raw.reshape(h, row_bytes)
        else:
            return None, (f"size mismatch: data={raw.size}B "
                          f"expected>={h * row_bytes}B (step={step})")
    except ValueError as e:
        return None, f"reshape failed: {e}"

    if enc in _YUV_CVT:
        # yuv422 계열: (H, W, 2) 로 보고 cv2 가 디코드
        yuv = arr.reshape(h, w, 2)
        return cv2.cvtColor(yuv, _YUV_CVT[enc]), f"ok:{enc}"

    arr = arr.reshape(h, w, bpp)
    if enc == "bgr8":
        # frombuffer 는 읽기전용 뷰 → 복사해서 반환 (호출측이 그리기 가능)
        return arr.copy(), "ok:bgr8"
    if enc == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), "ok:rgb8"
    if enc == "rgba8":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR), "ok:rgba8"
    if enc == "bgra8":
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR), "ok:bgra8"
    if enc == "mono8":
        return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR), "ok:mono8"
    return None, f"unsupported encoding: {msg.encoding!r}"  # 도달 불가 방어
