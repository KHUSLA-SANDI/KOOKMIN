#!/usr/bin/env bash
# 지면 H 캘리브용 한 방 캡처 — /image_raw 사진 N장 + /scan 1개(npz) 를 차에서 받아둔다.
#
# 역할: 센서(카메라/라이다)가 안 떠 있으면 sensors.launch.py 로 띄우고 12초 대기 후,
#       calib_ground_h.py 에 넣을 사진과 verify_lidar_cam.py 에 넣을 스캔을 저장.
# 입력: $1 = 출력 폴더(기본 ~/calib_shots), $2 = 사진 장수(기본 3)
# 출력: <out>/shot_NNNN.jpg (원본 raw — undistort 는 도구가 한다), <out>/scan.npz
# 주의: 모터/주행 토픽은 건드리지 않는다. 이 스크립트가 띄운 센서는 종료하지 않는다
#       (여러 번 찍게 되므로). 끝나면 안내되는 명령으로 직접 내릴 것.
#
# 사용법 (차에서):
#   bash ~/xycar_ws/src/track_drive/tools/calib_shot.sh ~/calib_shots 3
# 노트북에서 원격으로:
#   ssh xytron@10.42.0.1 "bash ~/xycar_ws/src/track_drive/tools/calib_shot.sh"
#
# ※ set -u 금지 — /opt/ros/humble/setup.bash 가 unbound 변수를 건드린다.

source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

OUT="${1:-$HOME/calib_shots}"
NSHOT="${2:-3}"
mkdir -p "$OUT"

STARTED=0
if pgrep -f usb_cam_node_exe >/dev/null; then
  echo "[센서] 이미 기동돼 있음 — 그대로 사용 (3초 안정화)"
  sleep 3
else
  echo "[센서] sensors.launch.py 기동 (카메라+라이다+IMU) — 12초 대기"
  nohup ros2 launch track_drive sensors.launch.py > /tmp/calib_sensors.log 2>&1 &
  STARTED=1
  sleep 12
fi

echo "[토픽]"
ros2 topic list | grep -E "^/(image_raw|scan|imu)$" || echo "  [경고] 기대 토픽 일부 없음"

python3 - "$OUT" "$NSHOT" <<'PYEOF'
import sys, time
import numpy as np, cv2, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

out, nshot = sys.argv[1], int(sys.argv[2])
rclpy.init()
node = rclpy.create_node("calib_shot")
st = {"img": None, "scan": None, "warned": set()}

def img_cb(m):
    enc = m.encoding.lower()
    h, w, step = int(m.height), int(m.width), int(m.step)
    buf = np.frombuffer(m.data, np.uint8)
    try:
        if enc in ("rgb8", "bgr8"):
            a = buf.reshape(h, step)[:, :w*3].reshape(h, w, 3)
            st["img"] = a[..., ::-1].copy() if enc == "rgb8" else a.copy()
            return
    except ValueError:
        pass
    if enc not in st["warned"]:
        st["warned"].add(enc)
        print("[경고] 지원하지 않는 인코딩 %r — rgb8 기대" % enc)

def scan_cb(m):
    if st["scan"] is None:
        st["scan"] = (np.asarray(m.ranges, np.float32),
                      float(m.angle_min), float(m.angle_increment))

node.create_subscription(Image, "/image_raw", img_cb, qos_profile_sensor_data)
node.create_subscription(LaserScan, "/scan", scan_cb, qos_profile_sensor_data)

# 기존 연번 이어가기
import os, re
idx = -1
for f in os.listdir(out):
    mm = re.match(r"shot_(\d+)\.jpg$", f)
    if mm:
        idx = max(idx, int(mm.group(1)))
idx += 1

saved = 0
t0 = time.time()
while saved < nshot and time.time() - t0 < 25.0:
    rclpy.spin_once(node, timeout_sec=0.05)
    if st["img"] is not None:
        p = os.path.join(out, "shot_%04d.jpg" % idx)
        cv2.imwrite(p, st["img"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("사진 저장: %s" % p)
        idx += 1; saved += 1; st["img"] = None
        time.sleep(0.7)

t0 = time.time()
while st["scan"] is None and time.time() - t0 < 8.0:
    rclpy.spin_once(node, timeout_sec=0.05)

if st["scan"] is not None:
    r, amin, ainc = st["scan"]
    np.savez(os.path.join(out, "scan.npz"), ranges=r,
             angle_min=np.float64(amin), angle_inc=np.float64(ainc))
    ok = np.isfinite(r) & (r > 0.05) & (r < 12.0)
    print("스캔 저장: %s/scan.npz (%d빔, 유효 %d)" % (out, r.size, int(ok.sum())))
else:
    print("[경고] /scan 미수신 — 라이다 미기동? verify_lidar_cam.py 는 못 돌린다")

node.destroy_node(); rclpy.shutdown()
print("사진 %d장 저장 완료" % saved)
if saved == 0:
    sys.exit(1)
PYEOF
RC=$?

echo
if [ "$STARTED" = "1" ]; then
  echo "[안내] 이 스크립트가 센서를 띄웠다. 계속 찍을 거면 그대로 두고,"
  echo "       다 끝났으면:  pkill -f 'sensors.launch.py'; pkill -f usb_cam_node_exe"
fi
echo "RC=$RC  출력: $OUT"
exit $RC
