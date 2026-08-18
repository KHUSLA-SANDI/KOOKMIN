#!/bin/bash
# 카메라 기동 — 기본 1920x1080 MJPG (센서 전체 화각).
#
# 이 카메라는 저해상도 모드가 '줄인 것' 이 아니라 '센서를 잘라낸 것' 이라
# 해상도마다 화각이 다르다 (실측):
#   1920x1080 : 가로 100%  세로 100%   <- 센서 전체
#   1024x768  : 가로  75%  세로  99%
#    640x480  : 가로  67%  세로  89%   (예전 설정)
#   1280x720  : 가로  67%  세로  67%
# 그래서 기본을 1920x1080 으로 둔다. YOLO 는 어차피 imgsz 로 줄여 넣으므로
# 추론 속도는 해상도와 무관하고(실측), 늘어나는 건 MJPG 디코딩뿐이다.
#
# framerate 는 카메라가 실제 지원하는 값만 먹는다 (안 맞으면 노드가 죽는다):
#   MJPG 1920x1080=30 / 1280x720=60 / 1024x768=30 / 640x480=120
#   YUYV  640x480=30 / 1920x1080=6
#
# 사용:  bash start_cam.sh                      (1920x1080 MJPG 30fps)
#        bash start_cam.sh 640 480 yuyv2rgb 30
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash 2>/dev/null
export ROS_DOMAIN_ID=7

W=${1:-1920}
H=${2:-1080}
FMT=${3:-mjpeg2rgb}
FPS=${4:-30}

PAT='usb_cam_node_exe'
if pgrep -f "$PAT" > /dev/null; then
  echo "이미 실행중 - 내리고 다시 띄운다"
  pkill -f "$PAT"; sleep 2
fi
if fuser /dev/video0 > /dev/null 2>&1; then
  echo "[경고] /dev/video0 를 다른 프로그램이 잡고 있다 (cam_modes.py 등). 먼저 끌 것:"
  fuser -v /dev/video0 2>&1 | tail -2
  exit 1
fi

echo "카메라 기동: ${W}x${H} [$FMT] ${FPS}fps"
setsid nohup ros2 run usb_cam usb_cam_node_exe --ros-args -r __node:=xycar_cam \
    -p video_device:=/dev/video0 -p pixel_format:="$FMT" \
    -p image_width:=$W -p image_height:=$H -p framerate:=${FPS}.0 \
    > /tmp/cam.log 2>&1 < /dev/null &
sleep 10

if ! pgrep -f "$PAT" > /dev/null; then
  echo "  [실패] 기동 실패 - 로그:"
  grep -iE "terminate|error|unsupported|invalid" /tmp/cam.log | head -3 | sed 's/^/     /'
  echo "  (해상도/포맷/fps 조합이 카메라 지원 목록에 있는지 확인)"
  exit 1
fi
echo "=== 프로세스 ==="
pgrep -af "$PAT" | head -2
echo "=== 실제 발행 ==="
timeout 10 ros2 topic hz /image_raw 2>&1 | grep -oE "average rate: [0-9.]+" | head -1 \
  || echo "  [실패] 발행 안 됨"
timeout 5 ros2 topic info /image_raw 2>&1 | head -3
echo "=== 로그 마지막 ==="
tail -2 /tmp/cam.log
