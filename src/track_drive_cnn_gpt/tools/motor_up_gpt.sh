#!/bin/bash
# 모터 기동: ROS1 모터 구독자가 실제 준비된 뒤 demand-driven bridge 시작.
set -eo pipefail

source /home/xytron/env.sh
export ROS_MASTER_URI=http://localhost:11311
source /home/xytron/ros-humble-ros1-bridge/install/local_setup.bash

if pgrep -f 'ros1_bridge dynamic_bridge' >/dev/null 2>&1; then
    echo "[motor] dynamic_bridge가 이미 실행 중입니다. 기존 창을 사용하세요."
    exit 0
fi

echo "[1/2] ROS1 컨테이너 시작"
docker start ros1_container >/dev/null

echo "[wait] roscore + vesc_driver + xycar_motor 준비 대기"
ready=0
for _ in $(seq 1 30); do
    if docker inspect -f '{{.State.Running}}' ros1_container 2>/dev/null | grep -qx true; then
        if docker exec ros1_container bash -lc 'source /opt/ros/noetic/setup.bash; source /root/noetic_ws/devel/setup.bash; rostopic info /xycar_motor 2>/dev/null' | grep -q '/xycar_motor'; then
            ready=1
            break
        fi
    fi
    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo "[ERROR] 30초 안에 ROS1 /xycar_motor 구독자가 준비되지 않았습니다."
    echo "        docker logs --tail 100 ros1_container 로 확인하세요."
    exit 1
fi

echo "[2/2] ROS1↔ROS2 dynamic bridge 시작"
echo "      이 창을 그대로 두세요. 종료는 주행 정지 후 Ctrl+C"
# --bridge-all-topics를 쓰지 않는다. ROS1 /xycar_motor 구독자와 ROS2
# /xycar_motor 발행자가 존재할 때 필요한 토픽만 demand-driven으로 연결한다.
exec ros2 run ros1_bridge dynamic_bridge
