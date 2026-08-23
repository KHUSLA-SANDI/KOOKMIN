#!/bin/bash
# 모터 기동: ROS1 모터 구독자가 실제 준비된 뒤 demand-driven bridge 시작.
set -eo pipefail

source /home/xytron/env.sh
export ROS_MASTER_URI=http://localhost:11311
source /home/xytron/ros-humble-ros1-bridge/install/local_setup.bash

# 대회 제공 motor 실행기는 --bridge-all-topics 브리지를 먼저 띄운다.
# 그 상태를 정상 브리지로 오인하면 1080p 카메라와 모든 image_transport
# 변종이 ROS1로 넘어가 CPU를 고갈시키므로, 나쁜 브리지만 골라 종료한다.
mapfile -t bridge_pids < <(pgrep -f '[d]ynamic_bridge' || true)
bad_bridge_pids=()
plain_bridge_pids=()
for pid in "${bridge_pids[@]}"; do
    [ -r "/proc/$pid/cmdline" ] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ "$cmdline" == *"--bridge-all-topics"* ]]; then
        bad_bridge_pids+=("$pid")
    else
        plain_bridge_pids+=("$pid")
    fi
done

if [ "${#bad_bridge_pids[@]}" -gt 0 ]; then
    echo "[motor] --bridge-all-topics 브리지 제거: PID ${bad_bridge_pids[*]}"
    kill "${bad_bridge_pids[@]}"
    for _ in $(seq 1 20); do
        alive=0
        for pid in "${bad_bridge_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        [ "$alive" -eq 0 ] && break
        sleep 0.1
    done
    if [ "$alive" -ne 0 ]; then
        echo "[ERROR] 잘못된 dynamic_bridge가 종료되지 않았습니다: ${bad_bridge_pids[*]}"
        exit 1
    fi
fi

# 이미 정상 demand-driven 브리지가 있으면 중복 실행하지 않는다.
mapfile -t remaining_bridge_pids < <(pgrep -f '[d]ynamic_bridge' || true)
if [ "${#remaining_bridge_pids[@]}" -gt 0 ]; then
    echo "[motor] 정상 dynamic_bridge가 이미 실행 중입니다: PID ${remaining_bridge_pids[*]}"
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
