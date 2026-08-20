#!/usr/bin/env python3
"""키보드 수동주행 노드 — 홈의 _teleop.py 계승 + CarInterface/mux 계약 적용.

역할: termios cbreak 키 입력(w/s/a/d/A/D/스페이스/c/x/k/q)으로 수동 조작.
      기본 모드는 /teleop_cmd 발행(mux 경유), --direct 인자 시 CarInterface를 거쳐
      /xycar_motor 직발행. 워치독(3s 무입력 감속) + 종료 시 0 발행 유지.
입력: 키보드 (stdin, TTY 필수)
출력: /teleop_cmd [angle, speed, estop] 20Hz (기본) 또는 /xycar_motor [angle, speed] (--direct)
예선 대비 변경점: _teleop.py의 트림/기계 스톱/램프 로직을 CarInterface 모듈로 교체
      (수치 산재 제거 — car.yaml 파라미터 공유). 조향 상태는 논리 단위(±100 스케일)로
      변경 — 기계 변환은 CarInterface(직발행) 또는 motion(mux)이 담당.

사용: ros2 run track_drive teleop [--direct] [--ros-args --params-file car.yaml]
"""

import sys
import time
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Bool, Float32MultiArray

from .lib import drive_cmd
from .lib.car_interface import CarInterface

# 표시용 서보 환산 (실측: servo = out×0.0082518+0.5004) — 발행값 아님
SERVO_GAIN = 0.0082518
SERVO_OFF = 0.5004


class TeleopNode(Node):
    def __init__(self, direct):
        super().__init__("teleop_node")
        self.direct = direct

        # ---------------- 조작감 파라미터 ----------------
        self.declare_parameter("rate_hz", 20.0)        # 발행 주기 (모터는 연속 수신 필요)
        self.declare_parameter("speed_step", 2.0)      # w/s 한 번당 목표속도 변화
        # ---------------- 조향: 연속 증감이 아니라 이산 단계 ----------------
        # 예전에는 a/d 가 논리값을 조금씩(5씩) 밀었는데, 좌회전 한 번 하려고 a 를
        # 십수 번 눌러야 했다(논리 범위가 -76~+148 이라 5씩으로는 턱없이 느리다).
        # 그래서 좌우 최대~중립을 홀수 단계로 나누고 a/d 가 한 칸씩 옮기게 바꿨다.
        # 5단계면 1,2=좌 / 3=중립 / 4,5=우.
        self.declare_parameter("steer_steps", 5)       # 홀수여야 한다 (중앙 칸이 있어야 직진)
        # True = a/d 칸 크기를 좌우 같게 (좁은 쪽 기준). False = 옛 동작(각 방향 최대까지)
        self.declare_parameter("steer_symmetric", True)
        # 아래 둘은 이제 조향에 쓰지 않는다. 예전 params 파일 호환을 위해 선언만 남긴다.
        self.declare_parameter("angle_step", 5.0)
        self.declare_parameter("angle_step_big", 24.0)
        self.declare_parameter("speed_max", 100.0)
        self.declare_parameter("speed_min", -50.0)     # 후진 한계 (직발행 시 allow_reverse 필요)
        # 무입력 시 자동 감속까지의 시간. 0 이면 끈다 — 키를 안 눌러도 계속 달린다.
        # ★ 0 으로 둬도 teleop 프로세스가 죽거나 창을 닫으면 차는 선다:
        #   /teleop_cmd 가 0.3초(drive_cmd.TELEOP_STALE_SEC) 끊기면 motion 이 자율로
        #   되돌아가고, 자율 입력이 없으면 정지한다. 여기서 끄는 건 "사람이 자리를
        #   비웠는데 프로세스는 살아있는" 경우뿐이다.
        self.declare_parameter("watchdog_sec", 0.0)

        # ---------------- CarInterface 파라미터 (car.yaml 공유 — motion과 동일 키) ----------------
        # 영점은 ROS1 xycar_motor의 calibration 파일에서 일괄 적용한다.
        self.declare_parameter("steer_trim", 0.0)
        self.declare_parameter("steer_scale_left", 0.625933146)
        self.declare_parameter("steer_scale_right", 0.585923661)
        self.declare_parameter("steer_limit_left", -62.593314622)
        self.declare_parameter("steer_limit_right", 58.592366078)
        self.declare_parameter("slew_angle_per_tick", 8.0)
        self.declare_parameter("slew_speed_per_tick", 1.5)  # 센서리스 기동 램프 (예선 teleop과 동일)
        self.declare_parameter("speed_deadzone", 4.0)      # 이 미만 명령은 안 구름 (실측)
        self.declare_parameter("allow_reverse", True)      # 텔레옵은 후진 허용 (motion은 False)
        self.declare_parameter("cmd_to_ms", 0.08)          # speed 명령 ×0.08 = m/s (실측)

        p = lambda name: self.get_parameter(name).value
        self.rate_hz = float(p("rate_hz"))
        self.speed_step = float(p("speed_step"))
        self.angle_step = float(p("angle_step"))
        self.angle_step_big = float(p("angle_step_big"))
        self.speed_max = float(p("speed_max"))
        self.speed_min = float(p("speed_min"))
        self.watchdog_sec = float(p("watchdog_sec"))

        cfg = {k: p(k) for k in (
            "steer_trim", "steer_scale_left", "steer_scale_right",
            "steer_limit_left", "steer_limit_right",
            "slew_angle_per_tick", "slew_speed_per_tick", "speed_deadzone",
            "allow_reverse", "cmd_to_ms")}
        self.car = CarInterface(cfg)
        # 논리 조향 키 상태 한계 — 기계 스톱을 논리 단위로 환산 (한계 탐색 가능하도록)
        self.steer_lo = (
            self.car.steer_limit_left / self.car.steer_scale_left)
        self.steer_hi = (
            self.car.steer_limit_right / self.car.steer_scale_right)

        # ---------------- 조향 단계 테이블 ----------------
        n = int(p("steer_steps"))
        if n < 3:
            n = 3
        if n % 2 == 0:
            n += 1          # 짝수면 중앙 칸이 없어 직진을 못 만든다
        self.steer_steps = n
        self.steer_mid = (n + 1) // 2          # 1-based 중앙 칸 번호 (5단계면 3)
        half = self.steer_mid - 1              # 중앙에서 끝까지 몇 칸인가
        # 좌우 VESC 끝점이 조금 비대칭이라 각 방향을 따로 등분한다.
        # 그래야 3번이 정확히 직진이고, 1번/5번이 각 방향 최대가 된다.
        # ★ 예전엔 각 방향을 "그 방향 최대까지" 등분했다. 그런데 좌우 스톱이
        #   비대칭이면 같은 칸 수에서도 실제 출력 크기가 달라질 수 있다.
        #   조작이 안 맞으니 기본은 좁은 쪽(좌)에 맞춰 양쪽을 같은 크기로 나눈다.
        #   우회전 최대 성능이 필요하면 D 키가 기계 스톱까지 그대로 간다.
        self.steer_symmetric = bool(p("steer_symmetric"))
        lim = min(abs(self.steer_lo), abs(self.steer_hi))
        self.steer_table = []
        for i in range(1, n + 1):
            f = (i - self.steer_mid) / float(half)     # -1.0 ~ +1.0
            if self.steer_symmetric:
                self.steer_table.append(lim * f)
            else:
                self.steer_table.append(
                    self.steer_lo * -f if f < 0 else self.steer_hi * f)

        if direct:
            self.pub = self.create_publisher(Float32MultiArray, "/xycar_motor", 1)
        else:
            self.pub = self.create_publisher(Float32MultiArray, "/teleop_cmd", 1)

        # 녹화 토글 — r 키를 누르면 recorder_node 로 상태를 보낸다.
        # ★ 여기서 파일을 직접 쓰지 않는 이유: 1080p JPEG 인코딩(10~20ms)과 디스크
        #   지연이 이 20Hz 루프에 들어오면, 밀리는 동안 차가 직전 명령을 그대로
        #   유지한 채 굴러간다. 그래서 토픽만 쏘고 저장은 별도 노드가 한다.
        self.pub_rec = self.create_publisher(Bool, "/record_toggle", 10)

    # ---------------- 조향 단계 helper ----------------

    def steer_of(self, step):
        """단계 번호(1-based) → 논리 조향값. 범위를 벗어나면 잘라낸다."""
        s = max(1, min(int(step), self.steer_steps))
        return self.steer_table[s - 1]

    def steer_bar(self, step):
        """현재 단계를 눈으로 보이게. 예: [· · ▣ · ·]"""
        cells = ["▣" if i == step else "·" for i in range(1, self.steer_steps + 1)]
        return "[" + " ".join(cells) + "]"

    def send_record(self, on):
        self.pub_rec.publish(Bool(data=bool(on)))

    def send_motor(self, angle_out, speed_out):
        m = Float32MultiArray()
        m.data = [float(angle_out), float(speed_out)]
        self.pub.publish(m)

    def send_teleop(self, angle, speed, estop):
        m = Float32MultiArray()
        m.data = drive_cmd.encode_teleop_cmd(
            {"angle": angle, "speed": speed, "estop": 1.0 if estop else 0.0})
        self.pub.publish(m)


def _help_text(node):
    mode = "직발행(/xycar_motor)" if node.direct else "mux 경유(/teleop_cmd)"
    return f"""
╔══════════════ XYCAR 키보드 주행 [{mode}] ══════════════╗
  w / s   : 속도 +/-  (스텝 {node.speed_step:.0f})
  a / d   : 조향 한 칸 좌/우   ({node.steer_steps}단계, {node.steer_mid}번이 직진)
            {"  ".join(f"{i}:{node.steer_of(i):+.0f}" for i in range(1, node.steer_steps + 1))}
  A / D   : 조향 좌/우 최대로 즉시
            기계 변환(트림 {node.car.steer_trim:+.0f}, 스톱 {node.car.steer_limit_left:+.0f}~{node.car.steer_limit_right:+.0f})은
            CarInterface{"" if node.direct else "(motion측)"}가 담당
  space   : 즉시 정지 (estop)   c : 조향 중립({node.steer_mid}번)
  x       : 전부 0 (estop 해제)
  k /엔터 : 유지 (워치독 갱신, 값 변화 없음)
  1~5     : 속도 프리셋 10/20/30/40/50
  r       : 녹화 시작/정지 (recorder 노드가 따로 떠 있어야 한다)
  q       : 종료 (정지 후 종료)
  ※ {"자동 감속 없음 — 속도를 준 채로 두면 계속 달린다 (정지: space / x / q)"
     if node.watchdog_sec <= 0 else f"{node.watchdog_sec:.0f}초간 키 없으면 자동 감속"}
╚═══════════════════════════════════════════════════════╝
"""


def main(args=None):
    rclpy.init(args=args)
    argv = remove_ros_args(sys.argv if args is None else args)
    direct = "--direct" in argv
    node = TeleopNode(direct)

    if not sys.stdin.isatty():
        node.get_logger().error("stdin이 TTY가 아님 — 터미널에서 직접 실행할 것")
        node.destroy_node()
        rclpy.shutdown()
        return

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)

    target_speed = 0.0             # 목표 속도 (명령단위, 키로 조절)
    steer_step = node.steer_mid    # 조향 단계 (1=좌최대 … 중앙=직진 … n=우최대)
    # A/D 는 칸 테이블을 넘어 기계 스톱까지 간다 (대칭 모드에서 우회전 성능을
    # 잃지 않게 하는 탈출구). -1=좌 스톱, 0=평소, +1=우 스톱.
    hard_lock = 0
    estop = False        # mux 모드에서 /teleop_cmd [2]로 발행
    recording = False    # r 키 토글 상태 (실제 저장은 recorder_node 가 한다)
    last_key_t = time.time()
    running = True

    print(_help_text(node))
    print("시작합니다. (안전 확보 확인!)\n")

    try:
        tty.setcbreak(fd)   # Ctrl+C 살아있는 모드
        dt = 1.0 / node.rate_hz

        while running and rclpy.ok():
            loop_start = time.time()

            # ── 키 입력 (논블로킹) ──
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                last_key_t = time.time()

                if ch == 'w':
                    target_speed = min(target_speed + node.speed_step, node.speed_max)
                    estop = False
                elif ch == 's':
                    target_speed = max(target_speed - node.speed_step, node.speed_min)
                    estop = False
                elif ch == 'a':
                    steer_step = max(1, steer_step - 1)              # 한 칸 좌
                    hard_lock = 0
                elif ch == 'd':
                    steer_step = min(node.steer_steps, steer_step + 1)  # 한 칸 우
                    hard_lock = 0
                elif ch == 'A':
                    steer_step = 1                                   # 좌 기계 스톱까지
                    hard_lock = -1
                elif ch == 'D':
                    steer_step = node.steer_steps                    # 우 기계 스톱까지
                    hard_lock = +1
                elif ch == ' ':
                    target_speed = 0.0
                    estop = True
                    node.car.reset()   # 직발행: 램프 무시 즉시 0부터
                elif ch == 'c':
                    steer_step = node.steer_mid   # 직진 (트림은 CarInterface가 적용)
                    hard_lock = 0
                elif ch == 'x':
                    target_speed = 0.0
                    steer_step = node.steer_mid
                    estop = False
                    hard_lock = 0
                    node.car.reset()
                elif ch in ('k', '\n', '\r'):
                    pass               # keepalive: 워치독 갱신만
                elif ch in '12345':
                    target_speed = float(int(ch) * 10)
                    estop = False
                elif ch == 'r':
                    recording = not recording
                    node.send_record(recording)
                elif ch == 'q':
                    running = False

            # ── 워치독: 키가 끊기면 감속 (watchdog_sec=0 이면 비활성) ──
            if node.watchdog_sec > 0.0:
                wd_left = max(0.0, node.watchdog_sec - (time.time() - last_key_t))
                if wd_left <= 0.0 and target_speed != 0.0:
                    target_speed = 0.0
            else:
                wd_left = None   # 표시에서 "off"로 나타낸다

            # ── 발행 ──
            # 평소엔 칸 테이블, A/D 를 눌렀을 때만 기계 스톱까지 (대칭 모드에서
            # 우회전 최대치를 못 쓰게 되는 걸 막는 탈출구)
            if hard_lock < 0:
                steer = node.steer_lo
            elif hard_lock > 0:
                steer = node.steer_hi
            else:
                steer = node.steer_of(steer_step)   # 단계 → 논리 조향값
            if direct:
                # CarInterface가 트림/스톱/데드존/램프 전부 흡수
                angle_out, speed_out = node.car.to_motor(
                    steer, 0.0 if estop else target_speed)
                node.send_motor(angle_out, speed_out)
                shown_out = speed_out
            else:
                # mux 경유 — 기계 변환/램프는 motion측 CarInterface가 담당
                node.send_teleop(steer, target_speed, estop)
                # 표시용 변환만 로컬 계산 (발행값 아님)
                rel = node.car.steering_relative(steer)
                angle_out = node.car.steer_trim + rel
                shown_out = target_speed

            servo = angle_out * SERVO_GAIN + SERVO_OFF
            if estop:
                flag = " ESTOP"
            elif wd_left is not None and wd_left <= 0.0:
                flag = "  WD! "
            else:
                flag = "      "
            rec = " ●REC" if recording else "     "
            wd_txt = "WD off  " if wd_left is None else f"WD {wd_left:3.1f}s"
            sys.stdout.write(
                f"\r 속도{target_speed:6.1f}({node.car.cmd_to_ms(shown_out):5.2f}m/s) "
                f"| 조향 {steer_step}/{node.steer_steps} {node.steer_bar(steer_step)} "
                f"논리{steer:6.1f} out{angle_out:6.1f} "
                f"|{flag}|{rec}| {wd_txt}  ")
            sys.stdout.flush()

            sleep_left = dt - (time.time() - loop_start)
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        pass
    finally:
        # 녹화가 켜져 있으면 먼저 끈다 — 안 끄면 recorder 가 계속 디스크를 채운다
        if recording:
            try:
                node.send_record(False)
                time.sleep(0.1)
            except Exception:      # noqa: BLE001 — 종료 경로에서 정지를 막지 않는다
                pass
        # 반드시 정지 명령을 충분히 보냄 (종료 시 0 발행 유지)
        node.car.reset()
        for _ in range(25):
            if direct:
                node.send_motor(node.car.steer_trim, 0.0)   # 트림 위치 + 정지
            else:
                node.send_teleop(0.0, 0.0, False)           # 0.3s 뒤 mux 자동 비활성
            time.sleep(0.02)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print("\n\n정지 완료. 종료합니다.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
