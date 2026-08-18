"""차량 고유 변환층 — 논리 명령(angle, speed) → /xycar_motor 실제 발행값.

역할: 조향 트림/비대칭 기계 스톱 클램프/slew, 속도 클립/데드존 스냅/램프(slew)를
  한 곳에 모은 순수 파이썬 클래스 (rclpy 금지 — 유닛테스트/rosbag 재생용).
입력: 논리 조향 angle_cmd(예선 ±100 스케일), 논리 속도 speed_cmd(명령단위, ×0.08=m/s).
출력: /xycar_motor Float32MultiArray data=[angle_out, speed_out]에 그대로 넣을 값.
예선 대비 변경점: 신규 파일. 예선은 시뮬 모터라 변환 없이 직접 발행했음 —
  실차 실측값(직진 트림 -20, 기계 스톱 -32~+62, 데드존 4, 센서리스 램프 필수)을
  여기서 전부 흡수한다. 스무딩 상태는 이 클래스 내부에만 존재 (재도입 금지 버그 1 대응
  — 외부에서 _prev 상태를 직접 오염시킬 수 없다).

사용 규칙 (motion_node):
  - 20Hz 틱마다 to_motor() 호출 (slew가 틱 단위라 호출 주기 = 발행 주기 전제).
  - last_angle_out/last_speed_out = '마지막 실제 발행값'. 계획값이 아니라 이 값을
    다음 판단 기준으로 쓴다.
  - 제어권 재장악/estop 직후에는 reset() — 속도 slew가 0(정지 가정)부터 다시 램프.
"""

import math


# ============================================================
# 기본 cfg — 전부 실차 실측 유래 (시뮬 값 아님)
# ============================================================

DEFAULT_CFG = {
    "steer_trim": -20.0,          # 직진 트림 (실측: 명령 -20이 직진)
    "steer_scale": 0.42,          # 논리 조향(±100) → 기계 조향 스케일
    "steer_limit_left": -32.0,    # 트림 기준 상대 좌 스톱 (실측 -34, 마진 포함)
    "steer_limit_right": 62.0,    # 트림 기준 상대 우 스톱 (실측 +64, 마진 포함)
    "slew_angle_per_tick": 8.0,   # 틱당 조향 출력 변화 상한
    "slew_speed_per_tick": 1.5,   # 틱당 속도 출력 변화 상한 (센서리스 기동 램프)
    "speed_deadzone": 4.0,        # 이 미만 명령은 안 구름 (실측)
    "allow_reverse": False,       # False면 속도 [0, 100] 클립
    "cmd_to_ms": 0.08,            # speed 명령 ×0.08 = m/s (실측검증)
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# CarInterface
# ============================================================

class CarInterface:
    """cfg dict로 초기화. 누락 키는 DEFAULT_CFG 값 사용 (키 목록도 DEFAULT_CFG 참조)."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.steer_trim = float(cfg.get("steer_trim", DEFAULT_CFG["steer_trim"]))
        self.steer_scale = float(cfg.get("steer_scale", DEFAULT_CFG["steer_scale"]))
        self.steer_limit_left = float(
            cfg.get("steer_limit_left", DEFAULT_CFG["steer_limit_left"]))
        self.steer_limit_right = float(
            cfg.get("steer_limit_right", DEFAULT_CFG["steer_limit_right"]))
        self.slew_angle_per_tick = float(
            cfg.get("slew_angle_per_tick", DEFAULT_CFG["slew_angle_per_tick"]))
        self.slew_speed_per_tick = float(
            cfg.get("slew_speed_per_tick", DEFAULT_CFG["slew_speed_per_tick"]))
        self.speed_deadzone = float(
            cfg.get("speed_deadzone", DEFAULT_CFG["speed_deadzone"]))
        self.allow_reverse = bool(cfg.get("allow_reverse", DEFAULT_CFG["allow_reverse"]))
        self._cmd_to_ms = float(cfg.get("cmd_to_ms", DEFAULT_CFG["cmd_to_ms"]))
        if self._cmd_to_ms <= 0.0:
            raise ValueError("cmd_to_ms는 0보다 커야 한다: %r" % self._cmd_to_ms)
        self.reset()

    # --------------------------------------------------------
    # slew 상태
    # --------------------------------------------------------

    def reset(self):
        """slew 상태 초기화 — 제어권 재장악/estop 후 호출.

        조향: 서보 현재 위치를 모르므로 다음 첫 틱은 slew 미적용 (기준 없음).
        속도: 정지 가정 — 0부터 다시 램프 (센서리스라 정지→고속 직행 불가).
        """
        self._last_angle_out = None
        self._last_speed_out = 0.0

    @property
    def last_angle_out(self):
        """마지막 실제 발행 조향값. reset 직후(첫 발행 전)에는 None."""
        return self._last_angle_out

    @property
    def last_speed_out(self):
        """마지막 실제 발행 속도값. reset 직후에는 0.0 (정지 가정)."""
        return self._last_speed_out

    # --------------------------------------------------------
    # 변환 본체
    # --------------------------------------------------------

    def to_motor(self, angle_cmd, speed_cmd):
        """논리 명령 → (angle_out, speed_out). 매 20Hz 틱 호출 전제.

        조향: rel = clamp(angle_cmd×scale, left, right); out = trim + rel;
              이전 출력 대비 slew 제한.
        속도: allow_reverse 아니면 [0,100] 클립 → 데드존 스냅 → slew 제한(램프).
        """
        # 비정상 입력이 slew 상태를 오염시키지 않도록 안전값으로 치환
        angle_cmd = float(angle_cmd)
        speed_cmd = float(speed_cmd)
        if not math.isfinite(angle_cmd):
            angle_cmd = 0.0   # 직진
        if not math.isfinite(speed_cmd):
            speed_cmd = 0.0   # 정지

        # --- 조향: 스케일 → 기계 스톱 클램프(트림 기준 상대) → 트림 → slew ---
        rel = _clamp(angle_cmd * self.steer_scale,
                     self.steer_limit_left, self.steer_limit_right)
        angle_out = self.steer_trim + rel
        if self._last_angle_out is not None:
            angle_out = _clamp(angle_out,
                               self._last_angle_out - self.slew_angle_per_tick,
                               self._last_angle_out + self.slew_angle_per_tick)

        # --- 속도: 클립 → 데드존 스냅 → slew(램프) ---
        if self.allow_reverse:
            v = _clamp(speed_cmd, -100.0, 100.0)
        else:
            v = _clamp(speed_cmd, 0.0, 100.0)
        v = self._snap_deadzone(v)
        speed_out = _clamp(v,
                           self._last_speed_out - self.slew_speed_per_tick,
                           self._last_speed_out + self.slew_speed_per_tick)

        self._last_angle_out = float(angle_out)
        self._last_speed_out = float(speed_out)
        return self._last_angle_out, self._last_speed_out

    def _snap_deadzone(self, v):
        """0 < |v| < deadzone 인 안 구르는 명령을 스냅.

        |v| >= deadzone/2 → deadzone(구르는 최소값), 아니면 0(정지 의도로 간주).
        allow_reverse 시 후진도 크기 기준 대칭 적용. (재도입 금지 버그 7 —
        데드존 미만 명령이 그대로 나가 정지·기어감 상태가 되는 것 방지)
        """
        mag = abs(v)
        if mag <= 0.0 or mag >= self.speed_deadzone:
            return v
        if mag >= self.speed_deadzone * 0.5:
            return math.copysign(self.speed_deadzone, v)
        return 0.0

    # --------------------------------------------------------
    # 단위 변환 헬퍼
    # --------------------------------------------------------

    def cmd_to_ms(self, cmd):
        """속도 명령단위 → m/s (실측: ×0.08)."""
        return float(cmd) * self._cmd_to_ms

    def ms_to_cmd(self, ms):
        """m/s → 속도 명령단위."""
        return float(ms) / self._cmd_to_ms
