#!/usr/bin/env python3
"""Check Xycar steering endpoints without driving the traction motor."""

import math
import time

from my_motor.steering_center import read_saved_offset
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class SteeringLimitChecker(Node):
    """Publish steering commands while always keeping the speed at zero."""

    def __init__(self):
        super().__init__('steering_limit_checker')
        self.publisher = self.create_publisher(
            Float32MultiArray, 'xycar_motor', 1)

        # Values currently used by xycar_motor.py and vesc.yaml.
        self.input_to_angle = self.declare_parameter(
            'input_to_angle', 0.0068).value
        self.servo_gain = self.declare_parameter(
            'servo_gain', -1.2135).value
        self.servo_offset = self.declare_parameter(
            'servo_offset', 0.5004).value
        self.servo_min = self.declare_parameter('servo_min', 0.0).value
        self.servo_max = self.declare_parameter('servo_max', 1.0).value

        self.current_input = 0.0
        self.has_published = False

    def input_for_servo(self, servo_value):
        """Convert a VESC servo value to the upstream xycar input value."""
        # xycar_motor negates the Ackermann angle before this VESC mapping:
        # servo = gain * (-input * input_to_angle) + offset
        denominator = -self.servo_gain * self.input_to_angle
        return (servo_value - self.servo_offset) / denominator

    def commanded_angle_deg(self, input_value):
        """Return the Ackermann command angle, not a measured tyre angle."""
        return math.degrees(-input_value * self.input_to_angle)

    def publish_zero_speed(self, steering_input, seconds=0.6):
        """Hold one steering command; the second array item is always zero."""
        message = Float32MultiArray()
        message.data = [float(steering_input), 0.0]
        end_time = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end_time:
            self.has_published = True
            self.publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.03)
        self.current_input = float(steering_input)

    def move_slowly(self, target, increment=1.0):
        """Ramp steering to avoid an abrupt servo movement."""
        value = self.current_input
        direction = 1.0 if target > value else -1.0
        while rclpy.ok() and abs(target - value) > increment:
            value += direction * increment
            self.publish_zero_speed(value, seconds=0.04)
        self.publish_zero_speed(target)

    def print_report(self, left_input, right_input):
        print('\n현재 설정으로 계산한 조향 서보 한계')
        print('  왼쪽 끝: 입력 {:+.2f}, 명령각 {:+.2f} deg'.format(
            left_input, self.commanded_angle_deg(left_input)))
        print('  오른쪽 끝: 입력 {:+.2f}, 명령각 {:+.2f} deg'.format(
            right_input, self.commanded_angle_deg(right_input)))
        print('\n주의: 명령각은 소프트웨어상의 Ackermann 각도입니다.')
        print('실제 좌/우 타이어 각도는 위치 센서가 없어 각도기로 측정해야 합니다.')
        print('모든 명령은 [조향 입력, 0.0]으로 발행되어 구동 속도는 항상 0입니다.\n')

    def interactive(self):
        endpoint_a = self.input_for_servo(self.servo_min)
        endpoint_b = self.input_for_servo(self.servo_max)
        saved_offset = read_saved_offset()

        # Endpoints above are raw motor inputs. Applications publish logical
        # inputs, and xycar_motor adds the persistent centre offset.
        left_input = min(endpoint_a, endpoint_b) - saved_offset
        right_input = max(endpoint_a, endpoint_b) - saved_offset
        print('저장된 영점 오프셋: {:+.3f}'.format(saved_offset))
        self.print_report(left_input, right_input)

        # The dynamic ROS 1 bridge is created only after this publisher is
        # discovered. Wait for it instead of publishing commands into a topic
        # with no subscriber, which would make the program appear to work
        # while the physical servo receives nothing.
        bridge_deadline = time.monotonic() + 5.0
        while (rclpy.ok() and
               self.count_subscribers('/xycar_motor') == 0 and
               time.monotonic() < bridge_deadline):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.count_subscribers('/xycar_motor') == 0:
            print('안전 중지: /xycar_motor 구독자가 없습니다.')
            print('motor_up.sh로 ROS 1 브리지를 시작한 뒤 다시 실행하세요.')
            return

        # A second driving node could overwrite our zero-speed command.
        publisher_count = len(self.get_publishers_info_by_topic(
            '/xycar_motor'))
        if publisher_count > 1:
            print('안전 중지: /xycar_motor 발행 노드가 이 프로그램 외에 '
                  '{}개 더 있습니다.'.format(publisher_count - 1))
            print('기존 주행 노드를 종료한 뒤 다시 실행하세요.')
            return

        answer = input(
            '바퀴 주변을 비우고 차를 바닥에 고정했으면 Enter, '
            '계산 결과만 볼 경우 q: ').strip().lower()
        if answer == 'q':
            return

        self.publish_zero_speed(0.0, seconds=1.0)
        print('\n명령: l=왼쪽 끝, r=오른쪽 끝, c=중립, q=중립 복귀 후 종료')
        while rclpy.ok():
            command = input('steering> ').strip().lower()
            if command == 'l':
                self.move_slowly(left_input)
                print('왼쪽 끝: 입력 {:+.2f}, 명령각 {:+.2f} deg'.format(
                    left_input, self.commanded_angle_deg(left_input)))
            elif command == 'r':
                self.move_slowly(right_input)
                print('오른쪽 끝: 입력 {:+.2f}, 명령각 {:+.2f} deg'.format(
                    right_input, self.commanded_angle_deg(right_input)))
            elif command == 'c':
                self.move_slowly(0.0)
                print('중립: 입력 0.00')
            elif command == 'q':
                return
            else:
                print('l, r, c, q 중 하나를 입력하세요.')


def main(args=None):
    rclpy.init(args=args)
    node = SteeringLimitChecker()
    try:
        node.interactive()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        # Always request centre and zero speed on normal exit/Ctrl-C.
        try:
            if node.has_published:
                node.move_slowly(0.0)
                node.publish_zero_speed(0.0, seconds=0.5)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
