#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')

        # QoS 설정 (LiDAR는 Best Effort 사용)
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # LiDAR 데이터 구독
        self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile
        )

        # 초기화
        self.ranges = None

        # Matplotlib 설정
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-150, 150)
        self.ax.set_aspect('equal')
        self.lidar_points, = self.ax.plot([], [], 'bo')

        plt.ion()  # 인터랙티브 모드 활성화
        plt.show()

        self.get_logger().info("Lidar Visualizer Ready ----------")

        # 0.1초마다 실행되는 타이머 설정
        self.create_timer(0.1, self.timer_callback)

    def lidar_callback(self, msg):
        """LiDAR 데이터를 저장하는 콜백 함수"""
        self.ranges = np.array(msg.ranges[1:505])

    def timer_callback(self):
        """LiDAR 데이터를 주기적으로 업데이트하는 함수"""
        if self.ranges is not None:
            # 각도 계산 (0 ~ 2π로 변환, -90도 보정)
            angles = np.linspace(0, 2 * np.pi, len(self.ranges)) - np.pi / 2

            # 극좌표 -> 직교좌표 변환 (cm 단위)
            x = self.ranges * np.cos(angles) * 100
            y = self.ranges * np.sin(angles) * 100

            # 그래프 업데이트
            self.lidar_points.set_data(x, y)
            self.fig.canvas.draw_idle()
            plt.pause(0.01)

            # 샘플 데이터 출력 (36개 샘플)
            ranges_cm = [int(distance * 100) for distance in self.ranges]
            step = len(ranges_cm) // 36
            self.get_logger().info(f"Selected distance values (cm): {ranges_cm[::step]}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

