#!/usr/bin/env python

import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        # QoS 설정: LiDAR 데이터는 Best Effort로 설정하는 것이 일반적
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # LiDAR 데이터 구독
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            qos_profile
        )

        self.lidar_ranges = None
        
        self.get_logger().info("Waiting for lidar data...")
        self.wait_for_message()
        self.get_logger().info("Lidar Ready ----------")

        # 주기적으로 데이터 출력 (1.0초마다)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def lidar_callback(self, msg):
        """LiDAR 데이터를 저장하는 콜백 함수"""
        self.lidar_ranges = msg.ranges[1:505]

    def wait_for_message(self):
        """LiDAR 데이터가 올 때까지 대기"""
        while self.lidar_ranges is None and rclpy.ok():
            rclpy.spin_once(self)
            time.sleep(0.1)  # 0.1초 대기

    def timer_callback(self):
        """1.0초마다 LiDAR 데이터를 출력"""
        if self.lidar_ranges:
            self.get_logger().info(f"# of data : {len(self.lidar_ranges)}")
            ranges_cm = [int(distance * 100) for distance in self.lidar_ranges]
            step = max(1, len(ranges_cm) // 18)
            self.get_logger().info(f"\nDistances(cm): {ranges_cm[::step]}")

def main(args=None):
    rclpy.init(args=args)
    lidar_node = LidarNode()

    try:
        rclpy.spin(lidar_node)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

