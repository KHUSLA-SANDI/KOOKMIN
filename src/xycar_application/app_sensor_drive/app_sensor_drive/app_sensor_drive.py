#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
#from xycar_msgs.msg import XycarMotor
from std_msgs.msg import Float32MultiArray
import math

class LidarDriverNode(Node):
    def __init__(self):
        super().__init__('lidar_driver')

        # QoS 설정 (LiDAR는 Best Effort 사용)
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # LiDAR 데이터 구독
        self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            qos_profile
        )
        
        # 모터 퍼블리셔 생성
        #self.motor_publisher = self.create_publisher(XycarMotor, 'xycar_motor', 1)
        self.motor_publisher = self.create_publisher(Float32MultiArray, 'xycar_motor', 1)
        
        # 모터 메시지 초기화
        #self.motor_msg = XycarMotor()
        self.motor_msg = Float32MultiArray()
        
        # 노드 시작 로그
        self.get_logger().info("Lidar Ready ----------")

        # 초기 모터 정지
        self.drive(0, 0)

        # 초기 대기 (2초)
        time.sleep(2)

        # 라이다 데이터 저장 변수
        self.lidar_ranges_cm = None

        # 0.1초마다 실행되는 타이머 설정
        self.timer = self.create_timer(0.1, self.timer_callback)

    def lidar_callback(self, msg):
        """LiDAR 데이터를 받아서 센티미터 단위로 변환하여 저장"""
        self.lidar_ranges_cm = [int(distance * 100) if distance > 0 and distance < float('inf') else 9999 for distance in msg.ranges[1:505]]

    def drive(self, angle, speed):
        """모터 메시지를 생성하여 퍼블리시"""
        #self.motor_msg.angle = float(angle)
        #self.motor_msg.speed = float(speed)
        self.motor_msg.data = [float(angle), float(speed)] 
        self.motor_publisher.publish(self.motor_msg)

    
    def lidar_drive(self):
        """LiDAR 데이터를 기반으로 조향각 결정"""
        if not self.lidar_ranges_cm:
            return 0  # 데이터가 없으면 직진

        left_distance = self.lidar_ranges_cm[252+63]  # 왼쪽 거리
        right_distance = self.lidar_ranges_cm[252-63]  # 오른쪽 거리
        
        self.get_logger().info(f"\n Left: {left_distance} Right: {right_distance}")

        # 유효한 거리값인지 체크 (9999는 유효하지 않은 값)
        if left_distance == 9999 or right_distance == 9999:
            return 0  # 거리값이 잘못되었으면 직진

        # 왼쪽이 더 가까우면 오른쪽으로 조향
        if left_distance < right_distance:
            angle = 50  
        elif left_distance > right_distance:
            angle = -50  
        else:
            angle = 0  

        return angle

    def timer_callback(self):
        """0.1초마다 실행되는 주행 로직"""
        if self.lidar_ranges_cm:
            angle = self.lidar_drive()  # 라이다 데이터 기반으로 조향각 결정
            self.drive(angle=angle, speed=10)  # 속도 12로 주행

def main(args=None):
    rclpy.init(args=args)
    node = LidarDriverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

