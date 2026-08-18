#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

class UltraNode(Node):
    def __init__(self):
        # 'ultra_node'라는 이름으로 ROS2 노드를 초기화합니다.
        super().__init__('ultra_node')

        # 'xycar_ultrasonic' 토픽을 구독하기 위한 구독자를 생성합니다.
        # 구독자는 Int32MultiArray 메시지를 받고, 초음파 데이터 콜백 함수를 호출합니다.
        self.subscription = self.create_subscription(
            Int32MultiArray,
            'xycar_ultrasonic',
            self.ultra_callback,
            10  # 큐 크기 설정
        )

        # 초음파 센서 데이터 저장을 위한 변수 초기화
        self.ultra_msg = None
        
        # 초기 메시지가 수신될 때까지 대기하는 로그 출력
        self.get_logger().info("Waiting for ultrasonic data...")
        self.wait_for_message()

        # 초기 메시지 수신 완료 후 로그 출력
        self.get_logger().info("Ultrasonic Ready ----------")

        # 주기적인 작업을 수행하기 위한 타이머를 설정합니다.
        # 타이머는 1초마다 timer_callback 함수를 호출합니다.
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1초마다 콜백 호출

    def ultra_callback(self, msg):
        # 초음파 센서로부터 수신한 메시지를 처리하는 콜백 함수입니다.
        # 메시지의 데이터를 self.ultra_msg 변수에 저장합니다.
        self.ultra_msg = msg.data

    def wait_for_message(self):
        # 초기에 메시지가 수신될 때까지 대기하는 함수입니다.
        # 초음파 데이터가 수신될 때까지 spin_once와 sleep_for를 사용하여 대기합니다.
        while self.ultra_msg is None and rclpy.ok():
            rclpy.spin_once(self)  # 노드의 상태를 업데이트
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))  # 짧은 대기

    def timer_callback(self):
        # 타이머에 의해 주기적으로 호출되는 콜백 함수입니다.
        # 수신된 초음파 센서 데이터를 로그에 출력합니다.
        if self.ultra_msg is not None:
            self.get_logger().info(f"Ultrasonic Data: {self.ultra_msg}")

def main(args=None):
    # ROS2를 초기화합니다.
    rclpy.init(args=args)
    
    # UltraNode의 인스턴스를 생성합니다.
    ultra_node = UltraNode()
    
    try:
        # UltraNode를 실행 상태로 유지하며 콜백을 처리합니다.
        rclpy.spin(ultra_node)
    except KeyboardInterrupt:
        # 사용자 인터럽트 (Ctrl+C)가 발생하면 예외를 처리합니다.
        pass
    finally:
        # 노드를 종료하고 ROS2를 정리합니다.
        ultra_node.destroy_node()
        rclpy.shutdown()

# 스크립트가 직접 실행될 때 main() 함수를 호출합니다.
if __name__ == '__main__':
    main()
