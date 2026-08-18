#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################################################################
# 프로그램 명 : app_8_drive.py
# 작  성  자 : (주)자이트론
# 본 프로그램은 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
####################################################################

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import rclpy
import time
from rclpy.node import Node
#from xycar_msgs.msg import XycarMotor
from std_msgs.msg import Float32MultiArray

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
class MotorDriver(Node):
    def __init__(self):
        super().__init__('motor_driver')
        
        # 퍼블리셔 초기화
        #self.motor_publisher = self.create_publisher(XycarMotor, 'xycar_motor', 1)
        self.motor_publisher = self.create_publisher(Float32MultiArray, 'xycar_motor', 1)
        
        #self.motor_msg = XycarMotor()
        self.motor_msg = Float32MultiArray()

        # 파라미터 초기화
        self.speed = self.declare_parameter("speed", 12).value

        # 초기 설정 출력
        self.get_logger().info("----- Xycar self driving -----")
        self.max_angle = 100
        self.angle = 0
        self.drive(0, 0)
        time.sleep(3)

        # 메인 루프 실행
        self.main_loop()

    #=============================================
    # 모터 토픽을 발행하는 함수.
    #=============================================
    def drive(self, angle, speed):
        #self.motor_msg.angle = float(angle)
        #self.motor_msg.speed = float(speed)
        self.motor_msg.data = [float(angle), float(speed)] 
        self.motor_publisher.publish(self.motor_msg)

    #=============================================
    # 실질적인 메인 루프 함수
    #=============================================
    def main_loop(self):
        while rclpy.ok():
            for i in range(50):
                self.angle += 5
                self.angle = min(self.angle, self.max_angle)
                self.drive(self.angle, self.speed)
                time.sleep(0.1)

            for i in range(50):
                self.angle -= 5
                self.angle = max(self.angle, -self.max_angle)
                self.drive(self.angle, self.speed)
                time.sleep(0.1)

        # 종료 시 모터를 정지
        self.drive(angle=0, speed=0)

#=============================================
# 메인 함수 - start() 함수를 호출.
#=============================================
def main(args=None):
    rclpy.init(args=args)
    motor_driver_node = MotorDriver()
    rclpy.spin(motor_driver_node)

    # 노드 종료 및 클린업
    motor_driver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
