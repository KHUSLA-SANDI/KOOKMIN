#!/usr/bin/env python3

import rclpy, time
from rclpy.node import Node
#from xycar_msgs.msg import XycarMotor
from std_msgs.msg import Float32MultiArray

class DriverNode(Node):
    def __init__(self):
        super().__init__('driver')
        #self.motor_publisher = self.create_publisher(XycarMotor, 'xycar_motor', 1)
        self.motor_publisher = self.create_publisher(Float32MultiArray, 'xycar_motor', 1)
        
        #self.motor_msg = XycarMotor()
        self.motor_msg = Float32MultiArray()

        # 파라미터 초기화
        self.speed = self.declare_parameter("speed", 50).value
        self.angle = 0
        self.delta = 10
        self.get_logger().info('----- Xycar self-driving node started -----')

    def drive(self, angle, speed):
        #self.motor_msg.angle = float(angle)
        #self.motor_msg.speed = float(speed)
        self.motor_msg.data = [float(angle), float(speed)] 
        self.motor_publisher.publish(self.motor_msg)

    def main_loop(self):
    
        for i in range(20):
            self.angle = 0
            self.speed = 0
            self.drive(self.angle, self.speed)
            time.sleep(0.1)

        while rclpy.ok():
        
            '''
            if (self.speed >= 100):
                self.delta = -10
            elif (self.speed <= -100):
                self.delta = +10
            self.speed = self.speed + self.delta
  
            for i in range(20):
                self.angle = 0                
                self.drive(self.angle, self.speed)
                time.sleep(0.1)
            '''
          
            self.angle = 0
            self.speed = 20
            self.drive(self.angle, self.speed)
            time.sleep(0.1)
         
          
            rclpy.spin_once(self, timeout_sec=0.1)
            
def main(args=None):
    rclpy.init(args=args)
    driver_node = DriverNode()

    try:
        driver_node.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        driver_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
