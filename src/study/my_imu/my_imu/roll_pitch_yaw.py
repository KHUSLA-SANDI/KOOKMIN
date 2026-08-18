#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf_transformations import euler_from_quaternion
import time

class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_print')

        # '/imu/data' Topic
        self.subscription = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10  
        )

        self.imu_msg = None        
        
        self.get_logger().info("Waiting for IMU data...")
        #self.wait_for_message()
        
        self.get_logger().info("IMU Ready ----------")
        
        self.timer = self.create_timer(1.0, self.timer_callback)  

    def imu_callback(self, msg):
        
        self.imu_msg = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

    def wait_for_message(self):
        
        while self.imu_msg is None and rclpy.ok():
            rclpy.spin_once(self)  
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))  

    def timer_callback(self):
        
        if self.imu_msg is not None:
            (roll, pitch, yaw) = euler_from_quaternion(self.imu_msg)
            self.get_logger().info(f'Roll: {roll:.4f}, Pitch: {pitch:.4f}, Yaw: {yaw:.4f}')

def main(args=None):
    
    rclpy.init(args=args)
    
    
    imu_node = ImuNode()
    
    try:
        
        rclpy.spin(imu_node)
    except KeyboardInterrupt:
        
        pass
    finally:
        
        imu_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

