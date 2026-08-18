#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import math
from xycar_msgs.msg import XycarMotor
from std_msgs.msg import Int32MultiArray

motor_pub = None
motor_msg = XycarMotor()
ultrasonicData = None

def callback(msg): 
    global ultrasonicData
    ultrasonicData = msg.data  

rclpy.init()
node = Node('drive')
motor_pub = node.create_publisher(XycarMotor, 'xycar_motor', 1)
subscription = node.create_subscription(
          Int32MultiArray,
          'ultrasonic',
          callback,
          1)

while rclpy.ok():
    rclpy.spin_once(node)
    while ultrasonicData is None:
        continue

    R = ultrasonicData[3] 
    L = ultrasonicData[1] 
    
    print(R,L)
    Q = R - L

    angle = 0
    if Q > 0 and abs(Q) > 0.1:           
        angle = Q
    elif Q < 0 and abs(Q) > 0.1:
        angle = Q

    motor_msg.angle = float(angle)
    motor_msg.speed = 10.0
    
    motor_pub.publish(motor_msg)

