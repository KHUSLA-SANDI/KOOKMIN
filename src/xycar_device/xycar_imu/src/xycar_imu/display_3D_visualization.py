import sys

import rclpy
from rclpy.node import Node
from vpython import *
import math
import wx

from sensor_msgs.msg import Imu
from transforms3d.euler import quat2euler as euler_from_quaternion

rad2degrees = 180.0 / math.pi
precision = 2  # round to this number of digits


class Display3DNode(Node):
    def __init__(self):
        super().__init__('display_3D_visualization_node')

        sub = self.create_subscription(Imu, 'imu', self.process_imu_message, 1)
        sub  # prevent unused variable warning

        self.yaw_offset = 0  # used to align animation upon key press

        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # Main scene
        self.scene = canvas(title="9DOF Razor IMU Main Screen")
        self.scene.range = 1.3
        self.scene.forward = vector(1, 0, -0.25)
        # Change reference axis (x,y,z) - z is up
        self.scene.up = vector(0, 0, 1)
        self.scene.width = 500
        self.scene.height = 500

        # Second scene (Roll, Pitch, Yaw)
        scene2 = canvas(title='9DOF Razor IMU Roll, Pitch, Yaw', x=550, y=0, width=500, height=500,
                        center=vector(0, 0, 0), background=vector(0, 0, 0))
        scene2.range = 1
        scene2.select()
        # Roll, Pitch, Yaw
        # Default reference, i.e. x runs right, y runs up, z runs backward (out of screen)
        self.cil_roll = cylinder(pos=vector(-0.5, 0.3, 0), axis=vector(0.2, 0, 0), radius=0.01,
                                 color=color.red)
        self.cil_roll2 = cylinder(pos=vector(-0.5, 0.3, 0), axis=vector(-0.2, 0, 0), radius=0.01,
                                  color=color.red)
        self.cil_pitch = arrow(pos=vector(0.5, 0.3, 0), axis=vector(0, 0, -0.4), shaftwidth=0.02,
                               color=color.green)
        self.arrow_course = arrow(pos=vector(0.0, -0.4, 0), color=color.cyan,
                                  axis=vector(0, 0.2, 0),
                                  shaftwidth=0.02,
                                  fixedwidth=1)

        # Roll,Pitch,Yaw labels
        label(pos=vector(-0.5, 0.6, 0), text="Roll (degrees / radians)", box=0, opacity=0)
        label(pos=vector(0.5, 0.6, 0), text="Pitch (degrees / radians)", box=0, opacity=0)
        label(pos=vector(0.0, 0.02, 0), text="Yaw (degrees / radians)", box=0, opacity=0)
        label(pos=vector(0.0, -0.16, 0), text="N", box=0, opacity=0, color=color.yellow)
        label(pos=vector(0.0, -0.64, 0), text="S", box=0, opacity=0, color=color.yellow)
        label(pos=vector(-0.24, -0.4, 0), text="W", box=0, opacity=0, color=color.yellow)
        label(pos=vector(0.24, -0.4, 0), text="E", box=0, opacity=0, color=color.yellow)
        label(pos=vector(0.18, -0.22, 0), height=7, text="NE", box=0, color=color.yellow)
        label(pos=vector(-0.18, -0.22, 0), height=7, text="NW", box=0, color=color.yellow)
        label(pos=vector(0.18, -0.58, 0), height=7, text="SE", box=0, color=color.yellow)
        label(pos=vector(-0.18, -0.58, 0), height=7, text="SW", box=0, color=color.yellow)

        self.roll_label = label(pos=vector(-0.5, 0.52, 0), text="-", box=0, opacity=0, height=12)
        self.pitch_label = label(pos=vector(0.5, 0.52, 0), text="-", box=0, opacity=0, height=12)
        self.yaw_label = label(pos=vector(0, -0.06, 0), text="-", box=0, opacity=0, height=12)

        # acceleration labels
        label(pos=vector(0, 0.9, 0), text="Linear Acceleration x / y / z (m/s^2)", box=0, opacity=0)
        label(pos=vector(0, -0.8, 0), text="Angular Velocity x / y / z (rad/s)", box=0, opacity=0)
        self.lin_acc_label = label(pos=vector(0, 0.82, 0), text="-", box=0, opacity=0, height=12)
        self.ang_vel_label = label(pos=vector(0, -0.88, 0), text="-", box=0, opacity=0, height=12)

        # Main scene objects
        scene.select()
        # Reference axis (x,y,z) - using ROS conventions (REP 103) - z is up, y left (west, 90 deg), x is forward (north, 0 deg)
        # In visual, z runs up, x runs forward, y runs left (see scene.up command earlier)
        # So everything is good
        arrow(color=color.green, axis=vector(1, 0, 0), shaftwidth=0.04, fixedwidth=1)
        arrow(color=color.green, axis=vector(0, 1, 0), shaftwidth=0.04, fixedwidth=1)
        arrow(color=color.green, axis=vector(0, 0, 1), shaftwidth=0.04, fixedwidth=1)

        # labels
        label(pos=vector(0, 0, -1.2), text="Press 'a' to align", box=0, opacity=0)
        label(pos=vector(1, 0.1, 0), text="X", box=0, opacity=0)
        label(pos=vector(0, 1, -0.1), text="Y", box=0, opacity=0)
        label(pos=vector(0, -0.1, 1), text="Z", box=0, opacity=0)
        # IMU object
        self.platform = box(length=1.0, height=0.05, width=0.65, color=color.red,
                            up=vector(0, 0, 1),
                            axis=vector(-1, 0, 0))
        self.p_line = box(length=1.1, height=0.08, width=0.1, color=color.yellow,
                          up=vector(0, 0, 1),
                          axis=vector(-1, 0, 0))
        self.plat_arrow = arrow(length=-0.8, color=color.cyan, up=vector(0, 0, 1),
                                axis=vector(-1, 0, 0),
                                shaftwidth=0.06,
                                fixedwidth=1)
        self.plat_arrow_up = arrow(length=0.4, color=color.cyan, up=vector(-1, 0, 0),
                                   axis=vector(0, 0, 1),
                                   shaftwidth=0.06,
                                   fixedwidth=1)

        # check for align key press - if pressed, next refresh will be aligned
        scene.bind('keydown', self.align)

    def align(self, ev):
        if ev.key == 'a':
            # align key pressed - align
            self.yaw_offset += -self.yaw

    def process_imu_message(self, imu_msg):
        quaternion = (
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w)
        self.roll, self.pitch, self.yaw = euler_from_quaternion(quaternion)

        # add align offset to yaw
        self.yaw += self.yaw_offset

        axis = vector(-cos(self.pitch) * cos(self.yaw), -cos(self.pitch) * sin(self.yaw),
                      sin(self.pitch))
        up = vector(
            sin(self.roll) * sin(self.yaw) + cos(self.roll) * sin(self.pitch) * cos(self.yaw),
            -sin(self.roll) * cos(self.yaw) + cos(self.roll) * sin(self.pitch) * sin(self.yaw),
            cos(self.roll) * cos(self.pitch))
        self.platform.axis = axis
        self.platform.up = up
        self.platform.length = 1.0
        self.plat_arrow_up.axis = up
        self.plat_arrow_up.up = axis
        self.plat_arrow_up.length = 0.4
        self.plat_arrow.axis = axis
        self.plat_arrow.up = up
        self.plat_arrow.length = -0.8
        self.p_line.axis = axis
        self.p_line.up = up
        self.p_line.length = 1.1
        self.cil_roll.axis = vector(-0.2 * cos(self.roll), 0.2 * sin(self.roll), 0)
        self.cil_roll2.axis = vector(0.2 * cos(self.roll), -0.2 * sin(self.roll), 0)
        self.cil_pitch.axis = vector(0, -0.4 * sin(self.pitch), -0.4 * cos(self.pitch))
        # remove yaw_offset from yaw display
        self.arrow_course.axis = vector(-0.2 * sin(self.yaw - self.yaw_offset),
                                        0.2 * cos(self.yaw - self.yaw_offset), 0)

        # display in degrees / radians
        self.roll_label.text = str(round(self.roll * rad2degrees, precision)) + " / " + str(
            round(self.roll, precision))
        self.pitch_label.text = str(round(self.pitch * rad2degrees, precision)) + " / " + str(
            round(self.pitch, precision))
        # remove yaw_offset from yaw display
        self.yaw_label.text = str(
            round((self.yaw - self.yaw_offset) * rad2degrees, precision)) + " / " + str(
            round((self.yaw - self.yaw_offset), precision))

        self.lin_acc_label.text = str(
            round(imu_msg.linear_acceleration.x, precision)) + " / " + str(
            round(imu_msg.linear_acceleration.y, precision)) + " / " + str(
            round(imu_msg.linear_acceleration.z, precision))
        self.ang_vel_label.text = str(round(imu_msg.angular_velocity.x, precision)) + " / " + str(
            round(imu_msg.angular_velocity.y, precision)) + " / " + str(
            round(imu_msg.angular_velocity.z, precision))

    # Create shutdown hook to kill visual displays
    def shutdown_hook(self):
        print("Killing displays")
        wx.Exit()


def main(args=None):
    rclpy.init(args=args)

    node = Display3DNode()
    # Might be affected by this bug https://github.com/ros2/rclpy/issues/532
    # rclpy.get_default_context().on_shutdown(node.shutdown_hook)

    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
