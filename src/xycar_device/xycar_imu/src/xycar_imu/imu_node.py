import rclpy
from rclpy.node import Node
import serial
import math
import sys
from time import sleep
from xycar_imu.lib.serial_commands import *
import yaml
import time

# from time import time
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from transforms3d.euler import euler2quat as quaternion_from_euler
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

degrees2rad = math.pi / 180.0


class RazorImuDriver(Node):

    def __init__(self):

        super().__init__('imu_node')

        pub_imu = self.create_publisher(
            msg_type=Imu,
            topic='imu',
            # we only care about the most recent measurement, i.e. queue_size=1
            qos_profile=1,
        )

        diag_pub = self.create_publisher(
            msg_type=DiagnosticArray,
            topic='diagnostics',
            qos_profile=1
        )
        diag_pub_time = self.get_clock().now()

        imu_msg = Imu()

        # TODO arrays not supported as parameter type ROS2
        imu_msg.orientation_covariance = [0.0025, 0.0, 0.0,
                                          0.0, 0.0025, 0.0,
                                          0.0, 0.0, 0.0025]
        # self.declare_parameter('orientation_covariance').value
        imu_msg.angular_velocity_covariance = [0.002, 0.0, 0.0,
                                               0.0, 0.002, 0.0,
                                               0.0, 0.0, 0.002]
        # self.declare_parameter('velocity_covariance').value
        imu_msg.linear_acceleration_covariance = [0.04, 0.0, 0.0,
                                                  0.0, 0.04, 0.0,
                                                  0.0, 0.0, 0.04]
        # self.declare_parameter('acceleration_covariance').value
        imu_msg.header.frame_id = self.declare_parameter('frame_header', 'base_imu_link').value

        publish_magnetometer = self.declare_parameter('publish_magnetometer', False).value

        if publish_magnetometer:
            pub_mag = self.create_publisher(MagneticField, 'mag', 1)
            mag_msg = MagneticField()
            mag_msg.magnetic_field_covariance = [0.00, 0.0, 0.0,
                                                 0.0, 0.00, 0.0,
                                                 0.0, 0.0, 0.00]
            # self.declare_parameter('magnetic_field_covariance').value
            mag_msg.header.frame_id = self.get_parameter_or('frame_header', 'base_imu_link').value
            # should a separate diagnostic for the Magnetometer be done?

        port = self.declare_parameter('port', '/dev/ttyIMU').value

        # read calibration parameters
        self.calib_dict = {}

        # accelerometer
        self.calib_dict['accel_x_min'] = self.declare_parameter('accel_x_min', -250.0).value
        self.calib_dict['accel_x_max'] = self.declare_parameter('accel_x_max', 250.0).value
        self.calib_dict['accel_y_min'] = self.declare_parameter('accel_y_min', -250.0).value
        self.calib_dict['accel_y_max'] = self.declare_parameter('accel_y_max', 250.0).value
        self.calib_dict['accel_z_min'] = self.declare_parameter('accel_z_min', -250.0).value
        self.calib_dict['accel_z_max'] = self.declare_parameter('accel_z_max', 250.0).value

        # magnetometer
        self.calib_dict['magn_x_min'] = self.declare_parameter('magn_x_min', -600.0).value
        self.calib_dict['magn_x_max'] = self.declare_parameter('magn_x_max', 600.0).value
        self.calib_dict['magn_y_min'] = self.declare_parameter('magn_y_min', -600.0).value
        self.calib_dict['magn_y_max'] = self.declare_parameter('magn_y_max', 600.0).value
        self.calib_dict['magn_z_min'] = self.declare_parameter('magn_z_min', -600.0).value
        self.calib_dict['magn_z_max'] = self.declare_parameter('magn_z_max', 600.0).value
        self.calib_dict['magn_use_extended'] = self.declare_parameter(
            'calibration_magn_use_extended', False).value
        self.calib_dict['magn_ellipsoid_center'] = self.declare_parameter('magn_ellipsoid_center',
                                                                          [0, 0, 0]).value
        # TODO Array of arrays not supported as parameter type ROS2
        self.calib_dict['magn_ellipsoid_transform'] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # self.declare_parameter('magn_ellipsoid_transform',[[0, 0, 0], [0, 0, 0], [0, 0, 0]]).value

        # gyroscope
        self.calib_dict['gyro_average_offset_x'] = self.declare_parameter('gyro_average_offset_x',
                                                                          0.0).value
        self.calib_dict['gyro_average_offset_y'] = self.declare_parameter('gyro_average_offset_y',
                                                                          0.0).value
        self.calib_dict['gyro_average_offset_z'] = self.declare_parameter('gyro_average_offset_z',
                                                                          0.0).value

        imu_yaw_calibration = self.declare_parameter('imu_yaw_calibration', 0.0).value

        # Check your COM port and baud rate
        self.get_logger().info(f'Razor IMU -> Opening {port}...')
        connection_attempts = 5
        for connection_tries in range(0, connection_attempts + 1):
            try:
                ser = serial.Serial(port=port, baudrate=57600, timeout=1)
                break
            except serial.serialutil.SerialException:
                self.get_logger().error(
                    f'Razor IMU not found at port {port}. '
                    f'Did you specify the correct port in the launch file? '
                    f'Trying {str(5 - connection_tries)} more times...')

            if connection_tries == connection_attempts:
                # exit
                sys.exit(0)
            time.sleep(3)

        roll = 0
        pitch = 0
        yaw = 0
        seq = 0
        accel_factor = 9.806 / 256.0  # sensor reports accel as 256.0 = 1G (9.8m/s^2). Convert to m/s^2.

        ### configure board ###
        # stop datastream
        self.send_command(ser, STOP_DATASTREAM)
        self.write_and_check_config(ser, self.calib_dict)

        if publish_magnetometer:
            self.send_command(ser, SET_TEXT_EXTENDED_FORMAT_WITH_MAG)
            line_start = LINE_START_WITH_MAG
        else:
            self.send_command(ser, SET_TEXT_EXTENDED_FORMAT_NO_MAG)
            line_start = LINE_START_NO_MAG

        self.send_command(ser, START_DATASTREAM)

        self.get_logger().info('Razor IMU up and running')

        # TODO: This forever blocking loop will prevent the node (executor) to spin
        #       -> no callbacks will be ever run, the parameter server won't work, ...
        #       -> This is not a way to go. Rethink.
        while rclpy.ok():
            # TODO: Why not use binary messages ( and avoid unnecessary int > string > int conversions
            line = ser.readline().decode('utf-8')
            if not line.startswith(line_start):
                self.get_logger().error('Did not find correct line start in the received IMU message')
                continue
            line = line.replace(line_start, '')  # Delete "#YPRAG=" or "#YPRAGM="
            words = line.split(',')  # Fields split
            if len(words) > 2:
                # in AHRS firmware z axis points down, in ROS z axis points up (see REP 103)
                yaw_deg = -float(words[2])
                yaw_deg = yaw_deg + imu_yaw_calibration
                if yaw_deg > 180.0:
                    yaw_deg = yaw_deg - 360.0
                if yaw_deg < -180.0:
                    yaw_deg = yaw_deg + 360.0
                yaw = yaw_deg * degrees2rad
                # in AHRS firmware y axis points right, in ROS y axis points left (see REP 103)
                pitch = float(words[1]) * degrees2rad
                roll = -float(words[0]) * degrees2rad

                # Publish message
                # AHRS firmware accelerations are negated
                # This means y and z are correct for ROS, but x needs reversing
                imu_msg.linear_acceleration.x = -float(words[3]) * accel_factor
                imu_msg.linear_acceleration.y = float(words[4]) * accel_factor
                imu_msg.linear_acceleration.z = float(words[5]) * accel_factor

                imu_msg.angular_velocity.x = float(words[6])
                # in AHRS firmware y axis points right, in ROS y axis points left (see REP 103)
                imu_msg.angular_velocity.y = -float(words[7])
                # in AHRS firmware z axis points down, in ROS z axis points up (see REP 103)
                imu_msg.angular_velocity.z = -float(words[8])

                if publish_magnetometer:
                    # according to line 178 the units published of the mag are mGauss
                    # REP103 specifies the units of magnetic field to be Tesla(s)
                    # The axis of the MPU magnetometer are X forward, Y right and Z down
                    #  when the chip is facing forward, in the sparkfun board, the chip is facing the left side
                    # but Sparkfun the firmware interchanges x and y and changes the sign of y
                    # so to get it to REP103 we need to make X and Z negative
                    # The magn frames form sparkfun can be seen in line 178 from Sensors.ino
                    mag_msg.magnetic_field.x = float(words[9]) * 1e-7
                    mag_msg.magnetic_field.y = -float(words[10]) * 1e-7
                    mag_msg.magnetic_field.z = -float(words[11]) * 1e-7
                    # check frame orientation and units

            q = quaternion_from_euler(roll, pitch, yaw)
            imu_msg.orientation.x = q[0]
            imu_msg.orientation.y = q[1]
            imu_msg.orientation.z = q[2]
            imu_msg.orientation.w = q[3]
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            pub_imu.publish(imu_msg)

            if publish_magnetometer:
                mag_msg.header.stamp = imu_msg.header.stamp
                pub_mag.publish(mag_msg)

            if (diag_pub_time < self.get_clock().now()):
                diag_arr = DiagnosticArray()
                diag_arr.header.stamp = self.get_clock().now().to_msg()
                diag_arr.header.frame_id = '1'
                diag_msg = DiagnosticStatus()
                diag_msg.name = 'Razor_Imu'
                diag_msg.level = DiagnosticStatus.OK
                diag_msg.message = 'Received AHRS measurement'

                for obj in [{'roll (deg)': str(roll * (180.0 / math.pi))},
                            {'pitch (deg)': str(pitch * (180.0 / math.pi))},
                            {'yaw (deg)': str(yaw * (180.0 / math.pi))}]:
                    kv = KeyValue()
                    for k, v in obj.items():
                        kv.key = k
                        kv.value = v
                        diag_msg.values.append(kv)

                diag_arr.status.append(diag_msg)
                diag_pub.publish(diag_arr)

        ser.close()

    def send_command(self, serial_instance, command, value=None):
        if value is None:
            cmd = command + chr(13)
        else:
            cmd = command + str(value) + chr(13)
        self.get_logger().info(f'Razor IMU -> Sending: {cmd}')
        expected_len = len(cmd)
        res = serial_instance.write(str.encode(cmd))
        if res != expected_len:
            self.get_logger().error(
                f'Razor IMU -> Expected serial command len ({str(expected_len)}) '
                f'didn\'t match amount of bytes written ({str(res)}) for command {command}')
        sleep(0.05)  # Don't spam serial with too many commands at once

    def write_and_check_config(self, serial_instance, calib_dict):

        self.get_logger().info('Razor IMU setting calibration values ...')

        self.send_command(serial_instance, SET_CALIB_ACC_X_MIN, calib_dict['accel_x_min'])
        self.send_command(serial_instance, SET_CALIB_ACC_X_MAX, calib_dict['accel_x_max'])
        self.send_command(serial_instance, SET_CALIB_ACC_Y_MIN, calib_dict['accel_y_min'])
        self.send_command(serial_instance, SET_CALIB_ACC_Y_MAX, calib_dict['accel_y_max'])
        self.send_command(serial_instance, SET_CALIB_ACC_Z_MIN, calib_dict['accel_z_min'])
        self.send_command(serial_instance, SET_CALIB_ACC_Z_MAX, calib_dict['accel_z_max'])

        if not calib_dict['magn_use_extended']:
            self.send_command(serial_instance, SET_MAG_X_MIN, calib_dict['magn_x_min'])
            self.send_command(serial_instance, SET_MAG_X_MAX, calib_dict['magn_x_max'])
            self.send_command(serial_instance, SET_MAG_Y_MIN, calib_dict['magn_y_min'])
            self.send_command(serial_instance, SET_MAG_Y_MAX, calib_dict['magn_y_max'])
            self.send_command(serial_instance, SET_MAG_Z_MIN, calib_dict['magn_z_min'])
            self.send_command(serial_instance, SET_MAG_Z_MAX, calib_dict['magn_z_max'])
        else:
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_CENTER_0,
                              self.calib_dict['magn_ellipsoid_center'][0])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_CENTER_1,
                              self.calib_dict['magn_ellipsoid_center'][1])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_CENTER_2,
                              self.calib_dict['magn_ellipsoid_center'][2])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_0_0,
                              self.calib_dict['magn_ellipsoid_transform'][0][0])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_0_1,
                              self.calib_dict['magn_ellipsoid_transform'][0][1])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_0_2,
                              self.calib_dict['magn_ellipsoid_transform'][0][2])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_1_0,
                              self.calib_dict['magn_ellipsoid_transform'][1][0])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_1_1,
                              self.calib_dict['magn_ellipsoid_transform'][1][1])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_1_2,
                              self.calib_dict['magn_ellipsoid_transform'][1][2])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_2_0,
                              self.calib_dict['magn_ellipsoid_transform'][2][0])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_2_1,
                              self.calib_dict['magn_ellipsoid_transform'][2][1])
            self.send_command(serial_instance, SET_MAG_ELLIPSOID_TRANSFORM_2_2,
                              self.calib_dict['magn_ellipsoid_transform'][2][2])

        self.send_command(serial_instance, SET_GYRO_AVERAGE_OFFSET_X,
                          self.calib_dict['gyro_average_offset_x'])
        self.send_command(serial_instance, SET_GYRO_AVERAGE_OFFSET_Y,
                          self.calib_dict['gyro_average_offset_y'])
        self.send_command(serial_instance, SET_GYRO_AVERAGE_OFFSET_Z,
                          self.calib_dict['gyro_average_offset_z'])

        self.send_command(serial_instance, GET_CALIBRATION_VALUES)
        config = ''
        for _ in range(0, 21):
            # Format each line received from serial into proper yaml with lowercase variable names
            config += serial_instance.readline().decode('utf-8').lower().replace(':', ': ')
            # TODO: round the numbers, otherwise we will get false negatives in the check phase

        print("========== RAW CONFIG ==========")
        print(config)
        print("================================")

        config_parsed = yaml.safe_load(config) or {}
        # config_parsed = yaml.safe_load(config)
        
        for key in self.calib_dict:
            if key not in config_parsed:
                self.get_logger().warning(
                    f'The calibration value and key of [{key}] is missing from your config file')

            elif config_parsed[key] != self.calib_dict[key]:
                self.get_logger().warning(
                    f'The calibration value of [{key}] did not match. '
                    f'Expected: {str(self.calib_dict[key])} , received: {str(config_parsed[key])}')


def main(args=None):
    rclpy.init(args=args)

    node = RazorImuDriver()

    # TODO: the code below will never run because RazorImuDriver's constructor never exits

    rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
