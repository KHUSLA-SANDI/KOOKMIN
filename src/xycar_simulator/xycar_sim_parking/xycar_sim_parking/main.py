#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.clock import Clock
import time, os, pygame, base64, sys
from xycar_sim_parking.wall import *
from xycar_sim_parking.game import *
from xycar_sim_parking.map import *
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray
from xycar_msgs.msg import XycarMotor
from ros2_aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import Pose


class Simul(Node):
  
    def __init__(self):
        super().__init__('simulator')
        
        self.set_param()
        self.ros_init()
        self.start()
        
    def set_param(self):
        self.qos_profile = QoSProfile(depth=1)
        self.angle = 0.0
        self.speed = 0.0
        self.timeCheck = 0
        self.DISTANCE_DEVICE = 2 # 0: choumpha, 1: lidar, 2: camera
        
    def ros_init(self):
        self.motor_subscriber = self.create_subscription(
            XycarMotor, '/xycar_motor', self.motor_callback, self.qos_profile)
            
        if self.DISTANCE_DEVICE == 0:
            self.choumpha_pub = self.create_publisher(Int32MultiArray, 'ultrasonic', self.qos_profile)
            self.choumpha_msg = Int32MultiArray()
        elif self.DISTANCE_DEVICE == 1:
            self.lidar_pub = self.create_publisher(LaserScan, 'scan', self.qos_profile)
            self.lidar_msg = LaserScan()
            self.lidar_msg.header.frame_id = "laser"
            self.lidar_msg.header.seq = 0
            self.lidar_msg.angle_min = -3.12413907051
            self.lidar_msg.angle_max = 3.14159274101
            self.lidar_msg.angle_increment = 0.0174532923847
            self.lidar_msg.scan_time = 0.075
            self.lidar_msg.time_increment = self.lidar_msg.scan_time/359
            self.lidar_msg.range_min = 0.15000000596
            self.lidar_msg.range_max = 12.0
        elif self.DISTANCE_DEVICE == 2:
            self.artag_pub = self.create_publisher(ArucoMarkers, 'ar_pose_marker', self.qos_profile)
        
        
    def motor_callback(self, msg):
        self.timeCheck = time.time()
        self.angle = -msg.angle
        self.speed = msg.speed

    def start(self):

        keymode = False
        episode = 1

        pos_x = 0
        pos_y = 0
        word_pos_x = 300
        word_pos_y = 288
        ticks = 120
        add = 30

        TURN_VECTOR = 0 # 0: clock, 1: reverse clock
        STOP_MODE = 0 # 0: floor game end, 1: move

        os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (pos_x,pos_y)
        os.environ['SDL_VIDEO_CENTERED'] = '0'

        pygame.init()
        pygame.display.set_caption("Simulator")
        screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        
        obs = obs_make()
        o = obs.get_list()
        
        F = pygame.font.SysFont("Arial", 20, bold=False, italic=False)    
        D = pygame.font.SysFont("Arial", 15, bold=False, italic=False)
        
        game = Game(True, state_num=6, distance_device=self.DISTANCE_DEVICE, stop_mode=STOP_MODE, turn=TURN_VECTOR, key_mode=keymode, pygame_screen=screen, Font_1=F, Font_2=D)
        game.map_input(obs.get_all_list(), o[0], o[1])

        x = 0
        y = 0
        suc_code = 0

        self.angle = 0
        self.speed = 0

        self.timeCheck = time.time()
        tc = time.time()
        rst = 0

        while rclpy.ok():
            while rst == -1:
                rst = game.restart_button()

            clock = pygame.time.Clock()
            game.game_init(episode, rst)
            step = 0

            while rclpy.ok():
                eventList = pygame.event.get()
                for event in eventList:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                dt = float(clock.get_time()) / 1000.0

                rst = -1
                if keymode:
                    game.key_return(dt)
                else:
                    rst = game.restart_button()

                if (0 <= rst <= 3):
                    break

                #print("start-angle", angle)
                if (time.time() - self.timeCheck) > 1:
                    suc_code, dddv, car_angle = game.run(0, 0, step, dt)
                    game.run(0, 0, step, dt)
                else:
                    suc_code, dddv, car_angle = game.run(self.angle, self.speed, step, dt)
                    #game.run(angle, speed, step, dt)
                #print("print-angle-1", angle)

                if self.DISTANCE_DEVICE == 0:
                    dddv = [int(dddv[0]), int(dddv[1]), int(dddv[2]), int(dddv[3]), int(dddv[4]), int(dddv[5]), int(dddv[6]), int(dddv[7])]
                    self.choumpha_msg.data = dddv
                    self.choumpha_pub.publish(self.choumpha_msg)
                    if (time.time() - tc) > 0.5:
                        tc = time.time()
                        print("fl: " + str(dddv[0]) + ", fm: " + str(dddv[1]) + ", fr: " + str(dddv[2]) + ", bl: " + str(dddv[3]) + ", bm: " + str(dddv[4]) + ", br: " + str(dddv[5]) + ", r: " + str(dddv[6]) + ", l: " + str(dddv[7]))
                    
                elif self.DISTANCE_DEVICE == 1:
                    self.lidar_msg.ranges = dddv
                    self.lidar_msg.header.stamp = Clock().now().to_msg()

                    self.lidar_pub.publish(self.lidar_msg)

                elif self.DISTANCE_DEVICE == 2:
                    artag_msg = ArucoMarkers()
                    artag_msg.header.frame_id = "usb_cam"

                    if game.car.arDetect:
                        posed = Pose()
                        ar_distance = game.car.ar_distance
                        ar_angle = game.car.MD2CAM_ang
                        
                        artag_msg.marker_ids.append(1)
                        
                        posed.position.x = ar_distance[0]
                        posed.position.y = ar_distance[1]
                        posed.position.z = 0.0

                        q = game.car.AR.get_quaternion(game.car.ar_angle)

                        posed.orientation.x = q[0]
                        posed.orientation.y = q[1]
                        posed.orientation.z = q[2]
                        posed.orientation.w = q[3]

                        artag_msg.poses.append(posed)

                    artag_msg.header.stamp = Clock().now().to_msg()
                    self.artag_pub.publish(artag_msg)


                #dddv_rr =  F.render("    Right     : " + str(dddv[0]), True, (28,0,0))
                #dddv_rl =  F.render(" Bottom Right : " + str(dddv[1]), True, (28,0,0))
                #dddv_cc =  F.render("    Center    : " + str(dddv[2]), True, (28,0,0))
                #dddv_lr =  F.render("   Top Left   : " + str(dddv[3]), True, (28,0,0))
                #dddv_ll =  F.render("     Left     : " + str(dddv[4]), True, (28,0,0))

                screen.blit(game.car.rotated, [game.car.car_img_x, game.car.car_img_y])

                #screen.blit(dddv_rr, (word_pos_x, word_pos_y + 0 * add))
                #screen.blit(dddv_rl, (word_pos_x, word_pos_y + 1 * add))
                #screen.blit(dddv_cc, (word_pos_x, word_pos_y + 2 * add))
                #screen.blit(dddv_lr, (word_pos_x, word_pos_y + 3 * add))
                #screen.blit(dddv_ll, (word_pos_x, word_pos_y + 4 * add))

                pygame.display.flip()
                clock.tick(ticks)
            
                if suc_code > 0:
                    break

                step += 1
                rclpy.spin_once(self)
                
            episode += 1

        #signal.signal(signal.SIGINT, signal_handler)
        
def main(args=None):
    rclpy.init(args=args)
    node = Simul()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

