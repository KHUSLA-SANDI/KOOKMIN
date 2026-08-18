#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import os
import time, pygame
from xycar_sim_driving.wall import *
from xycar_sim_driving.game import *

from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import XycarMotor
from std_msgs.msg import Int32MultiArray
from rclpy.clock import Clock
    
class Simul(Node):

    def __init__(self):
        super().__init__('simulator')
        
        # ROS 2 Clock 추가
        self.ros_clock = Clock()        
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_name', 'new_map'),
                ('range_sensor', 'ultrasonic'),
                ('drive_mode', 'ros'),
            ])
        self.map_name = self.get_parameter('map_name').value
        self.device = self.get_parameter('range_sensor').value
        self.drive_mode = self.get_parameter('drive_mode').value
        
        self.set_param()
        self.ros_init()
        self.start()          
        
    def set_param(self):
        self.qos_profile = QoSProfile(depth=1)
        self.angle = 0.0
        self.speed = 0.0
        self.timeCheck = 0
        self.scan_time_utc = time.time()
        self.episode = 1
        self.word_pos_x = 300
        self.word_pos_y = 288
        self.ticks = 120
        self.add = 30
        self.keymode = False
        self.obs = obs_make(self.map_name)
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self.obs.get_wh()
        self.DISTANCE_DEVICE = 0
        self.TURN_VECTOR = 0 # 0: clock, 1: reverse clock
        self.STOP_MODE = 0 # 0: floor game end, 1: move
        self.altKey = False
        self.fsMode = 0
        self.CurrentScreen = False
        #self.suc_code = 0
        self.INIT = list(self.obs.get_xyt())
        self.GOAL = list(self.obs.get_goal())
        self.LOGO = list(self.obs.get_logo())
        self.OBS = self.obs.get_obs()
        self.WARP = list(self.obs.get_warp())
        if self.drive_mode == 'keyboard':
            self.keymode = True
            
        if self.device == "lidar":
            self.DISTANCE_DEVICE = 1
            
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
            self.lidar_msg.angle_min = -3.12413907051
            self.lidar_msg.angle_max = 3.14159274101
            self.lidar_msg.angle_increment = 0.174533
            self.lidar_msg.range_min = 0.15000000596
            self.lidar_msg.range_max = 12.0
            
    def motor_callback(self, msg):
        self.timeCheck = time.time()
        self.angle = msg.angle
        self.speed = msg.speed
            
    def start(self):
        pygame.init()
        pygame.display.set_caption("Xycar Simulator")
        
        screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        
        o = self.obs.get_list()
        
        F = pygame.font.SysFont("Arial", 20, bold=False, italic=False)    
        D = pygame.font.SysFont("Arial", 15, bold=False, italic=False)
        
        game = Game(True, state_num=6, INIT=self.INIT, GOAL=self.GOAL, LOGO=self.LOGO, OBS=self.OBS, WARP=self.WARP, distance_device=self.DISTANCE_DEVICE, stop_mode=self.STOP_MODE, turn=self.TURN_VECTOR, key_mode=self.keymode, Font_1=F, Font_2=D, pygame_screen=screen)
        game.map_input(self.obs.get_all_list(), o[0], o[1])
        #clock = pygame.time.Clock()
        self.pygame_clock = pygame.time.Clock()
        
        self.angle = 0.0
        self.speed = 0.0
        
        self.timeCheck = time.time()      
        self.tc = time.time()
        
        while rclpy.ok():
            game.game_init(self.episode)
            step = 0

            while rclpy.ok():
                eventList = pygame.event.get()
                for event in eventList:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LALT:
                            self.altKey = True
                        if (event.key == pygame.K_RETURN) and self.altKey:
                            self.CurrentScreen = not self.CurrentScreen
                            if self.CurrentScreen:
                                self.fsMode = 1
                            else:
                                self.fsMode = -1
                        if (event.key == pygame.K_ESCAPE) and self.CurrentScreen and not self.fsMode:
                            pygame.quit()
                            exit()
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_LALT:
                            self.altKey = False
            
                if self.fsMode == 1:
                    pygame.display.quit()
                    pygame.display.init()
                    screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT], pygame.FULLSCREEN)
                    game.change_screen(screen)
                    self.fsMode = 0
                elif self.fsMode == -1:
                    pygame.display.quit()
                    pygame.display.init()
                    screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
                    game.change_screen(screen)
                    self.fsMode = 0
                else:
                    self.fsMode = 0

                #dt = float(clock.get_time()) / 1000.0
                dt = float(self.pygame_clock.get_time()) / 1000.0
                
                game.key_space_bar()

                if self.keymode:
                    game.key_return(dt)

                if (time.time() - self.timeCheck) > 1:
                    suc_code, dddv, car_angle = game.run(0, 0, step, dt)
                else:
                    suc_code, dddv, car_angle = game.run(self.angle, self.speed, step, dt)

                if self.DISTANCE_DEVICE == 0:
                    ddddv = [int(round(i)) for i in dddv]
                    self.choumpha_msg.data = ddddv
                    self.choumpha_pub.publish(self.choumpha_msg)
                    if ((time.time() - self.tc) > 0.5) and (game.car.goal_count < 3):
                        self.tc = time.time()
                    
                elif self.DISTANCE_DEVICE == 1:
                    if self.scan_time_utc is None:
                        self.scan_time_utc = time.time()
                    self.lidar_msg.ranges = dddv
                    
                    #self.lidar_msg.header.stamp = Clock().now().to_msg()
                    self.lidar_msg.header.stamp = self.ros_clock.now().to_msg()

                    self.lidar_msg.scan_time = time.time() - self.scan_time_utc
                    
                    self.lidar_msg.time_increment = self.lidar_msg.scan_time/180
                    self.lidar_pub.publish(self.lidar_msg)
                    self.scan_time_utc = time.time()

                if game.car.goal_count == 3:
                    First_time =  F.render(" 1 : " + str(game.car.first), True, (255,0,0))
                    Second_time =  F.render(" 2 : " + str(game.car.second), True, (255,0,0))
                    Third_time =  F.render(" 3 : " + str(game.car.third), True, (255,0,0))
                    Average_time =  F.render(" Average: " + str(game.car.time_aver), True, (255,0,0))

                if game.car.goal_count < 3:
                    screen.blit(game.car.rotated, [game.car.car_img_x, game.car.car_img_y])

                if game.car.goal_count == 3:
                    screen.blit(First_time, (self.word_pos_x, self.word_pos_y + 0 * self.add))
                    screen.blit(Second_time, (self.word_pos_x, self.word_pos_y + 1 * self.add))
                    screen.blit(Third_time, (self.word_pos_x, self.word_pos_y + 2 * self.add))
                    screen.blit(Average_time, (self.word_pos_x, self.word_pos_y + 3 * self.add))

                pygame.display.flip()
                
                #clock.tick(self.ticks)
                self.pygame_clock.tick(self.ticks)
            
                if suc_code > 0:
                    break

                step += 1
                rclpy.spin_once(self)

            self.episode += 1
        

def main(args=None):
    rclpy.init(args=args)
    node = Simul()
    
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()

