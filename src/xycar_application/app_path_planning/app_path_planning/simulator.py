#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import numpy as np
import math
import rclpy
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from math import radians
from app_path_planning.parking import *

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (216, 241, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
NAVY = (70, 130, 180)
ORANGE = (240, 128, 128)
LIGHT_PINK = (255, 163, 212)
PURPLE = (72, 62, 139)
SKY_BLUE = (0, 255, 255)
PINK = (255, 0, 255)

AR_CENTER = [1142.5, 62.5]
AR_END = [1175.0, 95.0]

game = None

class Config:
    # PID config
    KP = 1.0
    # System config
    K = 1.0
    MAX_STEER = np.radians(20.0)

class Map(pygame.sprite.Sprite):
    def __init__(self, screen, width, height):
        super(Map, self).__init__()
        self.screen = screen
        self.logo_path = "/home/xytron/xycar_ws/src/xycar_application/app_path_planning/images/logo.png"  
        #self.logo_path = "/workspaces/isaac_ros-dev/src/xycar/xycar_application/app_path_planning/images/logo.png"
        self.logo = pygame.image.load(self.logo_path).convert_alpha()
        self.font = pygame.font.SysFont('notosansmonocjkkrblack', 39)
        self.font2 = pygame.font.SysFont('notosansmonocjkkrblack', 45)
        self.plan = self.font2.render('Planning', True, WHITE)
        self.track = self.font2.render('Tracking', True, WHITE)

    def update(self, finish):
        pygame.draw.rect(self.screen, WHITE, [0, 0, 1190, 850])
        pygame.draw.line(self.screen, BLACK, [1000, -100], [1220, 120], 28)
        pygame.draw.line(self.screen, GREEN, [1110, 30], [1175, 95], 10)

        pygame.draw.circle(self.screen, BLUE, [100, 350], 64)
        pygame.draw.circle(self.screen, BLUE, [100, 750], 64)
        pygame.draw.circle(self.screen, BLUE, [500, 700], 64)
        pygame.draw.circle(self.screen, BLUE, [900, 700], 64)
        pygame.draw.circle(self.screen, BLUE, [1100, 300], 64)

        pygame.draw.rect(self.screen, NAVY, [250, 783, 200, 54])
        pygame.draw.rect(self.screen, ORANGE, [480, 783, 200, 54])
        self.screen.blit(self.plan, (285, 795))
        self.screen.blit(self.track, (518, 795))
        self.screen.blit(self.logo, (40, 30))

        if finish == 0 or finish == 2:
            line_color = RED
        elif finish == 1:
            line_color = GREEN

        points = [[1096, 36], [1003, 129], [1069, 195], [1162, 102]]

        for i in range(len(points) - 1):
            pygame.draw.line(self.screen, line_color, points[i], points[i + 1], 4)

        pygame.draw.line(self.screen, line_color, points[-1], points[0], 4)

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, screen, angle=0.0, max_acceleration=1000.0):
        super(Car, self).__init__()
        self.screen = screen
        self.x = x
        self.y = y
        self.yaw = angle
        self.max_acceleration = max_acceleration
        self.linear_velocity = 0.0
        self.max_velocity = 50
        self.steering_angle = 0.0
        self.wheel_base = 84
        self.car_img_x = 0
        self.car_img_y = 0

        self.car_x_ori = [-64, -64, 64, 64]
        self.car_y_ori = [-32, 32, -32, 32]

        self.car_x = [0, 0, 0, 0]
        self.car_y = [0, 0, 0, 0]

        self.car_center = [0.0, 0.0]
        self.car_front_center = [0.0, 0.0]
        self.car_back_center = [0.0, 0.0]

        self.car_x_list = []
        self.car_y_list = []

        self.img_path = "/home/xytron/xycar_ws/src/xycar_application/app_path_planning/images/car.png"
        #self.img_path = "/workspaces/isaac_ros-dev/src/xycar/xycar_application/app_path_planning/images/car.png"
        self.image = pygame.image.load(self.img_path).convert_alpha()

        self.rotated = pygame.transform.rotate(self.image, self.yaw)
        game.finish = 0

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    def update(self, linear_velocity, delta, dt):
        delta = self.limit_input(np.radians(delta))
        self.linear_velocity = min(max(-self.max_velocity, linear_velocity), self.max_velocity)

        self.angular_velocity = 0.0

        if delta != 0.0:
            self.angular_velocity = (self.linear_velocity / self.wheel_base) * np.tan((delta))

        self.yaw += (np.degrees(self.angular_velocity) * dt)
        self.yaw = np.degrees(self.normalize_angle(np.radians(self.yaw)))

        if game.finish == 1 or game.finish == 2:
            linear_velocity = 0

        self.x += (self.linear_velocity * np.cos(np.radians(-self.yaw))) * dt
        self.y += (self.linear_velocity * np.sin(np.radians(-self.yaw))) * dt

        self.car_x = [0, 0, 0, 0]
        self.car_y = [0, 0, 0, 0]

        for i in range(4):
            self.car_x[i] = self.car_x_ori[i] * np.cos(-radians(self.yaw)) - self.car_y_ori[i] * np.sin(-radians(self.yaw)) + self.x
            self.car_y[i] = self.car_x_ori[i] * np.sin(-radians(self.yaw)) + self.car_y_ori[i] * np.cos(-radians(self.yaw)) + self.y

        self.car_img_x = int(round(min(self.car_x)))
        self.car_img_y = int(round(min(self.car_y)))

        self.rotated = pygame.transform.rotate(self.image, self.yaw)

        self.screen.blit(self.rotated, [self.car_img_x, self.car_img_y])

        center_x = sum(self.car_x) / 4
        center_y = sum(self.car_y) / 4
        self.car_center = [center_x, center_y]
        self.car_front_center = [(self.car_x[2] + self.car_x[3]) / 2, (self.car_y[2] + self.car_y[3]) / 2]
        self.car_back_center = [(self.car_x[0] + self.car_x[1]) / 2, (self.car_y[0] + self.car_y[1]) / 2]

        self.car_x_list.append(self.car_front_center[0])
        self.car_y_list.append(self.car_front_center[1])

    @staticmethod
    def limit_input(delta):
        if delta > Config.MAX_STEER:
            return Config.MAX_STEER

        if delta < -Config.MAX_STEER:
            return -Config.MAX_STEER

        return delta

class Game(Node):
    def __init__(self):
        super().__init__('simulator')
        pygame.init()
        pygame.display.set_caption("Car Simulator")
        self.screen_width = 1200
        self.screen_height = 850
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.ticks = 60

        self.exit = False

        self.subscription = self.create_subscription(
            XycarMotor,
            'xycar_motor',
            self.motor_callback,
            10
        )
        self.target_idx = 0
        self.d = 0
        self.re = 1

        self.finish = 0
        self.angle = 90.0
        self.ang = 0.0
        self.spd = 0.0
                
    def motor_callback(self, msg):
        self.ang = int(msg.angle)
        self.spd = int(msg.speed)

    def is_within_parking_rect(self, rect_coords, center_coords):
        x, y = center_coords
        x_coords = [coord[0] for coord in rect_coords]
        y_coords = [coord[1] for coord in rect_coords]

        if min(x_coords) <= x <= max(x_coords) and min(y_coords) <= y <= max(y_coords):
            return True
        return False

    def run(self):
        mapped = Map(self.screen, self.screen_width, self.screen_height)
        car = Car(100, 350, self.screen, self.angle - 45)

        t, rx, ry, ryaw, v, a, j = [], [], [], [], [], [], []

        while not self.exit:
            car_yaw = 270 - car.yaw
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if pos[0] > 36 and pos[0] < 164 and pos[1] > 286 and pos[1] < 446:
                        car = Car(100, 350, self.screen, self.angle - 45)
                        self.d = 0

                    if pos[0] > 36 and pos[0] < 164 and pos[1] > 686 and pos[1] < 846:
                        car = Car(100, 750, self.screen, self.angle - 90)
                        self.d = 0

                    if pos[0] > 436 and pos[0] < 564 and pos[1] > 636 and pos[1] < 764:
                        car = Car(500, 700, self.screen, self.angle)
                        self.d = 0

                    if pos[0] > 836 and pos[0] < 964 and pos[1] > 636 and pos[1] < 764:
                        car = Car(900, 700, self.screen, self.angle - 45)
                        self.d = 0

                    if pos[0] > 1036 and pos[0] < 1164 and pos[1] > 236 and pos[1] < 364:
                        car = Car(1100, 300, self.screen, self.angle - 45)
                        self.d = 0

                    if pos[0] > 250 and pos[0] < 450 and pos[1] > 783 and pos[1] < 837:
                        #dt = float(self.clock.get_time()) / 1000.0
                        rx, ry = planning(car.x, car.y, car_yaw, car.max_acceleration, dt)

                    if pos[0] > 480 and pos[0] < 680 and pos[1] > 783 and pos[1] < 837:
                        print("Start Tracking")
                        self.d = 1
                        # car.linear_velocity = 50.0

                if event.type == pygame.QUIT:
                    pygame.quit()

            dt = float(self.clock.get_time()) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            self.screen.fill((255, 255, 255))

            if math.sqrt((1190 - car.car_front_center[0]) ** 2 + (0 - car.car_front_center[1]) ** 2) < 95:
                self.finish = 2
                if self.is_within_parking_rect([[1096, 36], [1003, 129], [1069, 195], [1162, 102]], [car.x, car.y]):
                    self.finish = 1

            if car.car_front_center[0] < 0 or car.car_front_center[1] < 0:
                self.finish = 2

            mapped.update(self.finish)

            if len(rx) != 0:
                for i, _ in enumerate(rx):
                    pygame.draw.circle(game.screen, SKY_BLUE, [int(rx[i]), int(ry[i])], 3)

            if len(car.car_x_list) != 0:
                for i, _ in enumerate(car.car_x_list):
                    pygame.draw.circle(game.screen, PINK, [int(car.car_x_list[i]), int(car.car_y_list[i])], 3)

            # path tracking
            if self.d != 0:
                self.ang, self.spd = tracking(game.screen, car.x, car.y, car.yaw, car.linear_velocity, car.max_acceleration, dt)
                
                #self.ang = int(msg.angle)
                #self.spd = int(msg.speed)
                
                car.update(self.spd, -self.ang, dt)

            if self.d == 0:
                car.linear_velocity = 0
                car.update(0, 0, dt)

            pygame.display.update()

            self.clock.tick(self.ticks)

        pygame.quit()

def main(args=None):
    global game
    
    rclpy.init(args=args)   
    game = Game()

    #init_parking_module() 
    
    game.run()
    game.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()                  
