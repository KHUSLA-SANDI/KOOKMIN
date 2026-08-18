#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xycar_sim_driving.confirm import *
from xycar_sim_driving import *
from xycar_sim_driving.resource import *
from io import BytesIO
from PIL import Image
from math import sin, cos, radians, degrees, copysign
import pygame, os, time, random, base64

class Car:
    location = [[-64,  0], [-63,-22], [-63, 22], [-53,-27], [-54, 27],
                [-45,-30], [-21,-30], [  2,-30], [ 24,-30], [ 46,-30],
                [-47, 29], [-24, 30], [  0, 30], [ 25, 30], [ 46, 30],
                [ 54,-23], [ 55, 22], [ 60,-14], [ 60, 14], [ 63,  0]]
    
    def __init__(self, pygame_go, state_num, turn, stop_mode, INIT, OBS, WARP, GOAL, mode=0, screen=None):
        self.mode = mode
        if self.mode == 0:
            self.choumpha_ang = [-90, -52, 0, 52, 90, 125, 180, -125]
            self.choumpha_posi = [[4,28], [55,-23], [62,-1], [57,20], [1,-27], [-51, 24], [-64, 0], [-51, -24]]
            
            self.choumpha = distance_device()
            self.choumpha_distance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        elif self.mode == 1:
            self.lidar_angle = 0
            self.lidar_posi = [62, -1]
            self.lidar = distance_device()
            #self.lidar_data_cnt = state_num - 1
            self.lidar_data_cnt = 180
            self.lidar_init(self.lidar_data_cnt)
            self.lidar_vector_aver = 0
    
        self.turn = turn

        self.init_x = INIT[0]
        self.init_y = INIT[1]
        self.init_angle = INIT[2]

        self.OBS = OBS
        self.GOAL = GOAL

        self.warp = WARP[0]
        self.warp_angle = WARP[1]

        self.location_cnt = len(self.location)
        self.screen = screen
        self.goalin = False

        self.first = 0.0
        self.second = 0.0
        self.third = 0.0
        self.stop_cnt = 0 

        self.pygame_on = pygame_go
        if self.pygame_on:
            self.stop_mode = stop_mode
            self.car_x_ori = [-64,-64, 64, 64]
            self.car_y_ori = [-32, 32,-32, 32]
            #self.img_path = rospkg.RosPack().get_path('xycar_sim_drive') + "/image/car.png"
            car_bytes = base64.b64decode(car)
            car_file = BytesIO(car_bytes)
            self.colorImage  = Image.open(car_file)
            self.rotated = self.PIL2PygameIMG(self.colorImage, 0)
            self.transparent = (0, 0, 0, 0)
            self.car_image = pygame.sprite.Sprite()
            self.car_img_x = 0
            self.car_img_y = 0

    def change_screen(self, scr):
        self.screen = scr
        
    def set_init_car(self, next):
        if self.turn == 0:
            x = self.init_x
            #y = float(random.randint(70, 160))/100.0
            y = self.init_y
            #A = float(random.randint(-25, 25))
        elif self.turn == 1:
            x = float(random.randint(70, 160))/100.0
            y = 3.5
            #A = float(random.randint(-115, -65))
        self.position = [x, y]
        self.polor_coordinate = [0.0, self.init_angle]
        self.Linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.length = 1.28
        self.steering = 0.0
        self.max_steering = 30.0

        self.end = False
        self.suc = None
        if next:
            self.first = 0.0
            self.second = 0.0
            self.third = 0.0
            self.goal_count = 0
            self.stop_cnt = 0 
            self.time_lab = 0

    def choumpha_add(self, posi, ang):
        self.choumpha_posi.append(posi)
        self.choumpha_ang.append(ang)
        self.choumpha_distance(0.0)
    
    def lidar_init(self, data_cnt):
        self.lidar_distance = []
        for _ in range(data_cnt):
            self.lidar_distance.append(0.0)

    def stop(self):
        self.Linear_velocity = 0.0
        self.steering = 0.0
        self.set_init_car(False)
        #self.end = True
        #if accident:
        #    self.suc = False
        #else:
        #    self.suc = True
        self.time_lab_func()
        self.goal_count += 1

    def time_lab_func(self):
        if self.goal_count == 0:
            self.first = self.time_lab

        elif self.goal_count == 1:
            self.second = self.time_lab - self.first
            
        elif self.goal_count == 2:
            self.third = self.time_lab - self.first - self.second

        self.time_aver = (self.first + self.second + self.third) / 3.0
        
    def set_time(self, time_stamp):
        self.time_lab = time_stamp

    def PIL2PygameIMG(self, imgData, angle, rgb='RGBA', center=(64, 32), expand=True):
        IMG = imgData.rotate(angle, expand=expand, center=center)
        return pygame.image.fromstring(IMG.tobytes("raw", rgb), IMG.size, rgb)

    def update(self, dt):
        if (self.goal_count == 3):
            self.Linear_velocity = 0

        self.angular_velocity = 0.0
        
        if self.steering != 0.0:
            self.angular_velocity = (self.Linear_velocity / self.length) * sin(radians(self.steering))
        
        self.polor_coordinate[0] = dt * self.Linear_velocity
        
        self.position[0] += self.polor_coordinate[0] * cos(-radians(self.polor_coordinate[1]))
        self.position[1] += self.polor_coordinate[0] * sin(-radians(self.polor_coordinate[1]))

        self.polor_coordinate[1] += degrees(self.angular_velocity) * dt

        car_center = [
            self.position[0] * 100, 
            self.position[1] * 100
        ]
        

        if self.pygame_on:
            car_x = [0,0,0,0]
            car_y = [0,0,0,0]

            self.rotated = self.PIL2PygameIMG(self.colorImage, self.polor_coordinate[1])

            for i in range(4):
                car_x[i] = self.car_x_ori[i] * cos(-radians(self.polor_coordinate[1])) - self.car_y_ori[i] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
                car_y[i] = self.car_x_ori[i] * sin(-radians(self.polor_coordinate[1])) + self.car_y_ori[i] * cos(-radians(self.polor_coordinate[1])) + car_center[1]

            self.car_img_x = int(round(min(car_x)))
            self.car_img_y = int(round(min(car_y)))

        if self.mode == 0:
            choumpha_position = []
            choumpha_angle = []

            for i in range(len(self.choumpha_ang)):
                choumpha_position.append([0.0, 0.0])
                choumpha_angle.append(0.0)

                choumpha_position[i][0] = self.choumpha_posi[i][0] * cos(-radians(self.polor_coordinate[1])) - self.choumpha_posi[i][1] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
                choumpha_position[i][1] = self.choumpha_posi[i][0] * sin(-radians(self.polor_coordinate[1])) + self.choumpha_posi[i][1] * cos(-radians(self.polor_coordinate[1])) + car_center[1]
                choumpha_angle[i] = float(self.choumpha_ang[i] - self.polor_coordinate[1])

                if self.pygame_on and (self.goal_count < 3):
                    pygame.draw.circle(self.screen, (0,0,255), [int(round(choumpha_position[i][0])), int(round(choumpha_position[i][1]))], 5) 

                _, distance, cordinate = self.choumpha.confirm_device(choumpha_angle[i], choumpha_position[i], self.OBS, 1, min_angle=0, max_angle=0)             

                self.choumpha_distance[i] = distance[0]

                if self.pygame_on and (self.goal_count < 3):
                    pygame.draw.line(self.screen, [255,0,0], choumpha_position[i], (cordinate[0][0], cordinate[0][1]), 3)\

        elif self.mode == 1:        

            lidar_position = [0.0, 0.0]
            lidar_position[0] = self.lidar_posi[0] * cos(-radians(self.polor_coordinate[1])) - self.lidar_posi[1] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
            lidar_position[1] = self.lidar_posi[0] * sin(-radians(self.polor_coordinate[1])) + self.lidar_posi[1] * cos(-radians(self.polor_coordinate[1])) + car_center[1]

            if self.pygame_on and (self.goal_count < 3):
                pygame.draw.circle(self.screen, (0,0,255), [int(round(lidar_position[0])), int(round(lidar_position[1]))], 5)

            _, distance, cordinate = self.lidar.confirm_device(-self.polor_coordinate[1], lidar_position, self.OBS, 180, min_angle=-90, max_angle=90)

            if len(distance) == 181:
                s, sr, cr = self.lidar.vector_aver(-self.polor_coordinate[1], distance, self.lidar_data_cnt, min_angle=0, max_angle=180)
                self.lidar_distance = []
                for i in range(180):
                    self.lidar_distance.append(distance[i])

                if (self.pygame_on) and (self.goal_count < 3):
                    for i in range(0,181,10):
                        pygame.draw.line(self.screen, [255,0,0], lidar_position, (cordinate[i][0], cordinate[i][1]))

        outline = [0.0, 0.0]
        for loc in range(self.location_cnt):
            outline[0] = self.location[loc][0] * cos(-radians(self.polor_coordinate[1])) - self.location[loc][1] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
            outline[1] = self.location[loc][0] * sin(-radians(self.polor_coordinate[1])) + self.location[loc][1] * cos(-radians(self.polor_coordinate[1])) + car_center[1]
            
            for ww in self.warp:
                if self.mode == 0: 
                    warp, bun = self.choumpha.rect_in_point(outline, ww)
                elif self.mode == 1: 
                    warp, bun = self.lidar.rect_in_point(outline, ww)

                if warp:
                    #self.position = WARP_POS
                    self.polor_coordinate = [0.0, self.warp_angle]
                    break

            accidents = []
            for wall in self.OBS:
                if self.mode == 0: 
                    accident, bun = self.choumpha.rect_in_point(outline, wall)
                if self.mode == 1: 
                    accident, bun = self.lidar.rect_in_point(outline, wall)
                
                if accident:
                    if (bun == "L" or bun == "R") and (self.stop_mode == 1): 
                        self.position[0] -= self.polor_coordinate[0] * cos(-radians(self.polor_coordinate[1]))
                    if (bun == "U" or bun == "D") and (self.stop_mode == 1):
                        self.position[1] -= self.polor_coordinate[0] * sin(-radians(self.polor_coordinate[1]))
                    accidents.append(accident)
  
            if (True in accidents) and (self.stop_mode == 0):
                self.start_time -= 30.0
                self.time_lab = time.time() - self.start_time
                print("-------------stop-------------" + "                " + str(self.time_lab) + " Sec")
                self.stop()
                break

        GOALS = []
        for gg in self.GOAL:
            if self.mode == 0: 
                goal, Gbun = self.choumpha.rect_in_point(car_center, gg)
            elif self.mode == 1: 
                goal, Gbun = self.lidar.rect_in_point(car_center, gg)

            GOALS.append(goal)
                
        if (True in GOALS) and (self.goalin==False):
            self.goalin = True
            self.goal_count += 1
            self.time_lab_func()
        elif (not (True in GOALS)) and (self.goalin == True):
            self.goalin = False
            
        if self.goal_count > 3:
            #print("-------------stop-------------" + "                " + str(self.time_aver) + " Sec")
            self.end = True

         
