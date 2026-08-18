#!/usr/bin/env python
from math import sin, cos, radians, degrees, copysign
from xycar_sim_parking.confirm import *
from xycar_sim_parking.map import *
from xycar_sim_parking.tag import *
from xycar_sim_parking.resource import *
from io import BytesIO
import pygame, os, time, random, base64
from PIL import Image

class Car:
    location = [[-64,  0], [-63,-22], [-63, 22], [-53,-27], [-54, 27],
                [-45,-30], [-21,-30], [  2,-30], [ 24,-30], [ 46,-30],
                [-47, 29], [-24, 30], [  0, 30], [ 25, 30], [ 46, 30],
                [ 54,-23], [ 55, 22], [ 60,-14], [ 60, 14], [ 63,  0]]
    
    def __init__(self, pygame_go, state_num, turn, stop_mode, screen, mode=0):
        self.mode = mode
        self.screen = screen
        self.AR = AR()
        self.AR.set_status(780,497,45)

        if self.mode == 0:
            #self.choumpha_ang = [90, -52, 0, 52, -90]
            self.choumpha_ang = [-90, -45, 0, 45, 90]
            self.choumpha_posi = [[1,-27], [55,-23], [62,-1], [57,20], [4,28]]
            self.choumpha = distance_device()
            self.choumpha_distance = [0.0, 0.0, 0.0, 0.0, 0.0]

        elif self.mode == 1:
            self.lidar_angle = 0
            self.lidar_posi = [62, -1]
            self.lidar = distance_device()
            self.lidar_data_cnt = state_num - 1
            self.lidar_init(self.lidar_data_cnt)
            self.lidar_vector_aver = 0

        elif self.mode == 2:
            self.camera_posi = [40, 0]
            self.camera_angle = 0
            self.camera = distance_device()
            self.arDetect = False
            pass

        self.turn = turn
        self.time_lab = 0

        self.location_cnt = len(self.location)
        
        self.pygame_on = pygame_go
        if self.pygame_on:
            self.stop_mode = stop_mode
            self.car_x_ori = [-64,-64, 64, 64]
            self.car_y_ori = [-32, 32,-32, 32]
            car_bytes = base64.b64decode(car)
            car_file = BytesIO(car_bytes)
            self.colorImage  = Image.open(car_file)
            self.rotated = self.PIL2PygameIMG(self.colorImage, 0)
            self.car_image = pygame.sprite.Sprite()
            self.car_img_x = 0
            self.car_img_y = 0
            self.PLine = "red"
        
    def set_init_car(self, R):
        if self.turn == 0:
            x = 3.5
            y = float(random.randint(70, 160))/100.0
            a = 0.0
        elif self.turn == 1:
            x = float(random.randint(70, 160))/100.0
            y = 4.5
            a = 0.0

        if self.mode == 2:
            #r = random.randint(0, 1)
            #r = 0
            if R==0:
                x = 2.5
                y = 1.0
                a = 0
            elif R==1:
                x = 5.4
                y = 1.0
                a = -90.0
            elif R==2:
                x = 1.3
                y = 3.5
                a = 0.0
            elif R==3:
                x = 6.5
                y = 1.0
                a = -90.0
        
        self.position = [x, y]
        self.polor_coordinate = [0.0, a]

        self.Linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.length = 1.28
        self.steering = 0.0
        self.max_steering = 30.0

        self.end = False
        self.suc = None
        self.goal_count = 0

    def choumpha_add(self, posi, ang):
        self.choumpha_posi.append(posi)
        self.choumpha_ang.append(ang)
        self.choumpha_distance(0.0)

    def tag_detact(self):
        pass
    
    def lidar_init(self, data_cnt):
        self.lidar_distance = []
        for _ in range(data_cnt):
            self.lidar_distance.append(0.0)

    def stop(self, accident):
        self.Linear_velocity = 0.0
        self.steering = 0.0
        self.end = True
        if accident:
            self.suc = False
        else:
            self.suc = True

    def set_time(self, time_stamp):
        self.time_lab = time_stamp

    def PIL2PygameIMG(self, imgData, angle, rgb='RGBA', center=(64, 32), expand=True):
        IMG = imgData.rotate(angle, expand=expand, center=center)
        return pygame.image.fromstring(IMG.tobytes("raw", rgb), IMG.size, rgb)

    def update(self, dt):
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

                if self.pygame_on:
                    pygame.draw.circle(self.screen, (0,0,255), [int(round(choumpha_position[i][0])), int(round(choumpha_position[i][1]))], 5) 

                _, distance, cordinate = self.choumpha.confirm_device(choumpha_angle[i], choumpha_position[i], OBS, 1, min_angle=0, max_angle=0)             

                self.choumpha_distance[i] = distance[0]

                if self.pygame_on:
                    pygame.draw.line(self.screen, [255,0,0], choumpha_position[i], (cordinate[0][0], cordinate[0][1]), 3)
            
            self.choumpha_distance = [self.choumpha_distance[1], self.choumpha_distance[2], self.choumpha_distance[3], 0.0, 0.0, 0.0, self.choumpha_distance[4], self.choumpha_distance[0]]


        elif self.mode == 1:        

            lidar_position = [0.0, 0.0]
            lidar_position[0] = self.lidar_posi[0] * cos(-radians(self.polor_coordinate[1])) - self.lidar_posi[1] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
            lidar_position[1] = self.lidar_posi[0] * sin(-radians(self.polor_coordinate[1])) + self.lidar_posi[1] * cos(-radians(self.polor_coordinate[1])) + car_center[1]

            if self.pygame_on:
                pygame.draw.circle(self.screen, (0,0,255), [int(round(lidar_position[0])), int(round(lidar_position[1]))], 5)

            _, distance, cordinate = self.lidar.confirm_device(-self.polor_coordinate[1], lidar_position, OBS, 180, min_angle=-90, max_angle=90)

            if len(distance) == 181:
                s, sr, cr = self.lidar.vector_aver(-self.polor_coordinate[1], distance, self.lidar_data_cnt, min_angle=0, max_angle=180)
                self.lidar_distance = []
                for i in range(180):
                    self.lidar_distance.append(distance[i])

                if self.pygame_on:
                    for i in range(0,181,45):
                        pygame.draw.line(self.screen, [255,0,0], lidar_position, (cordinate[i][0], cordinate[i][1]))

        elif self.mode == 2:
            camera_position = [0.0, 0.0]
            camera_position[0] = self.camera_posi[0] * cos(-radians(self.polor_coordinate[1])) - self.camera_posi[1] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
            camera_position[1] = self.camera_posi[0] * sin(-radians(self.polor_coordinate[1])) + self.camera_posi[1] * cos(-radians(self.polor_coordinate[1])) + car_center[1]

            if self.pygame_on:
                pygame.draw.circle(self.screen, (0,0,255), [int(round(camera_position[0])), int(round(camera_position[1]))], 5)
            
            _, distance, cordinate = self.camera.confirm_device(-self.polor_coordinate[1], camera_position, OBS, 2, min_angle=-45, max_angle=45)

            if distance == float('inf'):
                return

            if len(distance) == 2:
 
                min_angle = float(-45) + self.polor_coordinate[1]
                max_angle = float(45) + self.polor_coordinate[1]
                tmp = 0.0

                if min_angle > max_angle:
                    tmp = min_angle
                    min_angle = max_angle
                    max_angle = tmp

                C = [self.AR.RP, self.AR.LP, self.AR.MP]
                D = []
                A = []

                OK = 0

                for i in C:
                    angle, distance = self.AR.get_data(camera_position[0], camera_position[1], i[0], i[1])
                    if min_angle < -angle < max_angle:
                        OK += 1
                    D.append(distance)
                    A.append(angle)

                self.arDetect = False
                self.ar_distance = [0.0, 0.0]
                self.ar_angle = 0.0
                self.MD2CAM_ang = 0.0 
                if OK == 3:
                    m1 = -(float(camera_position[0]) - float(C[2][0]))/(float(camera_position[1]) - float(C[2][1]))
                    m2 = (float(camera_position[1]) - float(C[2][1]))/(float(camera_position[0]) - float(C[2][0]))

                    car_target_angle = degrees(math.atan(m2)) - (-self.polor_coordinate[1])

                    m2angle1 = math.atan(m1)
                    m2angle2 = math.radians(-53.0)
                    
                    self.arDetect = True
                    self.ar_distance = [D[2]*sin(radians(car_target_angle)), D[2]*cos(radians(car_target_angle))]
                    self.ar_angle = m2angle2-m2angle1
                    self.MD2CAM_ang = A[2]

                if self.pygame_on:
                    pygame.draw.line(self.screen, [255,0,0], camera_position, (cordinate[0][0], cordinate[0][1]))
                    pygame.draw.line(self.screen, [255,0,0], camera_position, (cordinate[1][0], cordinate[1][1]))


        cresh = []
        for loc in range(self.location_cnt):
            outline = [0.0, 0.0]
        
        #for loc in range(self.location_cnt):
            outline[0] = self.location[loc][0] * cos(-radians(self.polor_coordinate[1])) - self.location[loc][1] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
            outline[1] = self.location[loc][0] * sin(-radians(self.polor_coordinate[1])) + self.location[loc][1] * cos(-radians(self.polor_coordinate[1])) + car_center[1]

            cresh.append(outline)

        self.PLine = "red"
        if self.mode == 2:
            OKGOOD = False
            INOK = 0
            for loc in cresh:
                if self.AR.stopQ(loc[0], loc[1]):
                    self.stop(True)
                    break

                OKGOOD = self.AR.in_rect_detect(loc[0], loc[1]) 
                if OKGOOD:
                    INOK += 1
            if INOK == 20:
                self.PLine = "green"

        for loc in cresh:
            outline = [loc[0], loc[1]]
                
            accidents = []
            for wall in OBS:
                if self.mode == 0: 
                    accident, bun = self.choumpha.rect_in_point(outline, wall)
                if self.mode == 1: 
                    accident, bun = self.lidar.rect_in_point(outline, wall)
                if self.mode == 2:
                    accident, bun = self.camera.rect_in_point(outline, wall)                    
                
                if accident:
                    if (bun == "L" or bun == "R") and (self.stop_mode == 1): 
                        self.position[0] -= self.polor_coordinate[0] * cos(-radians(self.polor_coordinate[1]))
                    if (bun == "U" or bun == "D") and (self.stop_mode == 1):
                        self.position[1] -= self.polor_coordinate[0] * sin(-radians(self.polor_coordinate[1]))
                    accidents.append(accident)
  
            if (True in accidents) and (self.stop_mode == 0):
                print("-------------stop-------------" + "                " + str(self.time_lab) + " Sec")
                self.stop(True)
                break
                
            goals = []
            for aword_point in GOAL:
                if self.mode == 0: 
                    goal, Gbun = self.choumpha.rect_in_point(outline, aword_point)
                if self.mode == 1: 
                    goal, Gbun = self.lidar.rect_in_point(outline, aword_point)
                
                goals.append(goal)
                
            if True in goals:
                print("-------------goal-------------" + "                " + str(self.time_lab) + " Sec")
                self.goal_count += 1
                if self.goal_count == len(GOAL):
                    self.stop(False)
                break
        
