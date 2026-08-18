#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BytesIO
from xycar_sim_driving.resource import *
from xycar_sim_driving import *
from xycar_sim_driving.car import *
import pygame, time, random, base64

class Game:
    def __init__(self, pygame_on, state_num, distance_device, stop_mode, turn, key_mode, INIT, GOAL, LOGO, OBS, WARP, Font_1=None, Font_2=None, pygame_screen=None):
        self.device_sel = distance_device
        self.key_mode = key_mode
        self.set_random = True

        self.goal_x = GOAL[0]
        self.goal_y = GOAL[1]
        self.goal_angle = GOAL[2]

        self.car = Car(pygame_on, state_num, INIT=INIT, OBS=OBS, WARP=WARP, GOAL=GOAL[3], turn=turn, mode=self.device_sel, stop_mode=stop_mode)
        self.pygame_on = pygame_on

        self.logo_x = LOGO[0]
        self.logo_y = LOGO[1]
        self.logo_angle = LOGO[2]

        if self.pygame_on:
            self.FF = Font_1
            self.DF = Font_2
            self.screen = pygame_screen
            self.car = Car(self.pygame_on, state_num, INIT=INIT, OBS=OBS, WARP=WARP, GOAL=GOAL[3], turn=turn, mode=self.device_sel, stop_mode=stop_mode, screen=self.screen)
            self.stop_space = 0
        
        self.steering_AOC = 100 * 2

        self.start_time = time.time()
        self.switch = True

        goal_bytes = base64.b64decode(goal)
        goal_file = BytesIO(goal_bytes)

        logo_bytes = base64.b64decode(logo)
        logo_file = BytesIO(logo_bytes)

        self.goalImg  = Image.open(goal_file)
        self.logoImg  = Image.open(logo_file)
        self.logoImg  = self.logoImg.resize((217, 57))

    def change_screen(self, scr):
        self.screen = scr
        self.car.change_screen(scr)

    def map_input(self, all_sprite_list, wall_list, goal_list):
        self.all_sprite_list = all_sprite_list
        self.wall_list = wall_list
        self.goal_list = goal_list
        
    def game_init(self, epi_num):
        self.time_chk = time.time()

        self.epi_num = epi_num
        self.success = None
        self.exit = False
 
        self.action = 0
        self.angle = 0
        self.car.start_time = round(time.time(),2)
        self.chk_time = time.time()
        self.car.set_init_car(True)

    def key_return(self, dt):
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            self.car.Linear_velocity += 0.1
        elif pressed[pygame.K_DOWN]:
            self.car.Linear_velocity -= 0.1
        if pressed[pygame.K_RIGHT]:
            self.car.steering -= self.steering_AOC * dt
        elif pressed[pygame.K_LEFT]:
            self.car.steering += self.steering_AOC * dt
        else:
            self.car.steering = 0
        self.car.steering = max(-self.car.max_steering, min(self.car.steering, self.car.max_steering))

    def key_space_bar(self):
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_SPACE] and (self.car.goal_count == 3):
            self.car.end = True

    def run(self, A, S, step, dt):
        if not self.key_mode:
            self.car.steering = -0.6 * float(A)
            self.car.steering = max(-self.car.max_steering, min(self.car.steering, self.car.max_steering))
            
            S = max(-50.0, min(S, 50.0))
            self.car.Linear_velocity = float(S) / 25.0

        if self.pygame_on:
            self.screen.fill([255,255,255])

            if self.goal_angle == 0:
                goal_rotate = 0

            elif self.goal_angle == 1:
                goal_rotate = 90

            goalimg = self.car.PIL2PygameIMG(self.goalImg, goal_rotate, rgb='RGB', center=(55,110), expand=True)
            self.screen.blit(goalimg, [self.goal_x, self.goal_y])

            if self.logo_angle == 0:
                logo_rotate = 0

            elif self.logo_angle == 1:
                logo_rotate = 90

            logoimg = self.car.PIL2PygameIMG(self.logoImg, logo_rotate, rgb='RGB', center=(108,28), expand=True)
            self.screen.blit(logoimg, [self.logo_x, self.logo_y])

            self.all_sprite_list.update(dt)

        self.car.update(dt)

        
        if self.car.goal_count < 3:
            self.car.time_lab = time.time() - self.car.start_time
    
        self.viewtime = math.trunc(self.car.time_lab)

        if self.pygame_on:
            self.runtime = self.FF.render("Running time : " + str(self.viewtime), True, (28,0,0))
            self.car.time_lab_func()
            self.first = self.DF.render("1 : " + str(self.car.first), True, (28,0,0))
            self.second = self.DF.render("2 : " + str(self.car.second), True, (28,0,0))
            self.third = self.DF.render("3 : " + str(self.car.third), True, (28,0,0))

            self.all_sprite_list.draw(self.screen)
            self.screen.blit(self.runtime,(15,15))
            self.screen.blit(self.first,(15,40))
            self.screen.blit(self.second,(15,55))
            self.screen.blit(self.third,(15,70))

        if self.device_sel == 0:
            return self.car.end, self.car.choumpha_distance, self.car.polor_coordinate[1]
        if self.device_sel == 1:
            return self.car.end,self.car.lidar_distance, self.car.polor_coordinate[1]

        

