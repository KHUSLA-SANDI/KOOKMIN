#!/usr/bin/env python
import pygame, time, random, base64
from PIL import Image
from xycar_sim_parking.map import *
from xycar_sim_parking.car import *
from xycar_sim_parking.resource import *
from io import BytesIO

class Game:
    def __init__(self, pygame_on, state_num, distance_device, stop_mode, turn, key_mode, pygame_screen, Font_1=None, Font_2=None):
        self.device_sel = distance_device
        self.key_mode = key_mode
        self.set_random = True
        self.pygame_on = pygame_on
        self.FF = Font_1
        self.DF = Font_2
        self.screen = pygame_screen
        self.car = Car(self.pygame_on, state_num, turn=turn, mode=self.device_sel, screen=self.screen, stop_mode=stop_mode)
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
        #self.logoImg.resize((435, 115))
        self.logoImg  = self.logoImg.resize((217, 57))
        #self.goalImg = pygame.image.load(path)

    def map_input(self, all_sprite_list, wall_list, goal_list):
        self.all_sprite_list = all_sprite_list
        self.wall_list = wall_list
        self.goal_list = goal_list
        
    def game_init(self, epi_num, rst):
        self.time_chk = time.time()

        self.epi_num = epi_num
        self.success = None
        self.exit = False
 
        self.action = 0
        self.angle = 0
        self.start_time = round(time.time(),2)
        self.chk_time = time.time()
        self.car.set_init_car(rst)

    def key_return(self, dt):
        pygame.event.get()
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

    def restart_button(self):
        pygame.event.get()
        pressed = pygame.key.get_pressed()
        #if pressed[pygame.K_SPACE]:
        #   return True
        if pressed[pygame.K_1]:
           return 0
        elif pressed[pygame.K_2]:
           return 1
        elif pressed[pygame.K_3]:
           return 2
        elif pressed[pygame.K_4]:
           return 3
        else:
           return -1

    def run(self, A, S, step, dt):
        #print("print-angle-2", A)
        if not self.key_mode:
            self.car.steering = 0.6 * float(A)
            #print("A:", A, self.car.steering)
            self.car.steering = max(-self.car.max_steering, min(self.car.steering, self.car.max_steering))
            
            S = max(-50.0, min(S, 50.0))
            self.car.Linear_velocity = float(S) / 25.0
        #print("print-angle-3", A)

        if self.pygame_on:
            self.screen.fill([255,255,255])
            self.car.AR.pygame_view(self.screen, self.car.PLine)

            if GOAL_ANGLE == 0:
                goal_rotate = 0

            elif GOAL_ANGLE == 1:
                goal_rotate = 90

            goalimg = self.car.PIL2PygameIMG(self.goalImg, goal_rotate, rgb='RGB', center=(55,110), expand=True)
            self.screen.blit(goalimg, GOAL_PICT)

            if LOGO_ANGLE == 0:
                logo_rotate = 0

            elif LOGO_ANGLE == 1:
                logo_rotate = 90

            logoimg = self.car.PIL2PygameIMG(self.logoImg, logo_rotate, rgb='RGB', center=(108,28), expand=True)
            self.screen.blit(logoimg, LOGO_POSI)

            self.all_sprite_list.update(dt)

        #print("print-angle-4", A)

        self.time_lab = time.time() - self.start_time
        self.car.set_time(self.time_lab)
        self.viewtime = math.trunc(self.time_lab)

        if self.pygame_on:
            self.runtime = self.FF.render("Running time : " + str(self.viewtime), True, (28,0,0))
            self.all_sprite_list.draw(self.screen)
            self.screen.blit(self.runtime,(15,15))

        #print("print-angle-5", A)
            
        self.car.update(dt)

        #print("print-angle-6", A)

        suc_code = 0
        if self.car.end == False:
            suc_code = 0
        elif self.car.end and (self.car.suc == False):
            suc_code = 1
        elif self.car.end and self.car.suc:
            suc_code = 2

        #print("print-angle-7", A)

        if self.device_sel == 0:
            return suc_code, self.car.choumpha_distance, self.car.polor_coordinate[1]
        if self.device_sel == 1:
            return suc_code, self.car.lidar_distance, self.car.polor_coordinate[1]
        if self.device_sel == 2:
            return suc_code, None, self.car.polor_coordinate[1]

        

