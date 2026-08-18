#!/usr/bin/env python

import pygame, math, base64
from io import BytesIO
from PIL import Image
from xycar_sim_parking.map import *
from xycar_sim_parking.resource import *
from tf_transformations import quaternion_from_euler, euler_from_quaternion

class AR:
    def __init__(self):
        self.position = [0, 0]
        self.incline = 0.0
        self.camera_VA = 130.0 # view angle
        blackPanel_bytes = base64.b64decode(blackPanel)
        blackPanel_file = BytesIO(blackPanel_bytes)
        self.ARImg = Image.open(blackPanel_file)
        self.sizeX = 250
        self.sizeY = 30
        self.ARImg  = self.ARImg.resize((self.sizeX, self.sizeY))
        self.rotated = self.PIL2PygameIMG(self.ARImg, 53)

    def set_status(self, x, y, incline):
        self.car_length = 128
        self.car_width = 64
        self.alpha = 30
        self.beta = 50
        
        self.position = [x, y]
        self.position = [x+60, y+60]
        self.middle = []
        self.incline = incline

        self.RP = [SCREEN_WIDTH-10, y+62]

        self.LP = [x+95, SCREEN_HEIGHT-10]
        self.MP = [int((self.RP[0]+self.LP[0])/2),int((self.RP[1]+self.LP[1])/2)]

        xR, yR = self.calc(self.MP[0], self.MP[1], math.tan(math.radians(53+75)), int(self.car_width/2 + self.alpha))
        self.RP = [int(round(xR)), int(round(yR))]
        xL, yL = self.calc(self.MP[0], self.MP[1], math.tan(math.radians(53+75)), int(-self.car_width/2 - self.alpha))
        self.LP = [int(round(xL)), int(round(yL))]

        self.ar_m = (self.RP[1]-self.LP[1])/(self.RP[0]-self.LP[0])
        self.ar_b = self.RP[1] - self.ar_m*self.RP[0]

        self.x1, self.y1 = self.calc(self.RP[0],self.RP[1], 0.72, -(self.alpha))
        self.x2, self.y2 = self.calc(self.LP[0],self.LP[1], 0.72, -(self.alpha))
        self.x3, self.y3 = self.calc(self.RP[0],self.RP[1], 0.72, -(self.beta+self.car_length))
        self.x4, self.y4 = self.calc(self.LP[0],self.LP[1], 0.72, -(self.beta+self.car_length))

        self.x1 = int(round(self.x1))
        self.x2 = int(round(self.x2))
        self.x3 = int(round(self.x3))
        self.x4 = int(round(self.x4))
        self.y1 = int(round(self.y1))
        self.y2 = int(round(self.y2))
        self.y3 = int(round(self.y3))
        self.y4 = int(round(self.y4))

    def get_data(self, camX, camY, objX, objY):
        distance = (((objX-camX)**2) + ((objY-camY)**2)) ** 0.5
        return math.degrees(math.acos(float(abs(camX - objX))/float(distance))), distance

    def get_quaternion(self, angle):
        roll = 0.0
        pitch = 0.0
        yaw = angle
        return quaternion_from_euler(roll, pitch, yaw)

    def PIL2PygameIMG(self, imgData, angle):
        IMG = imgData.rotate(angle, expand=True, fillcolor='white', center=(int(self.sizeX/2), int(self.sizeY/2)))
        return pygame.image.fromstring(IMG.tobytes("raw", 'RGB'), IMG.size, 'RGB')

    def calc(self, poX, poY, m, length):
        b = poY - poX * m
        
        x = length/((m*m+1)**(0.5)) + poX
        y = m*x + b

        return x, y

    def stopQ(self, x, y):
        A = self.ar_m*x + self.ar_b
        if A < y:
            return True
        return False

    def makeMB(self, x1, y1, x2, y2):
        m = (float(y2)-float(y1))/(float(x2)-float(x1))
        b = float(y1) - float(x1)*m
        return m, b

    def in_rect_detect(self, x, y):
        OK = 0

        m1, b1 = self.makeMB(self.x1, self.y1, self.x3, self.y3)
        if ((m1 * x + b1) < y):
            OK += 1 

        m2, b2 = self.makeMB(self.x2, self.y2,self. x4, self.y4)
        if ((m2 * x + b2) > y):
            OK += 1

        m3, b3 = self.makeMB(self.x1, self.y1, self.x2, self.y2)
        if ((m3 * x + b3) > y):
            OK += 1

        m4, b4 = self.makeMB(self.x3, self.y3, self.x4, self.y4)
        if ((m4 * x + b4) < y):
            OK += 1

        if OK == 4:
            return True

        return False


    def pygame_view(self, screen, sel_color):
        color = {"red":[255,0,0], "green":[0,255,0]}

        screen.blit(self.rotated, self.position)
        pygame.draw.line(screen, [0,255,0], self.RP, self.LP, 5)

        pygame.draw.line(screen, color[sel_color], (self.x1,self.y1), (self.x3,self.y3), 5)
        pygame.draw.line(screen, color[sel_color], (self.x2,self.y2), (self.x4,self.y4), 5)
        pygame.draw.line(screen, color[sel_color], (self.x1,self.y1), (self.x2,self.y2), 5)
        pygame.draw.line(screen, color[sel_color], (self.x3,self.y3), (self.x4,self.y4), 5)


