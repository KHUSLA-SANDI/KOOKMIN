#!/usr/bin/env python
import pygame
from xycar_sim_parking.map import *

class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([width,height])
        self.image.fill([0,0,0])

        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x

class obs_make:
  all_sprite_list = pygame.sprite.Group()
  wall_list = pygame.sprite.Group()
  goal_list = pygame.sprite.Group()
  
  def __init__(self):
    for i in OBS:
      wall = Wall(i[0],i[1],i[2],i[3])
      self.wall_list.add(wall)
      self.all_sprite_list.add(wall)

    for i in GOAL:
      wall = Wall(i[0],i[1],i[2],i[3])
      wall.image.fill([255,255,255])
      self.goal_list.add(wall)
      self.all_sprite_list.add(wall)

  def get_list(self): 
    return [self.wall_list, self.goal_list]

  def get_all_list(self):
    return self.all_sprite_list

  def get_wh(self):
    return SCREEN_WIDTH, SCREEN_HEIGHT
