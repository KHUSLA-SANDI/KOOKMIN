#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame, math, sys

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
  warp_list = pygame.sprite.Group()
  
  def __init__(self, name):
    self.import_map(name)
    self.map_set()

  def import_map(self, name):
    map_name = 'xycar_sim_driving.' +name
    self.racing_name = name
    self.racing_map = __import__('%s' %(map_name), globals(), locals())
    self.racing_map = getattr(self.racing_map, name)

    if not (("LOGO_POSI" in dir(self.racing_map)) and ("LOGO_ANGLE" in dir(self.racing_map))):
        print("맵 파일 안에 LOGO_POSI, LOGO_ANGLE 변수를 반드시 넣어야 합니다.")
        sys.exit()

    self.logo_x = self.racing_map.LOGO_POSI[0]
    self.logo_y = self.racing_map.LOGO_POSI[1]
    self.logo_a = self.racing_map.LOGO_ANGLE

    logo_orix = [-217, 217, -217, 217]
    logo_oriy = [-57, -57, 57, 57]

    logo_xs = [0,0,0,0]
    logo_ys = [0,0,0,0]

    for i in range(4):
        logo_xs[i] = logo_orix[i] * math.cos(-math.radians(self.logo_a)) - logo_oriy[i] * math.sin(-math.radians(self.logo_a))
        logo_ys[i] = logo_orix[i] * math.sin(-math.radians(self.logo_a)) + logo_oriy[i] * math.cos(-math.radians(self.logo_a))

    w = int(round(max(logo_xs))) - int(round(min(logo_xs)))
    h = int(round(max(logo_ys))) - int(round(min(logo_ys)))

    if not ((0 < self.logo_x < (self.racing_map.SCREEN_WIDTH - w)) and (0 < self.logo_y < (self.racing_map.SCREEN_HEIGHT - h))):
        print("LOGO는 반드시 게임 화면 안에 위치해야 합니다.")
        sys.exit()

  def map_set(self):
    for i in self.racing_map.OBS:
      wall = Wall(i[0],i[1],i[2],i[3])
      self.wall_list.add(wall)
      self.all_sprite_list.add(wall)

    for i in self.racing_map.GOAL:
      wall = Wall(i[0],i[1],i[2],i[3])
      wall.image.fill([255,255,255])
      self.goal_list.add(wall)
      self.all_sprite_list.add(wall)

  def get_list(self): 
    return [self.wall_list, self.goal_list]

  def get_all_list(self):
    return self.all_sprite_list

  def get_wh(self):
    return self.racing_map.SCREEN_WIDTH, self.racing_map.SCREEN_HEIGHT

  def get_obs(self):
    return self.racing_map.OBS

  def get_xyt(self):
    return self.racing_map.INIT_POSITION[0], self.racing_map.INIT_POSITION[1], self.racing_map.INIT_ANGLE

  def get_logo(self):
    return self.racing_map.LOGO_POSI[0], self.racing_map.LOGO_POSI[1], self.racing_map.LOGO_ANGLE

  def get_goal(self):
    return self.racing_map.GOAL_PICT[0], self.racing_map.GOAL_PICT[1], self.racing_map.GOAL_ANGLE, self.racing_map.GOAL

  def get_warp(self):
    return self.racing_map.WARP, self.racing_map.WARP_ANGLE
