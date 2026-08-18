#!/usr/bin/env python
import math, time
import numpy as np

class distance_device:

  def vector_aver(self, theta, distance, beam_cnt, min_angle=-90, max_angle=90):
    s = 0.0
    c = 0.0

    cnt = 0

    #print(distance)

    S = []
    C = []

    for i in range(min_angle, max_angle+1, int((abs(min_angle) + abs(max_angle))/(beam_cnt-1))): 
      S.append(distance[cnt] * math.sin(math.radians(i)))
      C.append(distance[cnt] * math.cos(math.radians(i)))
      cnt += 1

    #print(S, sum(S)/5.0, C, sum(C)/5.0)

    s = sum(S) / float(beam_cnt)
    c = sum(C) / float(beam_cnt)

    #print(C, sum(C), c)

    sr = math.asin(s/(((s**2) + (c**2)) ** (0.5)))
    cr = math.acos(c/(((s**2) + (c**2)) ** (0.5)))

    return s, sr, cr

  def confirm_device(self, theta, point, OBS, beam_cnt, min_angle=-90, max_angle=90):
    dts = []
    loc = []

    mn = int(round(min_angle))
    mx = int(round(max_angle + 1))

    if beam_cnt == 1:
        period = 1
    else:
        period = int(round((abs(min_angle) + abs(max_angle))/(beam_cnt-1)))
    
    for i in range(mn, mx, period): 
      T = float(i) - theta
      ok, dt, lo = self.line_to_sprite_group([point[0], point[1]], OBS, T)
      if not ok:
        continue
      dts.append(dt)
      loc.append(lo)
    
    #dts.reverse()
    #loc.reverse()

    if len(dts) == 0:
      return False, float('inf'), [float('inf'), float('inf')]
    return True, dts, loc
    
    
  def line_to_sprite_group(self, point, OBS, theta):
    distance = []
    location = []
    for wall_num in range(len(OBS)):
      OK, dist, loca = self.line_to_rect(point, theta, OBS[wall_num])
      if OK:
        distance.append(dist)
        location.append(loca)

    if len(distance) == 0:
      return False, float('inf'), [float('inf'), float('inf')]

    min_distance = min(distance)
    min_index = distance.index(min_distance)

    return True, min_distance, location[min_index]
    
    
  def line_to_rect(self, point, theta, wall):
    m = math.tan(math.radians(-theta))

    rectLine_x = [
        wall[0],
        wall[0] + wall[2]
    ]

    rectLine_y = [
        wall[1],
        wall[1] + wall[3]
    ]

    cross = []
    distance = []

    bx = float(point[1])-(m*float(point[0]))
  
    for rl in rectLine_x:
      sol = float(m*float(rl) + bx)

      if sol > rectLine_y[1]:
        continue
      if sol < rectLine_y[0]:
        continue

      croc = [rl, sol]
      dit = self.distance_calc(point, croc)


      if abs(float(sol) - (dit*math.sin(math.radians(-theta)) + point[1])) > 0.01:
        continue
      if abs(float(rl) - (dit*math.cos(math.radians(-theta)) + point[0])) > 0.01:
        continue

      cross.append(croc)
      distance.append(self.distance_calc(point, croc))

    by = -1 * float('inf')
    if m != 0:
      by = float(point[0])-(float(point[1])/m)

    for rl in rectLine_y:
      sol = float('inf')
      if m != 0:
        sol = float(float(rl)/m + by)

      if sol > rectLine_x[1]:
        continue
      if sol < rectLine_x[0]:
        continue

      croc = [sol, rl]
      dit = self.distance_calc(point, croc)

      if abs(float(sol) - (dit*math.cos(math.radians(-theta)) + point[0])) > 0.01:
        continue
      if abs(float(rl) - (dit*math.sin(math.radians(-theta)) + point[1])) > 0.01:
        continue

      cross.append(croc)
      distance.append(self.distance_calc(point, croc))

    if len(distance) == 0:
      return False, -1, [-1, -1]

    min_distance = min(distance)
    min_index = distance.index(min_distance)

    return True, min_distance, cross[min_index]
    
    
  def distance_calc(self, point_1, point_2):
    if float('inf') in point_1:
        return float('inf')
    if float('inf') in point_2:
        return float('inf')
    return (((point_1[0] - point_2[0])**2) + ((point_1[1] - point_2[1])**2)) ** (0.5)
    
  def rect_in_point(self, point, wall):
    a = {"L":abs(point[0] - wall[0]), "R":abs(point[0] - (wall[0]+wall[2])), "U":abs(point[1] - wall[1]), "D":abs(point[1] - (wall[1]+wall[3]))}
    b = min(a.values())
    rtn = ""
    for key, value in a.items():
      if value == b:
          rtn = key
          break
    if (wall[0] <= point[0] <= (wall[0]+wall[2])) and (wall[1] <= point[1] <= (wall[1]+wall[3])):
      return True, rtn
    return False, ""
