#!/usr/bin/env python
#-- coding:utf-8 --

#=============================================
# parking_3_Polyfit_Pure_Pursuit.py

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import pygame
import numpy as np
import math
import rospy
import rospkg
from xycar_msgs.msg import xycar_motor
import time
#=============================================
# 모터 토픽을 발행할 것임을 선언
#============================================= 
motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
xycar_msg = xycar_motor()

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#============================================= 
rx, ry = [300, 350, 400, 450], [300, 350, 400, 450]
CHANGE = False #방향이 전환되는 여부: 후진에서 전진으로 바뀌는 경우 CHANGE = True

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
AR = (1142, 62) # AR 태그의 위치
P_ENTRY = (1036, 162) # 주차라인 진입 시점의 좌표

P_END = (1129, 69) # 주차라인 끝의 좌표
GAP = 30 # 도착지점 앞부분 길이 설정에 필요한 gap

X_MAX = 1036 # 도착지점 앞부분의 마지막 x 좌표
X_MIN = X_MAX - GAP # 도착지점 앞부분의 처음 x좌표

Y_MIN = 162 #도착지점 앞부분의 처음 y 좌표
Y_MAX = Y_MIN +GAP # 도착지점 앞부분의 마지막 y 좌표

HALF_CAR = 64 # 자동차의 절반 길이
#=============================================
# 모터 토픽을 발행하는 함수
# 입력으로 받은 angle과 speed 값을
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):
    xycar_msg.angle = int(angle)
    xycar_msg.speed = int(speed)
    motor_pub.publish(xycar_msg)

#차량의 머리 좌표 계산 함수
def head_point(x, y, yaw):
    x = x + HALF_CAR * np.cos(np.radians(yaw))
    y = y - HALF_CAR * np.sin(np.radians(yaw))
    return x, y

#차량의 꼬리 좌표 계산 함수
def back_point(x, y, yaw):
    x = x - HALF_CAR * np.cos(np.radians(yaw))
    y = y + HALF_CAR * np.sin(np.radians(yaw))
    return x, y

#=============================================
# 주차 구역과의 거리가 먼 경우 경로의 궤적을 생성하는 함수
# 현재 좌표값과 deg(차수) 값을 받아 주차구역으로 향하는
# deg 차 함수로 이루어진 궤적 형성
#=============================================
def get_approximate_traj(sx, sy, deg):
    ''' [Tool] 근사 궤적을 구하는 함수 '''
    # 도착 지점입구 부분에 기울기가 1인 포인트 리스트 생성
    x = list(range(X_MAX, X_MIN, -1))
    y = list(range(Y_MIN, Y_MAX, 1))
    
    #현재 좌표 리스트화
    start_x = [sx]
    start_y = [sy]

    #현재 좌표와 생성한 도착 지점 리스트 결합
    x_point = start_x + x
    y_point = start_y + y

    #결합한 포인트들을 가지고 deg차수의 함수 생성
    args = np.polyfit(x_point, y_point, deg)
    func = np.poly1d(args)

    #현재 좌표에서부터 도착 지점 부근 경로까지 리스트 생성 후 return
    if sx < X_MIN:
        X = np.arange(sx, X_MIN)
        Y = func(X)

    else:
        X = np.arange(X_MIN,sx)
        Y = func(X)

    return X, Y

#=============================================
# 주차 구역과의 거리가 가까운 경우 경로의 궤적을 생성하는 함수
# 현재 좌표값과 deg(차수) 값을 받아 주차구역으로 향하는
# deg 차 함수로 이루어진 궤적 형성
#=============================================
def get_approximate_traj2(sx, sy, deg):
    ''' [Tool] 근사 궤적을 구하는 함수 '''

    #추자 구역에서 일정거리 떨거진 점의 좌표 리스트화
    x = [X_MIN-300]
    y = [Y_MAX+300]
    
    #현재 좌표 리스트화
    start_x = [sx]
    start_y = [sy]

    #현재 좌표와 생성한 도착 지점 리스트 결합
    x_point = start_x + x
    y_point = start_y + y

    #결합한 포인트들을 가지고 deg차수의 함수 생성
    args = np.polyfit(x_point,y_point, deg)
    func = np.poly1d(args)

    #현재 좌표에서부터 주차 구역에서 일정거리 떨어진 점까지 리스트 생성 후 return
    if sx < X_MIN-300:
        X = np.arange(X_MIN-300, sx, -1)
        Y = func(X)

    else:
        X = np.arange(sx, X_MIN-300, -1)
        Y = func(X)

    return X, Y

#=============================================
# 경로를 생성하는 함수
# 차량의 시작위치 sx, sy, 시작각도 syaw
# 최대가속도 max_acceleration, 단위시간 dt 를 전달받고
# 경로를 리스트를 생성하여 반환한다.
#=============================================
def planning(sx, sy, syaw, max_acceleration, dt):
    global rx, ry, CHANGE #경로 좌표, 방향전환 여부
    print("Start Planning")

    CHANGE = False #hit_point_num 초기화

    #시작 순간의 현재 좌표를 차량의 head point로 전환
    sx = sx - HALF_CAR * np.sin(np.radians(syaw))
    sy = sy + HALF_CAR * np.cos(np.radians(syaw))

    #경로 생성부분
    if (X_MIN - sx) >30: #주차 구역과 시작점의 거리가 먼 경우
        rx, ry = get_approximate_traj(sx, sy, 2) # 출발지점부터 주차 지점까지 2차 궤적 형성 후 경로로 저장
        x = list(range(X_MIN, 1129 ,1)) # 도착 지점 일정구간 앞에서부터는 직진구간 (x좌표)
        y = list(range(Y_MAX,69, -1)) # (y좌표)

        #전체 경로 합성
        rx = list(rx) + x 
        ry = list(ry) + y

    else: #주차 구역과 시작점의 거리가 가까운 경우
        rx, ry = get_approximate_traj2(sx, sy, 2) #궤적 생성
        x = list(range(X_MIN-300, 1129 ,1)) # 도착 지점에서 추자 구역에서 일정거리 떨어진 점까지 직진구간 (x좌표)
        y = list(range(Y_MAX+300, 69, -1)) # (y좌표)

        #전체 경로 합성
        rx = list(rx) + x 
        ry = list(ry) + y

    return rx, ry

#=============================================
# 생성된 경로를 따라가는 함수
# 파이게임 screen, 현재위치 x,y 현재각도, yaw
# 현재속도 velocity, 최대가속도 max_acceleration 단위시간 dt 를 전달받고
# 각도와 속도를 결정하여 주행한다.
#=============================================

def tracking(screen, x, y, yaw, velocity, max_acceleration, dt):
    global rx, ry, CHANGE

    is_look_forward_point = False #lfd 비활성
    x_h, y_h = head_point(x, y, yaw) #차량의 head_point 좌표
    x, y = back_point(x, y, yaw) #차량의 back_point 좌표

    #속도, 조향각 초기화
    speed = 50
    angle = 0

    FORWARD = True #전진 활성화(초기화)
    lfd = 200 #look forward distance: purepursuit 알고리즘에서 전방주시거리
    L = 2 * HALF_CAR #차량의 길이
    
    ######################################
    #           purepursuit              #
    ######################################
    for i in range(len(rx)-1): #path 포인트 순회
        #현재 순회하는 포인트
        path_point_x = rx[i]
        path_point_y = ry[i]

        next_point_x = rx[i + 1] #다음 포인트의 x 좌표
        point_x_gap = next_point_x - path_point_x #현재 순회하는 포인트와 다음 포인트의 x 좌표 차이
        
        #차량의 후방 좌표와 경로 포인트의 x, y 좌표 차이 
        dx = path_point_x - x
        dy = path_point_y - y

        #purepursuit 알고리즘의 조향각 계산을 위한 수치 계산
        arpha = math.atan2(dy, dx) 
        yaw_radian = math.radians(yaw)
        theta = (math.pi/2) - (arpha + yaw_radian)

        rotated_x = math.sin(theta) * lfd #자동차기준 전방을 x 축으로 하였을때 포인트와의 x거리 (차량 전방기준 포인트 위치)
        dis = math.sqrt(pow(dx, 2) + pow(dy, 2)) #차량의 후방에서부터 path 까지의 거리
        
        #lfd 선정하는 부분
        if CHANGE is False: #방향전환 되지 않았을 때
            if point_x_gap < 0: #point 의 x 좌표가 줄어드는 경우
                FORWARD = False #후진

                if rotated_x < 0: #차량보다 뒤에 있는 점들만 순회
                    if dis >= lfd : #dis 가 lfd 보다 큰 경우 break
                        is_look_forward_point = True #lfd 활성화
                        break

            else: #point 의 x 좌표가 증가하는 경우
                if abs(math.sin(theta) * dis) <= 5: #차량의 후방과 point의 거리가 5 미만이 됐을때 전진으로 변경
                    FORWARD = True #전진

                if FORWARD == False: #후진하는 동안 차량이 포인트에 접근하기 전까지는 포인트 유지
                    break

                if rotated_x > 0: #차량보다 앞에 있는 점들만 순회
                    if dis >= lfd : #dis 가 lfd 보다 큰 경우 break
                        is_look_forward_point = True #lfd 활성화
                        CHANGE = True #방향전환 활성화
                        break

        else: #방향전환 되었을 때
            if point_x_gap > 0: #point 의 x 좌표가 증가하는 경우
                if rotated_x > 0: #차량보다 앞에 있는 점들만 순회
                    if dis >= lfd : #dis 가 lfd 보다 큰 경우 break
                        is_look_forward_point = True #lfd 활성화
                        break

    pygame.draw.line(screen, (255,0,0), (x+np.cos(arpha)*lfd, y+np.sin(arpha)*lfd), (x,y), 1) #차량에서 point 까지 연결하는 직선 출력(빨간선)
    pygame.draw.line(screen, (0,0,255), (x+dx, y), (x,y), 1) #차량에서 부터 dx 표시(파란선)
    pygame.draw.line(screen, (0,0,255), (x+dx, y), (x+dx,y+dy), 1) #차량에서부터 dy 표시 (파란선)

    distance = math.sqrt(pow(1129 - x_h, 2) + pow(69 - y_h , 2)) #차량의 head에서부터 주차지점의 end 값까지 거리

    if is_look_forward_point : #lfd 가 활성화 되었을때
        steering=math.atan2((4* math.cos(theta)*L),lfd)#조향각 계산
        angle = int(math.degrees(steering)) #조향각을 angle로 할당

        #조향각이 50보다 큰경우 50으로 고정
        if angle >= 50:
            angle = 50

        #조향각이 -50보다 작은경우 -50으로 고정
        elif angle <= -50:
            angle = -50

        #후진이 활성화 돼있으면 speed -50
        if FORWARD == False:
            speed = -50

        drive(angle, speed)

    else : #목표지점과 거리가 3이하가 되었을때 정지
        if distance <= 3:
            angle = 0
            speed = 0
    
            drive(angle, speed)
    
    beta = math.radians(-angle+yaw) #조향각을 차량 기준으로 전환

    pygame.draw.line(screen, (0,255,0), (x_h+np.cos(beta)*lfd, y_h-np.sin(beta)*lfd), (x_h,y_h), 3) #차량 기준으로 조향각 출력 (초록선)
    pygame.draw.circle(screen, (255,0,0), (int(x), int(y)), lfd, 1) #차량 기준으로 lfd 범위 출력 (빨간 원)
    pygame.draw.circle(screen, (0,0,255), (int(path_point_x), int(path_point_y)), 5, 5) #목표지점 (lfd 포인트) 표시(파란점)
    pygame.draw.circle(screen, (0,0,255), (int(x), int(y)), 10, 10) #차량후방에 점 출력(파란 점)