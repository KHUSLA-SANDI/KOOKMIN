#!/usr/bin/env python
#-- coding:utf-8 --

#=============================================
# parking_2_Bezier_Pure_Pursuit.py

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import pygame
import numpy as np
import math
import rospy
from xycar_msgs.msg import xycar_motor

#=============================================
# 모터 토픽을 발행할 것임을 선언
#============================================= 
motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
xycar_msg = xycar_motor()

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#============================================= 

# 타겟팅 좌표 리스트
rx, ry = [], []
# 현재 타겟팅 좌표에서의 차량 이동 속도 저장 리스트
rspeed = []
# 현재 타겟팅 좌표 인덱스를 저장하는 변수
idx = 0

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
AR = (1142, 62) # AR 태그의 위치
P_ENTRY = (1036, 162) # 주차라인 진입 시점의 좌표
P_END = (1129, 69) # 주차라인 끝의 좌표

#=============================================
# 사용자 정의 함수 선언부
#=============================================

# 두 점을 입력 받아 m:n으로 내분해주는 함수
def introversion(xy1, xy2, m, n):
    # 1번 좌표 언팩
    x1, y1 = xy1
    # 2번 좌표 언팩
    x2, y2 = xy2
    # x 좌표에 대한 내분
    result_x = (n * x1 + m * x2)/(m + n)
    # y 좌표에 대한 내분
    result_y = (n * y1 + m * y2)/(m + n)
    # 내분 결과 좌표 패킹
    result_xy = result_x, result_y
    # 내분 결과 반환
    return result_xy

# 2차 베지에 곡선을 만들어 주는 함수(입력: 1번 좌표, 2번 좌표, 3번 좌표, 곡선 샘플링 개수)
def quadraticBezier(xy1, xy2, xy3, n_samples):
    # 샘플링 좌표의 x 좌표를 저장하는 리스트
    x_list = []
    # 샘플링 좌표의 y 좌표를 저장하는 리스트
    y_list = []
    # 샘플링 개수만큼 점을 만들어주기 위해 반복문 수행
    for i in range(n_samples):
        # 첫번째 선분 내분점
        p1 = introversion(xy1, xy2, i, n_samples - i)
        # 두번째 선분 내분점
        p2 = introversion(xy2, xy3, i, n_samples - i)
        # 위의 두 내분점에 대한 내분점
        p_x, p_y = introversion(p1, p2, i, n_samples - i)
        # 결과 저장
        x_list.append(p_x)
        y_list.append(p_y)
    # 2차 베지에 곡선 생성 결과 반환
    return x_list, y_list

# 베지에 곡선을 rx, ry변수에 바로 이어 붙여주는 함수
def addQuadraticBezier(xy1, xy2, xy3, n_samples):
    # 2차 베지에 곡선 생성
    x_list, y_list = quadraticBezier(xy1, xy2, xy3, n_samples)
    # 현재 rx, ry 뒤에 이어 붙이기
    rx.extend(x_list)
    ry.extend(y_list)

# 각도에 대한 n의 크기를 가지는 벡터 반환(차량 출발 각도용)
def deg2unitVec(deg, n):
    # 라디안으로 변환
    rad = math.radians(deg % 360 - 180)
    # x 좌표 계산
    vec_x = n * math.sin(rad)
    # y 좌표 계산
    vec_y = -n * math.cos(rad)
    # 계산 결과 반환
    return vec_x, vec_y

# 시작점과 끝점으로 부터 각도 추출 후 반환
def vec2deg(xy1, xy2):
    # 벡터 계산
    x1, y1 = xy1
    x2, y2 = xy2
    vec_x, vec_y = x2 - x1, y2 - y1
    # 라디안으로 변환
    rad = math.atan(-vec_y/vec_x)
    # 각도로 변환하여 반환
    return np.degrees(rad)
    

# 각도에 대한 n의 크기를 가지는 벡터 반환(차량용)
def getVec(yaw, n):
    # 각도를 라디안으로 변환
    rad = math.radians(yaw)
    # 크기 n의 벡터 생성
    vec_x = n * math.cos(rad)
    vec_y = -n * math.sin(rad)
    # 결과 반환
    return vec_x, vec_y

# 두 좌표 입력에 대한 유클리드 거리 반환
def distance(xy1, xy2):
    # 좌표1 언팩
    x1, y1 = xy1
    # 좌표2 언팩
    x2, y2 = xy2
    # 거리 계산 후 반환
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

# Pure Pursuit 기법으로 조향각 계산 후 반환하는 함수
def getSteer(target_xy, now_xy, yaw):
    # 차량의 전륜과 후륜까지의 길이
    L = 84
    # 목표 좌표까지의 거리
    d = distance(target_xy, now_xy)
    # 차량의 정면 중앙 좌표
    dx, dy = getVec(yaw, 64)
    front_xy = now_xy[0] + dx, now_xy[1] + dy
    # 좌측 센서
    dx, dy = getVec(yaw - 30, 64)
    l_xy = now_xy[0] + dx, now_xy[1] + dy
    # 우측 센서
    dx, dy = getVec(yaw + 30, 64)
    r_xy = now_xy[0] + dx, now_xy[1] + dy

    # 좌측 센서와 타겟 좌표간의 거리
    l_dist = distance(target_xy, l_xy)
    # 우측 센서와 타겟 좌표간의 거리
    r_dist = distance(target_xy, r_xy)
    
    # 조향 방향 결정
    direct = -1
    # 좌측 센서가 더 가깝다면 핸들을 오른쪽으로 조향
    if l_dist < r_dist:
        direct = 1
    # 차량과 타겟 점간의 차량 진행방향의 수직 방향 거리를 타겟 좌표와 차량 정면 좌표의 거리로 근사
    x = distance(target_xy, front_xy)

    # Pure Pursuit 공식 대입하여 조향각 계산
    steer = math.atan(2 * x * L / d**2) * (180 / math.pi) * direct

    # 만약 조향각이 최대 조향각보다 크면 최대 조향각 이내로 제한
    if abs(steer) > 50:
       steer = steer * 50 / abs(steer) 
    # 조향값 반환
    return steer

#=============================================
# 모터 토픽을 발행하는 함수
# 입력으로 받은 angle과 speed 값을
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):
    xycar_msg.angle = int(angle)
    xycar_msg.speed = int(speed)
    motor_pub.publish(xycar_msg)

#=============================================
# 경로를 생성하는 함수
# 차량의 시작위치 sx, sy, 시작각도 syaw
# 최대가속도 max_acceleration, 단위시간 dt 를 전달받고
# 경로를 리스트를 생성하여 반환한다.
#=============================================
def planning(sx, sy, syaw, max_acceleration, dt):
    global rx, ry
    global rspeed
    global idx

    # 타겟 인덱스 초기화
    idx = 0

    rx = []
    ry = []
    rspeed = []
    # 주차 완료 각도
    eyaw = vec2deg(P_ENTRY, P_END)
    # 주차라인 진입 시점의 좌표
    ex, ey = P_ENTRY
    # 2차 베지에 곡선 샘플링 개수
    n_samples = 100
    # 경로 설정 시작
    print("Start Planning")

    # 차량의 중심이 아닌 정면 기준으로 계산하기 위해 차량 정면 시작 위치로 변환
    # 주차 방향에 대한 64크기의 벡터 구하기
    dx, dy = deg2unitVec(syaw, 64)
    # 시작 위치 변환
    sx += dx
    sy += dy
    
    # 후진 주차 필요
    if sx > ex:
        # 후진 후 전진 주차 시나리오
        # 차량 정면 위치로 변환(반대 방향)
        sx -= 2 * dx
        sy -= 2 * dy
        # 후진 경로 추가
        # 후진 길이
        back_dist = 500
        # 시작 위치
        s_xy = sx, sy
        # 후진 방향으로 후진 거리 만큼 이동했을 때 좌표 계산
        dx, dy = deg2unitVec(syaw - 180, back_dist)
        b_xy = sx + dx, sy + dy
        # 반복문과 내분을 통해 시작위치부터 후진 종료 위치까지 경로 추가
        for i in range(back_dist):
            # 내분점 구하기
            x, y = introversion(s_xy, b_xy, i, back_dist - i)
            rx.append(x)
            ry.append(y)
            # 후진이므로 -50 추가
            rspeed.append(-50)

        # 후진 후 시작 위치 재조정
        sx += dx
        sy += dy

        #전진 경로 생성
        # 시작 방향 직진점 생성
        dx, dy = deg2unitVec(syaw, 200)
        sm_x, sm_y = sx + dx, sy + dy

        # 끝방향 직진점 생성
        dx, dy = deg2unitVec(eyaw, 200)
        em_x, em_y = ex + dx, ey + dy

        # 중앙점 생성(끝방향 직진점과 시작방향 직진점과의 중점)
        m_x, m_y = int((sm_x + em_x)/ 2), int((sm_y + em_y)/ 2)

        # 첫번째 2차 베지에 곡선(시작점, 시작방향직진점, 중앙점:끝방향 직진점과 시작방향 직진점과의 중점)
        addQuadraticBezier((sx, sy), (sm_x, sm_y), (m_x, m_y), n_samples)
        rspeed.extend([50 for _ in range(n_samples)])

        # 두번째 2차 베지에 곡선(중앙점:끝방향 직진점과 시작방향 직진점과의 중점, 끝방향 직진점, 끝점)
        addQuadraticBezier((m_x, m_y), (em_x, em_y), (ex, ey), n_samples)
        rspeed.extend([50 for _ in range(n_samples)])

    else:
        #전진 경로 생성
        # 시작 방향 직진점 생성
        dx, dy = deg2unitVec(syaw, 100)
        sm_x, sm_y = sx + dx, sy + dy

        # 끝방향 직진점 생성
        dx, dy = deg2unitVec(eyaw, 100)
        em_x, em_y = ex + dx, ey + dy

        # 중앙점 생성(끝방향 직진점과 시작방향 직진점과의 중점)
        m_x, m_y = int((sm_x + em_x)/ 2), int((sm_y + em_y)/ 2)

        # 첫번째 2차 베지에 곡선(시작점, 시작방향직진점, 중앙점:끝방향 직진점과 시작방향 직진점과의 중점)
        addQuadraticBezier((sx, sy), (sm_x, sm_y), (m_x, m_y), n_samples)
        rspeed.extend([50 for _ in range(n_samples)])

        # 두번째 2차 베지에 곡선(중앙점:끝방향 직진점과 시작방향 직진점과의 중점, 끝방향 직진점, 끝점)
        addQuadraticBezier((m_x, m_y), (em_x, em_y), (ex, ey), n_samples)
        rspeed.extend([50 for _ in range(n_samples)])


    # 마지막 거리 직진 주차 경로 생성
    # 반복문과 내분점 계산을 통해 직진 경로 생성
    for i in range(n_samples):
        # 내분점 계산
        x, y = introversion(P_ENTRY, P_END, i, n_samples - i)
        # 경로에 추가
        rx.append(x)
        ry.append(y)
        # 직진이므로 50 추가
        rspeed.append(50)
    # 경로 생성 결과 반환
    return rx, ry

#=============================================
# 생성된 경로를 따라가는 함수
# 파이게임 screen, 현재위치 x,y 현재각도, yaw
# 현재속도 velocity, 최대가속도 max_acceleration 단위시간 dt 를 전달받고
# 각도와 속도를 결정하여 주행한다.
#=============================================



def tracking(screen, x, y, yaw, velocity, max_acceleration, dt):
    global rx, ry
    global idx
    global rspeed

    # 현재 위치 좌표
    xy = x, y
    
    # 주행
    # 경로가 남았다면 주행
    if idx < len(rx):
        # 현재 타겟 좌표 가져오기
        target_xy = rx[idx], ry[idx]
        # 현재 목표 속도 가져오기
        speed = rspeed[idx]
        # 타겟 좌표와 현재 좌표간의 거리 구하기
        dist = distance(target_xy, xy)

        # 타겟점과 차량간의 간격 조정
        look_ahead_dist = 20

        # 마지막 직진 주차 시  
        if len(rx) - idx < 100:
            look_ahead_dist = 0

        # 현재 타겟 좌표가 임계 거리안에 들어왔다면 다음 목표로 전환
        while(dist < 64 + look_ahead_dist):
            # 인덱스 증가
            idx += 1
            # 만약 다음 목표 없다면 종료
            if idx >= len(rx):
                break
            # 다음 목표로 전환
            target_xy = rx[idx], ry[idx]
            # 남은 거리 계산
            dist = distance(target_xy, xy)

        # 현재 타겟팅 하고 있는 좌표를 화면에 빨간점으로 표시
        pygame.draw.circle(screen, [255,0,0], (int(target_xy[0]),int(target_xy[1])), 3)

        # 조향값 계산(Pure Pursuit 방식)
        angle = getSteer(target_xy, xy, yaw)
    # 아니면 정지
    else:
        angle = 0
        speed = 0
    
    # 제어값 전달
    drive(angle, speed)

