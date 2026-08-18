#!/usr/bin/env python
# -*- coding: utf-8 -*- 7

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, math

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
Blue = (255,0,0) # 파란색
Green = (0,255,0) # 녹색
Red = (0,0,255) # 빨간색
Yellow = (0,255,255) # 노란색

#=============================================
# 차선 인식 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30  # 카메라 FPS 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
ROI_START_ROW = 250  # 차선을 찾을 ROI 영역의 시작 Row값
ROI_END_ROW = 450  # 차선을 찾을 ROT 영역의 끝 Row값
ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW  # ROI 영역의 세로 크기  
L_ROW = 110  # 차선의 위치를 찾기 위한 ROI 안에서의 기준 Row값 
View_Center = WIDTH//2  

def lane_detect(image, lane_row):

    """
         차선을 감지하고 기준수평선과의 교점을 계산하는 함수.

    Args:
        image (np.ndarray): 640x480 해상도의 입력 이미지.
        lane_row (int): 기준수평선의 Y 좌표값.

    Returns:
        tuple: (found, x_left, x_right)
            - found (bool): 차선을 찾았는지 여부.
            - x_left (int): 왼쪽 차선과 기준수평선 교점의 X좌표값.
            - x_right (int): 오른쪽 차선과 기준수평선 교점의 X좌표값.
    """

    # 그레이스케일 변환 및 가우시안 블러
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지 감지
    edges = cv2.Canny(blurred, 50, 150)

    # 화면에 표시
    cv2.imshow("Edge", edges)
    cv2.waitKey(1)

    # 관심영역(ROI) 설정
    height, width = edges.shape
    roi = np.array([[(0, height), (0, height-100), (width // 2 - 50, lane_row-150), 
                     (width // 2 + 50, lane_row-150), (width, height-100), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 화면에 표시
    cv2.imshow("Masked", masked_edges)
    cv2.waitKey(1)

    # 허프 변환을 이용한 선 감지
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)

    x_left, x_right = None, None
    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # 기울기 계산
            if slope < -0.5:  # 왼쪽 차선
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:  # 오른쪽 차선
                right_lines.append((x1, y1, x2, y2))

    def extrapolate(lines, y_target):
        if not lines:
            return None
        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]
        poly = np.polyfit(y_coords, x_coords, 1)  # 1차원 직선으로 근사
        return int(np.polyval(poly, y_target))

    # 기준수평선과 교점 계산
    x_left = extrapolate(left_lines, lane_row)
    x_right = extrapolate(right_lines, lane_row)

    # 디버깅용 시각화
    debug_image = image.copy()
    if x_left is not None:
        cv2.line(debug_image, (x_left, lane_row), (x_left, height), (255, 0, 0), 2)
        cv2.rectangle(debug_image, (x_left - 5, lane_row - 5), (x_left + 5, lane_row + 5), (0, 255, 0), -1)
    if x_right is not None:
        cv2.line(debug_image, (x_right, lane_row), (x_right, height), (255, 0, 0), 2)
        cv2.rectangle(debug_image, (x_right - 5, lane_row - 5), (x_right + 5, lane_row + 5), (0, 255, 0), -1)

    # 기준수평선 그리기
    cv2.line(debug_image, (0, lane_row), (width, lane_row), (0, 255, 255), 2)

    # 화면에 표시
    cv2.imshow("Lane Detection", debug_image)
    cv2.waitKey(1)

    cv2.waitKey() 

    found = x_left is not None and x_right is not None
    return found, x_left if x_left is not None else -1, x_right if x_right is not None else -1

  
#=============================================
# 실질적인 메인 함수 
# 각종 영상처리와 알고리즘을 통해 차선의 위치를 파악
#=============================================
def start():

    global image

    image = cv2.imread('line_pic2.png', cv2.IMREAD_COLOR)

    found, x_left, x_right = lane_detect(image, 360)

#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()
