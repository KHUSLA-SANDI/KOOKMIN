import sys
import cv2
import numpy as np
import pygame
import os

class Camera:
    def __init__(self, device_dir):
        self.device_dir = device_dir
        self.cam = cv2.VideoCapture(self.device_dir, cv2.CAP_V4L2)
        command = f"v4l2-ctl -d {self.device_dir} -c auto_exposure=1"
        os.system(command)
        self.exposure = 100
        self.set_exposure(self.exposure)

    def set_exposure(self, value):
        value = max(min(value, 5000), 1)
        command = f"v4l2-ctl -d {self.device_dir} -c exposure_time_absolute={value}"
        os.system(command)

    def read_frame(self):
        ret, frame = self.cam.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 시계 방향 90도 회전
            return frame
        return None

    def release(self):
        self.cam.release()

class App:
    def __init__(self, camera):
        pygame.init()
        self.camera = camera
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Camera Control")

        self.font = pygame.font.Font(None, 36)
        self.running = True

    def draw_text(self, text, x, y):
        # 반투명한 검은색 배경 위에 텍스트를 표시
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.topleft = (x, y)

        # 반투명 배경 추가
        bg_surface = pygame.Surface((text_rect.width + 10, text_rect.height + 5))
        bg_surface.set_alpha(150)  # 투명도 설정 (0: 완전 투명, 255: 완전 불투명)
        bg_surface.fill((0, 0, 0))  # 검은색 배경
        self.screen.blit(bg_surface, (x - 5, y - 3))
        self.screen.blit(text_surface, text_rect.topleft)

    def run(self):
        while self.running:
        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.camera.exposure += 1
                        self.camera.set_exposure(self.camera.exposure)
                    elif event.key == pygame.K_DOWN:
                        self.camera.exposure -= 1
                        self.camera.set_exposure(self.camera.exposure)
                
            frame = self.camera.read_frame()
            if frame is not None:
                # OpenCV는 BGR 포맷으로 이미지를 읽으므로 RGB로 변환
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Pygame의 surface로 변환
                frame_surface = pygame.surfarray.make_surface(frame)
                self.screen.blit(frame_surface, (0, 0))
                
            self.draw_text(f"Exposure: {self.camera.exposure}", 10, 10)
            self.draw_text(f"press up/down key", 10, 45)

            pygame.display.flip()
            #pygame.time.delay(30)

        self.camera.release()
        pygame.quit()

if __name__ == "__main__":
    device_dir = "/dev/videoCAM"  # 사용하려는 카메라 장치 경로
    camera = Camera(device_dir)
    app = App(camera)
    app.run()
