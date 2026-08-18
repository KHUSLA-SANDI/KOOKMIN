#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################
# 프로그램명 : xycar_ultrasonic_viewer.py
# 버 전 : v2.1
# 본 프로그램은 상업 라이센스에 의해 제공되므로 무단 배포 및 상업적 이용을 금합니다.
####################################################################

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from threading import Thread
from std_msgs.msg import Int32MultiArray

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import * 

import sys
import os

class MyWindow(QMainWindow):
    
    update_gui_signal = pyqtSignal(list) 
    
    # PyQt 애플리케이션을 종료하기 위한 시그널 (UI_thread에서 메인 스레드로 종료 지시)
    # 이 시그널을 사용하면 UI_thread에서 QApplication.quit()를 직접 호출하지 않아도 됩니다.
    quit_app_signal = pyqtSignal()
    
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.work_init()
        self.show()
        
        self.update_gui_signal.connect(self._update_ultrasonic_labels)
        
        # 애플리케이션 종료 시그널을 앱 종료 슬롯에 연결
        self.quit_app_signal.connect(QApplication.instance().quit)

        # QApplication이 종료될 때 호출될 슬롯 연결
        QApplication.instance().aboutToQuit.connect(self.on_about_to_quit)
        
        UI = Thread(target=self.UI_thread)
        UI.daemon = True 
        UI.start()

    # ... (setupUi, work_init, call_back, worked 함수는 이전과 동일) ...
    def setupUi(self, MainWindow):
        MainWindow.resize(334, 595)
        MainWindow.setWindowTitle("Ultrasonic Viewer")
        
        self.centralwidget = QWidget(MainWindow)
        self.Stat = QStatusBar(MainWindow)
        
        self.subscribe = QLabel(self.centralwidget)
        self.Xytron_logo = QLabel(self.centralwidget)
        self.Xycar_B2 = QLabel(self.centralwidget)
        self.Left_num = QLabel(self.centralwidget)
        self.Back_mid_num = QLabel(self.centralwidget)
        self.Front_ri_num = QLabel(self.centralwidget)
        self.Front_le_num = QLabel(self.centralwidget)
        self.Back_ri_num = QLabel(self.centralwidget)
        self.Right_num = QLabel(self.centralwidget)
        self.Back_le_num = QLabel(self.centralwidget)
        self.Front_mid_num = QLabel(self.centralwidget)

        self.subscribe.setGeometry(QRect(250, 20, 51, 51)) 
        self.Xytron_logo.setGeometry(QRect(50, 20, 171, 51))
        self.Xycar_B2.setGeometry(QRect(80, 120, 181, 401))
        self.Left_num.setGeometry(QRect(10, 310, 71, 31))
        self.Back_mid_num.setGeometry(QRect(140, 530, 71, 31))
        self.Front_ri_num.setGeometry(QRect(250, 120, 71, 31))
        self.Front_le_num.setGeometry(QRect(20, 120, 71, 31))
        self.Back_ri_num.setGeometry(QRect(240, 500, 71, 31))
        self.Right_num.setGeometry(QRect(260, 310, 71, 31))
        self.Back_le_num.setGeometry(QRect(20, 500, 71, 31))
        self.Front_mid_num.setGeometry(QRect(140, 80, 71, 31))
        
        palette = QPalette()
        self.subscribe.setPalette(palette)
        self.subscribe.setAutoFillBackground(True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        car_img = os.path.join(script_dir, '../image', 'car.png')
        logo = os.path.join(script_dir, '../image', 'logo.png')
        
        if os.path.exists(logo):
            self.Xytron_logo.setPixmap(QPixmap(logo))
        else:
            print(f"Warning: Logo image not found at {logo}")
        
        if os.path.exists(car_img):
            self.Xycar_B2.setPixmap(QPixmap(car_img))
        else:
            print(f"Warning: Car image not found at {car_img}")

        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        
        self.Left_num.setFont(font)
        self.Back_mid_num.setFont(font)
        self.Front_ri_num.setFont(font)
        self.Front_le_num.setFont(font)
        self.Back_ri_num.setFont(font)
        self.Right_num.setFont(font)
        self.Back_le_num.setFont(font)
        self.Front_mid_num.setFont(font)
        
        self.Left_num.setText("INIT")
        self.Back_mid_num.setText("INIT")
        self.Front_ri_num.setText("INIT")
        self.Front_le_num.setText("INIT")
        self.Back_ri_num.setText("INIT")
        self.Right_num.setText("INIT")
        self.Back_le_num.setText("INIT")
        self.Front_mid_num.setText("INIT")
        
        self.Left_num.raise_()
        self.subscribe.raise_()
        self.Back_mid_num.raise_()
        self.Front_ri_num.raise_()
        self.Xytron_logo.raise_()
        self.Front_le_num.raise_()
        self.Back_ri_num.raise_()
        self.Right_num.raise_()
        self.Back_le_num.raise_()
        self.Front_mid_num.raise_()
        self.Xycar_B2.raise_()
        
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setStatusBar(self.Stat)
        QMetaObject.connectSlotsByName(MainWindow)
        
    def work_init(self):
        self.last_callback_value = ["INIT","INIT","INIT","INIT","INIT","INIT","INIT","INIT"]
        self.g_callback_value = ["INIT","INIT","INIT","INIT","INIT","INIT","INIT","INIT"]

        self.sub_data = None
        self.callback_time = time.time()
        self.qos_profile = QoSProfile(depth=1)
        
        self.node = Node('ultra_viewer')
        subscription = self.node.create_subscription(
            Int32MultiArray,
            'xycar_ultrasonic',
            self.call_back,
            self.qos_profile)

    def call_back(self, data):
        self.callback_time = time.time()
        self.sub_data = data.data
        self.worked()

    def worked(self):          
        if self.sub_data == None:
            return 0

        callback_value = list(self.sub_data)

        for i in range(0,8):
            if callback_value[i] >= 200:
                callback_value[i] = 'INF'
            elif callback_value[i] < 0:
                callback_value[i] = 'ERR'

            self.g_callback_value[i] = str(callback_value[i]).zfill(3)
        
        self.update_gui_signal.emit(self.g_callback_value)

    def _update_ultrasonic_labels(self, values):
        """메인 GUI 스레드에서 초음파 값을 업데이트하는 슬롯."""
        label_name = [self.Left_num,
                      self.Front_le_num, 
                      self.Front_mid_num,
                      self.Front_ri_num,
                      self.Right_num,
                      self.Back_ri_num,
                      self.Back_mid_num,
                      self.Back_le_num]

        palette = self.subscribe.palette()

        check_time = time.time() - self.callback_time
        if check_time > 1:
            palette.setColor(QPalette.Window, Qt.red)
        else:
            palette.setColor(QPalette.Window, Qt.green)
        self.subscribe.setPalette(palette)

        for i in range(0,8):
            if self.last_callback_value[i] != values[i]:
                label_name[i].setText(values[i]) 
                self.last_callback_value[i] = values[i]

    def UI_thread(self):
        try:
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1) 
                
        except Exception as e:
            print(f"UI Thread Exception: {e}")
        finally:
            # rclpy.ok()가 False가 되어 루프를 빠져나오면
            # 메인 스레드에 GUI 종료를 요청하는 시그널을 보냅니다.
            print("UI thread detected ROS2 shutdown or error. Requesting GUI quit.")
            self.quit_app_signal.emit()
            # 노드가 아직 파괴되지 않았다면 파괴합니다.
            if self.node.executor is not None:
                self.node.destroy_node()
            print("UI thread finished.")

    def on_about_to_quit(self):
        """QApplication이 종료될 때 호출됩니다."""
        print("QApplication is about to quit. Ensuring ROS2 shutdown...")
        # rclpy.ok() 체크는 혹시 모를 상황에 대비한 것.
        # 대부분의 경우 UI_thread에서 이미 rclpy.shutdown()을 유도했거나,
        # launch 파일에 의해 ROS2 context가 종료된 상태일 것입니다.
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS2 shutdown complete.")
        

def main(args=None):
    rclpy.init(args=args)
    
    app = QApplication(sys.argv) 

    myWindow = MyWindow()
    sys.exit(app.exec_())
            
if __name__ == "__main__":
    main()
