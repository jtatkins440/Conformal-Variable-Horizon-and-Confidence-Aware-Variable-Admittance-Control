#!/usr/bin/env python
import pyqtgraph as pg
from pyqtgraph import RectROI
from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtGui import QBrush, QPen, QColor
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from pyqtgraph import EllipseROI
import rospy
from std_msgs.msg import Bool
from rehab_msgs.msg import GUIInfo
import numpy as np

class GuiNode:
    def __init__(self, init_yaw=70, init_pitch=-40):
        rospy.init_node('gui_node')
        
        self.pitch = 0.0
        self.yaw = 0.0
        self.yaw_pred_center = []
        self.pitch_pred_center = []
        self.radii = []
        self.estimated_label = "Unknown"  # Default label

        self.robot_pose = []
        self.target_radius = 2.5

        rospy.Subscriber('gui_info', GUIInfo, self.get_data_callback, queue_size=25, tcp_nodelay=True)
        self.pub_1 = rospy.Publisher('gui_stop_switch', Bool, queue_size=1, tcp_nodelay=True)

        # PyQtGraph setup
        self.app = QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="Robot Visualization")
        self.win.resize(800, 800)
        self.win.setWindowTitle("Robot GUI with PyQtGraph")
        self.win.setBackground('w')  # 배경색 변경: 흰색

        self.plot = self.win.addPlot(title="Robot Path")
        self.plot.setXRange(40, 90, padding=0)
        self.plot.setYRange(-60, 30, padding=0)
        self.plot.invertX(True)  # X축 반전
        self.plot.setLabel('left', 'Y')
        self.plot.setLabel('bottom', 'X')

        # set plot parameters
        self.robot_scatter = pg.ScatterPlotItem(size=30, brush=pg.mkBrush(0, 0, 255, 255))  # Blue (current position)
        self.starting_scatter = pg.ScatterPlotItem(size=60, brush=pg.mkBrush(0, 255, 0))  # Red (starting position)
        self.target_scatter = pg.ScatterPlotItem(size=60, brush=pg.mkBrush(255, 165, 0)) # Grean (target position)
        
        self.prediction_line = pg.PlotDataItem(
            pen=pg.mkPen(color=(0, 0, 0), width=2), symbol='o', symbolBrush=(0, 0, 0)
        )
        self.plot.addItem(self.robot_scatter)
        self.plot.addItem(self.starting_scatter)
        self.plot.addItem(self.target_scatter)
        self.plot.addItem(self.prediction_line)
        # Task label signal with QGraphicsEllipseItem

        # Behavior Label (circle + text)
        self.behavior_circle = pg.ScatterPlotItem(size=70, brush=pg.mkBrush(255, 0, 0))  # 빨간색 (초기값)
        self.behavior_text = pg.TextItem(anchor=(0.5, 0.5), color='k')  # 검은색 텍스트
        self.behavior_text.setText("STOP")
        self.behavior_text.setPos(80, 20)  # 초기 위치 설정
        font = QFont("Arial", 20)  # 폰트 이름과 크기 설정 (예: Arial, 크기 16)
        self.behavior_text.setFont(font)
        self.plot.addItem(self.behavior_circle)
        self.plot.addItem(self.behavior_text)
        
        # 원을 저장할 리스트
        self.prediction_circles = pg.ScatterPlotItem(size=30, brush=pg.mkBrush(0, 255, 255, 50))  # 청록색 투명
        self.plot.addItem(self.prediction_circles)
        
        # Init label
        self.behavior_label = "Ready"

        # 주기적인 업데이트 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 50ms 주기로 업데이트

    def get_data_callback(self, data):
        rpy_state = data.state
        self.yaw = rpy_state.RPY.z
        self.pitch = rpy_state.RPY.y

        self.yaw_pred_center = [pset.center.z for pset in data.prediction_sets.prediction_sets]
        self.pitch_pred_center = [pset.center.y for pset in data.prediction_sets.prediction_sets]
        self.radii = [pset.radius for pset in data.prediction_sets.prediction_sets]
        
        classification = data.classification  # This is of type ClassificationOutput
        self.maxprob_estimated_label = classification.expected_class_index
        self.estimated_label = classification.classes[self.maxprob_estimated_label].label
        self.probability = classification.classes[self.maxprob_estimated_label].probability

        self.robot_pose = [self.yaw, self.pitch]


        ### Exp related data
        protocol_data = data.protocol_state
        self.start_position_yaw = protocol_data.starting_position.z
        self.start_position_pitch = protocol_data.starting_position.y

        self.goal_position_yaw = protocol_data.goal_position.z
        self.goal_position_pitch = protocol_data.goal_position.y

        self.behavior_label = protocol_data.behavior_label
        self.countdown_time = protocol_data.countdown_time

    def update_plot(self):
        if not self.robot_pose:
            return

        # Adjusted Path 계산
        if self.yaw_pred_center and self.pitch_pred_center:
            # 중심점 계산
            offset_yaw = self.yaw_pred_center[0] - self.yaw
            offset_pitch = self.pitch_pred_center[0] - self.pitch
            yaw_adjusted = [y - offset_yaw for y in self.yaw_pred_center]
            pitch_adjusted = [p - offset_pitch for p in self.pitch_pred_center]

            # Adjusted Prediction Path 업데이트
            self.prediction_line.setData(yaw_adjusted, pitch_adjusted)

            # 기존 원 제거
            for item in self.plot.items[:]:
                if isinstance(item, EllipseROI):
                    self.plot.removeItem(item)

            # 원 업데이트 (진짜 원)
            for y, p, r in zip(yaw_adjusted, pitch_adjusted, self.radii):
                ellipse = EllipseROI([y - r, p - r], [r * 2, r * 2], pen=pg.mkPen(color=(0, 255, 255, 255), width=2))
                self.plot.addItem(ellipse)
        
        # 타겟 위치
        self.target_scatter.setData([self.goal_position_yaw], [self.goal_position_pitch])

        # Update behavior label
        if self.behavior_label == "Dynamic":
            brush = pg.mkBrush(0, 255, 0)  # 초록색
            self.behavior_text.setText("GO")
        elif self.behavior_label == "Static":
            brush = pg.mkBrush(255, 0, 0)  # 빨간색
            self.behavior_text.setText("STOP")
        else:
            brush = pg.mkBrush(0, 0, 255)  # 파란색 (기본값)
            self.behavior_text.setText("READY")

        # Update circle color and position
        self.behavior_circle.setData([80], [20], brush=brush)  # 고정 위치: x=80, y=-10
        self.behavior_text.setPos(80, 20)  # 텍스트 위치

        # 현재 로봇 위치
        self.robot_scatter.setData([self.yaw], [self.pitch])

    def run(self):
        self.app.exec_()


if __name__ == '__main__':
    gui_node = GuiNode()
    gui_node.run()