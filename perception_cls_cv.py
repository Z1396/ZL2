# 传统CV轨迹识别模块
# 使用颜色阈值分割和边缘检测来识别赛道线并判断行驶方向
"""
传统CV轨迹识别 - 基于颜色阈值和边缘检测

原理：
1. 颜色阈值分割：提取赛道线（白色/黄色/红色）
2. 边缘检测：增强线条特征
3. 滑动窗口：找到赛道线的中心位置
4. 计算偏差：判断车辆偏左/居中/偏右

适用场景：
- 固定颜色赛道线（白/黄/红/蓝）
- 光照相对稳定的室内/室外赛道

Usage:
    python3 perception_cls_cv.py --source 0 --show
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from pynput import keyboard

from cyber.python.cyber_py3 import cyber
from ZL2.proto.ZL2_pb2 import Car, Perception

break_loop = False


class CyberCarNode:
    def __init__(self):
        self.auto_mode = False
        self.lock = threading.Lock()

    def callback(self, msg):
        with self.lock:
            if msg.pattern == 1:
                self.auto_mode = True
            else:
                self.auto_mode = False
            print(f"Received pattern: {msg.pattern}, auto_mode: {self.auto_mode}")


import threading


def on_press(key):
    global break_loop
    try:
        if key.char.lower() == 'q':
            print("\n程序已退出")
            break_loop = True
            return False
    except AttributeError:
        pass


class TrackDetector:
    """
    赛道线检测器 - 基于颜色阈值和滑动窗口
    """

    def __init__(self):
        # ======================
        # 针对您的黄色赛道线优化
        # ======================
        # 黄色赛道线阈值（主赛道）
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        # 白色赛道线（可选，如果有白色辅助线）
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 40, 255])

        # 红色区域（不用于赛道跟踪，仅用于检测斑马线区域）
        self.lower_red1 = np.array([0, 120, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 120, 100])
        self.upper_red2 = np.array([180, 255, 255])

        # 滑动窗口参数
        self.window_height = 80  # 窗口高度
        self.window_num = 9      # 窗口数量
        self.margin = 50        # 窗口左右边界

        # 透视变换矩阵（可选，用于更精确的跟踪）
        self.M = None
        self.warped_width = 320
        self.warped_height = 240

    def set_white_track(self):
        """设置白色赛道线阈值"""
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])

    def set_yellow_track(self):
        """设置黄色赛道线阈值"""
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

    def set_red_track(self):
        """设置红色赛道线阈值"""
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

    def color_threshold(self, hsv):
        """
        颜色阈值分割 - 只提取黄色赛道线（主赛道）
        忽略白色斑马线和红色区域
        """
        # 只提取黄色赛道线
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # 如果需要，可以加上白色辅助线（但目前您的赛道主要是黄色）
        # mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        # mask = cv2.bitwise_or(mask_yellow, mask_white)

        # 只使用黄色
        mask = mask_yellow

        return mask

    def perspective_transform(self, img):
        """
        透视变换 - 将前视图转为俯视图
        """
        if self.M is None:
            # 定义源图像中的四边形顶点（赛道区域）
            # 需要根据摄像头安装位置调整
            src_points = np.float32([
                [img.shape[1] * 0.3, img.shape[0] * 0.6],  # 左下
                [img.shape[1] * 0.7, img.shape[0] * 0.6],  # 右下
                [img.shape[1] * 0.55, img.shape[0] * 0.3], # 右上
                [img.shape[1] * 0.45, img.shape[0] * 0.3],  # 左上
            ])

            # 定义目标图像的四边形顶点
            dst_points = np.float32([
                [0, self.warped_height],
                [self.warped_width, self.warped_height],
                [self.warped_width, 0],
                [0, 0],
            ])

            # 计算透视变换矩阵
            self.M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 应用透视变换
        warped = cv2.warpPerspective(img, self.M, (self.warped_width, self.warped_height))

        return warped

    def find_track_center(self, mask):
        """
        找到双黄色赛道线之间的中心
        返回：中心偏移值（负=偏左，正=偏右，0=居中）
        """
        height, width = mask.shape[:2]

        # 设置感兴趣区域（ROI）- 只看下半部分（更靠近车辆）
        roi_top = int(height * 0.5)
        roi = mask[roi_top:, :]
        roi_height, roi_width = roi.shape

        # 计算每列的黄色像素数量
        column_sum = np.sum(roi, axis=0)

        # 找到左右两侧的黄色赛道线
        left_line_x = -1
        right_line_x = -1

        # 从左往右找左赛道线
        threshold = 50
        for x in range(roi_width // 2):
            if column_sum[x] > threshold:
                left_line_x = x
                break

        # 从右往左找右赛道线
        for x in range(roi_width - 1, roi_width // 2, -1):
            if column_sum[x] > threshold:
                right_line_x = x
                break

        # 计算赛道中心
        if left_line_x != -1 and right_line_x != -1:
            # 两条线都找到了，取中间
            track_center = (left_line_x + right_line_x) // 2
        elif left_line_x != -1:
            # 只找到左线，假设在左侧
            track_center = left_line_x + 100
        elif right_line_x != -1:
            # 只找到右线，假设在右侧
            track_center = right_line_x - 100
        else:
            # 都没找到，保持当前方向
            return 0

        # 计算偏移（相对于图像中心）
        offset = track_center - roi_width // 2

        return offset

    def sliding_window(self, binary_warped):
        """
        滑动窗口法 - 更精确的赛道中心检测
        返回：中心偏移值
        """
        height, width = binary_warped.shape[:2]

        # 设置窗口参数
        window_height = height // self.window_num
        margin = self.margin

        # 初始化窗口位置（从底部开始）
        window_center = width // 2

        track_centers = []

        # 从下往上遍历窗口
        for i in range(self.window_num):
            # 窗口的上下边界
            win_y_low = height - (i + 1) * window_height
            win_y_high = height - i * window_height

            # 窗口的左右边界
            win_x_low = window_center - margin
            win_x_high = window_center + margin

            # 确保边界不超出图像
            win_x_low = max(0, win_x_low)
            win_x_high = min(width, win_x_high)

            # 在当前窗口内找非零像素（赛道线）
            window = binary_warped[win_y_low:win_y_high, win_x_low:win_x_high]
            if np.sum(window) > 0:
                # 找非零像素的x坐标
                nonzero = np.nonzero(window)
                if len(nonzero[1]) > 0:
                    window_center = int(np.mean(nonzero[1])) + win_x_low

            track_centers.append(window_center)

        # 使用加权平均（底部权重更大）
        if len(track_centers) > 0:
            weights = np.arange(1, len(track_centers) + 1)
            avg_center = np.average(track_centers, weights=weights)
            offset = avg_center - width // 2
        else:
            offset = 0

        return offset

    def detect_direction(self, frame):
        """
        主检测函数
        返回：(方向标签, 置信度)
        """
        # 调整图像大小（可选，加快处理速度）
        # frame = cv2.resize(frame, (320, 240))

        # 转换到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 颜色阈值分割
        mask = self.color_threshold(hsv)

        # 形态学操作 - 去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 可选：透视变换
        # warped = self.perspective_transform(mask)
        # offset = self.sliding_window(warped)

        # 直接使用简化方法
        offset = self.find_track_center(mask)

        # 根据偏移判断方向
        # offset < 0 表示偏左，需要右转
        # offset > 0 表示偏右，需要左转
        # 针对您的赛道调整的阈值

        offset_threshold_small = 20
        offset_threshold_large = 50

        if offset < -offset_threshold_large:
            direction = "145"  # 大右转
            confidence = min(abs(offset) / 80, 1.0)
        elif offset < -offset_threshold_small:
            direction = "115"  # 中右转
            confidence = min(abs(offset) / 40, 1.0)
        elif offset > offset_threshold_large:
            direction = "45"   # 大左转
            confidence = min(abs(offset) / 80, 1.0)
        elif offset > offset_threshold_small:
            direction = "65"  # 中左转
            confidence = min(abs(offset) / 40, 1.0)
        else:
            direction = "90"   # 直行
            confidence = 0.9

        return direction, confidence, mask


def run(source=0, show=True):
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    cyber.init()
    node = cyber.Node("cls_node")
    car_message = CyberCarNode()
    reader = node.create_reader("/car_message", Car, car_message.callback)

    cls_talker_node = cyber.Node("cls_talker_node")
    cls_talker = cls_talker_node.create_writer("/cls", Perception)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video source")
        return

    detector = TrackDetector()

    print("\n========================================")
    print("传统CV轨迹识别已启动")
    print("按 Q 键退出程序")
    print("========================================\n")

    while not cyber.is_shutdown():
        if break_loop:
            cyber.shutdown()
            break

        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            break

        with car_message.lock:
            auto_mode = car_message.auto_mode

        direction = "90"
        confidence = 0.0

        if auto_mode:
            direction, confidence, mask = detector.detect_direction(frame)

            # 发送结果
            msg = Perception()
            msg.cls_1 = direction
            cls_talker.write(msg)

            # 在mask上绘制调试信息
            debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            debug_img = cv2.resize(debug_img, (frame.shape[1]//2, frame.shape[0]//2))

        # 绘制结果
        if show:
            # 在原图上显示结果
            text = f"Direction: {direction} | Confidence: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = 10
            text_y = 50

            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 255, 0), 2, cv2.LINE_AA)

            # 显示模式
            mode_text = f"Mode: {'Auto' if auto_mode else 'Manual'}"
            cv2.putText(frame, mode_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if auto_mode else (0, 0, 255), 2, cv2.LINE_AA)

            # 缩小显示
            display_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            cv2.imshow('CV Track Detection', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cyber.shutdown()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=int, default=0, help='camera source')
    parser.add_argument('--show', action='store_true', help='show visualization')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
