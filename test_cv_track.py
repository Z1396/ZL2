# 传统CV轨迹识别 - 视频测试脚本
# 不依赖Cyber RT，可以单独运行
"""
用法：
    python3 test_cv_track.py --video 489ffb6738f71ce8afd72bf11f40e683.mp4
"""

import argparse
import cv2
import numpy as np


class TrackDetector:
    """
    赛道线检测器 - 基于颜色阈值
    """

    def __init__(self):
        # ======================
        # 针对黄色赛道线优化
        # ======================
        # 黄色赛道线阈值（主赛道）
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        # 白色赛道线（可选）
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 40, 255])

    def color_threshold(self, hsv):
        """
        颜色阈值分割 - 只提取黄色赛道线
        """
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        return mask_yellow

    def find_track_center(self, mask):
        """
        找到双黄色赛道线之间的中心
        返回：中心偏移值（负=偏左，正=偏右，0=居中）
        """
        height, width = mask.shape[:2]

        # 设置感兴趣区域（ROI）- 只看下半部分
        roi_top = int(height * 0.5)
        roi = mask[roi_top:, :]
        roi_height, roi_width = roi.shape

        # 计算每列的黄色像素数量
        column_sum = np.sum(roi, axis=0)

        # 找到左右两侧的黄色赛道线
        left_line_x = -1
        right_line_x = -1

        threshold = 50
        for x in range(roi_width // 2):
            if column_sum[x] > threshold:
                left_line_x = x
                break

        for x in range(roi_width - 1, roi_width // 2, -1):
            if column_sum[x] > threshold:
                right_line_x = x
                break

        # 计算赛道中心
        if left_line_x != -1 and right_line_x != -1:
            track_center = (left_line_x + right_line_x) // 2
        elif left_line_x != -1:
            track_center = left_line_x + 100
        elif right_line_x != -1:
            track_center = right_line_x - 100
        else:
            return 0, left_line_x, right_line_x, column_sum

        offset = track_center - roi_width // 2
        return offset, left_line_x, right_line_x, column_sum

    def detect_direction(self, frame):
        """
        主检测函数
        返回：(方向标签, 置信度, mask, offset)
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 颜色阈值分割
        mask = self.color_threshold(hsv)

        # 形态学操作 - 去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 找赛道中心
        offset, left_line_x, right_line_x, column_sum = self.find_track_center(mask)

        # 根据偏移判断方向
        offset_threshold_small = 20
        offset_threshold_large = 50

        if offset < -offset_threshold_large:
            direction = "145"
            confidence = min(abs(offset) / 80, 1.0)
        elif offset < -offset_threshold_small:
            direction = "115"
            confidence = min(abs(offset) / 40, 1.0)
        elif offset > offset_threshold_large:
            direction = "45"
            confidence = min(abs(offset) / 80, 1.0)
        elif offset > offset_threshold_small:
            direction = "65"
            confidence = min(abs(offset) / 40, 1.0)
        else:
            direction = "90"
            confidence = 0.9

        return direction, confidence, mask, offset, left_line_x, right_line_x, column_sum


def draw_results(frame, mask, direction, confidence, offset, left_line_x, right_line_x, column_sum):
    """
    在原图上绘制检测结果
    """
    result_frame = frame.copy()
    height, width = result_frame.shape[:2]

    # 绘制主方向文字
    text = f"Direction: {direction} | Confidence: {confidence:.2f} | Offset: {offset}"
    cv2.putText(result_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)

    # 绘制偏移箭头
    center_x = width // 2
    center_y = height // 2
    arrow_length = min(100, abs(offset))
    if offset != 0:
        direction_x = center_x + (offset / abs(offset)) * arrow_length
        cv2.arrowedLine(result_frame,
                       (center_x, center_y),
                       (int(direction_x), center_y),
                       (0, 0, 255), 3)

    # 绘制检测到的赛道线（在ROI区域）
    roi_top = int(height * 0.5)
    if left_line_x != -1:
        cv2.line(result_frame,
                (left_line_x, roi_top),
                (left_line_x, height),
                (255, 0, 0), 3)
    if right_line_x != -1:
        cv2.line(result_frame,
                (right_line_x, roi_top),
                (right_line_x, height),
                (255, 0, 0), 3)

    # 绘制中心参考线
    cv2.line(result_frame,
            (center_x, roi_top),
            (center_x, height),
            (0, 255, 255), 2, cv2.LINE_AA)

    # 绘制ROI区域
    cv2.rectangle(result_frame,
                 (0, roi_top),
                 (width, height),
                 (0, 255, 255), 1)

    return result_frame


def run_test(video_path, loop=False):
    """
    运行测试
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {video_path}")
        return

    detector = TrackDetector()

    print("\n========================================")
    print("传统CV轨迹识别测试")
    print("按 Q 键退出程序，按 R 键重新开始")
    print("========================================\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop:
                print("视频播放完毕，重新开始")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            else:
                print("视频播放完毕")
                break

        frame_count += 1

        # 检测方向
        direction, confidence, mask, offset, left_line_x, right_line_x, column_sum = detector.detect_direction(frame)

        # 绘制结果
        result_frame = draw_results(frame, mask, direction, confidence, offset, left_line_x, right_line_x, column_sum)

        # 准备mask显示
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 拼接显示
        display_top = np.hstack([result_frame, mask_color])
        display_top = cv2.resize(display_top, (1200, 400))

        # 显示
        cv2.imshow("CV Track Detection - Result & Mask", display_top)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n程序已退出")
            break
        elif key == ord('r'):
            print("\n重新开始播放")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0

    cap.release()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='489ffb6738f71ce8afd72bf11f40e683.mp4', help='video path')
    parser.add_argument('--loop', action='store_true', help='loop video')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run_test(**vars(opt))

