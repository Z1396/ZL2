# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run classification inference on images
分类模型推理脚本

Usage: 使用示例
    $ python classify/predict.py --weights yolov5s-cls.pt --source im.jpg
"""

# 导入命令行参数解析库，用于接收外部输入的配置（如权重路径、摄像头号）
import argparse
# 导入操作系统相关库，用于文件路径处理、文件夹操作
import os
# 导入Python系统库，用于修改Python解释器的环境变量、路径
import sys
# 导入键盘监听库，用于按Q键退出程序
from pynput import keyboard
# 导入路径处理库，用于更方便地操作文件路径
from pathlib import Path

# 导入OpenCV库，用于读取摄像头、显示图像、绘制文字
import cv2
# 导入PyTorch的神经网络函数库，softmax用于计算分类概率
import torch.nn.functional as F

# 获取当前脚本文件的绝对路径
FILE = Path(__file__).resolve()
# 获取当前文件的上两级目录，作为YOLOv5的根目录
ROOT = FILE.parents[1]
# 如果根目录不在Python的搜索路径里，就添加进去，保证能导入yolov5的模块
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# 把绝对路径转为相对路径，方便使用
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 从YOLOv5的训练文件导入图像显示函数
from classify.train import imshow_cls
# 从模型公共模块导入多后端检测模型加载器
from models.common import DetectMultiBackend
# 从数据增强模块导入分类任务的图像预处理函数
from utils.augmentations import classify_transforms
# 从通用工具模块导入日志、检查依赖、颜色文字、路径递增、打印参数
from utils.general import LOGGER, check_requirements, colorstr, increment_path, print_args
# 从PyTorch工具模块导入选择设备、推理模式、时间同步
from utils.torch_utils import select_device, smart_inference_mode, time_sync

# ========== Cyber RT 自动驾驶通信框架 相关导入 ==========
# 时间库，用于延时、计时
import time
# 线程库，用于多线程安全访问变量
import threading
# 导入Apollo Cyber RT Python API
from cyber.python.cyber_py3 import cyber
# 导入自定义的Protobuf消息类型：Car控制消息、Perception感知消息
from ZL2.proto.ZL2_pb2 import Car, Perception
# ========================================================

# 全局变量：标记程序是否退出循环，False=运行中，True=退出
break_loop = False

# Cyber RT节点类：用于接收Apollo的控制指令
class CyberCarNode:
    # 构造函数：创建对象时自动初始化
    def __init__(self):
        # 自动模式标志：True=允许AI推理并发送结果，False=不处理
        self.auto_mode = False
        # 线程锁：防止多线程同时修改变量导致数据错乱
        self.lock = threading.Lock()

    # 回调函数：每当Cyber RT收到/car_message消息时自动调用
    def callback(self, msg):
        # 加锁，保证线程安全
        with self.lock:
            # 如果收到的模式为1，开启自动模式
            if msg.pattern == 1:
                self.auto_mode = True
            # 其他模式都关闭自动模式
            else:
                self.auto_mode = False
            # 打印当前模式状态
            print(f"Received pattern: {msg.pattern}, auto_mode: {self.auto_mode}")

# 键盘按下事件处理函数
def on_press(key):
    # 声明使用全局变量break_loop
    global break_loop
    try:
        # 判断按下的键是否是q/Q（不区分大小写）
        if key.char.lower() == 'q':
            print("\n程序已退出")
            # 设置退出标志
            break_loop = True
            # 返回False，停止键盘监听
            return False
    # 捕获非字符按键（如Ctrl、Shift）的报错，忽略
    except AttributeError:
        pass

# 装饰器：启用PyTorch推理模式，节省显存、加速推理
@smart_inference_mode()
# 主运行函数，包含所有参数默认值
def run(
        weights=ROOT / 'yolov5s-cls.pt',  # 模型权重文件路径
        source=0,  # 视频源：0=摄像头，也可以填图片/视频路径
        imgsz=224,  # 模型输入图像尺寸
        device='',  # 运行设备：cuda:0或cpu
        half=False,  # 是否使用半精度FP16加速
        dnn=False,  # 是否使用OpenCV DNN后端
        show=True,  # 是否显示图像
        project=ROOT / 'runs/predict-cls',  # 结果保存目录
        name='exp',  # 保存文件夹名称
        exist_ok=False,  # 是否覆盖已有文件夹
):
    # 创建键盘监听器对象，绑定按键处理函数
    keyboard_listener = keyboard.Listener(on_press=on_press)
    # 启动键盘监听线程（后台运行，不阻塞主程序）
    keyboard_listener.start()

    # 初始化Cyber RT通信系统（必须先初始化）
    cyber.init()
    # 创建Cyber RT节点，节点名cls_node
    node = cyber.Node("cls_node")
    # 创建消息接收对象
    car_message = CyberCarNode()
    # 创建订阅者：监听/car_message话题，消息类型Car，收到消息调用callback
    reader = node.create_reader(
        "/car_message",
        Car,
        car_message.callback
    )

    # 创建第二个Cyber RT节点，用于发送分类结果
    cls_talker_node = cyber.Node("cls_talker_node")
    # 创建发布者：向/cls话题发送Perception类型消息
    cls_talker = cls_talker_node.create_writer("/cls", Perception)

    # 统计已处理的图像数量
    seen, dt = 1, [0.0, 0.0, 0.0]
    # 自动选择设备：GPU优先，没有则用CPU
    device = select_device(device)

    # 创建图像预处理转换器（尺寸归一化、归一化数值等）
    transforms = classify_transforms(imgsz)

    # 加载YOLOv5分类模型
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
    # 模型预热：跑一张空白图，让后续推理更快
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    # 打开摄像头/视频源
    cap = cv2.VideoCapture(source)
    # 如果打开失败
    if not cap.isOpened():
        LOGGER.error("Error opening video source")
        return

    # 主循环：只要Cyber RT没有关闭就一直运行
    while not cyber.is_shutdown():

        # 如果按下Q键，break_loop=True
        if break_loop:
            cyber.shutdown()
            break  # 退出循环

        # 读取一帧图像
        ret, frame = cap.read()
        # 读取失败
        if not ret:
            LOGGER.warning("Frame capture failed")
            break

        # 加锁读取当前模式，线程安全
        with car_message.lock:
            auto_mode = car_message.auto_mode

        # 只有auto_mode=True时才执行AI推理
        if auto_mode:
            # ====== 图像预处理 ======
            t1 = time_sync()  # 记录开始时间
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
            im = transforms(im).unsqueeze(0).to(device)  # 预处理+增加batch维度+传到设备
            im = im.half() if model.fp16 else im.float()  # 转为半精度/全精度
            t2 = time_sync()  # 记录结束时间
            dt[0] += t2 - t1  # 累计预处理时间

            # ====== 模型推理 ======
            results = model(im)  # 把图像送入模型得到结果
            p = F.softmax(results, dim=1)  # 用softmax转为0~1概率
            i = p.argsort(1, descending=True)[:, :5].squeeze()  # 取概率最高的5个类别索引
            t3 = time_sync()
            dt[1] += t3 - t2  # 累计推理时间

            p = F.softmax(results, dim=1)
            i = p.argsort(1, descending=True)[:, :5].squeeze()
            dt[2] += time_sync() - t3  # 累计后处理时间

            # 打印速度信息
            t = tuple(x / seen * 1E3 for x in dt)
            shape = (1, 3, imgsz, imgsz)
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}' % t)

            # ====== 获取结果 ======
            top_label = model.names[i[0]]  # 概率最高的类别名称
            top_prob = p[0, i[0]].item()   # 对应概率值

            # ====== 发送结果到Cyber RT ======
            msg = Perception()  # 创建消息对象
            msg.cls_1 = top_label  # 给消息字段赋值：分类结果
            cls_talker.write(msg)  # 发布消息

            # ====== 在画面上绘制结果 ======
            text = f"{top_label} {top_prob:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = 10
            text_y = frame.shape[0] - 30 if frame.shape[0] > 480 else 30
            
            # 黑色描边，让文字更清晰
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 4, cv2.LINE_AA)
            # 绿色文字
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 255, 0), 2, cv2.LINE_AA)

        # 显示图像窗口
        cv2.imshow('Classification Monitor', frame)

        # OpenCV自带键盘监听：按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭Cyber RT，释放资源
    cyber.shutdown()

# 解析命令行参数函数
def parse_opt():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 模型权重路径
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-cls.pt', help='model path(s)')
    # 摄像头/数据源
    parser.add_argument('--source', type=int, default=0, help='file')
    # 输入图像尺寸
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    # 运行设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 半精度加速
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # OpenCV DNN模式
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 保存路径
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls', help='save to project/name')
    # 保存文件夹名称
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 允许覆盖
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 解析参数并返回
    opt = parser.parse_args()
    # 打印参数
    print_args(vars(opt))
    return opt

# 主函数入口
def main(opt):
    # 检查依赖库是否安装完整
    check_requirements(exclude=('tensorboard', 'thop'))
    # 把参数传入run函数执行
    run(**vars(opt))

# Python标准写法：直接运行此脚本时才执行
if __name__ == "__main__":
    # 解析参数
    opt = parse_opt()
    # 运行主函数
    main(opt)