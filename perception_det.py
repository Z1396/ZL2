# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
# ====================== 官方注释 ======================
# 这是 YOLOv5 官方文件头
# 说明：本文件用于目标检测推理（图片/视频/摄像头/网络流）
# 列出了运行命令和支持的模型格式
# ======================================================

# ====================== 导入系统库 ======================
import argparse          # 命令行参数解析
import os                # 操作系统路径、文件管理
import platform          # 获取系统平台信息
import sys               # 系统相关操作，修改环境变量
from pynput import keyboard  # 键盘监听（按Q退出）
from pathlib import Path # 路径处理库，比os.path更简洁

# ====================== 导入深度学习库 ======================
import torch             # PyTorch深度学习框架
import torch.backends.cudnn as cudnn  # GPU加速配置

# ====================== 路径配置（官方固定写法） ======================
FILE = Path(__file__).resolve()          # 获取当前脚本的绝对路径
ROOT = FILE.parents[0]                   # 获取YOLOv5根目录（上一级）
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))           # 把根目录加入Python环境变量
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 转为相对路径

# ====================== 导入YOLOv5官方模块 ======================
from models.common import DetectMultiBackend  # 多框架模型加载类
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams  # 数据加载
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box  # 画框、颜色
from utils.torch_utils import select_device, smart_inference_mode, time_sync  # 设备选择、推理模式

# ========== 【自定义部分】Cyber RT 自动驾驶通信库导入 ==========
import time
import threading
from cyber.python.cyber_py3 import cyber  # 百度CyberRT框架
from ZL2.proto.ZL2_pb2 import Car, Perception  # 小车消息类型
# =================================================================

# 全局变量：标记是否退出程序
break_loop = False

# ====================== 【自定义】CyberRT 订阅节点类 ======================
# 作用：接收小车的模式消息（人工/自动驾驶）
class CyberCarNode:
    # 构造函数
    def __init__(self):
        self.auto_mode = False       # 是否是自动驾驶模式
        self.lock = threading.Lock() # 线程锁，防止多线程冲突

    # 回调函数：收到 /car_message 消息时自动执行
    def callback(self, msg):
        with self.lock:              # 加锁，保证线程安全
            if msg.pattern == 1:     # pattern=1 → 自动驾驶
                self.auto_mode = True
            else:                    # 其他模式 → 非自动驾驶
                self.auto_mode = False
            print(f"Received pattern: {msg.pattern}, auto_mode: {self.auto_mode}")

# ====================== 【自定义】键盘监听函数 ======================
# 作用：按 Q 键安全退出
def on_press(key):
    global break_loop
    try:
        if key.char.lower() == 'q':
            print("\n程序已退出")
            break_loop = True
            return False  # 停止监听
    except AttributeError:
        pass  # 忽略非字符按键

# ====================== YOLOv5 官方推理主函数 ======================
@smart_inference_mode()  # PyTorch推理模式装饰器，节省显存
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型权重文件路径
        source=ROOT / 'data/images',  # 数据源（图片/视频/摄像头）
        data=ROOT / 'data/coco128.yaml',  # 数据集配置
        imgsz=(640, 640),  # 推理图像尺寸
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU阈值
        max_det=1000,  # 单张图最大检测目标数
        device='',  # 运行设备（GPU/CPU）
        view_img=False,  # 是否显示图像
        save_txt=False,  # 是否保存结果到txt
        save_conf=False,  # 是否保存置信度
        save_crop=False,  # 是否保存裁剪结果
        nosave=False,  # 不保存图像/视频
        classes=None,  # 只检测指定类别
        agnostic_nms=False,  # 跨类别NMS
        augment=False,  # 增强推理
        visualize=False,  # 特征可视化
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 保存路径
        name='exp',  # 保存文件夹名称
        exist_ok=False,  # 覆盖已有文件夹
        line_thickness=3,  # 画框线条宽度
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # FP16半精度推理
        dnn=False,  # 使用OpenCV DNN
):

    # ====================== 【自定义】启动键盘监听 ======================
    # 创建键盘监听器对象：当按键按下时，调用 on_press 函数
    # keyboard.Listener：来自 keyboard 库，用于监听电脑键盘操作
    # on_press=on_press：绑定按键回调函数，按任意键就会触发 on_press 函数
    keyboard_listener = keyboard.Listener(on_press=on_press)

    # 启动键盘监听线程（非阻塞，后台运行）
    # start()：启动监听器，不影响后面代码运行
    keyboard_listener.start()

    # ====================== 【自定义】初始化 CyberRT ======================
    # 初始化 CyberRT 机器人通信框架
    # cyber.init()：必须先调用，才能创建节点、发布/订阅消息
    cyber.init()  # 启动CyberRT环境

    # 创建 CyberRT 节点，节点名：det_node
    # 节点 = 机器人系统里的一个功能模块（类似APP）
    node = cyber.Node("det_node")

    # 创建自定义消息处理对象：用于接收小车状态消息
    car_message = CyberCarNode()

    # 创建订阅者：监听 /car_message 通道的消息
    # node.create_reader(通道名, 消息类型, 收到消息后执行的回调函数)
    reader = node.create_reader(
        "/car_message",    # 订阅通道名：小车会往这个通道发状态
        Car,               # 消息类型：自定义的小车消息结构体
        car_message.callback  # 回调函数：收到消息自动执行
    )

    # 创建第二个 CyberRT 节点：det_talker_node
    det_talker_node = cyber.Node("det_talker_node")

    # 创建发布者：往 /det 通道发送目标检测结果
    # create_writer(通道名, 消息类型)
    det_talker = det_talker_node.create_writer("/det", Perception)

    # ====================== 官方：数据源处理 ======================
    # 把输入的数据源转成字符串类型（摄像头/视频/图片/网络流）
    source = str(source)

    # 是否保存图片：不禁止保存 + 数据源不是.txt文件 → 允许保存
    save_img = not nosave and not source.endswith('.txt')

    # 判断数据源是不是文件（图片/视频）
    # Path(source).suffix：获取文件后缀（.jpg/.mp4）
    # in (IMG_FORMATS + VID_FORMATS)：判断是否是支持的媒体格式
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # 判断数据源是不是网络流地址（rtsp/rtmp/http/https）
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 判断是否是摄像头/网络流
    # source.isnumeric()：摄像头编号（0、1、2）
    # endswith('.txt')：读取视频列表
    # (is_url and not is_file)：网络直播流
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # 如果是网络文件地址，下载到本地并返回本地路径
    if is_url and is_file:
        source = check_file(source)

    # ====================== 官方：加载模型 ======================
    # 选择计算设备：CPU 或 GPU（cuda）
    device = select_device(device)

    # 加载 YOLO 模型
    # DetectMultiBackend：YOLO 官方模型加载类
    # weights：权重文件（.pt）
    # device：运行设备
    # dnn：是否使用OpenCV DNN
    # data：数据集配置文件
    # fp16：是否使用半精度加速
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

    # 从模型中取出 3 个关键参数
    stride, names, pt = model.stride, model.names, model.pt
    # stride：模型步长（用于图像缩放）
    # names：类别名称（人、车、狗等）
    # pt：是否是PyTorch官方模型

    # 检查并调整图像尺寸，确保能被 stride 整除
    imgsz = check_img_size(imgsz, s=stride)

    # ====================== 官方：加载数据 ======================
    # 如果是摄像头/网络流
    if webcam:
        # 检查是否支持图像显示
        view_img = check_imshow()
        # 开启GPU加速（固定分辨率时有效）
        cudnn.benchmark = True
        # 加载摄像头数据流
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        # 批次大小 = 摄像头数量
        bs = len(dataset)
    # 如果是图片/视频文件
    else:
        # 加载图片/视频文件
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1

    # 初始化视频保存路径和写入器
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ====================== 官方：模型预热 ======================
    # 给模型预热（跑一次空推理），让后面推理更快
    # imgsz=(批次, 通道数, 高, 宽)
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

    # 初始化变量
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    # seen：统计检测过多少帧
    # windows：显示窗口列表
    # dt：存储3段推理时间（预处理、推理、NMS）

    # ====================== 官方：创建显示窗口 ======================
    window_name = 'Detection Monitor'

    # 如果允许显示图像
    if view_img:
        # 创建OpenCV可视化窗口
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # 设置窗口大小 640x480
        cv2.resizeWindow(window_name, 640, 480)

    # ====================== 官方：循环读取每一帧图像 ======================
    # 遍历数据集：每次循环拿到 路径、处理后图像、原始图像、视频对象、打印字符串
    for path, im, im0s, vid_cap, s in dataset:

        # 【自定义】按Q退出标志位
        # break_loop = True 时，退出循环
        if break_loop:
            # 关闭CyberRT通信
            cyber.shutdown()
            # 跳出for循环，结束程序
            break

        # 【自定义】线程安全读取自动驾驶模式
        # with 锁：防止多线程同时修改变量导致出错
        with car_message.lock:
            # 从小车消息对象中取出 auto_mode（True=自动驾驶，False=手动）
            auto_mode = car_message.auto_mode

        # ====================== 【核心逻辑】自动驾驶模式才运行YOLO ======================
        if auto_mode:
            # 记录当前时间（用于计算耗时）
            t1 = time_sync()

            # 把numpy图像 转成 PyTorch张量 并 搬到指定设备（CPU/GPU）
            im = torch.from_numpy(im).to(device)

            # 数据类型转换：半精度/浮点型
            im = im.half() if model.fp16 else im.float()

            # 像素值归一化：0~255 → 0~1（神经网络标准输入）
            im /= 255

            # 如果图像少了 batch 维度，就扩展一维
            if len(im.shape) == 3:
                im = im[None]  # 扩展batch维度

            # 记录预处理结束时间
            t2 = time_sync()
            # 累计预处理耗时
            dt[0] += t2 - t1

            # 官方：模型推理（核心！YOLO识别目标）
            pred = model(im, augment=augment, visualize=visualize)

            # 记录推理结束时间
            t3 = time_sync()
            # 累计推理耗时
            dt[1] += t3 - t2

            # 官方：非极大值抑制（去掉重复框，保留最优框）
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # 累计NMS耗时
            dt[2] += time_sync() - t3

            # 官方：遍历每一张图的检测结果（批次处理）
            for i, det in enumerate(pred):
                # 统计已检测帧数
                seen += 1

                # 如果是摄像头
                if webcam:
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                # 如果是文件
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # 把路径转成Path对象（方便操作）
                p = Path(p)

                # 拼接打印信息：图像尺寸
                s += '%gx%g ' % im.shape[2:]

                # 归一化系数
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                # 如果需要裁剪目标，复制原图
                imc = im0.copy() if save_crop else im0

                # 创建标注器：给图像画框、写文字
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                # ====================== 【自定义】创建感知消息 ======================
                # 创建要发送的CyberRT消息对象
                msg = Perception()
                # 默认值：没有检测到目标
                msg.det_1 = 'Nothing'

                # 如果有检测结果（det不为空）
                if len(det):
                    # 官方：把模型输出坐标 映射回 原图尺寸
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # 【自定义】找出置信度最高的那个目标
                    max_prob_idx = det[:, 4].argmax()  # 取置信度最大值索引
                    max_prob_det = det[max_prob_idx]   # 取出最优框
                    top_label = names[int(max_prob_det[-1])]  # 类别名称
                    top_prob = max_prob_det[4].item()        # 置信度

                    # 打印最高置信度目标
                    LOGGER.info(f"Top prediction: {top_label} (probability: {top_prob:.4f})")

                    # 置信度大于0.5，把识别结果写入消息
                    if top_prob >= 0.5:
                        msg.det_1 = top_label

                    # 官方：打印各类目标数量
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # 官方：遍历所有框，画在图像上
                    for *xyxy, conf, cls in reversed(det):
                        if view_img:
                            c = int(cls)
                            # 生成标签文字（类别+置信度）
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            # 画框 + 写标签
                            annotator.box_label(xyxy, label, color=colors(c, True))

                # ====================== 【自定义】发送识别结果到CyberRT ======================
                det_talker.write(msg)

                # 官方：显示处理后的图像
                if view_img:
                    cv2.imshow(window_name, annotator.result())
                # 延时0.5秒，控制显示速度
                time.sleep(0.5)

            # 打印这一帧的处理信息和耗时
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # 非自动驾驶模式，不运行YOLO
        else:
            # 监听按键：按 Q 退出
            if cv2.waitKey(1) == ord('q'):
                break

    # 【自定义】程序结束，关闭CyberRT
    cyber.shutdown()

# ====================== 官方：解析命令行参数 ======================
def parse_opt():
    #作用：创建命令行参数解析器，用于接收、解析用户在终端输入的参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augment inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

# ====================== 官方：主函数入口 ======================
def main(opt):
    #作用：检查你的电脑有没有装齐 YOLO 运行需要的所有库！,exclude排除不检查这两个库因为：
    #tensorboard = 训练可视化
    #thop = 计算模型参数量
    #这两个推理时不需要，所以跳过检查，避免报错。在requirements.txt文件中。
    check_requirements(exclude=('tensorboard', 'thop'))
    #作用：把所有参数 “打包” 扔进推理函数，正式开始目标检测！
    #1. vars(opt)
    #把 opt 参数对象
    #变成 字典（key-value）
    #2. run(**vars(opt))
    #把字典里的参数，一个一个“解包”，传给 run 函数。
    #2. **叫 关键字解包作用：把字典自动变成函数参数相当于自动帮你写：run(weights='best.pt', source='0', conf_thres=0.5, ...)
    run(**vars(opt))

# ====================== Python 固定执行入口 ======================
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)