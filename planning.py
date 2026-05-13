# 导入time库，用于延时、等待等时间相关操作
import time
# 导入threading库，用于多线程操作（保证Cyber RT消息接收线程安全）
import threading
# 导入argparse库，用于解析命令行传入的参数（如串口、波特率）
import argparse
# 导入serial库，用于控制串口，发送指令给硬件（小车/单片机）
import serial
# 导入librosa库，用于加载音频文件（.wav）
import librosa
# 导入sounddevice库，用于播放音频
import sounddevice as sd
# 导入soundfile库，辅助读写声音文件
import soundfile as sf
# 从pynput库导入键盘监听模块，用于检测键盘按键按下
from pynput import keyboard
# 导入Apollo Cyber RT Python API，用于自动驾驶消息通信
from cyber.python.cyber_py3 import cyber
# 导入自定义proto消息类型：Car（控制指令）、Perception（感知结果）
from ZL2.proto.ZL2_pb2 import Car, Perception

# 定义全局变量，用于标记是否退出主循环，初始为False（不退出）
break_loop = False

# 定义Cyber RT节点类：接收 /car_message 话题（模式控制）
class CyberCarNode:
    # 构造函数：创建对象时自动初始化
    def __init__(self):
        # 自动模式标志位：True=自动驾驶，False=手动
        self.auto_mode = False
        # 线程锁：防止多线程同时读写变量导致数据错乱
        self.lock = threading.Lock()

    # 回调函数：每当收到 /car_message 消息时自动调用
    def callback(self, msg):
        # 加锁：保证多线程环境下修改变量安全
        with self.lock:
            # 判断消息中的模式字段：pattern==1 表示开启自动驾驶
            if msg.pattern == 1:
                self.auto_mode = True
            # 其他模式都关闭自动驾驶
            else:
                self.auto_mode = False
            # 打印当前模式信息
            print(f"Received pattern: {msg.pattern}, auto_mode: {self.auto_mode}")

# 定义Cyber RT节点类：接收 /cls 话题（分类结果，如角度）
class ClsNode:
    def __init__(self):
        # 存储分类结果（转向角度），默认90
        self.cls_1 = 90
        self.lock = threading.Lock()

    # 收到 /cls 消息时执行
    def callback(self, msg):
        with self.lock:
            # 将消息中的 cls_1 字段赋值给成员变量
            self.cls_1 = msg.cls_1
            print(f"Received cls: {self.cls_1}")

# 定义Cyber RT节点类：接收 /det 话题（检测结果，如红灯）
class DetNode:
    def __init__(self):
        # 存储检测结果（如 red_light）
        self.det_1 = None
        self.lock = threading.Lock()

    def callback(self, msg):
        with self.lock:
            self.det_1 = msg.det_1
            print(f"Received det: {self.det_1}")   

# 定义Cyber RT节点类：接收雷达数据话题
class RadarNode:
    def __init__(self):
        # 存储雷达数据（1=右障碍，2=左障碍）
        self.radar = None
        self.lock = threading.Lock()

    def callback(self, msg):
        with self.lock:
            self.radar = msg.radar
            print(f"Received det: {self.radar}")  

# 解析命令行参数函数
def parse_arguments():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加 --port 参数：必须传入，串口设备路径（如 /dev/ttyUSB0）
    parser.add_argument('--port', type=str, required=True, 
                       help='串口设备路径 (如 /dev/ttyUSB0 或 COM3)')
    # 添加 --baud 参数：波特率，默认115200
    parser.add_argument('--baud', type=int, default=115200,
                       help='波特率 (默认115200)')
    # 解析并返回参数
    return parser.parse_args()

# 键盘按下事件回调函数
def on_press(key):
    # 声明使用全局退出标志
    global break_loop
    try:
        # 如果按下 q 或 Q 键
        if key.char.lower() == 'q':
            print("\n程序已退出")
            # 设置退出标志
            break_loop = True
            # 返回 False 停止监听
            return False
    # 忽略非字符按键（如Ctrl、Shift）
    except AttributeError:
        pass

# 串口发送数据包函数
def send_serial_packet(ser, gear, speed, angle, brake):
    """
    发送固定格式数据包 {200, gear, speed, angle, brake, 201}
    
    参数:
        ser: 已打开的串口对象
        gear: 档位 (0-255)
        speed: 速度 (0-255)
        angle: 转向角 (0-180)
        brake: 刹车状态 (0或1)
    """
    # 构建字节数组数据包
    packet = bytearray([
        0xC8,           # 报头 200 (十六进制 0xC8)
        gear & 0xFF,    # 档位：限制在0-255范围
        speed & 0xFF,   # 速度：限制在0-255范围
        angle & 0xFF,   # 转向角：限制在0-255范围
        brake & 0x01,   # 刹车：只取0或1
        0xC9            # 报尾 201 (十六进制 0xC9)
    ])
    # 通过串口发送数据包
    ser.write(packet)
    # 打印发送的内容（十六进制格式）
    print(f"发送: {[f'0x{x:02X}' for x in packet]}") 

# 主函数：程序入口
def main():
    # 解析命令行参数（获取port和baud）
    args = parse_arguments()

    # 创建键盘监听器，绑定回调函数
    keyboard_listener = keyboard.Listener(on_press=on_press)
    # 启动监听线程
    keyboard_listener.start()

    # 加载音频文件：红灯报警语音
    data, samplerate = librosa.load('/apollo_workspace/ZL2/data/audio/my_speak.wav', sr=48000)

    # 初始化Cyber RT通信系统
    cyber.init()

    # ========== 创建4个Cyber RT订阅者节点 ==========
    # 节点1：订阅 /car_message（模式）
    node_1 = cyber.Node("planning_node_1")
    car_message = CyberCarNode()
    reader_1 = node_1.create_reader("/car_message", Car, car_message.callback)

    # 节点2：订阅 /cls（分类角度）
    node_2 = cyber.Node("planning_node_2")
    cls_message = ClsNode()
    reader_2 = node_2.create_reader("/cls", Perception, cls_message.callback)

    # 节点3：订阅 /det（目标检测）
    node_3 = cyber.Node("planning_node_3")
    det_message = DetNode()
    reader_3 = node_3.create_reader("/det", Perception, det_message.callback)

    # 节点4：订阅雷达数据
    node_4 = cyber.Node("planning_node_4")
    radar_message = RadarNode()
    reader_4 = node_4.create_reader("/apollo/radar/data", Perception, radar_message.callback)

    # 打开串口
    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"已连接串口 {args.port} @ {args.baud}bps")

    # 主循环：只要Cyber RT未关闭就一直运行
    while not cyber.is_shutdown():

        # 如果按下 q 键，退出循环
        if break_loop:
            cyber.shutdown()
            break  

        # ========== 线程安全读取所有感知数据 ==========
        with car_message.lock:
            auto_mode = car_message.auto_mode       # 自动驾驶模式
            ori_angle = int(cls_message.cls_1)      # 分类角度
            det = det_message.det_1                 # 检测结果（红灯等）
            dir_data = radar_message.radar          # 雷达数据

        # ========== 初始化控制参数 默认值 ==========
        gear = 0    # 档位：0=前进，1=倒车
        speed = 30  # 默认速度
        angle = 98  # 默认转向角（90度对应中位）
        brake = 0   # 刹车：0=关闭，1=开启

        # ===================== 核心逻辑控制 =====================
        # 1. 如果检测到红灯，立即刹车
        if det == 'red_light':
            brake = 1
            speed = 0
        else:
            # 2. 雷达避障逻辑
            if dir_data == 1:       # 雷达检测到右侧有障碍物 → 左转
                angle=60
                speed=40
            elif dir_data == 2:     # 雷达检测到左侧有障碍 → 右转
                angle=120
                speed=40    
            else:
                # 3. 视觉分类角度映射（将模型输出角度转为实际控制角度）
                if ori_angle==90:
                    angle=98
                    speed=40
                elif ori_angle==45:
                    angle=43
                    speed=60
                elif ori_angle==145:
                    angle=143
                    speed=60
                elif ori_angle==115:
                    angle=120
                    speed=40
                elif ori_angle==65:
                    angle=40
                    speed=40
        # ======================================================
        
        # 打印当前所有状态
        print(f"自动驾驶模式: {auto_mode}, 转向角: {angle}, 目标: {det}, 雷达：{dir_data}, 刹车:{brake}")
        
        # 如果处于自动驾驶模式，发送控制指令
        if auto_mode == True:
            # 发送串口数据包（angle*0.8 是硬件比例缩放）
            send_serial_packet(ser, gear, speed, int(angle*0.8), brake)
            
            # 如果检测到红灯，播放报警语音
            if det == 'red_light':
                sd.play(data, 48000, device=2)  # 播放音频
                sd.wait()                       # 等待播放完毕
                time.sleep(2)                   # 延时2秒
    
    # 关闭Cyber RT，释放资源
    cyber.shutdown()

# Python标准写法：直接运行此脚本时才执行main()
if __name__ == '__main__':
    main()