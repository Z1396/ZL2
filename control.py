# 【导入Python标准库】
# time：时间相关工具，本代码中未直接使用，但属于工程常用库
import time

# argparse：命令行参数解析库
# 作用：允许用户在终端输入参数（如 -p /dev/ttyUSB0）传给程序
import argparse

# serial：pyserial库，用于Linux系统下的串口通信
# 作用：和小车底盘、手柄进行硬件数据交互
import serial

# pynput.keyboard：键盘监听库
# 作用：检测用户按键，实现按Q退出程序的功能
from pynput import keyboard

# 【导入CyberRT自动驾驶框架核心库】
# cyber：百度Apollo的CyberRT框架Python接口，用于节点创建、消息收发
from cyber.python.cyber_py3 import cyber

# 【导入自定义Protobuf消息类型】
# Car：小车专用通信消息结构体，包含模式、速度、角度等所有控制字段
from ZL2.proto.ZL2_pb2 import Car

# ====================== 全局变量定义 ======================
# 全局变量break_loop，标记程序是否需要退出
# global关键字修饰的变量，在函数内部可修改
break_loop = False

# ====================== 数据解析类 ======================
# class：Python定义类的关键字
# RemoteData：自定义类，作用是封装、解析串口原始数据
class RemoteData:
    # __init__：Python类的构造函数，创建对象时自动执行
    # self：必须写，代表类实例自身
    # raw_data：传入的原始串口数据（8字节）
    def __init__(self, raw_data):
        # 从8字节原始数据中，按顺序提取每个控制参数
        self.pattern = raw_data[0]            # 工作模式 0人工/1自动/2采集
        self.gear = raw_data[1]               # 档位 0前进/1倒车
        self.real_speed = raw_data[2]         # 底盘返回的实时车速
        self.ctrl_speed = raw_data[3]         # 控制目标车速
        self.angle = int(raw_data[4]*1.25)     # 转向角度，放大1.25倍并转整数
        self.brake = raw_data[5]              # 刹车状态 0=松开 1=刹车
        self.park = raw_data[6]               # 自动泊车状态
        self.custom_signals = raw_data[7]     # 自定义扩展信号

# ====================== 命令行参数解析函数 ======================
# def：Python定义函数的关键字
# parse_arguments：函数名，解析终端输入参数
def parse_arguments():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数：
    # -p / --port：短参数/长参数
    # required=True：必须输入，否则程序报错
    # help：提示信息
    parser.add_argument('-p', '--port', required=True, 
                       help='串口设备路径 (/dev/ttyUSB0, /dev/ttyUSB1...)')
    
    # 解析参数并返回，外部可通过args.port获取设备路径
    return parser.parse_args()

# ====================== 键盘按下回调函数 ======================
# on_press：按键按下时自动触发的函数
# key：pynput传入的按键对象
def on_press(key):
    # 声明break_loop是全局变量，函数内可修改
    global break_loop
    
    try:
        # key.char：获取按键对应的字符
        # .lower()：转小写，确保按Q或q都能退出
        if key.char.lower() == 'q':
            print("\n程序已退出")
            break_loop = True  # 修改全局退出标志
            return False       # 返回False，停止键盘监听
    except AttributeError:
        # 捕获非字符按键（如Ctrl、Shift），不做处理
        pass

# ====================== 数据包校验与解析函数 ======================
# parse_packet：校验一帧串口数据是否合法
# packet：完整的串口数据帧
def parse_packet(packet):
    # 数据帧规则：
    # 总长度=10字节 | 帧头=0x7F | 帧尾=0x7F
    if len(packet) != 10 or packet[0] != 0x7F or packet[-1] != 0x7F:
        return None  # 不合法，返回空
    
    # 合法：截取中间8个字节，传给RemoteData类解析
    return RemoteData(packet[1:-1])

# ====================== 串口数据读取核心生成器函数 ======================
# yield关键字 → 生成器函数，节省内存，逐条返回数据
# port：串口设备路径   baud：波特率（默认115200）
def serial_reader(port, baud=115200):
    try:
        # with语句：自动管理串口资源，退出时自动关闭
        # serial.Serial：打开串口，timeout=0.1 读取超时时间
        with serial.Serial(port, baud, timeout=0.1) as ser:
            print(f"已连接串口设备: {ser.name}")
            
            # bytearray()：创建空字节数组，作为数据缓冲区
            buffer = bytearray()
            
            # cyber.is_shutdown()：判断CyberRT是否关闭
            # 循环条件：未关闭则一直执行
            while not cyber.is_shutdown():
                # ser.in_waiting：获取串口缓存区的字节数
                # ser.read()：读取缓存区所有数据，拼接到buffer
                buffer += ser.read(ser.in_waiting or 1)
                
                # 内层循环：不断从缓冲区解析完整数据帧
                while True:
                    # buffer.find(0x7F)：查找帧头0x7F第一次出现的位置
                    header_pos = buffer.find(0x7F)
                    
                    if header_pos == -1:
                        break  # 找不到帧头，退出内层循环
                    
                    # 切片：丢弃帧头之前的无效数据
                    buffer = buffer[header_pos:]
                    
                    # 从下标1开始查找帧尾0x7F
                    footer_pos = buffer.find(0x7F, 1)
                    
                    if footer_pos == -1:
                        break  # 未收到完整帧尾，等待下一次读取
                    
                    # 切片：提取完整数据帧（从帧头到帧尾）
                    packet = buffer[:footer_pos+1]
                    # 剩余数据留在缓冲区，等待下一轮解析
                    buffer = buffer[footer_pos+1:]
                    
                    # 解析数据包，若合法则返回
                    if data := parse_packet(packet):
                        yield data  # 生成器：逐条返回解析好的数据
                        
    # 捕获串口打开失败、断开等异常
    except Exception as e:
        print(f"串口错误: {e}")
        cyber.shutdown()  # 关闭CyberRT，安全退出

# ====================== 主函数：程序入口 ======================
def main():
    # 1. 创建键盘监听对象，绑定按下回调函数
    keyboard_listener = keyboard.Listener(on_press=on_press)
    # 启动键盘监听（后台运行）
    keyboard_listener.start()

    # 2. 调用函数，解析命令行参数（获取串口）
    args = parse_arguments()
    
    # 3. 初始化CyberRT框架（必须先初始化才能创建节点）
    cyber.init()
    
    # 4. 创建CyberRT节点，节点名称：control_node_4
    control_node = cyber.Node("control_node_4")
    
    # 5. 创建Writer（发布者）：
    # 通道名：/car_message
    # 消息类型：Car
    control = control_node.create_writer("/car_message", Car)
    
    # 6. 主循环：程序核心运行逻辑
    while not cyber.is_shutdown():

        # 遍历生成器，不断获取解析好的串口数据
        for data in serial_reader(args.port):
            
            # 判断是否按下Q键
            if break_loop:
                cyber.shutdown()  # 关闭CyberRT
                break            # 跳出for循环
            
            # 元组解包：把data对象的属性一次性赋值给多个变量
            (pattern, gear, real_speed, ctrl_speed, 
             angle, brake, park, custom_signals) = (
                data.pattern, data.gear, data.real_speed,
                data.ctrl_speed, data.angle, data.brake,
                data.park, data.custom_signals
            )

            # 清屏指令：\033[2J 清屏 \033[H 光标回到左上角
            print("\033[2J\033[H")
            
            # 格式化打印车辆状态（三目运算符简化判断）
            print(f"""
            ============= 控制参数 =============
            工作模式: {
                '人工驾驶' if pattern == 0 else
                '自动驾驶' if pattern == 1 else
                '数据采集' if pattern == 2 else
                '未知模式'
            }
            档位状态: {
                '前进档' if gear == 0 else
                '倒车档' if gear == 1 else
                '未知档位'
            }
            实时车速: {real_speed} | 目标车速: {ctrl_speed}
            转向角度: {angle}°
            刹车状态: {'激活' if brake else '未激活'}
            自动泊车: {'已启用' if park else '未启用'}
            自定义信号: {custom_signals}
            ====================================
            """)

            # ====================== 核心：发送CyberRT消息 ======================
            msg = Car()  # 创建Car消息对象
            
            # 给Car消息对象赋值（元组赋值）
            (msg.pattern, msg.gear, msg.real_speed, msg.ctrl_speed, 
             msg.angle, msg.brake, msg.park, msg.custom_signals) = (pattern, 
             gear, real_speed, ctrl_speed, angle, brake, 
              park, custom_signals)
            
            # 调用write方法，将消息发送到 /car_message 通道
            control.write(msg)
        
    # 主循环结束，关闭CyberRT框架
    cyber.shutdown()

# ====================== Python程序入口判断 ======================
# __name__：Python内置变量
# 直接运行本文件时 __name__ == "__main__"
# 被导入时不执行main()
if __name__ == "__main__":
    main()