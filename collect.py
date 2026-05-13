# 导入time库，用于控制程序延时、休眠等操作
import time
# 导入OpenCV库，用于摄像头读取、图像显示、图像保存等计算机视觉功能
import cv2
# 导入os库，用于文件路径操作、创建文件夹等系统文件管理功能
import os
# 导入argparse库，用于解析命令行传入的参数（比如路径、摄像头编号）
import argparse
# 导入threading库，用于多线程操作（这里键盘监听是独立线程）
import threading
# 从pynput库导入键盘监听模块，用于检测键盘按键按下
from pynput import keyboard
# 导入Apollo Cyber RT Python API，用于和Apollo通信
from cyber.python.cyber_py3 import cyber
# 从自定义proto文件夹导入定义好的Car消息结构体（自定义消息类型）
from ZL2.proto.ZL2_pb2 import Car

# 定义全局变量，用于标记是否退出主循环，初始为False（不退出）
break_loop = False

# 定义Cyber RT节点类，用于接收Apollo消息并存储状态
class CyberCarNode:
    # 类的构造函数，创建对象时自动调用
    def __init__(self):
        # 自动模式标志位：True=自动保存图片，False=不保存
        self.auto_mode = False
        # 线程锁：防止多线程同时修改变量导致数据错乱
        self.lock = threading.Lock()
        # 角度值，默认90度，从Apollo消息中接收
        self.angle = 90

    # Cyber RT消息回调函数：每当收到/car_message话题就会自动调用
    def callback(self, msg):
        # 加锁：保证多线程下修改变量安全
        with self.lock:
            # 判断消息中的模式字段：pattern==2表示自动模式
            if msg.pattern == 2:
                self.auto_mode = True
            # 其他模式都视为非自动模式
            else:
                self.auto_mode = False
            # 打印当前模式信息
            print(f"Received pattern: {msg.pattern}, collect_mode: {self.auto_mode}")
        # 把消息中的角度值赋值给类成员变量
        self.angle = msg.angle

# 解析命令行参数函数
def parse_arguments():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 定义--save_path参数：图像保存路径，默认路径/apollo_workspace/ZL2/data/camera
    parser.add_argument('--save_path', type=str, default='/apollo_workspace/ZL2/data/camera', help='图像保存路径')
    # 定义--camera参数：摄像头编号，默认0（笔记本默认摄像头）
    parser.add_argument('--camera', type=int, default=0, help='摄像头编号')
    # 解析并返回所有命令行参数
    return parser.parse_args()

# 键盘按下事件回调函数：按下键盘时自动执行
def on_press(key):
    # 声明使用全局变量break_loop
    global break_loop
    try:
        # 判断按下的键是否是q/Q（不区分大小写）
        if key.char.lower() == 'q':
            # 打印退出提示
            print("\n程序已退出")
            # 将全局退出标志设为True
            break_loop = True
            # 返回False：停止键盘监听
            return False
    # 捕获非字符按键（如Shift/Ctrl）报错，忽略即可
    except AttributeError:
        pass

# 主函数：程序入口
def main():  
    # 调用函数解析命令行参数，得到args对象
    args = parse_arguments()
    
    # 创建键盘监听器对象，绑定按下事件处理函数
    keyboard_listener = keyboard.Listener(on_press=on_press)
    # 启动键盘监听线程（后台运行，不阻塞主程序）
    keyboard_listener.start()
    
    # 初始化Cyber RT系统（必须先初始化才能使用节点/读写话题）
    cyber.init()
    # 创建Cyber RT节点，节点名字为collect_node
    node = cyber.Node("collect_node")
    # 创建自定义消息处理类的实例对象
    car_message = CyberCarNode()
    # 创建Cyber RT订阅者（reader）：
    # 订阅话题 /car_message
    # 消息类型 Car
    # 收到消息后调用 car_message.callback 函数处理
    reader = node.create_reader(
        "/car_message",
        Car,
        car_message.callback
    )
    
    # 打开摄像头，参数是摄像头编号（0/1/2...）
    camera = cv2.VideoCapture(args.camera)
    # 判断摄像头是否成功打开
    if not camera.isOpened():
        # 打开失败则打印提示
        print("无法打开摄像头")
        # 退出函数，程序结束
        return
    
    # 摄像头打开成功，打印提示
    print(f"已成功打开摄像头 {args.camera}")
    
    # 图片计数变量：用于给图片命名（image0.jpg, image1.jpg...）
    image_counter = 0
    # 主循环：如果Cyber RT没有关闭，就一直循环
    while not cyber.is_shutdown():
        
        # 如果键盘按下q，break_loop变为True
        if break_loop:
            # 关闭Cyber RT系统
            cyber.shutdown()
            # 跳出while循环，结束程序
            break  
        
        # 从摄像头读取一帧图像
        # ret：是否读取成功（True/False）
        # frame：读取到的图像数据
        ret, frame = camera.read()
        # 如果读取失败
        if not ret:
            print("无法获取摄像头帧")
            # 退出循环
            break

        # 加锁读取模式和角度，保证线程安全
        with car_message.lock:
            auto_mode = car_message.auto_mode
            angle = car_message.angle

        # 判断是否为自动模式：自动模式才保存图片
        if auto_mode:
                
            # 拼接保存路径：基础路径 + 当前角度值作为文件夹名
            save_dir = f"{args.save_path}/{angle}"
            # 如果该文件夹不存在
            if not os.path.exists(save_dir):
                # 创建文件夹（支持多级目录）
                os.makedirs(save_dir)
            
            # 拼接图片完整路径：文件夹 + 图片名
            image_path = os.path.join(save_dir, f"image{image_counter}.jpg")
            # 使用OpenCV保存图像到硬盘
            cv2.imwrite(image_path, frame)
            # 打印保存成功信息
            print(f"已保存图像: {save_dir}/image{image_counter}.jpg")
            # 图片序号+1
            image_counter += 1
        
        # 显示图像窗口，窗口名Camera Monitor，显示内容frame
        cv2.imshow('Camera Monitor', frame)
        # 延时10毫秒，控制循环频率，同时让窗口可以刷新
        time.sleep(0.01)  
        
        # OpenCV键盘监听：检测是否按下q，按下则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 最终关闭Cyber RT系统，释放资源
    cyber.shutdown()        

# Python标准写法：当直接运行这个脚本时，才执行main()
if __name__ == '__main__':
    main()