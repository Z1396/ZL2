# 导入Python标准库
import os       # 文件/目录操作：遍历、创建路径、判断文件是否存在
import shutil   # 文件操作：复制(copy2)、移动(move)
import random   # 随机打乱数据集、固定随机种子
import argparse # 命令行参数解析：灵活传参运行脚本
from tqdm import tqdm  # 进度条库：处理大量文件时显示进度，更直观

def parse_arguments():
    """
    【函数1】解析命令行参数
    作用：接收用户在终端输入的参数，控制脚本行为
    所有参数都可以在运行脚本时自定义，无需修改代码
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="YOLO格式数据集划分脚本")
    
    # 必传参数：原始数据集根目录（必须包含 images 和 labels 两个子文件夹）
    parser.add_argument('--source_dir', type=str, required=True,
                       help='原始数据集目录路径（必须包含images和labels文件夹）')
    
    # 必传参数：划分后数据集的输出根目录
    parser.add_argument('--output_dir', type=str, required=True,
                       help='划分完成后的数据集输出目录路径')
    
    # 可选参数：训练集占比，默认 80%
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例（默认0.8，即80%）')
    
    # 可选参数：验证集占比，默认 20%
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例（默认0.2，即20%）')
    
    # 可选参数：随机种子，固定后每次划分结果完全一致
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（固定后划分结果可复现，默认42）')
    
    # 可选参数：是否移动文件，action='store_true' 表示传参就为True，不传为False
    # 默认：复制文件（保留原始数据）；传 --move_files 则移动文件（原始数据会被移走）
    parser.add_argument('--move_files', action='store_true',
                       help='使用移动文件而非复制（默认：复制文件，保留原始数据）')
    
    # 解析并返回所有参数
    return parser.parse_args()

def verify_ratios(args):
    """
    【函数2】参数合法性校验
    作用：防止用户输入错误比例导致脚本运行异常
    """
    # 校验：训练集、验证集比例必须在 0~1 之间
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1):
        raise ValueError("比例必须在0和1之间！")
    
    # 校验：训练集+验证集比例必须 = 1（容错浮点精度误差）
    if not abs(args.train_ratio + args.val_ratio - 1.0) < 0.001:
        raise ValueError("训练集和验证集比例之和必须等于1！")

def create_directory_structure(output_dir):
    """
    【函数3】创建标准YOLO数据集目录结构
    最终输出目录格式：
    output_dir/
    ├─ images/
    │   ├─ train/  训练集图片
    │   └─ val/    验证集图片
    └─ labels/
        ├─ train/  训练集标签
        └─ val/    验证集标签
    """
    # 定义需要创建的所有文件夹路径
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    
    # 循环创建文件夹
    # exist_ok=True：目录已存在也不会报错，安全创建
    for d in dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

def get_image_files(source_dir):
    """
    【函数4】获取原始数据中所有有效图片文件
    支持常见图片格式：jpg/jpeg/png/bmp/tif/webp
    自动过滤非图片文件，保证只处理有效数据
    """
    # 支持的图片后缀名（元组格式）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp')
    
    # 拼接原始图片文件夹路径：source_dir/images
    images_dir = os.path.join(source_dir, 'images')
    
    # 遍历images文件夹，筛选出所有图片文件并返回列表
    # f.lower()：统一转小写，避免 .JPG / .PNG 大写后缀被漏掉
    return [f for f in os.listdir(images_dir) 
            if f.lower().endswith(image_extensions)]

def split_dataset(args):
    """
    【核心函数5】数据集划分主逻辑
    1. 打乱图片顺序
    2. 按比例拆分训练集/验证集
    3. 同步复制/移动 图片+对应标签
    4. 带进度条显示处理过程
    """
    # 固定随机种子，保证每次打乱结果一致
    random.seed(args.seed)
    
    # 获取所有图片文件名列表
    image_files = get_image_files(args.source_dir)
    # 随机打乱图片顺序（数据集划分必须打乱，保证数据分布均匀）
    random.shuffle(image_files)
    
    # 计算划分索引：前 train_ratio 部分为训练集，后面为验证集
    split_idx = int(len(image_files) * args.train_ratio)
    train_files = image_files[:split_idx]  # 训练集文件列表
    val_files = image_files[split_idx:]    # 验证集文件列表
    
    # 根据参数选择操作方式：移动 或 复制
    # --move_files：shutil.move（移动）；默认：shutil.copy2（复制，保留元信息）
    file_op = shutil.move if args.move_files else shutil.copy2
    
    # ===================== 处理训练集 =====================
    # tqdm：给循环添加进度条，desc是进度条显示文字
    for img_file in tqdm(train_files, desc='处理训练集'):
        # 1. 处理图片：源路径 → 目标路径
        src_img = os.path.join(args.source_dir, 'images', img_file)
        dst_img = os.path.join(args.output_dir, 'images', 'train', img_file)
        file_op(src_img, dst_img)  # 执行复制/移动
        
        # 2. 处理对应标签：图片名 → 同名 .txt 标签名
        # os.path.splitext(img_file)[0]：去掉图片后缀，获取纯文件名
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(args.source_dir, 'labels', label_file)
        
        # 如果标签文件存在，才复制/移动，避免报错
        if os.path.exists(src_label):
            dst_label = os.path.join(args.output_dir, 'labels', 'train', label_file)
            file_op(src_label, dst_label)
        # 复制模式下标签缺失，直接跳过；移动模式必须严格对应
        elif not args.move_files:
            continue
    
    # ===================== 处理验证集 =====================
    for img_file in tqdm(val_files, desc='处理验证集'):
        # 1. 处理图片
        src_img = os.path.join(args.source_dir, 'images', img_file)
        dst_img = os.path.join(args.output_dir, 'images', 'val', img_file)
        file_op(src_img, dst_img)
        
        # 2. 处理对应标签
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(args.source_dir, 'labels', label_file)
        
        if os.path.exists(src_label):
            dst_label = os.path.join(args.output_dir, 'labels', 'val', label_file)
            file_op(src_label, dst_label)
        elif not args.move_files:
            continue

def main():
    """
    【主函数】程序总入口，按流程调用所有功能
    """
    # 1. 解析命令行参数
    args = parse_arguments()
    
    # 2. 校验比例参数是否合法
    verify_ratios(args)
    
    # 3. 检查原始目录是否规范：必须有 images 和 labels 文件夹
    required_dirs = ['images', 'labels']
    for d in required_dirs:
        if not os.path.exists(os.path.join(args.source_dir, d)):
            raise FileNotFoundError(f"原始数据目录缺少必要文件夹：'{d}'")
    
    # 4. 创建输出目录结构
    create_directory_structure(args.output_dir)
    
    # 5. 执行数据集划分（核心步骤）
    split_dataset(args)
    
    # 6. 统计并打印最终结果
    train_img_count = len(os.listdir(os.path.join(args.output_dir, 'images', 'train')))
    val_img_count = len(os.listdir(os.path.join(args.output_dir, 'images', 'val')))
    
    print(f"\n✅ 数据集划分完成！")
    print(f"操作方式: {'移动文件' if args.move_files else '复制文件'}")
    print(f"训练集图像: {train_img_count} 张")
    print(f"验证集图像: {val_img_count} 张")
    print(f"原始数据保留: {'否' if args.move_files else '是'}")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")

# Python 脚本固定写法：直接运行该文件时，才执行 main()
if __name__ == "__main__":
    main()