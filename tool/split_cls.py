# 导入Python标准库
import os       # 用于文件/目录操作（创建文件夹、遍历文件、路径拼接等）
import random   # 随机数工具（本脚本主要由sklearn控制随机种子）
import shutil   # 用于复制文件（保留文件元信息，比普通复制更稳定）
import argparse # 用于解析命令行参数（让脚本可以通过命令行传参运行）

# 导入机器学习库：用于数据集划分
from sklearn.model_selection import train_test_split

def parse_arguments():
    """
    【核心函数1】解析命令行参数
    作用：让用户可以在运行脚本时，通过命令行传入 源路径、输出路径、划分比例等参数
    无需修改代码，直接通过命令控制脚本行为
    """
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(description="图像分类数据集自动划分脚本")
    
    # 添加参数：--source_dir
    # type=str：参数类型为字符串
    # required=True：必须传入该参数，否则脚本报错
    # help：参数说明，用户输入-h时显示
    parser.add_argument('--source_dir', type=str, required=True,
                        help='原始数据根目录（内部必须包含各个类别的子文件夹）')
    
    # 添加参数：--output_dir
    # default='dataset'：如果用户不传该参数，默认值为dataset
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help='划分后数据集的输出目录（默认值：dataset）')
    
    # 添加参数：--train_ratio 训练集比例
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集占总数据的比例（默认0.8，即80%）')
    
    # 添加参数：--val_ratio 验证集比例
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集占总数据的比例（默认0.2，即20%）')
    
    # 添加参数：--seed 随机种子
    # 固定种子后，每次运行脚本划分结果完全一致，保证实验可复现
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（固定后划分结果不变，默认42）')
    
    # 解析所有命令行参数，并返回参数对象
    return parser.parse_args()

def split_classification_dataset(root_dir, output_dir, train_ratio, val_ratio, seed):
    """
    【核心函数2】数据集划分主逻辑
    功能：遍历原始数据 → 按类别拆分 → 按比例分配 → 复制文件到对应目录
    最终生成标准目录结构：
    输出目录/
    ├── train/        # 训练集
    │   ├── 类别1/
    │   └── 类别2/
    └── val/          # 验证集
        ├── 类别1/
        └── 类别2/
    
    参数说明：
    root_dir: 原始数据根目录
    output_dir: 划分后数据输出目录
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    seed: 随机种子
    """
    # 断言校验：强制要求 训练集比例+验证集比例=1
    # 1e-6 是浮点误差容错值，避免因为浮点数精度问题报错
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "训练集和验证集比例之和必须等于1"

    # 获取所有类别文件夹
    # 遍历源目录下的所有内容，只保留【文件夹】，排除文件
    classes = [d for d in os.listdir(root_dir) 
              if os.path.isdir(os.path.join(root_dir, d))]
    # 打印类别信息
    print(f"发现 {len(classes)} 个类别: {classes}")

    # 创建输出目录的 train 和 val 根文件夹
    # exist_ok=True：如果目录已存在，不会报错，直接跳过
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

    # 遍历每一个类别，逐个处理
    for cls in classes:
        # 为当前类别，在train和val目录下创建对应的子文件夹
        os.makedirs(os.path.join(output_dir, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val", cls), exist_ok=True)

        # 拼接当前类别的完整路径
        cls_dir = os.path.join(root_dir, cls)
        # 获取当前类别下所有图片文件
        # 只保留后缀为 jpg/png/jpeg 的文件（不区分大小写）
        images = [f for f in os.listdir(cls_dir) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # 核心：使用sklearn库划分训练集和验证集
        # train_test_split：自动随机打乱数据后按比例划分
        train_files, val_files = train_test_split(
            images, 
            train_size=train_ratio,   # 训练集比例
            test_size=val_ratio,      # 验证集比例
            random_state=seed         # 固定随机种子
        )

        # 复制文件：将训练集图片复制到对应目录
        for f in train_files:
            shutil.copy2(
                os.path.join(cls_dir, f),          # 源文件路径
                os.path.join(output_dir, "train", cls, f)  # 目标路径
            )
        # 复制文件：将验证集图片复制到对应目录
        for f in val_files:
            shutil.copy2(
                os.path.join(cls_dir, f), 
                os.path.join(output_dir, "val", cls, f)
            )

        # 打印当前类别的划分结果
        print(f"类别 {cls}: 训练集 {len(train_files)} 张 | 验证集 {len(val_files)} 张")

    # 划分完成，打印最终提示信息
    print("\n✅ 数据集划分完成！输出目录结构：")
    print(f"训练集路径: {os.path.abspath(os.path.join(output_dir, 'train'))}")
    print(f"验证集路径: {os.path.abspath(os.path.join(output_dir, 'val'))}")

# 程序主入口
if __name__ == "__main__":
    # 1. 解析命令行参数
    args = parse_arguments()
    # 2. 调用划分函数，传入解析后的参数
    split_classification_dataset(
        root_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )