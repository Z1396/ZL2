import os
import argparse
from pathlib import Path
from lxml import etree as ET

def convert_pascal_voc_to_yolov5(voc_root, output_dir, class_list):
    """
    将Pascal VOC格式数据集转换为YOLOv5格式
    """
    # 创建输出目录
    yolov5_images_dir = Path(output_dir) / 'images'
    yolov5_labels_dir = Path(output_dir) / 'labels'
    yolov5_images_dir.mkdir(parents=True, exist_ok=True)
    yolov5_labels_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有XML文件
    voc_images_dir = Path(voc_root) / 'JPEGImages'
    voc_annotations_dir = Path(voc_root) / 'Annotations'
    xml_files = list(voc_annotations_dir.glob('*.xml'))

    # 统计信息
    total_files = 0
    skipped_files = 0
    empty_files = 0

    for xml_file in xml_files:
        total_files += 1
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 检查图像文件名
            filename = root.find('filename').text
            if not filename:
                print(f"警告: {xml_file} 中缺少 filename，跳过")
                skipped_files += 1
                continue

            # 检查图像尺寸
            size = root.find('size')
            if size is None:
                print(f"警告: {xml_file} 缺少 size 信息，跳过")
                skipped_files += 1
                continue

            width = int(size.find('width').text)
            height = int(size.find('height').text)
            if width <= 0 or height <= 0:
                print(f"警告: {xml_file} 的图像尺寸无效 (width={width}, height={height})，跳过")
                skipped_files += 1
                continue

            # 处理每个对象
            label_path = yolov5_labels_dir / f"{Path(filename).stem}.txt"
            with open(label_path, 'w') as f:
                objects_written = 0
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in class_list:
                        print(f"跳过类别: {cls} (不在 class_list 中)")
                        continue

                    bbox = obj.find('bndbox')
                    if bbox is None:
                        print(f"警告: {xml_file} 中对象缺少 bndbox，跳过")
                        continue

                    try:
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                    except (AttributeError, ValueError):
                        print(f"警告: {xml_file} 中坐标值无效，跳过")
                        continue

                    # 计算YOLOv5格式的归一化坐标
                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    box_width = (xmax - xmin) / width
                    box_height = (ymax - ymin) / height

                    # 写入标签文件
                    f.write(f"{class_list.index(cls)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                    objects_written += 1

                if objects_written == 0:
                    empty_files += 1
                    print(f"警告: {xml_file} 没有有效的对象，生成空标签文件")

            # 复制图像文件
            image_path = voc_images_dir / filename
            if image_path.exists():
                import shutil
                shutil.copy2(image_path, yolov5_images_dir / image_path.name)
            else:
                print(f"警告: 图像文件 {image_path} 不存在")

        except Exception as e:
            print(f"处理 {xml_file} 时出错: {e}")
            skipped_files += 1

    print(f"\n转换完成！\n总文件数: {total_files}\n跳过文件数: {skipped_files}\n空标签文件数: {empty_files}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='将Pascal VOC格式转换为YOLOv5格式')
    parser.add_argument('--voc_root', required=True, help='Pascal VOC数据集根目录路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--classes', required=True, 
                        help='类别列表，用逗号分隔（例如: person,car,dog）')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 处理类别列表
    class_list = [cls.strip() for cls in args.classes.split(',')]
    
    # 验证输入路径
    if not Path(args.voc_root).exists():
        print(f"错误: 输入路径 {args.voc_root} 不存在")
        exit(1)
        
    # 确保输出目录存在
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n转换参数:")
    print(f"输入路径: {args.voc_root}")
    print(f"输出路径: {args.output_dir}")
    print(f"类别列表: {class_list}")
    
    # 执行转换
    convert_pascal_voc_to_yolov5(args.voc_root, args.output_dir, class_list)

if __name__ == '__main__':
    main()

