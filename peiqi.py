import os
import cv2
import numpy as np
import shutil


def read_label_file(label_file):
    with open(label_file, 'r') as f:
        data = f.readlines()
    return [list(map(float, line.strip().split()[1:])) for line in data]


def adjust_image(visible_image_path, infrared_image_path, visible_rect, infrared_rect):
    visible_image = cv2.imread(visible_image_path)
    infrared_image = cv2.imread(infrared_image_path)

    if visible_image is None or infrared_image is None:
        print(f"Warning: Could not read image files. Visible: {visible_image_path}, Infrared: {infrared_image_path}")
        return None

    visible_height, visible_width = visible_image.shape[:2]
    infrared_height, infrared_width = infrared_image.shape[:2]

    # 计算红外框在可见光图像中的位置
    infrared_rect_rescaled = [
        infrared_rect[0] * visible_width, infrared_rect[1] * visible_height,
        infrared_rect[2] * visible_width, infrared_rect[3] * visible_height
    ]

    # 计算可见光标签的框的位置
    visible_rect_rescaled = [
        visible_rect[0] * visible_width, visible_rect[1] * visible_height,
        visible_rect[2] * visible_width, visible_rect[3] * visible_height
    ]

    # 计算平移量(dx, dy)，使得红外标签框与可见光标签框对齐
    dx = infrared_rect_rescaled[0] - visible_rect_rescaled[0]
    dy = infrared_rect_rescaled[1] - visible_rect_rescaled[1]

    # 应用平移到可见光图像
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_visible_image = cv2.warpAffine(visible_image, M, (visible_width, visible_height), borderValue=(0, 0, 0))

    return translated_visible_image


def save_yolo_labels(label_data, label_path):
    with open(label_path, 'w') as f:
        for label in label_data:
            f.write("0 " + " ".join(map(str, label)) + "\n")


def process_folder():
    # 设置路径
    infrared_image_dir = r'F:\BaiduNetdiskDownload\Fixed-wing-UAV-F2\Fixed-wing-UAV-F2\Top_Down\Infrared_Imgs'
    visible_image_dir = r'F:\BaiduNetdiskDownload\Fixed-wing-UAV-F2\Fixed-wing-UAV-F2\Top_Down\Wide_Imgs'
    infrared_label_dir = r'F:\BaiduNetdiskDownload\Fixed-wing-UAV-F2\Fixed-wing-UAV-F2\Top_Down\labels_ir'
    visible_label_dir = r'F:\BaiduNetdiskDownload\Fixed-wing-UAV-F2\Fixed-wing-UAV-F2\Top_Down\labels_vis'

    # 设置输出路径
    output_dir_vis = r'F:\BaiduNetdiskDownload\MMFW-UAV\images'
    output_dir_ir = r'F:\BaiduNetdiskDownload\MMFW-UAV\image'
    output_dir_labels = r'F:\BaiduNetdiskDownload\MMFW-UAV\labels'

    output_visible_dir = os.path.join(output_dir_vis, 'images_33')
    output_infrared_dir = os.path.join(output_dir_ir, 'image_33')
    output_label_dir = os.path.join(output_dir_labels, 'labels_33')

    # 创建输出目录
    os.makedirs(output_visible_dir, exist_ok=True)
    os.makedirs(output_infrared_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 检查输入目录是否存在
    if not os.path.exists(infrared_image_dir):
        print(f"Error: Infrared image directory {infrared_image_dir} does not exist!")
        return

    if not os.path.exists(visible_image_dir):
        print(f"Error: Visible image directory {visible_image_dir} does not exist!")
        return

    if not os.path.exists(infrared_label_dir):
        print(f"Error: Infrared label directory {infrared_label_dir} does not exist!")
        return

    if not os.path.exists(visible_label_dir):
        print(f"Error: Visible label directory {visible_label_dir} does not exist!")
        return

    # 获取所有红外图像文件
    infrared_images = [f for f in os.listdir(infrared_image_dir) if
                       f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    total_images = len(infrared_images)

    print(f"Total infrared images to process: {total_images}")

    processed_images = 0

    # 遍历所有的红外图像
    for infrared_image in infrared_images:
        # 提取文件名（不含扩展名）
        image_name = os.path.splitext(infrared_image)[0]

        # 构建文件路径
        infrared_image_path = os.path.join(infrared_image_dir, infrared_image)
        visible_image_path = os.path.join(visible_image_dir, f'{image_name}.jpg')

        # 检查对应文件是否存在
        if not os.path.exists(visible_image_path):
            # 尝试其他常见图片格式
            possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            found = False
            for ext in possible_extensions:
                temp_path = os.path.join(visible_image_dir, f'{image_name}{ext}')
                if os.path.exists(temp_path):
                    visible_image_path = temp_path
                    found = True
                    break
            if not found:
                print(f"Warning: Visible image not found for {infrared_image}")
                continue

        # 构建标签文件路径
        infrared_label_path = os.path.join(infrared_label_dir, f'{image_name}.txt')
        visible_label_path = os.path.join(visible_label_dir, f'{image_name}.txt')

        # 检查标签文件是否存在
        if not os.path.exists(infrared_label_path) or not os.path.exists(visible_label_path):
            print(f"Warning: Label files not found for {image_name}")
            continue

        # 读取标签数据
        try:
            visible_rects = read_label_file(visible_label_path)
            infrared_rects = read_label_file(infrared_label_path)

            if len(visible_rects) == 0 or len(infrared_rects) == 0:
                print(f"Warning: No bounding boxes found in labels for {image_name}")
                continue

            # 假设每张图片只有一个目标，使用第一个标注
            visible_rect = visible_rects[0]
            infrared_rect = infrared_rects[0]
        except Exception as e:
            print(f"Error reading label files for {image_name}: {e}")
            continue

        # 调整可见光图像与红外图像对齐
        translated_visible_image = adjust_image(visible_image_path, infrared_image_path, visible_rect, infrared_rect)

        if translated_visible_image is None:
            print(f"Warning: Failed to adjust image for {image_name}")
            continue

        # 保存对齐后的图像
        output_visible_path = os.path.join(output_visible_dir, f'{image_name}.jpg')
        output_infrared_path = os.path.join(output_infrared_dir, f'{image_name}.jpg')

        cv2.imwrite(output_visible_path, translated_visible_image)
        shutil.copy2(infrared_image_path, output_infrared_path)  # 复制原始红外图像

        # 保存红外标签文件（使用原始红外标签）
        output_label_path = os.path.join(output_label_dir, f'{image_name}.txt')
        save_yolo_labels([infrared_rect], output_label_path)

        processed_images += 1
        if processed_images % 100 == 0:
            print(f"Processed {processed_images}/{total_images}: {image_name}")

    print(f"Processing completed! Processed {processed_images} images out of {total_images} total.")


def main():
    print("Starting image alignment process...")
    process_folder()
    print("Image alignment process completed!")


if __name__ == "__main__":
    main()
