import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查路径是否正确。")
        return None, None, None, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用双边滤波降噪，保留边缘
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 形态学闭运算填补字符空隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 形态学膨胀以增强字符（第一次）
    dilated_once = cv2.dilate(denoised, kernel, iterations=1)

    # 形态学膨胀以增强字符（第二次）
    dilated_twice = cv2.dilate(dilated_once, kernel, iterations=1)

    return image, gray, denoised, dilated_once, dilated_twice


def binarize_image(dilated):
    # 使用自适应阈值进行二值化
    thresh = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def vertical_projection(thresh):
    # 计算垂直投影
    projection = np.sum(thresh, axis=0)
    # 平滑投影以减少噪声
    smoothed_projection = gaussian_filter1d(projection, sigma=1)
    return smoothed_projection


def segment_by_projection(thresh, projection, min_width=5, max_width=60, min_segments=4):
    # 根据投影分割图像
    h, w = thresh.shape
    segments = []
    in_char = False
    start = 0

    for i in range(w):
        if projection[i] > 0 and not in_char:
            in_char = True
            start = i
        elif projection[i] == 0 and in_char:
            in_char = False
            end = i
            if min_width <= end - start <= max_width:  # 添加宽度过滤
                segments.append((start, end))

    # 检查是否有足够的分割
    if len(segments) < min_segments:
        segments = [(0, w)]  # 如果不满足最小分割数，将整幅图像作为一个分割

    return segments


def process_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []
    labels = []

    for folder in ['train', 'test']:
        folder_path = os.path.join(input_dir, folder)
        output_folder = os.path.join(output_dir, folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                label_text = filename[:-6].lower()  # 获取标签

                original_image, gray, denoised, dilated_once, dilated_twice = preprocess_image(image_path)
                if dilated_once is None or dilated_twice is None:
                    continue

                # 使用第二次膨胀的图像进行二值化和分割
                thresh = binarize_image(dilated_twice)
                projection = vertical_projection(thresh)
                segments = segment_by_projection(thresh, projection, min_width=5, max_width=60)

                if len(segments) == len(label_text):
                    for i, (start, end) in enumerate(segments):
                        # 输出第一次膨胀的图像
                        segment_img = dilated_once[:, start:end]
                        # 二值化增强输出的字符图像
                        segment_binarized = binarize_image(segment_img)
                        segment_resized = cv2.resize(segment_binarized, (30, 60))  # 调整大小为30x60
                        data.append(segment_resized)
                        labels.append(label_text[i])
                        segment_path = os.path.join(output_folder, f'{filename[:-4]}_char_{i}.png')
                        cv2.imwrite(segment_path, segment_resized)

    return np.array(data), np.array(labels)


def create_model():
    model = keras.Sequential([
        layers.Input(shape=(30, 60, 1)),
        layers.Conv2D(32, (4, 4), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # 加入Dropout层
        layers.Conv2D(64, (4, 4), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # 加入Dropout层
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # 调低Dropout层的值
        layers.Dense(36, activation='softmax')  # 36个输出，对应26个字母和10个数字
    ])

    # 设置Adam优化器并调整学习率
    optimizer = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True))
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    input_dir = 'dataset'  # 输入数据集路径
    output_dir = 'processed_dataset'  # 输出分割数据集路径

    # 处理数据集并生成标签
    data, labels = process_dataset(input_dir, output_dir)
    print(f"数据集包含 {len(data)} 个字符样本。")

    # 将数据标准化并调整维度
    data = data / 255.0  # 标准化
    data = data[..., np.newaxis]  # 添加通道维度

    # 将标签转换为数字编码
    char_to_num = {char: i for i, char in enumerate('abcdefghijklmnopqrstuvwxyz0123456789')}
    labels = np.array([char_to_num[char] for char in labels])

    # 拆分训练和测试集
    split_index = int(0.8 * len(data))
    x_train, y_train = data[:split_index], labels[:split_index]
    x_test, y_test = data[split_index:], labels[split_index:]

    # 创建和训练模型
    model = create_model()
    model.fit(x_train, y_train, epochs=15, validation_split=0.1, shuffle=True)

    # 测试模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"testing accuracy: {test_acc:.2f}")


if __name__ == "__main__":
    main()
