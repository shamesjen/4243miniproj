import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    model = keras.Sequential([
        layers.Input(shape=(30, 60, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
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
    input_dir = 'processed_dataset'  # 使用已处理的数据集

    # 加载处理好的数据
    data = []
    labels = []

    for folder in ['train', 'test']:
        folder_path = os.path.join(input_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                label_text = filename.split('_')[0]  # 获取完整字符串
                char_index = int(filename.split('_')[-1].split('.')[0])  # 获取字符的索引
                label_char = label_text[char_index]  # 获取对应的字符

                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    data.append(cv2.resize(img, (30, 60)))
                    labels.append(label_char)

    # 将数据标准化并调整维度
    data = np.array(data) / 255.0  # 标准化
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
    print(f"测试集准确率: {test_acc:.2f}")

if __name__ == "__main__":
    main()
