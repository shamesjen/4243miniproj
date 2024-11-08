import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d


def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查路径是否正确。")
        return None, None, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用双边滤波降噪，保留边缘
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 形态学闭运算填补字符空隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 形态学膨胀以增强字符
    dilated = cv2.dilate(denoised, kernel, iterations=2)

    return image, gray, denoised, dilated


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


def segment_by_projection(thresh, projection, min_width=5, max_width=100, min_segments=4):
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
            if min_width <= end - start <= max_width:
                segments.append((start, end))

    if len(segments) < min_segments:
        segments = [(0, w)]

    return segments


def test_segmentation_on_dataset(dataset_path):
    total_images = 0
    successfully_segmented = 0

    for filename in os.listdir(dataset_path):
        if filename.endswith(".png"):
            total_images += 1
            image_path = os.path.join(dataset_path, filename)
            label = filename[:-6]  # 获取标签
            expected_length = len(label)

            original_image, gray, denoised, dilated = preprocess_image(image_path)
            if dilated is None:
                continue

            thresh = binarize_image(dilated)
            projection = vertical_projection(thresh)
            segments = segment_by_projection(thresh, projection, min_width=5, max_width=100, min_segments=1)

            if len(segments) == expected_length:
                successfully_segmented += 1

    return total_images, successfully_segmented


def main():
    dataset_path = 'dataset/test'  # 替换为您的本地数据集路径
    total_images, successfully_segmented = test_segmentation_on_dataset(dataset_path)

    print(f"total image: {total_images}")
    print(f"correct: {successfully_segmented}")
    if total_images > 0:
        print(f"rate: {successfully_segmented / total_images * 100:.2f}%")


if __name__ == "__main__":
    main()
