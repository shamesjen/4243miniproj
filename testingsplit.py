import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 形态学膨胀以增强字符
    dilated = cv2.dilate(closed, kernel, iterations=1)

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
    smoothed_projection = gaussian_filter1d(projection, sigma=3)
    return smoothed_projection


def segment_by_projection(thresh, projection, min_width=5, max_width=100, min_segments=1):
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


def main():
    image_path = 'dataset/train/0bdm8-0.png'  # 替换为您的图像路径

    # 预处理
    original_image, gray, denoised, dilated = preprocess_image(image_path)
    if dilated is None:
        return

    # 二值化
    thresh = binarize_image(dilated)

    # 计算垂直投影
    projection = vertical_projection(thresh)

    # 分割字符
    segments = segment_by_projection(thresh, projection, min_width=5, max_width=100, min_segments=1)  # 调整最小和最大宽度

    # 显示处理步骤和分割结果
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Grayscale Image')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Dilated Image')
    plt.imshow(dilated, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Binarized Image')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Character Segmentation')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    # 绘制分割线
    for start, end in segments:
        plt.axvline(x=start, color='red', linewidth=1)
        plt.axvline(x=end, color='red', linewidth=1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
