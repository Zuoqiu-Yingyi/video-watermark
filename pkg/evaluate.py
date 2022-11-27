import math

# conda install numpy
import numpy as np


def MSE(img1: np.array, img2: np.array) -> float:
    """
    计算两张图片的均方误差 (Mean Squared Error, MSE)
    :params:
        img1: np.array
            图片1
        img2: np.array
            图片2
    :return:
        float
            均方误差
    """
    return np.sum((img1 - img2) ** 2) / (img1.shape[0] * img1.shape[1])


def PSNR(img1: np.array, img2: np.array) -> float:
    """
    计算两张图片的峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)
    :params:
        img1: np.array
            图片1
        img2: np.array
            图片2
    :return:
        float
            峰值信噪比
    """
    return 10 * math.log10(255 ** 2 / MSE(img1, img2))


def BER(watermark1: np.array, watermark2: np.array) -> float:
    """
    计算水印序列的误比特率 (Bit Error Rate, BER)
    :params:
        watermark1: np.array
            原始的水印序列
        watermark2: np.array
            提取的水印序列
    :return:
        float
            误比特率
    """
    return np.sum(watermark1 != watermark2) / len(watermark1)


def NC(watermark1: np.array, watermark2: np.array) -> float:
    """
    计算水印序列的相关系数 (Normalized Correlation, NC)
    :params:
        watermark1: np.array
            图片1
        watermark2: np.array
            图片2
    :return:
        float
            相关系数
    """
    return np.sum(watermark1 * watermark2) / (np.sum(watermark1 ** 2) * np.sum(watermark2 ** 2)) ** 0.5
