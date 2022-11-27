# -*-coding: UTF-8 -*-

import typing as T
import math

import numpy as np


def logistic(x: float, N: int, u: float = 4 - 2**(-50)) -> np.array:
    """
    混沌序列生成
    :params:
        x: float
            混沌变量初值
        N: int
            混沌序列长度
        u: float
            混沌控制参数
    :return:
        List[float]
            混沌序列
    """
    x, _ = math.modf(abs(x))  # 取绝对值并取小数部分
    logistic_array = np.zeros(N, dtype=float)
    for i in range(N):
        logistic_array[i] = x
        x = u * x * (1 - x)
    return logistic_array


def logistic2(x: float, N: int, k: float = 3001) -> T.List[float]:
    """
    改进的混沌序列生成
    :params:
        x: float
            混沌变量初值
        N: int
            混沌序列长度
        u: float
            混沌控制参数
    :return:
        List[float]
            混沌序列
    """
    x, _ = math.modf(abs(x))  # 取绝对值并取小数部分
    logistic_array = np.zeros(N, dtype=float)
    for i in range(N):
        logistic_array[i] = x
        x, _ = math.modf(k * x * (1 - x))
    return logistic_array
