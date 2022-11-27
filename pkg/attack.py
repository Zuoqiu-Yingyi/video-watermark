# -*-coding: UTF-8 -*-
import typing as T

import numpy as np
import cv2 as cv


# 无攻击
def none(src: np.array) -> np.array:
    return src.copy()


# 帧删除攻击
def frameDelete(src: np.array, count: int = 0) -> np.array:
    return src[..., count:].copy()


# 缩放攻击
def scale(src: np.array, rate: float) -> np.array:
    rst = src.copy()
    height, width, _ = src.shape
    dsize = (width, height)
    dsize_scale = (int(width * rate), int(height * rate))
    rst = cv.resize(cv.resize(src, dsize=dsize_scale), dsize=dsize)
    return rst


# 旋转攻击
def rotate(src: np.array, angle: float, recover: bool = True) -> np.array:
    h, w, k = src.shape
    if recover:
        # 不裁剪旋转恢复
        # h_stage, w_stage = h * 3, w * 3
        # dsize_stage = (w_stage, h_stage)
        # stage = np.zeros((h_stage, w_stage, k))
        # stage[h:h*2, w:w*2, :] = src.copy()
        # M1 = cv.getRotationMatrix2D(((w_stage-1)/2, (h_stage-1)/2), angle, 1)
        # M2 = cv.getRotationMatrix2D(((w_stage-1)/2, (h_stage-1)/2), -angle, 1)
        # return cv.warpAffine(cv.warpAffine(stage, M1, dsize=dsize_stage), M2, dsize=dsize_stage)[h:h*2, w:w*2, :].copy()

        # 裁剪旋转恢复
        dsize_stage = (w, h)
        M1 = cv.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1)
        M2 = cv.getRotationMatrix2D(((w-1)/2, (h-1)/2), -angle, 1)
        return cv.warpAffine(cv.warpAffine(src, M1, dsize=dsize_stage), M2, dsize=dsize_stage)
    else:
        dsize_stage = (w, h)
        M1 = cv.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1)
        return cv.warpAffine(src, M1, dsize=dsize_stage)


# 裁剪攻击
def cut(src: np.array, slices: T.List[T.Tuple[slice, slice]] = []) -> np.array:
    rst = src.copy()
    for s in slices:
        rst[s[0], s[1], ...] = 0
    return rst


# 噪声攻击
def noise(src: np.array, handler: T.Callable[[np.array], np.array], *args, **kw) -> np.array:
    return handler(src, *args, **kw)


# 滤波攻击
def filte(src: np.array, handler: T.Callable[[np.array], np.array], *args, **kw) -> np.array:
    return handler(src, *args, **kw)
