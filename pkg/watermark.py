# -*-coding: UTF-8 -*-

import typing as T
import math
import functools

import enum

# conda install opencv
import cv2 as cv

# conda install numpy
import numpy as np

# conda install moviepy
import moviepy.editor as editor

# conda install matplotlib
# from matplotlib import pyplot as plt

# conda install tensorly
from tensorly.decomposition._tucker import partial_tucker

# conda install pywavelets
import pywt

# conda install scipy
# import scipy

from . import chaos


# 模式
@enum.unique
class MODE(enum.Enum):
    DEV = 1  # 开发模式, 视频的一组帧转换为灰度图片后嵌入水印
    PROD = 2  # 生产模式, 视频所有符合条件的帧的三个颜色通道嵌入水印


# 变换
@enum.unique
class TRAN(enum.Enum):
    FORWARD = 1  # 正变换
    INVERSE = 2  # 反变换


class Watermark(object):
    def __init__(
        self,
        x: float,  # 混沌序列初值
        videoPath: str,  # 视频文件路径
        imagePath: str,  # 图像文件路径
        QP: float = 40,  # 量化参数
        windowSize: T.Tuple[int, int] = (8, 8),  # DCT 窗口大小(高, 宽)
        embedPosition: T.Tuple[int, int] = (4, 4),  # 数据嵌入位置
        embedOffsets: T.List[T.Tuple[int, int]] = [  # 数据嵌入位置相邻系数的偏移量
            (3, -3),
            (2, -2),
            (1, -1),
            (-1, 1),
            (-2, 2),
            (-3, 3),
        ],
        chaoticMaker: T.Callable[[float, int], np.array] = chaos.logistic2,  # 混沌序列生成器
        lockAspectRatio: bool = True,  # 水印图片是否锁定纵横比 (True: 缩放至合适大小后以比特序列方式嵌入, False: 缩放拉伸后嵌入)
        mode: MODE = MODE.DEV,  # 模式
        frames: slice = slice(0, 8),  # 帧切片
    ):
        self.x = x
        self.videoPath = videoPath
        self.imagePath = imagePath
        self.QP = QP
        self.windowSize = windowSize
        self.embedPosition = embedPosition
        self.embedOffsets = embedOffsets
        self.mode = mode
        self.frames = frames

        self.video = editor.VideoFileClip(videoPath)  # 载体视频
        self.viedo_fps = self.video.fps  # 帧率
        self.video_nframes = self.video.reader.nframes  # 总帧数
        self.video_size = self.video.size  # 视频尺寸(高, 宽)
        self.image = cv.imread(imagePath)  # 水印图片

        self.m_N = math.floor(self.video.h / 2 // self.windowSize[0])  # 分块后的行数
        self.n_N = math.floor(self.video.w / 2 // self.windowSize[1])  # 分块后的列数

        self.embedded_capacity = self.m_N * self.n_N  # 嵌入容量

        self.image_width = self.image.shape[1]  # 图片宽度
        self.image_height = self.image.shape[0]  # 图片高度

        self.image_aspect_ratio = self.image_width / self.image_height  # 图片宽高比

        if lockAspectRatio:  # 水印保持原始宽高比
            self.watermark_width = math.floor((self.embedded_capacity * self.image_aspect_ratio) ** 0.5)  # 水印的宽度
            self.watermark_height = math.floor(self.watermark_width / self.image_aspect_ratio)  # 水印的高度
        else:  # 水印拉伸为视频宽高比
            self.watermark_width = self.n_N  # 水印的宽度
            self.watermark_height = self.m_N  # 水印的高度

        self.watermark_shape = (self.watermark_height, self.watermark_width)
        self.watermark_size = self.watermark_width * self.watermark_height

        self.chaoticMaker = chaoticMaker
        self.logistic_vector = self.chaoticMaker(self.x, self.embedded_capacity)  # 混沌序列
        self.logistic_bits_vector = self.logistic_vector > 0.5  # 混沌比特序列

    # 彩色水印图片转换为灰度图片
    def imageColor2Gray(self) -> object:
        self.image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        return self

    # 灰度水印图片二值化
    def imageGray2Binary(
        self,
        thresh: None | float = None,
        **kw,
    ) -> object:
        """
        灰度图片二值化
        :params:
            thresh: None | float = None
                None: 动态确定二值化阈值
                float: 指定阈值
        """
        if thresh is None:
            self.thresh, self.image_binary = cv.threshold(self.image_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        else:
            self.thresh, self.image_binary = cv.threshold(self.image_gray, self.thresh, 255, cv.THRESH_BINARY)
        return self

    # 调整水印图像大小
    def imageResize(
        self,
        width: None | int = None,
        height: None | int = None,
    ) -> object:
        if width is not None:
            self.watermark_width = width
        if height is not None:
            self.watermark_height = height
        self.watermark_image = cv.resize(self.image_binary, dsize=(self.watermark_width, self.watermark_height))
        return self

    # 序列化/反序列化水印图像
    def serialize(
        self,
        tran: TRAN = TRAN.FORWARD,
    ) -> object:
        match tran:
            case TRAN.FORWARD:
                self.watermark_bits_image = self.watermark_image.astype(np.bool8)  # bool 表示的水印图片
                self.watermark_bits_vector = self.watermark_bits_image.ravel()  # bool 表示的水印向量

            case TRAN.INVERSE:
                self.extract_watermark_bits_image = self.extract_watermark_bits_vector.reshape(self.watermark_shape)  # bool 表示的提取水印图片
                self.extract_watermark_image = self.extract_watermark_bits_image.astype(np.uint8)  # uint8 表示的提取水印图片
        return self

    # 置乱/反置乱水印字节序列
    def scramble(
        self,
        tran: TRAN = TRAN.FORWARD,
    ) -> object:
        """
        置乱/反置乱水印字节序列
        :params:
            tran: TRAN
                变换方向
        :return:
            self
        """
        match tran:
            case TRAN.FORWARD:
                self.watermark_padded_bits_vector = np.concatenate((
                    self.watermark_bits_vector,
                    np.zeros(self.embedded_capacity - self.watermark_size, dtype=np.bool8),
                ))  # 填充后的水印比特序列
                self.watermark_scrambled_padded_bits_vector = self.watermark_padded_bits_vector ^ self.logistic_bits_vector  # 置乱后的水印比特序列
            case TRAN.INVERSE:
                self.extract_watermark_padded_bits_vector = self.extract_watermark_scrambled_padded_bits_vector ^ self.logistic_bits_vector  # 恢复后的水印填充比特序列
                self.extract_watermark_bits_vector = self.extract_watermark_padded_bits_vector[:self.watermark_size]  # 恢复后的水印比特序列

        return self

    # 提取/重建帧
    def frame(
        self,
        frames: None | slice = None,  # 视频切片
        tran: TRAN = TRAN.FORWARD,
    ) -> object:
        if frames is not None:
            self.frames = frames
        frame_range = list(range(*self.frames.indices(self.video_nframes)))
        match tran:
            case TRAN.FORWARD:
                self.tensor_k = len(frame_range)  # 张量第三维度的长度
                self.tensor = np.zeros((*reversed(self.video_size), self.tensor_k))  # (高, 宽, 帧数)
                k = 0
                for i in frame_range:
                    self.tensor[..., k] = cv.cvtColor(self.video.get_frame(i / self.viedo_fps), cv.COLOR_RGB2GRAY)
                    k += 1
            case TRAN.INVERSE:
                pass

        return self

    # 张量分解/重建
    def tensorDecompose(
        self,
        tran: TRAN = TRAN.FORWARD,
    ) -> object:
        match tran:
            case TRAN.FORWARD:
                self.tensor_core, self.tensor_U3 = partial_tucker(self.tensor, modes=[2])  # 张量 Tucker 分解
                self.tensor_U3 = self.tensor_U3[0]  # 获得 U_3 分量
            case TRAN.INVERSE:
                self.embedded_tensor_uint8 = self.embedded_tensor.astype(np.uint8)
        return self

    # 特征张量
    def tensorFeature(
        self,
        tran: TRAN = TRAN.FORWARD,
    ) -> object:
        match tran:
            case TRAN.FORWARD:
                self.tensor_zeta = np.einsum('ijk, lk->ijl', self.tensor, self.tensor_U3.T)  # 特征张量
            case TRAN.INVERSE:
                self.embedded_tensor = np.einsum('ijk, lk->ijl', self.embedded_tensor_zeta, self.tensor_U3)  # 嵌入水印后的张量
        return self

    # 特征图
    def tensorFeatureImage(
        self,
        tran: TRAN = TRAN.FORWARD,
    ) -> object:
        match tran:
            case TRAN.FORWARD:
                self.tensor_F = self.tensor_zeta[..., 0]  # 特征张量的特征图
            case TRAN.INVERSE:
                self.embedded_tensor_zeta = self.tensor_zeta.copy()  # 嵌入水印后的特征张量
                self.embedded_tensor_zeta[..., 0] = self.embedded_tensor_F
        return self

    # 离散小波变换
    def haar(
        self,
        tran: TRAN = TRAN.FORWARD,
    ) -> object:
        match tran:
            case TRAN.FORWARD:
                self.tensor_F_haar = pywt.dwt2(self.tensor_F, 'haar')  # 一阶 DWT 变换结果
                self.tensor_F_haar_LL = self.tensor_F_haar[0]  # 一阶 DWT 变换结果的 LL 子带
            case TRAN.INVERSE:
                self.embedded_tensor_F_haar = (self.embedded_tensor_F_haar_LL, self.tensor_F_haar[1])  # 嵌入后的一阶 DWT 变换结果
                self.embedded_tensor_F = pywt.idwt2(self.embedded_tensor_F_haar, 'haar')  # 一阶 DWT 逆变换
        return self

    #  DCT 变换
    def _dct(self, i: int, j: int) -> T.Tuple[slice, slice, np.array, float]:
        slice_i = slice(i * self.windowSize[0], (i + 1) * self.windowSize[0])  # 纵向窗口
        slice_j = slice(j * self.windowSize[1], (j + 1) * self.windowSize[1])  # 横向窗口
        block = cv.dct(self.tensor_F_haar_LL[slice_i, slice_j])  # 对窗口进行 DCT 变换
        avg = functools.reduce(
            lambda x, y: x + y,
            map(
                lambda offset: block[self.embedPosition[0] + offset[0], self.embedPosition[1] + offset[1]],
                self.embedOffsets,
            ),
        ) / len(self.embedOffsets)  # 参考平均值
        return slice_i, slice_j, block, avg

    # 水印嵌入
    def embed(self) -> object:
        bit_iter = iter(self.watermark_scrambled_padded_bits_vector)
        self.embedded_tensor_F_haar_LL = self.tensor_F_haar[0].copy()  # 嵌入后的一阶 DWT 变换结果的 LL 子带
        for i in range(self.m_N):
            for j in range(self.n_N):
                slice_i, slice_j, block, avg = self._dct(i, j)
                block[self.embedPosition[0]][self.embedPosition[1]] = avg + self.QP * (1 if next(bit_iter) else -1)  # 根据参考值嵌入中频系数
                self.embedded_tensor_F_haar_LL[slice_i, slice_j] = cv.idct(block)  # 嵌入水印后的 DCT 系数进行 DCT 逆变换
        return self

    # 水印提取
    def extract(self) -> object:
        self.extract_watermark_scrambled_padded_bits_vector = np.zeros(self.embedded_capacity, dtype=np.bool8)  # 提取的混淆填充水印比特向量
        count = 0
        for i in range(self.m_N):
            for j in range(self.n_N):
                _, _, block, avg = self._dct(i, j)
                self.extract_watermark_scrambled_padded_bits_vector[count] = block[self.embedPosition[0]][self.embedPosition[1]] > avg
                count += 1
        return self

    def transparency(
        self,
        evalfuncs: T.List[T.Callable[[np.array, np.array], float]],
        QPs: None | float | T.Iterable = None,
        slices: None | T.List[slice] = None,
    ) -> T.Dict[str, T.List[T.List[float]]]:
        """
        透明性评估
        :params:
            evalfuncs: T.List[T.Callable[[np.array, np.array], float]]
                水印透明性评估函数列表
            QPs: None | float | Iterable = None
                量化参数
                None: 量化参数为原设定值
                float: 量化参数为该值
                Iterable: 对量化参数进行扫描
            slices: None | T.List[slice] = None
                帧切片
                None: 使用默认的帧切片评估
                T.List[slice, slice]: 使用指定的帧切片
        :return:
            T.Dict[str, T.List[T.List[float]]]
                透明性评估值
                str: 评估函数名
                T.List[T.List[float]]: 多个量化参数的透明性评估值列表
                    T.List[float]: 一个量化参数的不同帧的透明性评估结果
                        float: 一个量化参数的一帧的透明性评估结果
        """
        eval_result = dict()  # 评估结果
        for evalfunc in evalfuncs:
            eval_result[evalfunc.__name__] = []

        if isinstance(QPs, T.Iterable):
            pass
        elif isinstance(QPs, float):
            QPs = [QPs]
        else:
            QPs = [self.QP]

        if isinstance(slices, T.Iterable):
            pass
        else:
            slices = [self.frames]

        self.imageColor2Gray() \
            .imageGray2Binary() \
            .imageResize() \
            .serialize() \
            .scramble()

        for self.frames in slices:
            self.frame() \
                .tensorDecompose() \
                .tensorFeature() \
                .tensorFeatureImage() \
                .haar()

            for self.QP in QPs:
                self.embed() \
                    .haar(tran=TRAN.INVERSE) \
                    .tensorFeatureImage(tran=TRAN.INVERSE) \
                    .tensorFeature(tran=TRAN.INVERSE) \
                    .tensorDecompose(tran=TRAN.INVERSE)
                for evalfunc in evalfuncs:
                    eval_result[evalfunc.__name__].append([
                        evalfunc(
                            self.tensor[..., i],
                            self.embedded_tensor_uint8[..., i]
                        ) for i in range(self.tensor.shape[2])
                    ])

        return eval_result

    def robustness(
        self,
        attackfunc: T.Callable[[np.array], np.array],
        evalfuncs: T.List[T.Callable[[np.array, np.array], float]],
        QPs: None | float | T.Iterable = None,
    ) -> T.Dict[str, T.List[float]]:
        """
        鲁棒性评估
        :params:
            attackfunc: T.Callable[[np.array], np.array]
                攻击方法
            evalfuncs: T.List[T.Callable[[np.array, np.array], float]]
                水印鲁棒性评估函数列表
            QPs: None | float | Iterable = None
                量化参数
                None: 量化参数为原设定值
                float: 量化参数为该值
                Iterable: 对量化参数进行扫描
        :return:
            T.Dict[str, T.List[float]]
                str: 评估函数名
                 T.List[float]]: 多个量化参数的鲁棒性评估值列表
                    float: 一个量化参数的鲁棒性评估结果
        """
        eval_result = dict()  # 评估结果
        for evalfunc in evalfuncs:
            eval_result[evalfunc.__name__] = []

        self.imageColor2Gray() \
            .imageGray2Binary() \
            .imageResize() \
            .serialize() \
            .scramble()

        if isinstance(QPs, T.Iterable):
            pass
        elif isinstance(QPs, float):
            QPs = [QPs]
        else:
            QPs = [self.QP]

        for self.QP in QPs:
            self.frame() \
                .tensorDecompose() \
                .tensorFeature() \
                .tensorFeatureImage() \
                .haar() \
                .embed() \
                .haar(tran=TRAN.INVERSE) \
                .tensorFeatureImage(tran=TRAN.INVERSE) \
                .tensorFeature(tran=TRAN.INVERSE) \
                .tensorDecompose(tran=TRAN.INVERSE)
            self.tensor = attackfunc(self.embedded_tensor_uint8)
            self.tensorDecompose() \
                .tensorFeature() \
                .tensorFeatureImage() \
                .haar() \
                .extract() \
                .scramble(tran=TRAN.INVERSE) \
                .serialize(tran=TRAN.INVERSE)
            for evalfunc in evalfuncs:
                eval_result[evalfunc.__name__].append(evalfunc(
                    self.watermark_bits_vector,
                    self.extract_watermark_bits_vector,
                ))

        return eval_result
