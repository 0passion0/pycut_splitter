"""
基于 Python 3.12 + OpenCV + 向量化 NumPy 的高级智能图像切割引擎模块。

对齐 ``smart-image-splitter`` Rust 核心契约，支持零拷贝 ROI 视口切片、
XY-Cut 递归投影切分、轮廓闭合检测与多策略统一尺寸规格化输出。

:author: tongliga
:date: 2026-11-18
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from .algorithms import (
    batch_crop_and_save_py,
    batch_crop_to_memory_py,
    detect_auto_py,
)
from .memory_safe import DEFAULT_MAX_PIXELS, decode_image_safe, validate_image_array

HAS_OPENCV_CORE = True

SplitMode = Literal["auto", "xycut", "contour", "grid"]
UniformMode = Literal["pad", "resize", "crop"]
_VALID_UNIFORM_MODES = ("pad", "resize", "crop")


@dataclass
class SplitBox:
    """
    单个检出的切片目标区域边界框与修边元数据载体。

    :author: tongliga
    :date: 2026-11-18
    """

    index: int
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    confidence: float
    strategy: str
    trimmed_edges: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (top, right, bottom, left)


@dataclass
class SplitResult:
    """
    整张输入图像的版面分析与切分综合结果载体。

    :author: tongliga
    :date: 2026-11-18
    """

    image_width: int
    image_height: int
    detected_mode: str
    backend: str
    elapsed_ms: float
    boxes: List[SplitBox] = field(default_factory=list)
    diagnostics: Optional[Dict[str, object]] = None


class SmartSplitter:
    """
    内存安全、向量化 OpenCV + NumPy 复合图像版面切分器。

    提供 ``analyze``、``split_to_bytes``、``split_file`` 以及外部机器学习分割器钩子，
    内置分块投影剖析、OpenCV C++ 形态学闭合与编解码，以及严格受控的瞬态内存开销。

    :author: tongliga
    :date: 2026-11-18
    """

    def __init__(
        self,
        mode: SplitMode = "auto",
        sensitivity: float = 0.65,
        min_region_ratio: float = 0.08,
        bleed_trim: int = 2,
        contour_gap_px: int = 8,
        ml_segmenter: Optional[Callable[[np.ndarray], List[Tuple[int, int, int, int]]]] = None,
        uniform_size: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        uniform_mode: UniformMode = "pad",
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        初始化智能图像切分器实例。

        :param mode: 切割模式，可选 "auto"（自动优选）、"xycut"（递归投影）、"contour"（连通域）、"grid"（等分网格）。
        :type mode: Literal["auto", "xycut", "contour", "grid"]
        :param sensitivity: 缝隙检测灵敏度，取值范围 (0.0, 1.0]，默认 0.65。
        :type sensitivity: float
        :param min_region_ratio: 最小有效子图面积相对全图的比例，过滤过小的噪点碎块，默认 0.08。
        :type min_region_ratio: float
        :param bleed_trim: 边缘去杂色去色渗透的内缩像素数，默认 2 像素。
        :type bleed_trim: int
        :param contour_gap_px: 轮廓模式下形态学闭合的膨胀间隙像素，默认 8 像素。
        :type contour_gap_px: int
        :param ml_segmenter: 可选的外部机器学习检测回调函数，接收图像数组并返回边界框四元组列表。
        :type ml_segmenter: Optional[Callable[[numpy.ndarray], List[Tuple[int, int, int, int]]]]
        :param uniform_size: 是否启用统一尺寸标准化输出，默认 False。
        :type uniform_size: bool
        :param target_size: 显式指定的统一切片目标尺寸 (宽度, 高度)；若未指定则自动采用检出切片的最大长宽。
        :type target_size: Optional[Tuple[int, int]]
        :param uniform_mode: 尺寸标准化对齐模式，可选 "pad"（补边）、"resize"（缩放）、"crop"（中心裁剪），默认 "pad"。
        :type uniform_mode: Literal["pad", "resize", "crop"]
        :param max_pixels: 允许加载与处理的单图最大像素数上限，默认 256M 像素。
        :type max_pixels: int
        :param max_workers: 批量切片导出与保存时的线程池最大工作线程数，默认根据 CPU 核心数自适应。
        :type max_workers: Optional[int]
        :raises ValueError: 当 uniform_mode 不在有效模式中，或 target_size 的高宽非正数时抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        if uniform_mode not in _VALID_UNIFORM_MODES:
            raise ValueError(
                f"uniform_mode must be one of {_VALID_UNIFORM_MODES}, got {uniform_mode!r}"
            )
        if target_size is not None and (target_size[0] <= 0 or target_size[1] <= 0):
            raise ValueError(
                f"target_size width and height must be positive, got {target_size!r}"
            )

        self.mode = mode
        self.sensitivity = sensitivity
        self.min_region_ratio = min_region_ratio
        self.bleed_trim = bleed_trim
        self.contour_gap_px = contour_gap_px
        self.ml_segmenter = ml_segmenter
        self.uniform_size = uniform_size or (target_size is not None)
        self.target_size = target_size
        self.uniform_mode: UniformMode = uniform_mode
        self.max_pixels = max_pixels
        self.max_workers = max_workers

    def analyze(
        self,
        image_bgr_or_rgb: np.ndarray,
        rows: int = 3,
        cols: int = 3,
        gutter_px: int = 0,
        collect_diagnostics: bool = False,
    ) -> SplitResult:
        """
        执行向量化版面布局检测算法，返回所有检出的子图边界框与元数据。

        :param image_bgr_or_rgb: 输入的三通道或四通道图像 NumPy 数组。
        :type image_bgr_or_rgb: numpy.ndarray
        :param rows: 网格模式或备用降级时的划分行数，默认 3。
        :type rows: int
        :param cols: 网格模式或备用降级时的划分列数，默认 3。
        :type cols: int
        :param gutter_px: 网格模式下行列间距的像素宽度，默认 0。
        :type gutter_px: int
        :param collect_diagnostics: 是否收集投影直方图与缝隙坐标等调试诊断数据，默认 False。
        :type collect_diagnostics: bool
        :return: 包含检测耗时、命中策略、边界框列表和诊断信息的结构体。
        :author: tongliga
        :date: 2026-11-18
        """
        # 第一步：记录起始计时并进行内存安全合规验证与只读视口绑定
        t0 = time.perf_counter()
        safe_img = validate_image_array(image_bgr_or_rgb, max_pixels=self.max_pixels)
        h, w = safe_img.shape[:2]

        # 第二步：若注入了外部 ML 机器学习分割器钩子，优先交由外部检测模型处理
        if self.ml_segmenter is not None:
            rects = self.ml_segmenter(safe_img)
            if rects:
                boxes = [
                    SplitBox(
                        index=i + 1,
                        row=i,
                        col=0,
                        x=max(0, min(int(rx), w - 1)),
                        y=max(0, min(int(ry), h - 1)),
                        width=max(1, min(int(bw), w - max(0, int(rx)))),
                        height=max(1, min(int(bh), h - max(0, int(ry)))),
                        confidence=0.99,
                        strategy="ml-segmentation",
                    )
                    for i, (rx, ry, bw, bh) in enumerate(rects)
                ]
                return SplitResult(
                    image_width=w,
                    image_height=h,
                    detected_mode="ml",
                    backend="ml-hook",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    boxes=boxes,
                )

        # 第三步：执行基于 OpenCV 矩阵运算的自适应版面分割算法
        detected_mode, raw_boxes, diag = detect_auto_py(
            safe_img,
            mode=self.mode,
            sensitivity=self.sensitivity,
            min_region_ratio=self.min_region_ratio,
            bleed_trim=self.bleed_trim,
            contour_gap_px=self.contour_gap_px,
            rows=rows,
            cols=cols,
            gutter_px=gutter_px,
            uniform_size=self.uniform_size,
            target_size=self.target_size,
            uniform_mode=self.uniform_mode,
            collect_diagnostics=collect_diagnostics,
        )

        # 第四步：封装并转换原始检测框为标准公共契约对象
        boxes = [
            SplitBox(
                index=b.index,
                row=b.row,
                col=b.col,
                x=b.x,
                y=b.y,
                width=b.width,
                height=b.height,
                confidence=b.confidence,
                strategy=b.strategy,
                trimmed_edges=(b.trim_top, b.trim_right, b.trim_bottom, b.trim_left),
            )
            for b in raw_boxes
        ]
        return SplitResult(
            image_width=w,
            image_height=h,
            detected_mode=detected_mode,
            backend="opencv-numpy-simd",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            boxes=boxes,
            diagnostics=diag,
        )

    def _resolve_target_size(self, boxes: List[SplitBox]) -> Optional[Tuple[int, int]]:
        """
        计算统一尺寸规格化输出时的目标分辨率 (宽, 高)。

        :param boxes: 当前图像检出的所有子图切片边界框列表。
        :type boxes: List[SplitBox]
        :return: 目标分辨率二元组 ``(target_w, target_h)``；若未开启统一模式或模式为 crop 则返回 None。
        :author: tongliga
        :date: 2026-11-18
        """
        if not self.uniform_size or self.uniform_mode == "crop" or not boxes:
            return None
        if self.target_size is not None:
            return self.target_size
        return (max(b.width for b in boxes), max(b.height for b in boxes))

    def split_to_bytes(
        self,
        image_bgr_or_rgb: np.ndarray,
        fmt: str = "png",
        rows: int = 3,
        cols: int = 3,
        gutter_px: int = 0,
    ) -> Tuple[SplitResult, List[bytes]]:
        """
        分析输入图像版面并将所有切片在内存中直接裁剪、规格化并编码为压缩字节流。

        :param image_bgr_or_rgb: 输入的三通道或四通道图像 NumPy 数组。
        :type image_bgr_or_rgb: numpy.ndarray
        :param fmt: 导出的压缩编码格式，可选 "png"、"jpg"、"webp"，默认 "png"。
        :type fmt: str
        :param rows: 网格模式或备用网格时的切分行数，默认 3。
        :type rows: int
        :param cols: 网格模式或备用网格时的切分列数，默认 3。
        :type cols: int
        :param gutter_px: 网格模式下切割沟槽间距像素，默认 0。
        :type gutter_px: int
        :return: 二元组包含 ``(版面分析结果结构体, 编码后字节列表)``。
        :author: tongliga
        :date: 2026-11-18
        """
        # 第一步：校验并分析图像获得所有子块的坐标
        safe_img = validate_image_array(image_bgr_or_rgb, max_pixels=self.max_pixels)
        res = self.analyze(safe_img, rows=rows, cols=cols, gutter_px=gutter_px)
        if not res.boxes:
            return res, []

        # 第二步：提取切片矩形坐标并释放 GIL 采用线程池执行并发编码
        coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
        blobs = batch_crop_to_memory_py(
            safe_img,
            coords,
            fmt=fmt,
            target_size=self._resolve_target_size(res.boxes),
            uniform_mode=self.uniform_mode,
            max_workers=self.max_workers,
        )
        return res, blobs

    def split_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        prefix: str = "slice",
        fmt: str = "png",
        rows: int = 3,
        cols: int = 3,
        gutter_px: int = 0,
    ) -> Tuple[SplitResult, List[str]]:
        """
        从磁盘安全解码图片、执行版面检测、并发裁剪、统一尺寸规格化并持久化到本地目录。

        :param input_path: 待切割的本地源图像文件路径。
        :type input_path: str | pathlib.Path
        :param output_dir: 保存切片文件的目标输出目录。
        :type output_dir: str | pathlib.Path
        :param prefix: 生成切片文件名称的前缀，默认 "slice"。
        :type prefix: str
        :param fmt: 输出图像格式，支持 "png"、"jpg"、"webp"，默认 "png"。
        :type fmt: str
        :param rows: 网格模式下的切分行数，默认 3。
        :type rows: int
        :param cols: 网格模式下的切分列数，默认 3。
        :type cols: int
        :param gutter_px: 网格缝隙消除像素，默认 0。
        :type gutter_px: int
        :return: 二元组包含 ``(版面分析结果结构体, 已成功持久化的切片绝对路径列表)``。
        :raises FileNotFoundError: 当源图像文件不存在时抛出。
        :author: tongliga
        :date: 2026-11-18
        """
        # 第一步：从磁盘安全解码源图像，并确保目标输出目录就绪
        img = decode_image_safe(input_path, max_pixels=self.max_pixels)
        res = self.analyze(img, rows=rows, cols=cols, gutter_px=gutter_px)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not res.boxes:
            return res, []

        # 第二步：提取目标区域坐标并调用并发批处理函数落地磁盘
        coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
        saved_paths = batch_crop_and_save_py(
            input_path,
            coords,
            out_dir,
            prefix=prefix,
            fmt=fmt,
            target_size=self._resolve_target_size(res.boxes),
            uniform_mode=self.uniform_mode,
            max_workers=self.max_workers,
        )
        return res, saved_paths
