"""
PyCut 图像切割引擎的内存安全守护、零拷贝只读视图与受限图像解码模块。

提供严格的像素上限截断（防止解压炸弹与 OOM）、图像数据格式合规校验、边界裁剪与零拷贝 ROI 切片。

:author: tongliga
:date: 2026-11-18
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

# 默认硬性防解压炸弹像素上限（256 Megapixels，约 16384x16384），防止恶意大图耗尽系统物理内存
DEFAULT_MAX_PIXELS = 256 * 1024 * 1024


def validate_image_array(
    image: np.ndarray,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> np.ndarray:
    """
    校验输入的三维图像矩阵（HxWx3 或 HxWx4）是否符合规范，并返回安全、非变异的只读视图。

    对 C 连续的 uint8 缓冲区实现零拷贝复用；对浮点型或非标准通道/跨步的数据执行安全归一化与类型收敛。

    :param image: 待校验的输入 NumPy 图像数组，形状必须为 HxWx3 (RGB/BGR) 或 HxWx4 (RGBA/BGRA)。
    :type image: numpy.ndarray
    :param max_pixels: 允许处理的单张图像最大像素总数 (宽 × 高)，默认 268,435,456 (256M 像素)。
    :type max_pixels: int
    :return: 经过安全校验且标记为 write=False 的只读 NumPy 图像视图。
    :raises ValueError: 当输入对象不是 numpy.ndarray、维度不为 3、宽高非正整数或通道数非 3/4 时抛出。
    :raises MemoryError: 当图像总像素数超出 max_pixels 安全阈值时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：校验输入基础类型与维度形状
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(image).__name__}.")
    if image.ndim != 3:
        raise ValueError(f"Expected HxWx3 or HxWx4 image array, got ndim={image.ndim}.")

    # 第二步：校验图像长宽有效性与通道数
    h, w, ch = image.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Image dimensions must be positive, got {w}x{h}.")
    if ch not in (3, 4):
        raise ValueError(f"Expected 3 (RGB/BGR) or 4 (RGBA/BGRA) channels, got {ch}.")

    # 第三步：防解压炸弹与内存溢出像素硬上限检查
    if w * h > max_pixels:
        raise MemoryError(
            f"Image pixel count ({w}x{h}={w * h:,}) exceeds safety limit ({max_pixels:,})."
        )

    # 第四步：像素数据类型规范化（将浮点型或非 uint8 数组截断并转换为 uint8）
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            max_val = float(np.nanmax(image)) if image.size else 1.0
            scale = 255.0 if max_val <= 1.0 else 1.0
            arr = np.clip(np.nan_to_num(image) * scale, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(image, 0, 255).astype(np.uint8)
    else:
        arr = image

    # 第五步：生成非破坏性零拷贝只读视图，锁定 write 标志防止底层缓存被下游意外修改
    ro_view = arr.view()
    ro_view.setflags(write=False)
    return ro_view


def clamp_box(x: int, y: int, bw: int, bh: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    将待裁剪的目标区域坐标严格约束在图像实际像素范围 [0, img_w] × [0, img_h] 之内。

    保证裁剪起点坐标非负、宽高至少为 1 像素，且边界不会溢出图像右下角。

    :param x: 区域左上角横坐标。
    :type x: int
    :param y: 区域左上角纵坐标。
    :type y: int
    :param bw: 区域建议宽度。
    :type bw: int
    :param bh: 区域建议高度。
    :type bh: int
    :param img_w: 图像原始宽度。
    :type img_w: int
    :param img_h: 图像原始高度。
    :type img_h: int
    :return: 严格约束后的四元组 ``(x0, y0, width, height)``。
    :author: tongliga
    :date: 2026-11-18
    """
    # 限制起始坐标在 [0, img - 1] 区间
    x0 = max(0, min(int(x), max(0, img_w - 1)))
    y0 = max(0, min(int(y), max(0, img_h - 1)))
    # 限制宽高至少为 1，且不超过剩余可用像素跨度
    cw = max(1, min(int(bw), img_w - x0))
    ch = max(1, min(int(bh), img_h - y0))
    return x0, y0, cw, ch


def safe_roi_view(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """
    基于绝对坐标边界，截取严格在图像尺寸范围内的零拷贝 NumPy 视口切片。

    :param img: 原始图像数组（HxWx3 或 HxWx4）。
    :type img: numpy.ndarray
    :param x0: 切片起始横坐标。
    :type x0: int
    :param y0: 切片起始纵坐标。
    :type y0: int
    :param x1: 切片结束横坐标。
    :type x1: int
    :param y1: 切片结束纵坐标。
    :type y1: int
    :return: 处于合法范围内的二维/三维切片视口（不发生底层像素数据拷贝）。
    :author: tongliga
    :date: 2026-11-18
    """
    # 获取图像物理高宽并双向截断切片端点
    h, w = img.shape[:2]
    cx0 = max(0, min(int(x0), w))
    cy0 = max(0, min(int(y0), h))
    cx1 = max(cx0, min(int(x1), w))
    cy1 = max(cy0, min(int(y1), h))
    return img[cy0:cy1, cx0:cx1]


def decode_image_safe(
    input_path: str | Path,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> np.ndarray:
    """
    从磁盘安全解码图片文件为 RGB 或 RGBA 格式的 uint8 NumPy 数组。

    采用先通过 Pillow 读取文件头元数据评估像素上限、再通过 OpenCV C++ 内存映射解码的高性能链路，
    完美支持 Windows 中文/Unicode 路径，并在特殊格式（如 CMYK、调色板图）下安全回落至 Pillow。

    :param input_path: 目标图像文件的本地磁盘绝对或相对路径。
    :type input_path: str | pathlib.Path
    :param max_pixels: 允许解码的最大像素总数，默认 256M 像素。
    :type max_pixels: int
    :return: 解码后标准化为 RGB 或 RGBA 排列的 uint8 类型图像数组。
    :raises FileNotFoundError: 当指定路径文件不存在或非普通文件时抛出。
    :raises ValueError: 当图像头尺寸解析异常（宽高非正）时抛出。
    :raises MemoryError: 当图像总像素数超出 max_pixels 安全阈值时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：验证文件物理存在性
    p = Path(input_path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {p}")

    # 第二步：利用 Pillow 快速探测图片头信息，先于全量解码防范解压炸弹攻击
    with Image.open(p) as probe:
        pw, ph = probe.size
        if pw <= 0 or ph <= 0:
            raise ValueError(f"Corrupt image dimensions: {pw}x{ph}")
        if pw * ph > max_pixels:
            raise MemoryError(
                f"Image {p.name} ({pw}x{ph}={pw * ph:,} px) exceeds safety limit ({max_pixels:,} px)."
            )

    # 第三步：采用 np.fromfile 绕过 OpenCV 在 Windows 下无法直接打开 Unicode 路径的问题
    raw_bytes = np.fromfile(str(p), dtype=np.uint8)
    decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)

    # 第四步：若 OpenCV 解码成功，标准化 BGR/BGRA 到 RGB/RGBA 颜色通道
    if decoded is not None and decoded.ndim == 3:
        if decoded.shape[2] == 4:
            return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
        if decoded.shape[2] == 3:
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif decoded is not None and decoded.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)

    # 第五步：对于特殊格式（如 CMYK 印刷图、调色板 PNG 等），回退由 Pillow 执行颜色空间转换
    with Image.open(p) as pil_img:
        mode_img = pil_img.convert("RGBA") if "A" in pil_img.getbands() else pil_img.convert("RGB")
        return np.asarray(mode_img, dtype=np.uint8)
