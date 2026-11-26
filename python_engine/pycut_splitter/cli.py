"""
PyCut 图像切割命令行工具模块（提供终端交互与批处理执行能力）。

支持多目录文件扫描、各切割模式参数指定、统一切片尺寸规格化与实时终端反馈。

:author: tongliga
:date: 2026-11-18
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from . import __version__
from .engine import HAS_OPENCV_CORE, SmartSplitter


def _parse_target_size(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    解析终端传入的统一尺寸字符串参数（格式如 "512x512"、"1920x1080"）。

    :param spec: 形如 "WIDTHxHEIGHT" 的字符串；若为 None 则返回 None。
    :type spec: Optional[str]
    :return: 包含 ``(width, height)`` 的二元整型元组，或 None。
    :raises ValueError: 当规格字符串无法以 'x' 分割为两个元素，或解析出的高宽非正整数时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    if spec is None:
        return None
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid --target-size '{spec}', expected WIDTHxHEIGHT such as 512x512")
    w = int(parts[0].strip())
    h = int(parts[1].strip())
    if w <= 0 or h <= 0:
        raise ValueError(f"--target-size dimensions must be positive, got {w}x{h}")
    return (w, h)


def main() -> None:
    """
    PyCut Splitter CLI 命令行主入口函数。

    解析命令行参数、构造 SmartSplitter 实例并驱动单图或批量图片的自动化切割与保存。

    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：构建命令行参数解析器并注册所有配置选项
    parser = argparse.ArgumentParser(
        prog="pycut-split",
        description="Memory-safe, vectorized OpenCV + NumPy composite image splitter.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"pycut-split {__version__}")
    parser.add_argument("input", type=Path, help="Input image path or directory of images")
    parser.add_argument("-o", "--output", type=Path, default=Path("./output_slices"), help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["auto", "xycut", "contour", "grid"],
        default="auto",
        help="Layout detection mode (default: auto)",
    )
    parser.add_argument("--sensitivity", type=float, default=0.65, help="Detection sensitivity (0.1 - 1.0)")
    parser.add_argument("--min-region-ratio", type=float, default=0.08, help="Min region ratio (0.03 - 0.35)")
    parser.add_argument("--bleed-trim", type=int, default=2, help="Bleed decontamination pixels")
    parser.add_argument("--contour-gap", type=int, default=8, help="Contour closing gap in px")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows")
    parser.add_argument("--cols", type=int, default=3, help="Grid columns")
    parser.add_argument("--gutter-px", type=int, default=0, help="Grid gutter pixels")
    parser.add_argument("--grid-refine", action="store_true", help="Refine cut positions in grid mode")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "webp"], help="Output image format")
    parser.add_argument("--uniform-size", action="store_true", help="Normalize all slices to uniform dimensions")
    parser.add_argument("--target-size", default=None, help="Explicit uniform size WIDTHxHEIGHT, e.g. 512x512")
    parser.add_argument(
        "--uniform-mode",
        default="pad",
        choices=["pad", "resize", "crop"],
        help="Uniform sizing strategy: pad, resize, or crop",
    )

    # 第二步：解析参数并实例化核心切割器
    args = parser.parse_args()
    parsed_target_size = _parse_target_size(args.target_size)

    splitter = SmartSplitter(
        mode=args.mode,
        sensitivity=args.sensitivity,
        min_region_ratio=args.min_region_ratio,
        bleed_trim=args.bleed_trim,
        contour_gap_px=args.contour_gap,
        uniform_size=args.uniform_size,
        target_size=parsed_target_size,
        uniform_mode=args.uniform_mode,
        grid_refine=args.grid_refine,
    )

    # 第三步：收集目标输入文件列表（支持单个文件或目录批处理）
    inputs = (
        sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if args.input.is_dir()
        else [args.input]
    )

    # 第四步：循环处理每张图像并输出控制台汇总信息
    print(f"PyCut Splitter v{__version__} | OpenCV+NumPy SIMD Core: {'ACTIVE' if HAS_OPENCV_CORE else 'INACTIVE'}")
    for img_path in inputs:
        sub_out = args.output / img_path.stem if args.input.is_dir() else args.output
        result, saved = splitter.split_file(
            img_path,
            sub_out,
            prefix=img_path.stem,
            fmt=args.format,
            rows=args.rows,
            cols=args.cols,
            gutter_px=args.gutter_px,
        )
        print(
            f"  [OK] {img_path.name} ({result.image_width}x{result.image_height}) -> "
            f"{len(saved)} slices [{result.detected_mode} / {result.backend}] in {result.elapsed_ms:.1f}ms"
        )


if __name__ == "__main__":
    main()
