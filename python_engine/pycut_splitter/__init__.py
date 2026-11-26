"""
PyCut Image Splitter 纯 Python + OpenCV SIMD + 向量化 NumPy 核心模块。

提供具备工业级内存安全守卫、极速吞吐、零拷贝视口切片与多策略自适应智能切图能力。

:author: tongliga
:date: 2026-11-18
"""

from .algorithms import (
    batch_crop_and_save_py,
    batch_crop_to_memory_py,
    detect_auto_py,
    estimate_border_bg,
)
from .engine import (
    HAS_OPENCV_CORE,
    SmartSplitter,
    SplitBox,
    SplitMode,
    SplitResult,
    UniformMode,
)

__version__ = "1.3.0"

__all__ = [
    "__version__",
    "HAS_OPENCV_CORE",
    "SmartSplitter",
    "SplitBox",
    "SplitMode",
    "SplitResult",
    "UniformMode",
    "batch_crop_and_save_py",
    "batch_crop_to_memory_py",
    "detect_auto_py",
    "estimate_border_bg",
]
