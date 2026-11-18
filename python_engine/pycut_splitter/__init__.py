"""PyCut Image Splitter: pure Python + OpenCV + vectorized NumPy engine."""

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

__all__ = [
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
