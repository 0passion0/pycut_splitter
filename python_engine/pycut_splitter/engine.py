"""
High-level SmartSplitter engine using pure Python 3.12 + OpenCV + vectorized NumPy.
Enforces strict memory-safety guardrails, strip-chunked projection, and zero-copy ROI views.
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
    """Single detected sub-region bounding box and edge-trim metadata."""

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
    """Layout analysis summary for the full input image."""

    image_width: int
    image_height: int
    detected_mode: str
    backend: str
    elapsed_ms: float
    boxes: List[SplitBox] = field(default_factory=list)
    diagnostics: Optional[Dict[str, object]] = None


class SmartSplitter:
    """
    Memory-safe, vectorized OpenCV + NumPy composite image layout splitter.

    Provides `analyze`, `split_to_bytes`, `split_file`, and `ml_segmenter` hook
    with strip-chunked SIMD profiling, OpenCV C++ morphological closing & codecs,
    and strictly bounded transient memory usage.
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
        Execute OpenCV + NumPy vectorized layout detection and return bounding boxes.
        """
        t0 = time.perf_counter()
        safe_img = validate_image_array(image_bgr_or_rgb, max_pixels=self.max_pixels)
        h, w = safe_img.shape[:2]

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
        Analyze layout and encode all cropped/normalized slices to compressed bytes in RAM.
        """
        safe_img = validate_image_array(image_bgr_or_rgb, max_pixels=self.max_pixels)
        res = self.analyze(safe_img, rows=rows, cols=cols, gutter_px=gutter_px)
        if not res.boxes:
            return res, []

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
        Safely decode image from disk, detect regions, crop/normalize, and save concurrently.
        """
        img = decode_image_safe(input_path, max_pixels=self.max_pixels)
        res = self.analyze(img, rows=rows, cols=cols, gutter_px=gutter_px)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not res.boxes:
            return res, []

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
