"""
Memory-safety guardrails, zero-copy read-only views, and bounded image decoding
for PyCut Image Splitter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

# Default hard ceiling (256 megapixels) to prevent accidental OOM / decompression bombs.
DEFAULT_MAX_PIXELS = 256 * 1024 * 1024


def validate_image_array(
    image: np.ndarray,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> np.ndarray:
    """
    Validate an HxWx3 or HxWx4 image array and return a safe, read-only view.
    Avoids copying C-contiguous uint8 buffers; safely normalizes non-uint8 or strided inputs.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(image).__name__}.")
    if image.ndim != 3:
        raise ValueError(f"Expected HxWx3 or HxWx4 image array, got ndim={image.ndim}.")

    h, w, ch = image.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Image dimensions must be positive, got {w}x{h}.")
    if ch not in (3, 4):
        raise ValueError(f"Expected 3 (RGB/BGR) or 4 (RGBA/BGRA) channels, got {ch}.")
    if w * h > max_pixels:
        raise MemoryError(
            f"Image pixel count ({w}x{h}={w * h:,}) exceeds safety limit ({max_pixels:,})."
        )

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            max_val = float(np.nanmax(image)) if image.size else 1.0
            scale = 255.0 if max_val <= 1.0 else 1.0
            arr = np.clip(np.nan_to_num(image) * scale, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(image, 0, 255).astype(np.uint8)
    else:
        arr = image

    # Create a lightweight read-only view without copying pixel memory
    ro_view = arr.view()
    ro_view.setflags(write=False)
    return ro_view


def clamp_box(x: int, y: int, bw: int, bh: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Clamp bounding box coordinates strictly within [0, img_w] x [0, img_h]."""
    x0 = max(0, min(int(x), max(0, img_w - 1)))
    y0 = max(0, min(int(y), max(0, img_h - 1)))
    cw = max(1, min(int(bw), img_w - x0))
    ch = max(1, min(int(bh), img_h - y0))
    return x0, y0, cw, ch


def safe_roi_view(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Return a bounds-clamped, zero-copy read-only NumPy slice."""
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
    Decode an image file safely into an RGB or RGBA uint8 NumPy array.
    Uses OpenCV C++ decoding via memory-mapped/raw bytes (Unicode-path safe),
    falling back to Pillow for exotic formats.
    """
    p = Path(input_path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {p}")

    # Peek header via Pillow first to enforce decompression-bomb ceiling before full decode
    with Image.open(p) as probe:
        pw, ph = probe.size
        if pw <= 0 or ph <= 0:
            raise ValueError(f"Corrupt image dimensions: {pw}x{ph}")
        if pw * ph > max_pixels:
            raise MemoryError(
                f"Image {p.name} ({pw}x{ph}={pw * ph:,} px) exceeds safety limit ({max_pixels:,} px)."
            )

    raw_bytes = np.fromfile(str(p), dtype=np.uint8)
    decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if decoded is not None and decoded.ndim == 3:
        if decoded.shape[2] == 4:
            return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
        if decoded.shape[2] == 3:
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif decoded is not None and decoded.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)

    # Fallback via Pillow for CMYK / palette / animated webp / etc.
    with Image.open(p) as pil_img:
        mode_img = pil_img.convert("RGBA") if "A" in pil_img.getbands() else pil_img.convert("RGB")
        return np.asarray(mode_img, dtype=np.uint8)
