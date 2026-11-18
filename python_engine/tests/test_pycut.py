"""Comprehensive pytest suite verifying layout parsing, memory safety, and large-image latency."""

from __future__ import annotations

import sys
import tempfile
import tracemalloc
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pycut_splitter import (
    HAS_OPENCV_CORE,
    SmartSplitter,
    batch_crop_and_save_py,
    batch_crop_to_memory_py,
    detect_auto_py,
)


def test_core_exports():
    assert HAS_OPENCV_CORE is True
    assert callable(SmartSplitter)


def test_bento_xycut():
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    img[:, :298] = [24, 52, 110]
    img[:198, 299:] = [215, 72, 65]
    img[200:, 299:] = [32, 145, 112]
    splitter = SmartSplitter(mode="xycut", sensitivity=0.65, bleed_trim=1)
    res = splitter.analyze(img)
    assert res.backend == "opencv-numpy-simd"
    assert len(res.boxes) == 3
    assert res.boxes[0].height > 390


def test_contour_objects():
    img = np.full((950, 1500, 3), 248, dtype=np.uint8)
    img[70:350, 80:400] = [29, 78, 216]
    img[90:330, 540:920] = [4, 120, 87]
    img[65:375, 1060:1400] = [190, 18, 60]
    img[490:830, 170:630] = [15, 23, 42]
    img[470:840, 810:1320] = [217, 119, 6]
    splitter = SmartSplitter(mode="auto", sensitivity=0.7)
    res, blobs = splitter.split_to_bytes(img, fmt="png")
    assert res.detected_mode == "contour"
    assert len(res.boxes) == 5
    assert len(blobs) == 5
    assert abs(res.boxes[0].width - 320) <= 2
    assert abs(res.boxes[0].height - 280) <= 2


def test_white_gutter_collage():
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    img[20:280, 20:380] = [30, 64, 175]
    img[20:280, 420:780] = [16, 185, 129]
    img[320:580, 20:380] = [225, 29, 72]
    img[320:580, 420:780] = [245, 158, 11]
    splitter = SmartSplitter(mode="xycut", sensitivity=0.65, bleed_trim=2)
    res = splitter.analyze(img)
    assert len(res.boxes) == 4
    for box in res.boxes:
        assert abs(box.width - 360) <= 2
        assert abs(box.height - 260) <= 2


def test_bleed_decontamination():
    img = np.full((400, 600, 3), 252, dtype=np.uint8)
    img[:, :299] = [18, 36, 74]
    img[:, 299] = [213, 214, 217]  # 1px gray divider hairline
    img[:, 300:] = [250, 251, 253]
    splitter = SmartSplitter(mode="xycut", sensitivity=0.7, bleed_trim=2)
    res = splitter.analyze(img)
    assert len(res.boxes) == 2
    left_box, right_box = sorted(res.boxes, key=lambda b: b.x)
    assert left_box.x + left_box.width <= 299
    assert right_box.x >= 300


def test_uniform_size_modes():
    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    img[20:220, 20:320] = [24, 52, 110]
    img[20:280, 360:760] = [215, 72, 65]
    img[320:540, 20:420] = [32, 145, 112]

    pad_splitter = SmartSplitter(mode="auto", uniform_size=True, uniform_mode="pad")
    res_pad, blobs_pad = pad_splitter.split_to_bytes(img, fmt="png")
    assert res_pad.backend == "opencv-numpy-simd"
    assert len(blobs_pad) == len(res_pad.boxes)
    decoded_pad = [Image.open(BytesIO(b)).size for b in blobs_pad]
    assert len(set(decoded_pad)) == 1

    resize_splitter = SmartSplitter(mode="auto", target_size=(256, 192), uniform_mode="resize")
    _, blobs_resize = resize_splitter.split_to_bytes(img, fmt="png")
    assert all(Image.open(BytesIO(b)).size == (256, 192) for b in blobs_resize)

    crop_splitter = SmartSplitter(mode="auto", uniform_size=True, uniform_mode="crop")
    res_crop, blobs_crop = crop_splitter.split_to_bytes(img, fmt="png")
    crop_box_dims = {(b.width, b.height) for b in res_crop.boxes}
    assert len(crop_box_dims) == 1
    assert all(Image.open(BytesIO(b)).size == next(iter(crop_box_dims)) for b in blobs_crop)


def test_grid_and_ml_hook_and_split_file():
    img = np.full((600, 900, 3), 240, dtype=np.uint8)
    grid_sp = SmartSplitter(mode="grid")
    res_g = grid_sp.analyze(img, rows=2, cols=3, gutter_px=10)
    assert res_g.detected_mode == "grid"
    assert len(res_g.boxes) == 6
    assert res_g.boxes[0].row == 0 and res_g.boxes[5].row == 1 and res_g.boxes[5].col == 2

    ml_sp = SmartSplitter(ml_segmenter=lambda arr: [(10, 10, 100, 100), (200, 200, 120, 80)])
    res_ml = ml_sp.analyze(img)
    assert res_ml.detected_mode == "ml"
    assert res_ml.backend == "ml-hook"
    assert len(res_ml.boxes) == 2

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.png"
        out_dir = Path(tmp) / "out"
        Image.fromarray(img).save(src)
        res_f, saved = grid_sp.split_file(src, out_dir, prefix="tile", fmt="webp", rows=2, cols=2)
        assert len(saved) == 4
        assert all(Path(p).exists() for p in saved)
        assert len(res_f.boxes) == 4


def test_rgba_transparency_and_low_level_apis():
    img = np.zeros((500, 800, 4), dtype=np.uint8)
    img[40:200, 50:320] = [20, 90, 200, 255]
    img[260:450, 420:740] = [200, 50, 60, 255]
    mode, boxes, _ = detect_auto_py(img, mode="auto")
    assert mode == "contour"
    assert len(boxes) == 2
    blobs = batch_crop_to_memory_py(img, [(b.x, b.y, b.width, b.height) for b in boxes], fmt="png")
    assert len(blobs) == 2
    with tempfile.TemporaryDirectory() as tmp:
        paths = batch_crop_and_save_py(
            _save_tmp(img, Path(tmp) / "rgba.png"),
            [(b.x, b.y, b.width, b.height) for b in boxes],
            Path(tmp) / "slices",
        )
        assert len(paths) == 2


def _save_tmp(arr: np.ndarray, path: Path) -> Path:
    Image.fromarray(arr).save(path)
    return path


def test_sample_slides_2848x1600_real_image():
    sample_path = Path(__file__).resolve().parent.parent.parent / "public" / "examples" / "sample_slides.png"
    assert sample_path.exists(), f"Missing {sample_path}"
    with Image.open(sample_path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    splitter = SmartSplitter(mode="auto", sensitivity=0.65, bleed_trim=2)
    res = splitter.analyze(arr)
    assert res.image_width == 2848
    assert res.image_height == 1600
    assert len(res.boxes) == 9, f"Expected 9 slides, got {len(res.boxes)}"
    assert res.elapsed_ms < 250.0


def test_memory_safety_and_large_image_low_buffer():
    # Reject invalid ndim / channels / uniform_mode / target_size
    with pytest.raises(ValueError):
        SmartSplitter(uniform_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SmartSplitter(target_size=(0, 100))
    with pytest.raises(ValueError):
        SmartSplitter().analyze(np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(ValueError):
        SmartSplitter().analyze(np.zeros((100, 100, 2), dtype=np.uint8))
    with pytest.raises(MemoryError):
        SmartSplitter(max_pixels=1000).analyze(np.zeros((100, 100, 3), dtype=np.uint8))

    # Large 4K image peak transient allocation stays tiny relative to raw image size (24MB)
    big = np.full((2160, 3840, 3), 255, dtype=np.uint8)
    big[40:1000, 40:1800] = [24, 52, 110]
    big[40:1000, 1900:3800] = [215, 72, 65]
    big[1100:2100, 40:1800] = [32, 145, 112]
    big[1100:2100, 1900:3800] = [217, 119, 6]

    sp = SmartSplitter(mode="xycut", sensitivity=0.65)
    tracemalloc.start()
    res = sp.analyze(big)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(res.boxes) == 4
    # Transient Python/NumPy buffer overhead must be < 15 MB (well below full int16 copy ~48MB)
    assert peak < 15 * 1024 * 1024, f"Peak transient memory too high: {peak / (1024*1024):.2f} MB"
