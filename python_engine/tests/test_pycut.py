"""Comprehensive pytest suite verifying layout parsing, memory safety, and large-image latency."""

from __future__ import annotations

import sys
import tempfile
import tracemalloc
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pycut_splitter import (
    HAS_OPENCV_CORE,
    SmartSplitter,
    __version__,
    batch_crop_and_save_py,
    batch_crop_to_memory_py,
    detect_auto_py,
)


def test_core_exports():
    assert HAS_OPENCV_CORE is True
    assert __version__ == "1.3.1"
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


def test_grid_refine_weak_blurred_jpeg_seams():
    y, x = np.indices((401, 601))
    image = np.repeat((95 + 0.02 * x + 0.01 * y).astype(np.uint8)[..., None], 3, axis=2)
    image[:, 208:] += 8
    image[:, 394:] += 8
    image[191:] += 8
    image = cv2.GaussianBlur(image, (0, 0), 6)
    noise = np.random.default_rng(9517).normal(0, 1.2, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    image = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

    plain = SmartSplitter(mode="grid").analyze(image, rows=2, cols=3)
    refined = SmartSplitter(mode="grid", grid_refine=True).analyze(image, rows=2, cols=3)
    assert [b.x for b in plain.boxes[:3]] == [0, 200, 401]
    assert refined.detected_mode == "grid"
    assert [(b.index, b.row, b.col, b.strategy) for b in refined.boxes] == [
        (b.index, b.row, b.col, b.strategy) for b in plain.boxes
    ]
    assert abs(refined.boxes[1].x - 208) <= 2
    assert abs(refined.boxes[2].x - 394) <= 2
    assert abs(refined.boxes[3].y - 191) <= 2
    assert sum(b.width * b.height for b in refined.boxes) == 401 * 601


def test_grid_refine_requires_distributed_unambiguous_evidence():
    y, x = np.indices((401, 601))
    gradient = np.repeat((95 + 0.02 * x + 0.01 * y).astype(np.uint8)[..., None], 3, axis=2)
    short_rule = gradient.copy()
    short_rule[75:155, 208:213] = 240
    double_peak = gradient.copy()
    double_peak[126:] += 8
    double_peak[140:] += 8
    splitter = SmartSplitter(mode="grid", grid_refine=True)
    equal_boxes = SmartSplitter(mode="grid").analyze(gradient).boxes
    for image in (np.zeros_like(gradient), gradient, short_rule, cv2.GaussianBlur(double_peak, (0, 0), 3)):
        assert splitter.analyze(image).boxes == equal_boxes


@pytest.mark.parametrize("sigma", [0, 2, 4])
def test_grid_refine_staggered_slide_seams(sigma):
    image = np.empty((1600, 2848, 3), dtype=np.uint8)
    y_cuts = (0, 532, 1046, 1600)
    x_cuts = [(0, 946, x, 2848) for x in (1825, 1823, 1812)]
    colors = ((40, 210, 50), (213, 230, 201), (203, 227, 55))
    expected = []
    for row, (top, bottom) in enumerate(zip(y_cuts, y_cuts[1:])):
        for col, (left, right) in enumerate(zip(x_cuts[row], x_cuts[row][1:])):
            image[top:bottom, left:right] = colors[row][col]
            expected.append((left, top, right, bottom))
    image[600:650, 1915:2170] = 25
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)

    splitter = SmartSplitter(mode="grid", grid_refine=True)
    for transposed in (False, True):
        source = image.swapaxes(0, 1) if transposed else image
        boxes = splitter.analyze(source, rows=3, cols=3).boxes
        assert len(boxes) == 9
        coverage = np.zeros(source.shape[:2], dtype=np.uint8)
        for box in boxes:
            right, bottom = box.x + box.width, box.y + box.height
            actual = (box.y, box.x, bottom, right) if transposed else (box.x, box.y, right, bottom)
            index = box.col * 3 + box.row if transposed else box.row * 3 + box.col
            assert actual == pytest.approx(expected[index], abs=2)
            coverage[box.y:bottom, box.x:right] += 1
        assert np.all(coverage == 1)


@pytest.mark.parametrize("height,width,cols,shift", [(319, 1801, 4, 0.30), (443, 1200, 3, 0.44)])
@pytest.mark.parametrize("sigma", [0, 3, 8])
def test_grid_refine_finds_large_moves_in_single_row(height, width, cols, shift, sigma):
    cell = width / cols
    cuts = [round(cell * (i + (-1) ** i * shift)) for i in range(1, cols)]
    image = np.full((height, width, 3), 45, dtype=np.uint8)
    for x in cuts:
        image[:, x:] += 35
    image[height // 3:height // 2, round(cell * 0.85):round(cell * 1.1)] = 240
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)

    splitter = SmartSplitter(mode="grid", grid_refine=True)
    boxes = splitter.analyze(image, rows=1, cols=cols).boxes
    assert [box.x for box in boxes[1:]] == pytest.approx(cuts, abs=1)
    boxes = splitter.analyze(image.swapaxes(0, 1), rows=cols, cols=1).boxes
    assert [box.y for box in boxes[1:]] == pytest.approx(cuts, abs=1)


@pytest.mark.parametrize("sigma", [0, 8])
def test_grid_refine_preserves_aligned_partial_seam_over_strong_wavy_edge(sigma):
    y, x = np.indices((512, 720))
    content_edge = 159 + 8 * np.sin(x * 6 * np.pi / x.shape[1])
    gray = 40 + 125 * (y >= content_edge) + 40 * ((y >= 256) & (x >= 418))
    image = np.repeat(gray.astype(np.uint8)[..., None], 3, axis=2)
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    boxes = SmartSplitter(mode="grid", grid_refine=True).analyze(image, rows=2, cols=1).boxes
    assert boxes[1].y == 256


@pytest.mark.parametrize("width,height,offsets", [(780, 610, (-109, 16, 100)), (800, 420, (0, 75, 125))])
@pytest.mark.parametrize("sigma", [0, 8])
def test_grid_refine_finds_large_staggered_moves(width, height, offsets, sigma):
    image = np.empty((height, width, 3), dtype=np.uint8)
    ys = [round(i * height / len(offsets)) for i in range(len(offsets) + 1)]
    xs = [width // 2 + offset for offset in offsets]
    for row, x in enumerate(xs):
        top, bottom = ys[row:row + 2]
        image[top:bottom, :x] = (20 + row * 20, 30 + row * 25, 100 + row * 40)
        image[top:bottom, x:] = (140 + row * 20, 100 + row * 25, 30 + row * 40)
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)

    boxes = SmartSplitter(mode="grid", grid_refine=True).analyze(image, rows=len(offsets), cols=2).boxes
    assert len(boxes) == 2 * len(offsets)
    for row, (left, right) in enumerate(zip(boxes[::2], boxes[1::2])):
        assert left.x + left.width == right.x == pytest.approx(xs[row], abs=1)
        assert right.x + right.width == width
        assert left.y == right.y == ys[row]
        assert left.y + left.height == right.y + right.height == ys[row + 1]


def test_grid_refine_uses_alpha_and_ignores_invisible_rgb():
    image = np.full((401, 601, 4), 120, dtype=np.uint8)
    image[:, :288, 3] = 0
    image[:, 288:, 3] = 255
    image = cv2.GaussianBlur(image, (0, 0), 5)
    splitter = SmartSplitter(mode="grid", grid_refine=True)
    assert abs(splitter.analyze(image, rows=1, cols=2).boxes[1].x - 288) <= 3

    image[..., 3] = 0
    image[:, 288:, :3] = 255
    assert splitter.analyze(image, rows=1, cols=2).boxes[1].x == round(601 / 2)


@pytest.mark.parametrize("gutter", [8, 9])
@pytest.mark.parametrize("sigma", [0, 2])
def test_grid_refine_centers_gutter_and_keeps_its_width(gutter, sigma):
    xs, ys = (0, 468, 899, 1367), (0, 244, 526, 768)
    image = np.full((ys[-1], xs[-1], 3), 225, dtype=np.uint8)
    start, end = gutter // 2, (gutter + 1) // 2
    for r in range(3):
        for c in range(3):
            image[
                ys[r] + (start if r else 0):ys[r + 1] - (end if r < 2 else 0),
                xs[c] + (start if c else 0):xs[c + 1] - (end if c < 2 else 0),
            ] = 25 + (r * 3 + c) * 17
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    boxes = SmartSplitter(mode="grid", grid_refine=True).analyze(image, gutter_px=gutter).boxes
    for i in (1, 2):
        assert abs(boxes[i].x - start - xs[i]) <= 1
        assert abs(boxes[3 * i].y - start - ys[i]) <= 1
        assert boxes[i].x - (boxes[i - 1].x + boxes[i - 1].width) == gutter
        assert boxes[3 * i].y - (boxes[3 * (i - 1)].y + boxes[3 * (i - 1)].height) == gutter
    assert (boxes[0].x, boxes[0].y) == (0, 0)
    assert (boxes[-1].x + boxes[-1].width, boxes[-1].y + boxes[-1].height) == (xs[-1], ys[-1])


@pytest.mark.parametrize("uniform_mode", ["pad", "resize", "crop"])
def test_grid_refine_uniform_export(uniform_mode):
    image = np.zeros((101, 201, 3), dtype=np.uint8)
    image[:, 93:] = 120
    splitter = SmartSplitter(mode="grid", grid_refine=True, target_size=(80, 80), uniform_mode=uniform_mode)
    result, blobs = splitter.split_to_bytes(image, rows=1, cols=2)
    assert len(result.boxes) == len(blobs) == 2
    assert all(Image.open(BytesIO(blob)).size == (80, 80) for blob in blobs)


def test_grid_refine_validates_geometry_before_writing(tmp_path):
    source = tmp_path / "tiny.png"
    output = tmp_path / "tiles"
    Image.new("RGB", (3, 2)).save(source)
    splitter = SmartSplitter(mode="grid", grid_refine=True)
    for rows, cols, gutter in ((3, 1, 0), (1, 3, 1)):
        with pytest.raises(ValueError, match="at least one pixel"):
            splitter.split_file(source, output, rows=rows, cols=cols, gutter_px=gutter)
    assert not output.exists()
    result = splitter.analyze(np.zeros((2, 3, 3), dtype=np.uint8), rows=1, cols=1, gutter_px=9)
    assert [(b.x, b.y, b.width, b.height) for b in result.boxes] == [(0, 0, 3, 2)]


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


def test_jpeg_compression_and_1px_dark_bezel_resilience():
    """Verify 2px border ring + soft L1 inliers resist 1px dark screenshot bezels and JPEG Q70 noise."""
    from pycut_splitter.algorithms import estimate_border_bg

    canvas = np.full((640, 840, 3), 248, dtype=np.uint8)
    # 1px dark screenshot bezel around the entire image
    canvas[0, :] = [28, 30, 34]
    canvas[-1, :] = [28, 30, 34]
    canvas[:, 0] = [28, 30, 34]
    canvas[:, -1] = [28, 30, 34]
    # 2x2 collage cards inside
    canvas[36:300, 36:400] = [34, 72, 168]
    canvas[36:300, 440:804] = [18, 148, 108]
    canvas[340:604, 36:400] = [196, 42, 74]
    canvas[340:604, 440:804] = [218, 134, 24]

    # Re-encode through JPEG quality 70 to introduce compression ringing and bin-edge quantization noise
    buf = BytesIO()
    Image.fromarray(canvas).save(buf, format="JPEG", quality=70)
    buf.seek(0)
    jpeg_arr = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

    bg_rgba, is_trans, unif = estimate_border_bg(jpeg_arr)
    assert not is_trans
    assert unif >= 0.45  # inner ring is clean light bg even though outer 1px is black bezel

    # Pure JPEG noise without bezel must remain high uniformity (>0.92)
    clean_canvas = canvas[1:-1, 1:-1].copy()
    buf2 = BytesIO()
    Image.fromarray(clean_canvas).save(buf2, format="JPEG", quality=70)
    buf2.seek(0)
    jpeg_clean = np.asarray(Image.open(buf2).convert("RGB"), dtype=np.uint8)
    _, _, unif_clean = estimate_border_bg(jpeg_clean)
    assert unif_clean > 0.92

    res = SmartSplitter(mode="auto", sensitivity=0.65, bleed_trim=2).analyze(jpeg_clean, collect_diagnostics=True)
    assert len(res.boxes) == 4
    assert res.diagnostics is not None
    assert "bg_noise_mad" in res.diagnostics


def test_mixed_gutter_and_hairline_per_edge_tracking():
    """Horizontal white gutter + vertical 1px touching hairline must split and snap cleanly on all 4 cards."""
    img = np.full((620, 800, 3), 255, dtype=np.uint8)
    # Row 0: two touching cards separated by a 1px gray hairline at x=400
    img[20:290, 20:400] = [28, 62, 136]
    img[20:290, 400] = [210, 212, 216]
    img[20:290, 401:780] = [214, 70, 62]
    # 40px white horizontal gutter (y=290..330)
    # Row 1: two touching cards separated by a 1px gray hairline at x=400
    img[330:600, 20:400] = [24, 142, 106]
    img[330:600, 400] = [210, 212, 216]
    img[330:600, 401:780] = [196, 128, 22]

    res = SmartSplitter(mode="xycut", sensitivity=0.65, bleed_trim=2).analyze(img)
    assert len(res.boxes) == 4
    # Verify vertical hairline was cleanly decontaminated and snapped
    left_boxes = [b for b in res.boxes if b.col == 0]
    right_boxes = [b for b in res.boxes if b.col == 1]
    assert len(left_boxes) == 2 and len(right_boxes) == 2
    assert all(b.x + b.width <= 401 for b in left_boxes)
    assert all(b.x >= 400 for b in right_boxes)


def test_three_column_bento_preserves_tall_column():
    """3-column Bento (tall left hero + 2x2 right stack) must keep left hero full-height."""
    img = np.full((800, 1200, 3), 250, dtype=np.uint8)
    # Tall column 0
    img[20:780, 20:390] = [24, 52, 110]
    # Column 1 split horizontally
    img[20:388, 414:790] = [215, 72, 65]
    img[412:780, 414:790] = [32, 145, 112]
    # Column 2 split horizontally
    img[20:388, 814:1180] = [180, 83, 9]
    img[412:780, 814:1180] = [67, 56, 202]

    res = SmartSplitter(mode="xycut", sensitivity=0.65, bleed_trim=1).analyze(img)
    assert len(res.boxes) == 5
    tall = [b for b in res.boxes if b.height >= 740]
    assert len(tall) == 1
    assert tall[0].x <= 24


def test_large_grid_multi_pass_sibling_propagation():
    """4x4 grid with faint aligned internal dividers across >3 leaves must recover all 16 cells."""
    img = np.full((800, 1200, 3), 246, dtype=np.uint8)
    # 4 distinct columns with strong vertical dividers
    col_w = 300
    row_h = 200
    for c in range(4):
        for r in range(4):
            y0, y1 = r * row_h, (r + 1) * row_h
            x0, x1 = c * col_w, (c + 1) * col_w
            if c == 0:
                # High-contrast alternating rows in col 0 establish the 3 horizontal grid lines
                base = (30 + (r % 2) * 140, 60 + (r * 35), 180 - (r * 30))
                img[y0:y1, x0:x1] = base
            else:
                # Subtle near-white cards separated by faint 1px divider lines in cols 1..3
                img[y0:y1, x0:x1] = (242 + (r % 2) * 6, 243 + (c % 2) * 4, 246)
            if r > 0:
                img[y0, x0:x1] = (195, 198, 205)  # faint 1px horizontal hairline
        if c > 0:
            img[:, c * col_w] = (90, 95, 110)  # strong vertical column divider

    res = SmartSplitter(mode="xycut", sensitivity=0.65, bleed_trim=1).analyze(img)
    assert len(res.boxes) == 16, f"Expected all 16 cells via multi-pass propagation, got {len(res.boxes)}"


def test_hollow_frame_containment_suppression_and_exact_gap():
    """Hollow frames must suppress inner nested boxes, and gap > contour_gap_px must NOT falsely merge."""
    img = np.full((700, 1100, 3), 252, dtype=np.uint8)
    # Hollow frame #1 with a disconnected inner icon
    img[60:320, 60:420] = [28, 64, 150]
    img[80:300, 80:400] = [252, 252, 252]
    img[140:240, 180:300] = [200, 50, 60]  # inner disconnected icon

    # Hollow frame #2 separated by 22px (> contour_gap_px=10)
    img[60:320, 442:800] = [20, 130, 95]
    img[80:300, 462:780] = [252, 252, 252]
    img[150:230, 560:680] = [190, 110, 20]  # inner disconnected icon

    # Separate bottom card separated by 18px vertically (> contour_gap_px=10)
    img[338:620, 220:680] = [40, 48, 72]

    res = SmartSplitter(mode="contour", sensitivity=0.70, contour_gap_px=10).analyze(img)
    assert len(res.boxes) == 3, f"Expected 3 outer objects (inner icons suppressed, no false merge), got {len(res.boxes)}"


def test_noisy_background_contour_speckle_stability():
    """1px background dust speckles on a noisy canvas must not bridge objects or spawn fake boxes."""
    rng = np.random.default_rng(42)
    # Slight Gaussian-like background texture (bg_noise_mad > 1.0)
    noise = rng.integers(-3, 4, size=(700, 1000, 3), dtype=np.int16)
    img = np.clip(248 + noise, 0, 255).astype(np.uint8)
    # Add 40 isolated 1px dark dust speckles across the canvas
    dust_y = rng.integers(10, 690, size=40)
    dust_x = rng.integers(10, 990, size=40)
    img[dust_y, dust_x] = [40, 40, 40]

    # 4 distinct foreground objects
    img[50:280, 60:420] = [30, 75, 195]
    img[60:270, 540:920] = [16, 150, 105]
    img[390:640, 80:450] = [205, 38, 76]
    img[400:630, 560:930] = [220, 140, 20]

    res = SmartSplitter(mode="contour", sensitivity=0.65, contour_gap_px=8).analyze(img)
    assert len(res.boxes) == 4, f"Expected 4 clean objects despite background noise/speckles, got {len(res.boxes)}"


def test_staggered_height_reading_order_stability():
    """Objects with staggered heights across 2 visual rows must cluster cleanly into row 0 and row 1."""
    img = np.full((620, 1050, 3), 252, dtype=np.uint8)
    # Row 0: 3 cards with staggered tops & heights (all overlap vertically >= 45%)
    img[30:230, 40:300] = [30, 70, 180]     # left
    img[75:210, 380:650] = [20, 150, 100]   # middle (shorter, starts 45px lower)
    img[40:250, 730:1000] = [200, 50, 70]   # right
    # Row 1: 3 cards with staggered tops
    img[340:560, 50:310] = [180, 100, 20]
    img[370:540, 390:660] = [80, 60, 190]
    img[330:570, 740:1010] = [35, 50, 75]

    res = SmartSplitter(mode="contour", sensitivity=0.70, contour_gap_px=8).analyze(img)
    assert len(res.boxes) == 6
    assert [(b.row, b.col) for b in res.boxes] == [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
    ]
    # Ensure left-to-right monotonic x within each row
    assert res.boxes[0].x < res.boxes[1].x < res.boxes[2].x
    assert res.boxes[3].x < res.boxes[4].x < res.boxes[5].x


def test_real_uploaded_slide_decks_white_seams_and_split_screen():
    """Verify LUT + 1D prominence + BSP size/aspect regularization cleanly slice 9/9 slides across sensitivity sweeps."""
    root_assets = Path(__file__).resolve().parent.parent.parent / "public" / "assets" / "aistudio"
    healthcare_path = root_assets / "bd30a154-cfe4-42d8-bd4f-d25eb6dd044a_c6ab570a.png"
    bluebird_path = root_assets / "9725e7ed-c93f-42e4-a57c-8e4d219622bd_d374a3d5.png"
    if not healthcare_path.exists() or not bluebird_path.exists():
        pytest.skip("Uploaded sample assets not present")

    with Image.open(healthcare_path) as im1:
        arr1 = np.asarray(im1.convert("RGB"), dtype=np.uint8)
    with Image.open(bluebird_path) as im2:
        arr2 = np.asarray(im2.convert("RGB"), dtype=np.uint8)

    for sens in (0.35, 0.65, 0.85):
        res1 = SmartSplitter(mode="auto", sensitivity=sens, bleed_trim=2).analyze(arr1)
        assert len(res1.boxes) == 9, f"Healthcare deck s={sens}: expected 9, got {len(res1.boxes)}"
        for b in res1.boxes:
            assert 940 <= b.width <= 956
            assert 528 <= b.height <= 538

        res2 = SmartSplitter(mode="auto", sensitivity=sens, bleed_trim=2).analyze(arr2)
        assert len(res2.boxes) == 9, f"Bluebird deck s={sens}: expected 9, got {len(res2.boxes)}"
        for b in res2.boxes:
            assert 940 <= b.width <= 956
            assert 502 <= b.height <= 564


def test_split_screen_and_three_column_internal_infographic_slides():
    """Slide deck with subtle near-white boundaries plus internal split-screen & 3-stripe content must yield 9/9 tiles."""
    sheet = np.full((1200, 1800, 3), 252, dtype=np.uint8)
    tones = [
        [(252, 253, 255), (245, 248, 252), (254, 251, 249)],
        [(246, 249, 253), (253, 253, 255), (244, 247, 251)],
        [(254, 252, 250), (246, 249, 252), (252, 254, 253)],
    ]
    for r in range(3):
        for c in range(3):
            y0, y1 = r * 400, (r + 1) * 400
            x0, x1 = c * 600, (c + 1) * 600
            sheet[y0:y1, x0:x1] = tones[r][c]
            # Place a clean title block & footer badge inside each slide
            sheet[y0 + 36 : y0 + 76, x0 + 40 : x0 + 280] = (24, 58, 120)
            sheet[y0 + 110 : y0 + 280, x0 + 48 : x0 + 320] = (90, 145, 210)
    # Slide (0, 2): full-height half-width dark split-screen hero block
    sheet[0:400, 1200:1470] = (18, 46, 98)
    # Slide (1, 1): full-height 3-column internal infographic strips
    sheet[400:800, 600:800] = (28, 65, 130)
    sheet[400:800, 800:1000] = (220, 232, 246)
    sheet[400:800, 1000:1200] = (42, 135, 110)

    buf = BytesIO()
    Image.fromarray(sheet).save(buf, format="JPEG", quality=75)
    buf.seek(0)
    arr = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

    res = SmartSplitter(mode="auto", sensitivity=0.65, bleed_trim=2).analyze(arr)
    assert len(res.boxes) == 9, f"Expected 9 intact slides, got {len(res.boxes)}"
    for b in res.boxes:
        assert 592 <= b.width <= 606
        assert 392 <= b.height <= 406


def test_pure_white_hairline_divided_slide_deck():
    """Pure #FFFFFF slide deck divided by 1px hairlines with internal slide padding must keep all 9 intact slides."""
    pw = np.full((1200, 1800, 3), 255, dtype=np.uint8)
    for r in range(3):
        for c in range(3):
            y0, x0 = r * 400, c * 600
            pw[y0 + 40 : y0 + 82, x0 + 42 : x0 + 300] = (24, 58, 120)
            pw[y0 + 115 : y0 + 290, x0 + 48 : x0 + 340] = (90, 145, 210)
            pw[y0 + 330 : y0 + 355, x0 + 42 : x0 + 180] = (140, 155, 175)
    # Full-height split-screen and 3-column infographic internals
    pw[0:400, 1200:1470] = (18, 46, 98)
    pw[400:800, 600:800] = (28, 65, 130)
    pw[400:800, 800:1000] = (225, 236, 248)
    pw[400:800, 1000:1200] = (42, 135, 110)
    for k in (1, 2):
        pw[k * 400, :] = (210, 216, 224)
        pw[:, k * 600] = (210, 216, 224)

    for sens in (0.35, 0.65, 0.85):
        res = SmartSplitter(mode="auto", sensitivity=sens, bleed_trim=2).analyze(pw)
        assert len(res.boxes) == 9, f"Pure white deck s={sens}: expected 9, got {len(res.boxes)}"
        for b in res.boxes:
            assert 594 <= b.width <= 604
            assert 394 <= b.height <= 404
