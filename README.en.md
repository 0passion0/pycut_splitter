# PyCut Image Splitter (`pycut-image-splitter`)

[简体中文](./README.md) | **English**

[![PyPI](https://img.shields.io/pypi/v/pycut-image-splitter?color=blue&label=PyPI)](https://pypi.org/project/pycut-image-splitter/)
[![Python](https://img.shields.io/pypi/pyversions/pycut-image-splitter)](https://pypi.org/project/pycut-image-splitter/)
[![CI](https://github.com/0passion0/pycut_splitter/actions/workflows/ci.yml/badge.svg)](https://github.com/0passion0/pycut_splitter/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-green.svg)](./LICENSE)

A memory-safe, ultra-fast **Python 3.12 + OpenCV SIMD + vectorized NumPy** composite image layout parser and batch slicer. Using **strip-chunked projection** and **zero-copy ROI views**, it keeps latency and transient memory overhead minimal even on 4K and 8K composite sheets.

---

## Table of Contents

1. [Highlights & Memory-Safety Architecture](#1-highlights--memory-safety-architecture)
2. [Pipeline & Detection Modes](#2-pipeline--detection-modes)
3. [Installation (PyPI / GitHub / Local)](#3-installation-pypi--github--local)
4. [Python API & CLI Usage](#4-python-api--cli-usage)
5. [Configuration Reference](#5-configuration-reference)
6. [Testing & PyCut Studio](#6-testing--pycut-studio)
7. [Repository Structure](#7-repository-structure)
8. [License](#8-license)

---

## 1. Highlights & Memory-Safety Architecture

- **Pure Python 3.12 + OpenCV SIMD / NumPy Kernels**: Seam projections, balanced uint8 RGB/BGR channel diffs, `INTER_NEAREST` zero-blur strip sampling, and morphological extraction execute directly on OpenCV C++ operators (`cv2.absdiff`, `cv2.reduce`, `cv2.addWeighted`, `cv2.dilate`, `cv2.connectedComponents`) with zero Python pixel loops.
- **Strip-Chunked Low-Memory Design**: Extracts compact strips to avoid full-image `float32`/`int16` copies—achieving **`150.9 ms` (219.8 MP/s)** parse time and **`19.18 MB` (20.2%)** peak transient buffer on an **8K (7680×4320, 94.92 MB)** test canvas.
- **Noise-Resilient Background, Weber-Log LUT & 1D Prominence SNR**: Uses 2px perimeter ring sampling + tight L1 inlier MAD (`<= 9`) to resist multi-tone white panel compression noise, combined with `_SEAM_LUT` sublinear contrast compression, 1D prominence baseline subtraction, and bilateral gutter density validation to isolate `1px` dividers from internal slide padding.
- **Per-Edge Gutter Tracking, Recursive XY-Cut BSP & Bottom-Up Geometric Regularization**: Tracks gutter edges independently per side with depth-0 rail consensus, runs **full multi-pass `±12px` sibling grid-line propagation**, and merges spurious narrow slivers via dominant tile clustering (`_regularize_bsp_tree`) while preserving Bento columns.
- **Coverage-Gated `±8px` Snap & Texture-Aware `bleed_trim`**: Snaps non-gutter edges with distance-prior decay and peels anti-aliased edge bleed or `1px` dividers via vectorized strip scans.
- **Zero-Copy ROI Views & GIL-Free Parallel Encoding**: Slices NumPy views and encodes PNG / JPG / WebP concurrently via `ThreadPoolExecutor` (`cv2.imencode`), with optional uniform output sizing (`pad` / `resize` / `crop`).
- **Strict Memory Guardrails (`memory_safe.py`)**: Built-in decompression-bomb pixel ceiling (`256 MP` default), array shape/dtype validation, safe coordinate clamping (`clamp_box`), and read-only zero-copy checks.

---

## 2. Pipeline & Detection Modes

```text
Input Image (NumPy ndarray / File Path)
  │
  ▼ [memory_safe.py: 256MP bomb guard & shape/dtype check]
[1. Strip-Chunked SIMD Profiling]  1024-row strips · step gradient + 1px hairline energy
  │
  ├─ mode = "auto"    ──► Selects xycut or contour based on component geometry & coverage
  ├─ mode = "xycut"   ──► Gutter collapse + recursive XY-Cut BSP + ±12px sibling propagation + ±8px snap
  ├─ mode = "contour" ──► Separable downsampling + adaptive closing (contour_gap_px) + 1:1 edge tightening
  └─ mode = "grid"    ──► Uniform grid partitioning with optional gutter_px deduction
  │
  ▼ [Optional Uniform Sizing: uniform_size / target_size (pad | resize | crop)]
[2. Parallel ThreadPool Encoding]  Zero-copy ROI slices + GIL-free disk/memory export
```

| Mode | Best Suited For | Core Mechanism |
|---|---|---|
| `auto` (default) | Mixed slide sheets, collages, scattered object sheets | Chooses between `xycut` and `contour` automatically |
| `xycut` | Shifted 3×3 slides, Bento layouts, white-guttered cards | Strip projection + gutter collapse + BSP + sibling propagation + `bleed_trim` |
| `contour` | Scattered icons, stickers, transparent RGBA sprites | Separable morphological closing + 8-connected components + 1:1 tightening |
| `grid` | Fixed uniform grids with optional `gutter_px` | Exact uniform row/column division |

---

## 3. Installation (PyPI / GitHub / Local)

- **Python `>= 3.12`**
- Core dependencies: `numpy >= 2.0`, `opencv-python-headless >= 4.10`, `pillow >= 10.0`

```bash
# Option 1: Install from PyPI (recommended)
pip install pycut-image-splitter

# Option 2: Install directly from GitHub
pip install git+https://github.com/0passion0/pycut_splitter.git

# Option 3: Editable local install with dev tools
pip install -e ".[dev]"
```

---

## 4. Python API & CLI Usage

```python
import numpy as np
from PIL import Image
from pycut_splitter import SmartSplitter

splitter = SmartSplitter(
    mode="auto",
    sensitivity=0.65,
    min_region_ratio=0.08,
    bleed_trim=2,
    contour_gap_px=8,
    uniform_size=False,
    uniform_mode="pad",
)

# 1. Split image file directly to disk
result, saved_paths = splitter.split_file(
    "public/examples/sample_slides.png",
    "./output_slices",
    fmt="png",
)
print(
    f"Mode: {result.detected_mode}, Backend: {result.backend}, "
    f"Slices: {len(result.boxes)}, Elapsed: {result.elapsed_ms:.1f}ms"
)

# 2. In-memory zero-disk slicing
img = np.asarray(Image.open("public/examples/sample_slides.png").convert("RGB"))
res, bytes_list = splitter.split_to_bytes(img, fmt="webp")
```

```bash
pycut-split public/examples/sample_slides.png -o ./slices --mode auto --format png
```

---

## 5. Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"auto"` | `"auto"` \| `"xycut"` \| `"contour"` \| `"grid"` |
| `sensitivity` | `0.65` | Edge & foreground threshold sensitivity (`0.1..=1.0`) |
| `min_region_ratio` | `0.08` | Minimum panel size ratio relative to short edge (`0.03..=0.35`) |
| `bleed_trim` | `2` | Anti-aliased seam edge trim in pixels |
| `contour_gap_px` | `8` | Morphological closing radius in `contour` mode |
| `uniform_size` | `False` | Normalize all output slices to equal width & height |
| `target_size` | `None` | Explicit `(width, height)` target size |
| `uniform_mode` | `"pad"` | `"pad"` (center-pad) \| `"resize"` \| `"crop"` |
| `max_pixels` | `268_435_456` | Decompression-bomb guard ceiling (`256 MP`) |
| `ml_segmenter` | `None` | Optional ML bounding-box callback returning `(x, y, w, h)` tuples |

---

## 6. Testing & PyCut Studio

```bash
# Run pytest suite including 4K transient memory benchmark
pytest -v python_engine/tests/test_pycut.py

# Build wheel & sdist and verify package metadata
python -m build --outdir dist
twine check dist/*

# Start PyCut Studio interactive workbench (port 3000)
npm run dev
```

---

## 7. Repository Structure

```text
.
├── LICENSE                         # Apache-2.0 OR MIT dual license
├── MANIFEST.in                     # PyPI source distribution manifest
├── pyproject.toml                  # Root PyPI packaging config (pycut-image-splitter)
├── README.md                       # Chinese documentation (default)
├── README.en.md                    # English documentation
├── .github/workflows/
│   ├── ci.yml                      # Python 3.12/3.13 pytest + package checks
│   └── release-pypi.yml            # GitHub Release & PyPI auto-publish workflow
├── python_engine/
│   ├── pyproject.toml              # Subdirectory editable install config
│   ├── pycut_splitter/             # Core Python 3.12 + OpenCV/NumPy engine
│   └── tests/test_pycut.py         # Pytest suite
├── server.ts                       # Express backend + Python 3.12 worker pool
└── src/                            # React + Tailwind CSS Studio UI
```

---

## 8. License

Dual-licensed under **[Apache-2.0 OR MIT](LICENSE)**.
