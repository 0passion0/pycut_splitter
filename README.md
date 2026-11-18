# PyCut 智能切图引擎（`pycut-image-splitter`）

**简体中文** | [English](./README.en.md)

[![PyPI](https://img.shields.io/pypi/v/pycut-image-splitter?color=blue&label=PyPI)](https://pypi.org/project/pycut-image-splitter/)
[![Python](https://img.shields.io/pypi/pyversions/pycut-image-splitter)](https://pypi.org/project/pycut-image-splitter/)
[![CI](https://github.com/0passion0/pycut_splitter/actions/workflows/ci.yml/badge.svg)](https://github.com/0passion0/pycut_splitter/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-green.svg)](./LICENSE)

基于 **Python 3.12 + OpenCV SIMD + NumPy 向量化加速** 构建的高性能、严格内存安全复合图像版面解析与批量裁切引擎。通过**分条（Strip-Chunked）投影计算**与**零拷贝 ROI 切片**，在 4K / 8K 超大图场景下保持极低延迟与极低瞬态内存占用。

---

## 目录

1. [核心能力与内存优化亮点](#1-核心能力与内存优化亮点)
2. [处理管线与工作模式](#2-处理管线与工作模式)
3. [快速安装（PyPI / GitHub / 本地源码）](#3-快速安装pypi--github--本地源码)
4. [Python API 与 CLI 用法](#4-python-api-与-cli-用法)
5. [核心参数速查表](#5-核心参数速查表)
6. [运行测试与 PyCut Studio 工作台](#6-运行测试与-pycut-studio-工作台)
7. [仓库目录结构](#7-仓库目录结构)
8. [开源协议](#8-开源协议)

---

## 1. 核心能力与内存优化亮点

- **全向量化 OpenCV + NumPy 内核**：核心投影、边缘能量计算与形态学提取完全基于 OpenCV C++ 底层算子（`cv2.absdiff`、`cv2.addWeighted`、`cv2.dilate`、`cv2.connectedComponentsWithStats`）与 NumPy `uint8` / `uint16` 运算，零 Python 像素循环。
- **分条（Strip-Chunked）低内存架构**：按 `1024` 行分块扫描图像背景差值与边界梯度能量，及时释放临时切片缓冲，避免创建全图 `float32` / `int16` 副本；在 **8K（7680×4320，94.92 MB 原图）** 压测下峰值瞬态缓冲仅约 **21 MB（22.1%）**。
- **方向步进 + 1px 细线接缝能量检测**：结合 `C^2.4` 非线性覆盖率抑制与切向平滑度过滤，避免图表、文字段落与表格内部出现误切。
- **谷底折叠与递归 XY-Cut BSP**：自动识别背景留白沟槽（Gutter）并折叠中点，配合 **±12px 兄弟网格线传播**，稳健应对未对齐网格、非对称 Bento 拼图以及浅白细线分隔。
- **±8px 阶跃吸附与边缘去污染（`bleed_trim`）**：对分割边界进行局部能量极大值吸附，并向内剥离抗锯齿残影与 `1px` 灰边。
- **零拷贝 ROI 裁切与多线程并发编码**：基于 NumPy 切片视图截取子图，通过 `ThreadPoolExecutor` 释放 GIL 并发执行 `cv2.imencode`（支持 `PNG` / `JPG` / `WebP` 及 Alpha 透明通道保留），并内置 `pad` / `resize` / `crop` 统一输出尺寸。
- **严格内存安全护栏（`memory_safe.py`）**：内建解压缩炸弹像素上限保护（默认 `256 MP`）、数组维度与 `dtype` 校验、边界坐标安全钳位（`clamp_box`）以及只读零拷贝 ROI 校验。

---

## 2. 处理管线与工作模式

```text
输入图像 (NumPy ndarray / 磁盘文件)
  │
  ▼ [memory_safe.py: 256MP 解压防爆拦截 & 维度/dtype 校验]
[1. 分条 SIMD 投影与背景估计]  1024 行分块扫描 · 阶跃梯度 + 1px 细线脉冲
  │
  ├─ mode = "auto"    ──► 基于前景连通域分布自动决策 xycut 或 contour
  ├─ mode = "xycut"   ──► 谷底折叠 + 递归 XY-Cut BSP + ±12px 兄弟网格传播 + ±8px 边缘吸附
  ├─ mode = "contour" ──► 可分离降采样 + 自适应闭运算 (contour_gap_px) + 1:1 边缘收紧
  └─ mode = "grid"    ──► 整数均分 + gutter_px 间距扣除
  │
  ▼ [可选统一长宽: uniform_size / target_size (pad | resize | crop)]
[2. ThreadPoolExecutor 并发编码]  零拷贝 ROI 切片 + 释放 GIL 并发写盘或内存 bytes
```

| 模式 (`mode`) | 适用场景 | 核心实现 |
| :--- | :--- | :--- |
| `auto`（默认） | 通用混排图、幻灯片拼图、散落元素图 | 根据前景连通域宽高比与空间重叠度自动决策 `xycut` 或 `contour` |
| `xycut` | 3×3 幻灯片、Bento 布局、带细线/留白的多栏拼贴 | 向量化行列接缝能量投影 + 谷底折叠 + 递归 BSP + 兄弟传播 + 边缘修剪 |
| `contour` | 散落图标、不规则排版卡片、透明背景素材 | 可分离降采样 + 自适应形态学闭运算（`contour_gap_px`）+ 1:1 紧致边界收紧 |
| `grid` | 规整宫格切图（支持自定义行列与间距） | 基于整数均分与 `gutter_px` 内缩的高速网格切分 |

---

## 3. 快速安装（PyPI / GitHub / 本地源码）

### 环境要求

- **Python `>= 3.12`**（推荐 Python `3.12` / `3.13`）
- 核心依赖：`numpy >= 2.0`、`opencv-python-headless >= 4.10`、`pillow >= 10.0`

```bash
# 方式 1：直接从 PyPI 安装（推荐）
pip install pycut-image-splitter

# 方式 2：从 GitHub 直接安装
pip install git+https://github.com/0passion0/pycut_splitter.git

# 方式 3：本地开发可编辑安装
pip install -e ".[dev]"
```

---

## 4. Python API 与 CLI 用法

### Python API 示例

```python
import numpy as np
from PIL import Image
from pycut_splitter import SmartSplitter

# 1. 初始化切分器
splitter = SmartSplitter(
    mode="auto",            # "auto" | "xycut" | "contour" | "grid"
    sensitivity=0.65,       # 分割灵敏度 (0.1 ~ 1.0)
    min_region_ratio=0.08,  # 最小有效区域占比
    bleed_trim=2,           # 边缘去污染像素数
    contour_gap_px=8,       # 轮廓模式形态学闭合间隙
    uniform_size=False,     # 是否统一输出尺寸
    uniform_mode="pad",     # "pad" | "resize" | "crop"
)

# 2. 直接切分文件并保存为 PNG / JPG / WebP
result, saved_paths = splitter.split_file(
    "public/examples/sample_slides.png",
    "./output_slices",
    fmt="png",
)
print(
    f"模式: {result.detected_mode}, 后端: {result.backend}, "
    f"切片数: {len(result.boxes)}, 耗时: {result.elapsed_ms:.1f}ms"
)

# 3. 或者在内存中处理 NumPy RGB/RGBA 数组（零磁盘落地）
img = np.asarray(Image.open("public/examples/sample_slides.png").convert("RGB"))
res, bytes_list = splitter.split_to_bytes(img, fmt="webp")
```

### 命令行工具（CLI）

```bash
pycut-split public/examples/sample_slides.png -o ./slices --mode auto --format png
```

---

## 5. 核心参数速查表

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mode` | `"auto"` | 布局检测模式：`"auto"` / `"xycut"` / `"contour"` / `"grid"` |
| `sensitivity` | `0.65` | 边缘与前景灵敏度（`0.1..=1.0`） |
| `min_region_ratio` | `0.08` | 子图边长占短边最小比例（`0.03..=0.35`） |
| `bleed_trim` | `2` | 内接缝抗锯齿去杂边像素数（px） |
| `contour_gap_px` | `8` | 轮廓模式形态学闭运算弥合半径（px） |
| `uniform_size` | `False` | 是否统一所有切片输出宽高 |
| `target_size` | `None` | 显式指定统一尺寸 `(width, height)` |
| `uniform_mode` | `"pad"` | 统一尺寸对齐方式：`"pad"` / `"resize"` / `"crop"` |
| `max_pixels` | `268_435_456` | 防解压炸弹最大像素限制（`256 MP`） |
| `ml_segmenter` | `None` | 可选机器学习检测回调（返回 `(x, y, w, h)` 列表） |

---

## 6. 运行测试与 PyCut Studio 工作台

本项目同时内置基于 **Express + Python 3.12 持久化 IPC Worker + React** 的 **PyCut Studio** 交互式工作台，可实时调节参数、检查像素边缘并运行 1080p–8K 实时基准压测。

```bash
# 运行 Python 单元测试与 4K 内存安全压测（10 项测试）
pytest -v python_engine/tests/test_pycut.py

# 构建 wheel 与 sdist 并检查分发规范
python -m build --outdir dist
twine check dist/*

# 启动 PyCut Studio 全栈工作台（默认监听 3000 端口）
npm run dev
```

---

## 7. 仓库目录结构

```text
.
├── LICENSE                         # Apache-2.0 OR MIT 双重开源协议
├── MANIFEST.in                     # PyPI 源码分发清单（排除前端/Node 非核心资源）
├── pyproject.toml                  # 根目录 PyPI 打包构建配置（pycut-image-splitter）
├── README.md                       # 默认中文使用说明
├── README.en.md                    # 英文说明文档
├── .github/workflows/
│   ├── ci.yml                      # Python 3.12/3.13 Pytest + 打包校验 CI
│   └── release-pypi.yml            # GitHub Release 与 PyPI 自动发布流
├── python_engine/
│   ├── pyproject.toml              # 子目录兼容安装配置
│   ├── pycut_splitter/             # 核心 Python 3.12 + OpenCV/NumPy 引擎
│   │   ├── algorithms.py           # SIMD 投影、XY-Cut BSP、Contour 与并发裁切
│   │   ├── memory_safe.py          # 解压缩炸弹防护、坐标钳位与零拷贝 ROI 校验
│   │   ├── engine.py               # SmartSplitter / SplitBox / SplitResult 主类
│   │   ├── worker.py               # 供 Node.js 后端调用的持久化 JSON-line IPC 进程
│   │   └── cli.py                  # 命令行批量切图入口
│   └── tests/test_pycut.py         # pytest 完整算法与内存安全测试集
├── server.ts                       # Express 后端与 Python 3.12 Worker 进程池
└── src/                            # React + Tailwind CSS 中文可视化工作台
```

---

## 8. 开源协议

本项目采用 **[Apache-2.0 OR MIT](LICENSE)** 双重协议开源。
