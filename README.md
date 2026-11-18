# PyCut 智能切图引擎（`pycut-image-splitter`）

基于 **Python 3.12 + OpenCV + NumPy 向量化加速** 构建的高性能、内存安全图像版面解析与批量裁切引擎，通过 OpenCV SIMD 与分条（Strip）计算在 4K / 8K 大图下保持极低延迟与极低瞬态内存占用。

## 核心能力与优化特性

- **全 Python 3.12 + OpenCV / NumPy 向量化内核**：核心投影、边缘能量计算与形态学提取完全基于 OpenCV C++ 底层算子（`cv2.absdiff`、`cv2.addWeighted`、`cv2.dilate`、`cv2.connectedComponentsWithStats`）与 NumPy `uint8` / `uint16` 向量运算，告别纯 Python 像素循环。
- **分条（Strip-Chunked）低内存架构**：按 `1024` 行分块扫描图像行/列背景差值与边界梯度能量，及时释放临时切片缓冲，避免创建全图 `float32` / `int16` 副本；在 **8K（7680×4320，94.92 MB 原图）** 压测下峰值瞬态缓冲仅约 **21 MB（22.1%）**。
- **方向步进 + 1px 细线接缝能量检测**：结合 `C^2.4` 非线性覆盖率抑制与切向平滑度过滤，防止图表、文字段落、表格内部出现误切。
- **谷底折叠与递归 XY-Cut BSP**：自动识别背景留白沟槽（Gutter）并折叠中点，配合 **±12px 兄弟网格线传播**，稳健应对未对齐网格、非对称 Bento 拼图以及浅白细线分隔。
- **±8px 阶跃吸附与边缘去污染（`bleed_trim`）**：对分割边界进行局部能量极大值吸附，并向内剥离抗锯齿残影与 1px 灰边。
- **可分离降采样连通域 + 1:1 边缘收紧（Contour Mode）**：在降采样图上执行自适应闭运算形态学融合与连通域标定，再映射回 1:1 原图执行精确的紧致边界收敛。
- **零拷贝 ROI 裁切与多线程并发编码**：基于 NumPy 切片零拷贝视图截取子图，通过 `ThreadPoolExecutor` 释放 GIL 并发执行 OpenCV `cv2.imencode`（支持 `PNG` / `JPG` / `WebP` 及 Alpha 透明通道保留），并支持 `pad` / `resize` / `crop` 统一输出尺寸。
- **严格内存安全护栏（`memory_safe.py`）**：内建解压缩炸弹像素上限保护（默认 `256 MP`）、数组维度与 `dtype` 校验、边界坐标安全钳位（`clamp_box`）以及只读零拷贝 ROI 校验。

## 算法架构与工作模式

| 模式 (`mode`) | 适用场景 | 核心实现 |
| :--- | :--- | :--- |
| `auto`（自动） | 通用混排图、幻灯片拼图、散落元素图 | 根据前景连通域宽高比与空间重叠度自动决策 `xycut` 或 `contour` |
| `xycut`（递归切分） | 3×3 幻灯片、Bento 布局、带细线/留白的多栏拼贴 | 向量化行列接缝能量投影 + 谷底折叠 + 递归 BSP + 兄弟传播 + 边缘修剪 |
| `contour`（轮廓提取） | 散落图标、不规则排版卡片、透明背景素材 | 可分离降采样 + 自适应形态学闭运算（`contour_gap_px`）+ 1:1 紧致边界收紧 |
| `grid`（均匀网格） | 规整宫格切图（支持自定义行列与间距） | 基于整数均分与 `gutter_px` 内缩的高速网格切分 |

## Python 3.12 安装与快速使用

### 环境要求

- **Python >= 3.12**（推荐 Python `3.12.16`）
- `opencv-python-headless >= 4.9`、`numpy >= 1.26`、`pillow >= 10.0`

```bash
# 使用 pip 安装本地引擎包
python3.12 -m pip install -e ./python_engine
```

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
print(f"模式: {result.detected_mode}, 后端: {result.backend}, 切片数: {len(result.boxes)}, 耗时: {result.elapsed_ms:.1f}ms")

# 3. 或者在内存中处理 NumPy RGB/RGBA 数组（不落盘）
img = np.asarray(Image.open("public/examples/sample_slides.png").convert("RGB"))
res, bytes_list = splitter.split_to_bytes(img, fmt="webp")
```

### 命令行工具（CLI）

```bash
PYTHONPATH=python_engine python3.12 -m pycut_splitter.cli public/examples/sample_slides.png -o ./slices --mode auto --format png
```

## 运行测试与 PyCut Studio 可视化工作台

本项目内置基于 **Express + Python 3.12 持久化 IPC Worker + React** 的 **PyCut Studio** 交互式工作台，可实时调节参数、叠加 NumPy 边缘能量曲线、检查像素边缘并运行 1080p–8K 实时性能基准测试。

```bash
# 运行 Python 3.12 单元测试与 4K 内存安全测试（10 项测试）
python3.12 -m pytest -v python_engine/tests/test_pycut.py

# 运行前端工具函数测试（4 项测试）
npx --yes tsx --test src/utils/splitterUtils.test.ts

# 启动 PyCut Studio 全栈工作台（默认监听 3000 端口）
npm run dev
```

## 项目结构

```text
├── python_engine/
│   ├── pyproject.toml              # Python 3.12 包配置
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
