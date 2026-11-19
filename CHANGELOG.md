# 更新日志（Changelog）

本项目遵循 [语义化版本控制（Semantic Versioning）](https://semver.org/lang/zh-CN/)。

## [1.1.0] - 2026-11-18

### 性能优化
- **OpenCV C++ SIMD 全链路重构**：在 `uint8` 空间引入双通道均衡色差（`_color_diff_u8`：`max(RGB2GRAY, BGR2GRAY)`）与 `cv2.reduce` 轴向归约，替代原有 `uint16` 跨步累加与 NumPy 轴向求和。
- **`INTER_NEAREST` 查表零模糊条带采样（`_sample_strip`）**：替代 `np.ascontiguousarray` 跨步列拷贝循环，在完整保留 1px 细线的前提下，将 4K/8K 水平条带抽取与 RGBA 通道分离提速 `3.5×–14×`。
- **端到端延迟大幅下降**：8K (`7680×4320`, `xycut`) 解析耗时由 `490.4 ms` 降至 **`150.9 ms`**（`219.8 MP/s`，峰值瞬态缓冲 `19.18 MB` / 占原图 `20.2%`）；4K (`3840×2160`, `auto`) 由 `355.9 ms` 降至 **`98.7 ms`**；`auto` 模式对高置信度紧密平铺网格自动短路跳过冗余轮廓扫描。
- **持久化工作进程线程优化**：`worker.py` 启动时配置 `cv2.setNumThreads(1)`，消除小条带算子的内部线程池调度抖动；`server.ts` 优先探测 `.venv/bin/python` 以确保 Python 3.12 环境稳定。

### 算法稳定性增强
- **多环边框采样 + 软 L1 内点背景估计（`_estimate_border_bg_stats`）**：采样外侧 2px 环带抵抗 1px 截图黑边，通过软内点聚合（`|rgb - seed|_1 <= 24`）消除 JPEG 压缩量化跨箱分裂问题，并输出 `bg_noise_mad` 驱动下游自适应阈值。
- **按边独立沟槽跟踪与深度门控 BSP**：为每个区域独立记录 `gutter_top/right/bottom/left`，结合覆盖率加权与沟槽优先的轴向仲裁保护 Bento 通高列；支持纯净沟槽深层嵌套至 6 层，并在 `depth >= 4` 时屏蔽非沟槽内部色带，避免误切幻灯片表格标题栏。
- **完整多轮同级兄弟割线传播**：修复每轮仅拆分单次且上限 3 次的局限，升级为单轮全叶子扫描（上限 6 轮、256 块），稳定召回 4×4 及以上大型网格中的微弱对齐分割线。
- **覆盖率门控 `±8px` 边缘吸附与向量化纹理感知去污染（`bleed_trim`）**：新增 `cov >= 0.55` 门控与距离先验衰减，并向量化背景边缘剥离与内缘渗色剥离。
- **高精度轮廓提取与稳定视觉阅读顺序**：修正轮廓形态学膨胀半径为精确半间隙（`contour_gap_px / (2 * scale)`），新增噪点背景 1px 孤立尘点过滤与 `≥85%` 嵌套包含框去重（`_suppress_contained_boxes`），并升级为基于纵向重叠率（`≥45%`）的视觉行聚类排序。

### 包与 CLI 改进
- 导出 `pycut_splitter.__version__`（`"1.1.0"`），并为 `pycut-split` CLI 增加 `-V` / `--version` 参数。
- 新增 7 项稳定性与边界场景回归测试（总计 `17` 项 `pytest` 测试全部通过）。

## [1.0.0] - 2026-11-18

### 初始发布
- 发布纯 Python 3.12 + OpenCV + NumPy 向量化图像版面解析与批量切分引擎 `pycut-image-splitter`。
- 支持 `auto` / `xycut` / `contour` / `grid` 四种切分模式、`pad` / `resize` / `crop` 统一输出尺寸、`256 MP` 内存安全护栏、多线程并发编码导出，以及中文交互式 **PyCut Studio** 工作台。
