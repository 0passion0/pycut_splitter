"""
PyCut 图像切割引擎的向量化 OpenCV + NumPy 核心算法模块。

高性能与高稳定性架构特性：
- 零拷贝步长跨度条带提取（仅 O(span * 200) 瞬态内存开销）
- OpenCV C++ SIMD 双通道均衡色差（RGB2GRAY / BGR2GRAY max，uint8 空间 8x 加速与 50% 内存缩减）
- OpenCV C++ SIMD `cv2.LUT` 韦伯感知对数对比度压缩 + 1D 局部突出度（Prominence）脉冲信噪比增强
- OpenCV C++ SIMD cv2.reduce 轴向均值归约（较 NumPy 轴向求和提速 3.5x~31x）
- 多环边框采样 + 软内点一致性背景估计（抗 1px 截图黑边与 JPEG 量化边界噪声）
- 按边独立跟踪沟槽属性（per-edge gutter flags），消除混合沟槽与细线场景的吸附冲突
- 覆盖率门控 + 距离先验的 ±8px 边缘吸附与纹理感知型去渗色（bleed_trim）修边
- 完整多轮同级兄弟割线对齐传播（±12px 突显度召回，支持同轴子树展平）
- 基于全盘尺寸共识与长宽比一致性的自底向上 BSP 几何正则化（消除卡片/幻灯片内部插图误切）
- 噪声自适应、精确间隙半径、防嵌套重复框与向量化 1:1 收紧的轮廓提取算子
- 基于纵向重叠率的视觉行聚类阅读顺序重排与释放 GIL 的多线程并发编码导出

:author: tongliga
:date: 2026-11-18
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .memory_safe import clamp_box, decode_image_safe, safe_roi_view, validate_image_array

_KERNEL_5x1 = np.ones((5, 1), dtype=np.uint8)
_KERNEL_1x5 = np.ones((1, 5), dtype=np.uint8)
_KERNEL_3x1 = np.ones((3, 1), dtype=np.uint8)
_KERNEL_1x3 = np.ones((1, 3), dtype=np.uint8)


def _build_seam_lut() -> np.ndarray:
    """
    预计算用于 OpenCV C++ SIMD `cv2.LUT` 的亚线性韦伯对数感知对比度查找表。

    抑制微小量化抖动（<=1）的同时，对浅色微弱拼接边缘（2..15）给予非线性提权，
    防止高对比度内部插图/摄影区块以线性幅度压制全贯通真实面板割线。

    :return: 长度为 256 的 uint8 查找表数组。
    :author: tongliga
    :date: 2026-11-25
    """
    x = np.arange(256, dtype=np.float32)
    lut = np.zeros(256, dtype=np.uint8)
    mask = x > 1.0
    lut[mask] = np.clip(np.rint(26.0 * np.log1p((x[mask] - 1.0) * 0.45)), 0, 255).astype(np.uint8)
    return lut


_SEAM_LUT = _build_seam_lut()


def _clamp(v: float, lo: float, hi: float) -> float:
    """
    将浮点数值严格限制在指定的区间 [lo, hi] 内。

    :param v: 输入原始数值。
    :type v: float
    :param lo: 允许的下界阈值。
    :type lo: float
    :param hi: 允许的上界阈值。
    :type hi: float
    :return: 处于 [lo, hi] 区间内的截断结果。
    :author: tongliga
    :date: 2026-11-18
    """
    return lo if v < lo else (hi if v > hi else v)


def _color_diff_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    """
    利用 OpenCV C++ SIMD 计算两幅 RGB uint8 图像的通道均衡感知色差矩阵（uint8，范围 0..255）。

    通过同时计算 RGB2GRAY 与 BGR2GRAY 并取逐像素极大值，确保 R、G、B 三通道均具有高灵敏度
    （权重分别为 >=0.299 / 0.587 / >=0.299，中性灰权重为 1.0），在保持精度的同时较 uint16
    跨步累加提速约 8 倍且瞬态内存减半。

    :param a_u8: 第一组三维 uint8 图像数组 (H, W, 3)。
    :type a_u8: numpy.ndarray
    :param b_u8: 第二组三维 uint8 图像数组 (H, W, 3) 或 4 元素标量元组。
    :type b_u8: numpy.ndarray
    :return: 形状为 (H, W) 的 uint8 均衡色差矩阵。
    :author: tongliga
    :date: 2026-11-18
    """
    diff = cv2.absdiff(a_u8, b_u8)
    g_rgb = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    g_bgr = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return cv2.max(g_rgb, g_bgr)


def _reduce_mean_axis0(mat_u8: np.ndarray) -> np.ndarray:
    """
    使用 OpenCV SIMD cv2.reduce 沿第 0 轴（按列）计算二维 uint8 矩阵的 float32 均值向量。

    :param mat_u8: 二维 uint8 矩阵 (H, W)。
    :type mat_u8: numpy.ndarray
    :return: 长度为 W 的一维 float32 均值数组。
    :author: tongliga
    :date: 2026-11-18
    """
    if mat_u8.shape[0] == 0 or mat_u8.shape[1] == 0:
        return np.zeros(mat_u8.shape[1], dtype=np.float32)
    return cv2.reduce(mat_u8, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).ravel()


def _reduce_mean_axis1(mat_u8: np.ndarray) -> np.ndarray:
    """
    使用 OpenCV SIMD cv2.reduce 沿第 1 轴（按行）计算二维 uint8 矩阵的 float32 均值向量。

    :param mat_u8: 二维 uint8 矩阵 (H, W)。
    :type mat_u8: numpy.ndarray
    :return: 长度为 H 的一维 float32 均值数组。
    :author: tongliga
    :date: 2026-11-18
    """
    if mat_u8.shape[0] == 0 or mat_u8.shape[1] == 0:
        return np.zeros(mat_u8.shape[0], dtype=np.float32)
    return cv2.reduce(mat_u8, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).ravel()


def _sample_strip(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    max_w: int,
    max_h: int,
    need_alpha: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用 OpenCV C++ INTER_NEAREST 查表 SIMD 快速提取连续采样条带（避免 NumPy 跨步拷低效循环）。

    不引入任何插值模糊，完整保留 1px 细线与原始像素值；对于 4 通道 RGBA 图像可在单次采样中
    同时分离出紧凑 RGB 条带与可选 Alpha 通道，在 4K/8K 图像上较 ``np.ascontiguousarray`` 跨步切片
    提速 3.5x~14x。

    :param img: 原始图像只读矩阵 (H, W, 3 或 4)。
    :type img: numpy.ndarray
    :param x0: ROI 左边界。
    :type x0: int
    :param y0: ROI 上边界。
    :type y0: int
    :param x1: ROI 右边界（不含）。
    :type x1: int
    :param y1: ROI 下边界（不含）。
    :type y1: int
    :param max_w: 采样目标最大宽度（<=0 或 >=roi_w 时保持原始宽度）。
    :type max_w: int
    :param max_h: 采样目标最大高度（<=0 或 >=roi_h 时保持原始高度）。
    :type max_h: int
    :param need_alpha: 当原图为 4 通道时是否同步提取 Alpha 通道矩阵。
    :type need_alpha: bool
    :return: 二元组 ``(rgb_u8_contiguous, alpha_u8_or_none)``。
    :author: tongliga
    :date: 2026-11-18
    """
    roi = img[y0:y1, x0:x1]
    rh, rw, ch = roi.shape
    dst_w = rw if (max_w <= 0 or rw <= max_w) else max_w
    dst_h = rh if (max_h <= 0 or rh <= max_h) else max_h
    if dst_w != rw or dst_h != rh:
        sampled = cv2.resize(roi, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    else:
        sampled = np.ascontiguousarray(roi)

    if ch >= 4:
        alpha = cv2.extractChannel(sampled, 3) if need_alpha else None
        rgb = cv2.cvtColor(sampled, cv2.COLOR_RGBA2RGB)
        return rgb, alpha
    return sampled, None


@dataclass(slots=True)
class RawSplitBox:
    """
    原始检测输出的切片包围盒与裁边修剪元数据。

    :author: tongliga
    :date: 2026-11-18
    """

    index: int
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    confidence: float
    strategy: str
    trim_top: int = 0
    trim_right: int = 0
    trim_bottom: int = 0
    trim_left: int = 0


@dataclass(slots=True)
class _SeamCandidate:
    """
    单个缝隙分割线候选点位信息。

    :author: tongliga
    :date: 2026-11-18
    """

    pos: int
    score: float
    coverage: float
    is_gutter: bool


@dataclass(slots=True)
class _SeamProfile:
    """
    投影视口区域内所有扫掠扫描线的缝隙评分剖面向量集合。

    :author: tongliga
    :date: 2026-11-18
    """

    positions: np.ndarray
    scores: np.ndarray
    coverage: np.ndarray
    is_gutter: np.ndarray
    is_hairline_sheet: bool = False


@dataclass(slots=True)
class _RawRegion:
    """
    XY-Cut 递归空间划分树中的未精修中间区域矩形（支持四边独立沟槽标记）。

    :author: tongliga
    :date: 2026-11-18
    """

    x0: int
    y0: int
    x1: int
    y1: int
    conf: float
    gutter_top: bool = False
    gutter_right: bool = False
    gutter_bottom: bool = False
    gutter_left: bool = False

    @property
    def is_gutter(self) -> bool:
        """
        判断该区域任意一条切割边界是否来自背景沟槽。

        :return: 若任一边为背景沟槽则返回 True。
        :author: tongliga
        :date: 2026-11-18
        """
        return self.gutter_top or self.gutter_right or self.gutter_bottom or self.gutter_left


@dataclass(slots=True)
class _BspNode:
    """
    表示递归 XY-Cut 二叉/多叉空间分割树节点的层次数据结构。

    显式保存每一层级的切割轴向、割线列表（含沟槽标记与得分）及子树集合，
    支持同轴展平、邻居割线传播挂载及自底向上几何尺度正则化剪枝。

    :author: tongliga
    :date: 2026-11-25
    """

    x0: int
    y0: int
    x1: int
    y1: int
    conf: float
    gutter_top: bool = False
    gutter_right: bool = False
    gutter_bottom: bool = False
    gutter_left: bool = False
    axis: Optional[str] = None
    cuts: List[Tuple[int, float, float, bool]] = field(default_factory=list)
    children: List["_BspNode"] = field(default_factory=list)

    def collect_leaves(self) -> List["_BspNode"]:
        """
        收集当前节点子树下按从前到后顺序排列的全部叶子节点。

        :return: 当前子树的叶子 `_BspNode` 列表。
        :author: tongliga
        :date: 2026-11-25
        """
        if not self.children:
            return [self]
        res: List[_BspNode] = []
        for child in self.children:
            res.extend(child.collect_leaves())
        return res

    def to_raw_region(self) -> _RawRegion:
        """
        将叶子 BSP 节点转换为包含四边独立沟槽标记的 `_RawRegion` 对象。

        :return: 对应的 `_RawRegion` 实例。
        :author: tongliga
        :date: 2026-11-25
        """
        return _RawRegion(
            self.x0,
            self.y0,
            self.x1,
            self.y1,
            self.conf,
            self.gutter_top,
            self.gutter_right,
            self.gutter_bottom,
            self.gutter_left,
        )


def _estimate_border_bg_stats(img: np.ndarray) -> Tuple[np.ndarray, bool, float, float]:
    """
    基于多环外框采样与软内点 L1 一致性投票估算主导背景色、平坦度与背景噪声水平。

    相比单环 15-bit 硬分箱直方图，本方法具有两项关键稳定性提升：
    1. 采样外侧 2 像素环带，抵抗 1px 截图边框/黑线污染；
    2. 找到量化峰值种子后在全部样本中执行软 L1 内点聚合（|rgb - seed|_1 <= 24），
       消除 JPEG 压缩噪声跨越量化分箱边界导致的平坦度骤降问题，同时估计背景噪声 MAD。

    :param img: 输入的原始图像 NumPy 数组（形状为 HxWx3 或 HxWx4）。
    :type img: numpy.ndarray
    :return: 四元组 ``(bg_rgba, is_transparent, uniformity, bg_noise_mad)``。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：沿图像四周外框提取 1~2 像素环带样本，增强抗 1px 边缘伪影能力
    h, w, ch = img.shape
    sx = max(1, w // 80)
    sy = max(1, h // 80)

    ring_parts = [img[0, ::sx], img[h - 1, ::sx], img[::sy, 0], img[::sy, w - 1]]
    if h >= 6 and w >= 6:
        ring_parts.extend([img[1, ::sx], img[h - 2, ::sx], img[::sy, 1], img[::sy, w - 2]])
    samples = np.concatenate(ring_parts, axis=0)
    n = max(1, len(samples))

    # 第二步：透明通道主导性判断（若半数以上边缘像素透明，则直接判定为透明背景）
    if ch >= 4:
        trans_cnt = int(np.count_nonzero(samples[:, 3] < 32))
        if trans_cnt > n // 2:
            return np.array([0, 0, 0, 0], dtype=np.uint8), True, float(trans_cnt) / float(n), 0.0

    # 第三步：15-bit 粗量化寻找众数种子色
    rgb = samples[:, :3].astype(np.int16)
    keys = ((rgb[:, 0].astype(np.int32) >> 3) << 10) | ((rgb[:, 1].astype(np.int32) >> 3) << 5) | (rgb[:, 2].astype(np.int32) >> 3)
    counts = np.bincount(keys, minlength=32768)
    best_key = int(np.argmax(counts))
    seed_mask = keys == best_key
    seed_rgb = np.median(rgb[seed_mask], axis=0).astype(np.int16)

    # 第四步：围绕种子色执行软 L1 内点共识聚合，并基于紧邻主众数子集估算高频噪声 MAD
    l1_dist = np.abs(rgb - seed_rgb).sum(axis=1)
    inliers = l1_dist <= 24
    inlier_cnt = max(1, int(np.count_nonzero(inliers)))
    inlier_rgb = rgb[inliers]
    mean_rgb = np.rint(inlier_rgb.mean(axis=0)).clip(0, 255).astype(np.uint8)
    # 仅在紧邻众数（|rgb - seed|_1 <= 9）内点上估计高频噪声 MAD，避免多色浅底拼接板误拉高噪声阈值
    tight_mask = l1_dist <= 9
    noise_rgb = rgb[tight_mask] if int(np.count_nonzero(tight_mask)) >= 8 else inlier_rgb
    tight_mean = np.rint(noise_rgb.mean(axis=0)).clip(0, 255).astype(np.int16)
    bg_noise_mad = float(np.abs(noise_rgb - tight_mean).mean())

    bg_rgba = np.array([int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]), 255], dtype=np.uint8)
    uniformity = float(inlier_cnt) / float(n)
    return bg_rgba, False, uniformity, bg_noise_mad


def estimate_border_bg(img: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """
    估算图像四周边框的主导背景色、是否透明背景及平坦度占比。

    :param img: 输入的原始图像 NumPy 数组（形状为 HxWx3 或 HxWx4）。
    :type img: numpy.ndarray
    :return: 三元组 ``(bg_rgba_uint8_4, is_transparent, uniformity_0_to_1)``。
    :author: tongliga
    :date: 2026-11-18
    """
    bg_rgba, is_trans, uniformity, _ = _estimate_border_bg_stats(img)
    return bg_rgba, is_trans, uniformity


def _validate_gutter_mask(
    g_mask: np.ndarray,
    bg_frac_full: np.ndarray,
    seam_inner: np.ndarray,
    cov_inner: np.ndarray,
    threshold: float,
    min_dim: int,
    allow_sparse_gutter: bool,
) -> np.ndarray:
    """
    从候选背景沟槽掩模中甄别真正的卡片间隔沟槽与白底幻灯片/文章内部的留白边距。

    甄别遵循三项图像布局拓扑几何原则：
    1. 双侧前景密度一致性：真实拼图沟槽两侧紧邻区域应存在实质前景实体（bg_frac 较低），而非单侧或双侧大面积留白；
    2. 锐利缝隙优先原则：若留白区一侧或内部紧邻全跨度高覆盖率非沟槽分割线（如 1px 细线或浅色底板接缝），
       或者在稀疏留白附近存在竞争性锐利面板边界，则该留白属于面板内边距而非独立卡片沟槽，应抑制以免遮蔽真接缝；
    3. 宽幅留白守卫：超出常规沟槽宽度的超大留白区域若紧邻接缝且非双侧致密前景包裹，一律判为面板留白。

    :param g_mask: 初步由平坦度与背景占比筛选出的内部沟槽布尔掩模。
    :type g_mask: numpy.ndarray
    :param bg_frac_full: 包含首尾边界的逐行/列背景占比数组。
    :type bg_frac_full: numpy.ndarray
    :param seam_inner: 对应内区间的锐利非沟槽缝隙评分数组。
    :type seam_inner: numpy.ndarray
    :param cov_inner: 对应内区间的精细覆盖率数组。
    :type cov_inner: numpy.ndarray
    :param threshold: 当前割线入选的能量阈值。
    :type threshold: float
    :param min_dim: 最小子块避让尺寸。
    :type min_dim: int
    :param allow_sparse_gutter: 是否允许在稀疏前景条件下保留窄幅独立沟槽（顶层或外缘已切沟槽区域）。
    :type allow_sparse_gutter: bool
    :return: 二元组 ``(validated_g_mask, is_hairline_sheet)``。
    :author: tongliga
    :date: 2026-11-25
    """
    n_inner = seam_inner.size
    n_full = bg_frac_full.size
    sharp_seam = (seam_inner >= threshold) & (cov_inner >= 0.72)
    # 若条带中存在双侧均为高背景留白（>0.65）的锋利 1~3px 分隔细线，说明当前为白底无缝分栏面板而非浮岛拼图，抑制内部留白误切
    sharp_indices = np.flatnonzero(sharp_seam)
    is_hairline_sheet = False
    for p_np in sharp_indices:
        p = int(p_np)
        far_l = bg_frac_full[max(0, p - 10) : max(0, p)]
        far_r = bg_frac_full[min(n_full, p + 3) : min(n_full, p + 13)]
        if far_l.size > 0 and far_r.size > 0 and float(far_l.mean()) > 0.65 and float(far_r.mean()) > 0.65:
            is_hairline_sheet = True
            break

    if is_hairline_sheet or not np.any(g_mask):
        return np.zeros_like(g_mask, dtype=bool), is_hairline_sheet

    validated = g_mask.copy()
    padded = np.empty(n_inner + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = g_mask
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1) - 1

    max_sparse_width = max(72, int(round(n_inner * 0.12)))

    for s_idx_np, e_idx_np in zip(starts, ends):
        s_idx = int(s_idx_np)
        e_idx = int(e_idx_np)
        run_len = e_idx - s_idx + 1

        # 注意：seam_inner[k] 与 bg_frac_full[k + 1] 一一对应
        fl_end = min(n_full, s_idx + 2)
        fl_start = max(0, fl_end - 12)
        fr_start = min(n_full, e_idx + 2)
        fr_end = min(n_full, fr_start + 12)

        bg_l_min = float(bg_frac_full[fl_start:fl_end].min()) if fl_end > fl_start else 1.0
        bg_r_min = float(bg_frac_full[fr_start:fr_end].min()) if fr_end > fr_start else 1.0
        max_flank_bg = max(bg_l_min, bg_r_min)

        # 双侧均为致密前景实体（<=0.45 背景占比）：检查边界是否仅为白底细线分隔伪影
        if max_flank_bg <= 0.45:
            # 确认边缘处的强锋利细线不是两侧均为高背景（>0.70）的 1px 隔断线
            edge_lo = max(0, s_idx - 3)
            edge_hi = min(n_inner, e_idx + 4)
            local_sharp = np.flatnonzero(sharp_seam[edge_lo:edge_hi])
            for rel_p in local_sharp:
                p = edge_lo + int(rel_p)
                far_l = bg_frac_full[max(0, p - 10) : max(0, p)]
                far_r = bg_frac_full[min(n_full, p + 3) : min(n_full, p + 13)]
                if far_l.size > 0 and far_r.size > 0 and float(far_l.mean()) > 0.70 and float(far_r.mean()) > 0.70:
                    validated[s_idx : e_idx + 1] = False
                    break
            continue

        # 单侧或双侧留白较多（max_flank_bg > 0.45）：严格校验是否真正无边界冲突且为窄沟槽
        if (not allow_sparse_gutter) or max_flank_bg > 0.72 or run_len > max_sparse_width:
            validated[s_idx : e_idx + 1] = False
            continue

        eff_s = max(s_idx, min_dim - 1)
        eff_e = min(e_idx, n_inner - min_dim)
        mid_eff = (eff_s + eff_e) // 2 if eff_e >= eff_s else (s_idx + e_idx) // 2
        chk_lo = max(0, mid_eff - min_dim + 1)
        chk_hi = min(n_inner, mid_eff + min_dim)
        if np.any(sharp_seam[chk_lo:chk_hi]):
            validated[s_idx : e_idx + 1] = False

    return validated, is_hairline_sheet


def _collapse_gutter_runs_vec(scores: np.ndarray, is_gutter: np.ndarray) -> None:
    """
    将连续成片的空白沟槽（is_gutter=True）就地合并并收敛至其确切的几何中心点。

    :param scores: 缝隙评分浮点数组，将在中心点处保留区间最大评分。
    :type scores: numpy.ndarray
    :param is_gutter: 标记各行/列是否属于空白沟槽的布尔数组，除中心点外就地清零。
    :type is_gutter: numpy.ndarray
    :author: tongliga
    :date: 2026-11-18
    """
    if not np.any(is_gutter):
        return
    # 第一步：在首尾补零以快速检测连通块的起止差分跃变
    padded = np.empty(is_gutter.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = is_gutter
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1) - 1  # 闭区间结束下标

    # 第二步：将每段连续沟槽归并到区间中点，置零非中点位置
    for s_idx, e_idx in zip(starts, ends):
        mid = (int(s_idx) + int(e_idx)) // 2
        max_s = float(scores[s_idx : e_idx + 1].max())
        scores[s_idx : e_idx + 1] = 0.0
        is_gutter[s_idx : e_idx + 1] = False
        scores[mid] = max_s
        is_gutter[mid] = True


def _prominence_1d(raw: np.ndarray, half_win: int = 6) -> np.ndarray:
    """
    计算一维缝隙能量曲线相对于两侧局部背景基线的突出度（Prominence SNR）。

    抑制平滑渐变背景带、连续图表纹理与 JPEG 压缩区块带来的宽域基准抬升，
    使 1~2px 锐利面板拼接边缘获得高信噪比峰值。

    :param raw: 原始轴向对数对比度均值曲线。
    :type raw: numpy.ndarray
    :param half_win: 邻域基线采样窗口半径（默认 6 像素，跳过紧邻中心 ±2px 过渡肩部）。
    :type half_win: int
    :return: 非负局部突出度向量（float32）。
    :author: tongliga
    :date: 2026-11-25
    """
    n = raw.size
    if n < half_win * 2:
        return raw
    pad = np.pad(raw, half_win, mode="edge")
    if half_win >= 6:
        l_min = np.minimum(np.minimum(pad[0:n], pad[1 : n + 1]), np.minimum(pad[2 : n + 2], pad[3 : n + 3]))
        r_min = np.minimum(
            np.minimum(pad[half_win + 3 : half_win + 3 + n], pad[half_win + 4 : half_win + 4 + n]),
            np.minimum(pad[half_win + 5 : half_win + 5 + n], pad[half_win + 6 : half_win + 6 + n]),
        )
    else:
        l_min = np.minimum(pad[0:n], pad[1 : n + 1])
        r_min = np.minimum(pad[2 * half_win - 1 : 2 * half_win - 1 + n], pad[2 * half_win : 2 * half_win + n])
    bg_1d = 0.5 * (l_min + r_min)
    return np.maximum(0.0, raw - 0.85 * bg_1d)


def _profile_horizontal_seams(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_dim: int,
    bg_rgb: np.ndarray,
    bg_uniform: bool,
    is_trans: bool,
    bg_noise: float = 0.0,
    return_curve: bool = False,
    threshold: float = 12.9,
    allow_sparse_gutter: bool = True,
) -> Tuple[Optional[_SeamProfile], Optional[np.ndarray]]:
    """
    在指定矩形区域内进行水平缝隙扫描与多因子能量评分（全 uint8 OpenCV SIMD + LUT + 突出度增强）。

    :param img: 全局只读图像矩阵。
    :type img: numpy.ndarray
    :param x0: 扫描区域左边界横坐标。
    :type x0: int
    :param y0: 扫描区域上边界纵坐标。
    :type y0: int
    :param x1: 扫描区域右边界横坐标。
    :type x1: int
    :param y1: 扫描区域下边界纵坐标。
    :type y1: int
    :param min_dim: 边缘避让最小尺寸，防止分割出过小碎块。
    :type min_dim: int
    :param bg_rgb: 背景色 RGB 数组。
    :type bg_rgb: numpy.ndarray
    :param bg_uniform: 背景是否为平坦纯色。
    :type bg_uniform: bool
    :param is_trans: 是否为透明背景。
    :type is_trans: bool
    :param bg_noise: 背景单通道噪声 MAD 估计值，用于自适应抗 JPEG 振铃。
    :type bg_noise: float
    :param return_curve: 是否返回完整的投影评分曲线（用于前端能量图展示）。
    :type return_curve: bool
    :return: 二元组包含 ``(_SeamProfile 缝隙剖面对象, 曲线数据数组或 None)``。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：计算扫描跨度并做基础有效性边界检查
    span_h = y1 - y0
    span_w = x1 - x0
    if span_h <= min_dim * 2 or span_w < 8:
        return None, None

    # 第二步：采用 OpenCV SIMD INTER_NEAREST 提取最大 200 采样列的紧凑条带（span_h × <=200 × 3）
    need_alpha = bg_uniform and is_trans and img.shape[2] >= 4
    strip, alpha_strip = _sample_strip(img, x0, y0, x1, y1, 200, span_h, need_alpha=need_alpha)

    c_mid = strip[1:-1]
    c_prev = strip[:-2]
    c_next = strip[2:]

    # 第三步：SIMD 色差 + cv2.LUT 韦伯对数压缩 + 1D 局部突出度计算
    ds_u8 = _color_diff_u8(c_mid, c_next)
    avg_pn = cv2.addWeighted(c_prev, 0.5, c_next, 0.5, 0.0)
    dl_u8 = _color_diff_u8(c_mid, avg_pn)
    del avg_pn

    raw_inner = 0.58 * _reduce_mean_axis1(cv2.LUT(ds_u8, _SEAM_LUT)) + 0.42 * _reduce_mean_axis1(
        cv2.LUT(dl_u8, _SEAM_LUT)
    )
    comb_u8 = cv2.add(ds_u8, dl_u8)
    del ds_u8, dl_u8

    # 第四步：在 ±2px 垂直窗口内通过 SIMD 形态学膨胀评估精细横贯覆盖度 C(y)
    _, act = cv2.threshold(comb_u8, 2, 1, cv2.THRESH_BINARY)
    del comb_u8
    act_dil = cv2.dilate(act, _KERNEL_5x1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    del act
    cov_inner = _reduce_mean_axis1(act_dil)
    del act_dil

    prom_inner = _prominence_1d(raw_inner, half_win=6)

    # 第五步：计算抗局部异常值干扰的切向平滑度（截断单点剧烈跳变，避免角标/页码破坏真接缝）
    luma_u8 = cv2.cvtColor(c_mid, cv2.COLOR_RGB2GRAY)
    if luma_u8.shape[1] > 1:
        tang = cv2.absdiff(luma_u8[:, 1:], luma_u8[:, :-1])
        tang_clipped = cv2.min(tang, 40)
        sm_inner = 3.0 * _reduce_mean_axis1(tang_clipped)
        del tang, tang_clipped
    else:
        sm_inner = np.zeros(span_h - 2, dtype=np.float32)
    del luma_u8

    sm_prev = np.empty_like(sm_inner)
    sm_prev[0] = sm_inner[0]
    sm_prev[1:] = sm_inner[:-1]
    sm_next = np.empty_like(sm_inner)
    sm_next[-1] = sm_inner[-1]
    sm_next[:-1] = sm_inner[1:]
    min_smooth = np.minimum(np.minimum(sm_prev, sm_inner), sm_next)

    seam_inner = (prom_inner * np.power(cov_inner, 1.8)) / (1.0 + min_smooth * 0.12)

    # 第六步：针对平坦背景图像激活噪声自适应空白沟槽评分
    gutter_inner = np.zeros(span_h - 2, dtype=np.float32)
    is_hairline_sheet = False
    if bg_uniform:
        if alpha_strip is not None:
            _, bg_bin = cv2.threshold(alpha_strip, 23, 1, cv2.THRESH_BINARY_INV)
            bg_frac_full = _reduce_mean_axis1(bg_bin)
            del alpha_strip, bg_bin
        else:
            bg_scalar = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 0)
            bg_diff_u8 = _color_diff_u8(strip, bg_scalar)
            bg_tol = int(min(18, max(7, round(7.0 + bg_noise * 2.5))))
            _, bg_bin = cv2.threshold(bg_diff_u8, bg_tol, 1, cv2.THRESH_BINARY_INV)
            bg_frac_full = _reduce_mean_axis1(bg_bin)
            del bg_diff_u8, bg_bin
        del strip

        fg_rows = np.flatnonzero(bg_frac_full < 0.90)
        if fg_rows.size >= 2:
            first_fg = int(fg_rows[0])
            last_fg = int(fg_rows[-1])
            ry_idx = np.arange(1, span_h - 1)
            sm_limit = 3.2 + bg_noise * 0.6
            g_mask = (
                (sm_inner < sm_limit)
                & (bg_frac_full[1:-1] > 0.94)
                & (ry_idx > first_fg)
                & (ry_idx < last_fg)
            )
        else:
            g_mask = np.zeros(span_h - 2, dtype=bool)
        g_mask, is_hairline_sheet = _validate_gutter_mask(
            g_mask,
            bg_frac_full,
            seam_inner,
            cov_inner,
            threshold=threshold,
            min_dim=min_dim,
            allow_sparse_gutter=allow_sparse_gutter,
        )
        gutter_inner[g_mask] = 110.0 / (1.0 + sm_inner[g_mask])
    else:
        del strip

    # 第七步：截取符合 min_dim 边缘避让的安全评分切片并折叠连续沟槽
    start_ry = min_dim
    end_ry = span_h - min_dim
    if end_ry < start_ry:
        return None, None

    i0 = max(0, start_ry - 1)
    i1 = min(span_h - 2, end_ry)
    s_slice = seam_inner[i0:i1]
    g_slice = gutter_inner[i0:i1]
    c_slice = cov_inner[i0:i1].copy()
    scores = np.maximum(s_slice, g_slice)
    is_gut = (g_slice > s_slice).copy()
    _collapse_gutter_runs_vec(scores, is_gut)
    positions = np.arange(y0 + i0 + 1, y0 + i0 + 1 + len(scores), dtype=np.int32)

    curve = None
    if return_curve:
        full_scores = np.zeros(span_h, dtype=np.float32)
        full_scores[1:-1] = np.maximum(seam_inner, gutter_inner)
        curve = full_scores

    return (
        _SeamProfile(
            positions=positions,
            scores=scores,
            coverage=c_slice,
            is_gutter=is_gut,
            is_hairline_sheet=is_hairline_sheet,
        ),
        curve,
    )


def _profile_vertical_seams(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_dim: int,
    bg_rgb: np.ndarray,
    bg_uniform: bool,
    is_trans: bool,
    bg_noise: float = 0.0,
    return_curve: bool = False,
    threshold: float = 12.9,
    allow_sparse_gutter: bool = True,
) -> Tuple[Optional[_SeamProfile], Optional[np.ndarray]]:
    """
    在指定矩形区域内进行垂直缝隙扫描与多因子能量评分（全 uint8 OpenCV SIMD + LUT + 突出度增强）。

    :param img: 全局只读图像矩阵。
    :type img: numpy.ndarray
    :param x0: 扫描区域左边界横坐标。
    :type x0: int
    :param y0: 扫描区域上边界纵坐标。
    :type y0: int
    :param x1: 扫描区域右边界横坐标。
    :type x1: int
    :param y1: 扫描区域下边界纵坐标。
    :type y1: int
    :param min_dim: 边缘避让最小尺寸。
    :type min_dim: int
    :param bg_rgb: 背景色 RGB 数组。
    :type bg_rgb: numpy.ndarray
    :param bg_uniform: 背景是否为平坦纯色。
    :type bg_uniform: bool
    :param is_trans: 是否为透明背景。
    :type is_trans: bool
    :param bg_noise: 背景单通道噪声 MAD 估计值。
    :type bg_noise: float
    :param return_curve: 是否返回完整的投影评分曲线。
    :type return_curve: bool
    :return: 二元组包含 ``(_SeamProfile 垂直缝隙剖面对象, 曲线数据数组或 None)``。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：计算扫描宽度并检查有效跨度
    span_w = x1 - x0
    span_h = y1 - y0
    if span_w <= min_dim * 2 or span_h < 8:
        return None, None

    # 第二步：采用 OpenCV SIMD INTER_NEAREST 提取最大 200 采样行的紧凑条带（<=200 × span_w × 3）
    need_alpha = bg_uniform and is_trans and img.shape[2] >= 4
    strip, alpha_strip = _sample_strip(img, x0, y0, x1, y1, span_w, 200, need_alpha=need_alpha)

    c_mid = strip[:, 1:-1]
    c_prev = strip[:, :-2]
    c_next = strip[:, 2:]

    # 第三步：SIMD 色差 + cv2.LUT 韦伯对数压缩 + 1D 局部突出度计算
    ds_u8 = _color_diff_u8(c_mid, c_next)
    avg_pn = cv2.addWeighted(c_prev, 0.5, c_next, 0.5, 0.0)
    dl_u8 = _color_diff_u8(c_mid, avg_pn)
    del avg_pn

    raw_inner = 0.58 * _reduce_mean_axis0(cv2.LUT(ds_u8, _SEAM_LUT)) + 0.42 * _reduce_mean_axis0(
        cv2.LUT(dl_u8, _SEAM_LUT)
    )
    comb_u8 = cv2.add(ds_u8, dl_u8)
    del ds_u8, dl_u8

    # 第四步：在 ±2px 水平窗口内通过 SIMD 形态学膨胀评估纵贯覆盖度 C(x)
    _, act = cv2.threshold(comb_u8, 2, 1, cv2.THRESH_BINARY)
    del comb_u8
    act_dil = cv2.dilate(act, _KERNEL_1x5, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    del act
    cov_inner = _reduce_mean_axis0(act_dil)
    del act_dil

    prom_inner = _prominence_1d(raw_inner, half_win=6)

    # 第五步：计算抗局部异常值干扰的纵向平滑度
    luma_u8 = cv2.cvtColor(c_mid, cv2.COLOR_RGB2GRAY)
    if luma_u8.shape[0] > 1:
        tang = cv2.absdiff(luma_u8[1:, :], luma_u8[:-1, :])
        tang_clipped = cv2.min(tang, 40)
        sm_inner = 3.0 * _reduce_mean_axis0(tang_clipped)
        del tang, tang_clipped
    else:
        sm_inner = np.zeros(span_w - 2, dtype=np.float32)
    del luma_u8

    sm_prev = np.empty_like(sm_inner)
    sm_prev[0] = sm_inner[0]
    sm_prev[1:] = sm_inner[:-1]
    sm_next = np.empty_like(sm_inner)
    sm_next[-1] = sm_inner[-1]
    sm_next[:-1] = sm_inner[1:]
    min_smooth = np.minimum(np.minimum(sm_prev, sm_inner), sm_next)

    seam_inner = (prom_inner * np.power(cov_inner, 1.8)) / (1.0 + min_smooth * 0.12)

    # 第六步：针对平坦背景激活噪声自适应垂直空白沟槽强化
    gutter_inner = np.zeros(span_w - 2, dtype=np.float32)
    is_hairline_sheet = False
    if bg_uniform:
        if alpha_strip is not None:
            _, bg_bin = cv2.threshold(alpha_strip, 23, 1, cv2.THRESH_BINARY_INV)
            bg_frac_full = _reduce_mean_axis0(bg_bin)
            del alpha_strip, bg_bin
        else:
            bg_scalar = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 0)
            bg_diff_u8 = _color_diff_u8(strip, bg_scalar)
            bg_tol = int(min(18, max(7, round(7.0 + bg_noise * 2.5))))
            _, bg_bin = cv2.threshold(bg_diff_u8, bg_tol, 1, cv2.THRESH_BINARY_INV)
            bg_frac_full = _reduce_mean_axis0(bg_bin)
            del bg_diff_u8, bg_bin
        del strip

        fg_cols = np.flatnonzero(bg_frac_full < 0.90)
        if fg_cols.size >= 2:
            first_fg = int(fg_cols[0])
            last_fg = int(fg_cols[-1])
            rx_idx = np.arange(1, span_w - 1)
            sm_limit = 3.2 + bg_noise * 0.6
            g_mask = (
                (sm_inner < sm_limit)
                & (bg_frac_full[1:-1] > 0.94)
                & (rx_idx > first_fg)
                & (rx_idx < last_fg)
            )
        else:
            g_mask = np.zeros(span_w - 2, dtype=bool)
        g_mask, is_hairline_sheet = _validate_gutter_mask(
            g_mask,
            bg_frac_full,
            seam_inner,
            cov_inner,
            threshold=threshold,
            min_dim=min_dim,
            allow_sparse_gutter=allow_sparse_gutter,
        )
        gutter_inner[g_mask] = 110.0 / (1.0 + sm_inner[g_mask])
    else:
        del strip

    # 第七步：截取符合安全跨度的评分切片
    start_rx = min_dim
    end_rx = span_w - min_dim
    if end_rx < start_rx:
        return None, None

    i0 = max(0, start_rx - 1)
    i1 = min(span_w - 2, end_rx)
    s_slice = seam_inner[i0:i1]
    g_slice = gutter_inner[i0:i1]
    c_slice = cov_inner[i0:i1].copy()
    scores = np.maximum(s_slice, g_slice)
    is_gut = (g_slice > s_slice).copy()
    _collapse_gutter_runs_vec(scores, is_gut)
    positions = np.arange(x0 + i0 + 1, x0 + i0 + 1 + len(scores), dtype=np.int32)

    curve = None
    if return_curve:
        full_scores = np.zeros(span_w, dtype=np.float32)
        full_scores[1:-1] = np.maximum(seam_inner, gutter_inner)
        curve = full_scores

    return (
        _SeamProfile(
            positions=positions,
            scores=scores,
            coverage=c_slice,
            is_gutter=is_gut,
            is_hairline_sheet=is_hairline_sheet,
        ),
        curve,
    )


def _pick_peaks_nms(
    profile: Optional[_SeamProfile],
    threshold: float,
    min_gap: int,
    min_coverage: float = 0.70,
) -> List[_SeamCandidate]:
    """
    通过非极大值抑制（NMS）从缝隙剖面中筛选出互不重叠的高置信度候选分割线。

    :param profile: 缝隙打分剖面对象。
    :type profile: Optional[_SeamProfile]
    :param threshold: 分割线入围的基础评分阈值。
    :type threshold: float
    :param min_gap: 两条平行分割线之间的最小间距（像素）。
    :type min_gap: int
    :param min_coverage: 非沟槽割线最低跨度覆盖率门限，默认为 0.70。
    :type min_coverage: float
    :return: 按坐标升序排序的高质量候选缝隙点位列表。
    :author: tongliga
    :date: 2026-11-18
    """
    if profile is None or profile.scores.size == 0:
        return []
    # 过滤出符合阈值要求或属于沟槽的候选索引
    valid_idx = np.flatnonzero(
        (profile.scores >= threshold) & (profile.is_gutter | (profile.coverage >= min_coverage))
    )
    if valid_idx.size == 0:
        return []

    # 按评分降序排列并执行非极大值抑制
    order = valid_idx[np.argsort(-profile.scores[valid_idx])]
    picked: List[_SeamCandidate] = []
    for idx in order:
        pos = int(profile.positions[idx])
        if all(abs(p.pos - pos) >= min_gap for p in picked):
            picked.append(
                _SeamCandidate(
                    pos=pos,
                    score=float(profile.scores[idx]),
                    coverage=float(profile.coverage[idx]),
                    is_gutter=bool(profile.is_gutter[idx]),
                )
            )
    picked.sort(key=lambda c: c.pos)
    return picked


def _is_region_empty_bg(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    bg_rgb: np.ndarray,
    is_trans: bool,
    bg_noise: float = 0.0,
) -> bool:
    """
    通过稀疏步长采样，快速判定指定矩形区域是否仅为纯净背景色块（用于剔除无内容空白废块）。

    :param img: 原始图像数组。
    :type img: numpy.ndarray
    :param x0: 区域左界。
    :type x0: int
    :param y0: 区域上界。
    :type y0: int
    :param x1: 区域右界。
    :type x1: int
    :param y1: 区域下界。
    :type y1: int
    :param bg_rgb: 背景颜色 RGB 数组。
    :type bg_rgb: numpy.ndarray
    :param is_trans: 是否为透明背景模式。
    :type is_trans: bool
    :param bg_noise: 背景噪声估计值。
    :type bg_noise: float
    :return: 若该区域前景占比低于 3%，则判定为空白背景返回 True，否则返回 False。
    :author: tongliga
    :date: 2026-11-18
    """
    if x1 - x0 <= 4 or y1 - y0 <= 4:
        return False
    step_x = max(1, (x1 - x0) // 32)
    step_y = max(1, (y1 - y0) // 32)
    sub = np.ascontiguousarray(img[y0 + 2 : y1 - 2 : step_y, x0 + 2 : x1 - 2 : step_x])
    if sub.size == 0:
        return False
    if is_trans and img.shape[2] >= 4:
        fg = int(np.count_nonzero(sub[:, :, 3] >= 24))
    else:
        bg_scalar = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 0)
        bg_tol = int(min(18, max(7, round(7.0 + bg_noise * 2.5))))
        diff = _color_diff_u8(sub[:, :, :3], bg_scalar)
        fg = int(np.count_nonzero(diff > bg_tol))
    total = sub.shape[0] * sub.shape[1]
    return total > 0 and (float(fg) / float(total)) < 0.03


def _filter_hierarchy_peaks(peaks: List[_SeamCandidate], global_rails: List[int]) -> List[_SeamCandidate]:
    """
    在单层子条带剖分时进行层次化主次分离：防止微弱的卡片内部纹路割线与全局强主干导轨混在同一多叉划分层。

    若存在强主干割线（max_score >= 30.0），仅在本层保留沟槽、全跨度高覆盖率或与顶层全局导轨对齐的强候选；
    其余较弱的局部子候选自然下放至子节点递归层级参与全局尺度正则化评估。

    :param peaks: 当前条带选出的非极大值抑制割线候选列表。
    :type peaks: List[_SeamCandidate]
    :param global_rails: 顶层全局投影分析抽取的贯穿导轨坐标列表。
    :type global_rails: List[int]
    :return: 经过层次主次过滤的割线候选列表。
    :author: tongliga
    :date: 2026-11-25
    """
    if len(peaks) <= 1:
        return peaks
    max_sc = max(p.score for p in peaks)
    if max_sc < 30.0:
        return peaks
    kept = [
        p
        for p in peaks
        if p.is_gutter
        or p.score >= max_sc * 0.30
        or p.coverage >= 0.90
        or any(abs(p.pos - gr) <= 16 for gr in global_rails)
    ]
    return kept if kept else peaks


def _build_bsp_tree(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    depth: int,
    min_dim: int,
    threshold: float,
    bg_rgb: np.ndarray,
    bg_uniform: bool,
    is_trans: bool,
    bg_noise: float,
    parent_conf: float,
    gutter_top: bool,
    gutter_right: bool,
    gutter_bottom: bool,
    gutter_left: bool,
    leaf_budget: List[int],
    global_h_rails: Optional[List[int]] = None,
    global_v_rails: Optional[List[int]] = None,
    top_curves: Optional[Dict[str, np.ndarray]] = None,
    disable_gutter: bool = False,
) -> _BspNode:
    """
    递归构建 XY-Cut 空间划分树（BSP Tree），跟踪独立四边沟槽标记与全局主干导轨共识。

    :param img: 原始图像数组。
    :type img: numpy.ndarray
    :param x0: 当前子空间左上角横坐标。
    :type x0: int
    :param y0: 当前子空间左上角纵坐标。
    :type y0: int
    :param x1: 当前子空间右下角横坐标。
    :type x1: int
    :param y1: 当前子空间右下角纵坐标。
    :type y1: int
    :param depth: 当前递归调用深度（上限 6）。
    :type depth: int
    :param min_dim: 最小允许子块跨度。
    :type min_dim: int
    :param threshold: 割线切分能量阈值。
    :type threshold: float
    :param bg_rgb: 背景色 RGB 数组。
    :type bg_rgb: numpy.ndarray
    :param bg_uniform: 背景是否平坦。
    :type bg_uniform: bool
    :param is_trans: 背景是否为透明通道。
    :type is_trans: bool
    :param bg_noise: 背景噪声估计值。
    :type bg_noise: float
    :param parent_conf: 父级分割线继承的置信度得分。
    :type parent_conf: float
    :param gutter_top: 上边缘是否来自空白沟槽。
    :type gutter_top: bool
    :param gutter_right: 右边缘是否来自空白沟槽。
    :type gutter_right: bool
    :param gutter_bottom: 下边缘是否来自空白沟槽。
    :type gutter_bottom: bool
    :param gutter_left: 左边缘是否来自空白沟槽。
    :type gutter_left: bool
    :param leaf_budget: 叶子节点总数量配额计数器（避免病态图深度爆发）。
    :type leaf_budget: List[int]
    :param global_h_rails: 顶层水平全局主干导轨坐标集合。
    :type global_h_rails: Optional[List[int]]
    :param global_v_rails: 顶层垂直全局主干导轨坐标集合。
    :type global_v_rails: Optional[List[int]]
    :param top_curves: 顶层投影曲线收集字典。
    :type top_curves: Optional[Dict[str, numpy.ndarray]]
    :param disable_gutter: 当上层已确认为细线分隔的白底拼版套图时，禁止深层退化为内部留白沟槽切分。
    :type disable_gutter: bool
    :return: 构建完成的 `_BspNode` 节点树。
    :author: tongliga
    :date: 2026-11-18
    """
    rw = x1 - x0
    rh = y1 - y0
    if depth >= 6 or leaf_budget[0] >= 256 or (rw < min_dim * 2 and rh < min_dim * 2):
        leaf_budget[0] += 1
        return _BspNode(x0, y0, x1, y1, parent_conf, gutter_top, gutter_right, gutter_bottom, gutter_left)

    capture = depth == 0 and top_curves is not None
    enable_gutter = bg_uniform and (not disable_gutter)
    allow_sparse_gutter = (depth == 0) or gutter_top or gutter_right or gutter_bottom or gutter_left
    h_prof, h_curve = _profile_horizontal_seams(
        img,
        x0,
        y0,
        x1,
        y1,
        min_dim,
        bg_rgb,
        enable_gutter,
        is_trans,
        bg_noise=bg_noise,
        return_curve=capture,
        threshold=threshold,
        allow_sparse_gutter=allow_sparse_gutter,
    )
    v_prof, v_curve = _profile_vertical_seams(
        img,
        x0,
        y0,
        x1,
        y1,
        min_dim,
        bg_rgb,
        enable_gutter,
        is_trans,
        bg_noise=bg_noise,
        return_curve=capture,
        threshold=threshold,
        allow_sparse_gutter=allow_sparse_gutter,
    )
    if capture and top_curves is not None:
        if h_curve is not None:
            top_curves["h"] = h_curve
        if v_curve is not None:
            top_curves["v"] = v_curve

    min_cov = 0.96 if (disable_gutter and depth >= 2) else 0.70
    h_peaks = _pick_peaks_nms(h_prof, threshold, min_dim, min_coverage=min_cov)
    v_peaks = _pick_peaks_nms(v_prof, threshold, min_dim, min_coverage=min_cov)

    if depth == 0:
        global_h_rails = [
            p.pos for p in h_peaks if p.is_gutter or (p.score >= max(threshold, 14.0) and p.coverage >= 0.72)
        ]
        global_v_rails = [
            p.pos for p in v_peaks if p.is_gutter or (p.score >= max(threshold, 14.0) and p.coverage >= 0.72)
        ]
    else:
        h_peaks = _filter_hierarchy_peaks(h_peaks, global_h_rails or [])
        v_peaks = _filter_hierarchy_peaks(v_peaks, global_v_rails or [])

    # 深度 >= 4 时仅允许纯净背景沟槽继续嵌套切分，防止误切幻灯片/卡片内部的标题栏与装饰色带
    if depth >= 4:
        h_peaks = [p for p in h_peaks if p.is_gutter]
        v_peaks = [p for p in v_peaks if p.is_gutter]

    if not h_peaks and not v_peaks:
        leaf_budget[0] += 1
        return _BspNode(x0, y0, x1, y1, parent_conf, gutter_top, gutter_right, gutter_bottom, gutter_left)

    # 采用覆盖率加权与沟槽优先原则仲裁横/纵切割轴，避免 Bento 布局误穿通高列
    h_has_gutter = any(p.is_gutter for p in h_peaks)
    v_has_gutter = any(p.is_gutter for p in v_peaks)
    h_eff = max((p.score * (0.55 + 0.45 * (1.0 if p.is_gutter else p.coverage)) for p in h_peaks), default=0.0)
    v_eff = max((p.score * (0.55 + 0.45 * (1.0 if p.is_gutter else p.coverage)) for p in v_peaks), default=0.0)

    if h_has_gutter and not v_has_gutter and h_eff >= v_eff * 0.65:
        choose_h = True
    elif v_has_gutter and not h_has_gutter and v_eff >= h_eff * 0.65:
        choose_h = False
    else:
        choose_h = bool(h_peaks) and (not v_peaks or h_eff >= v_eff * 0.85)

    any_hairline_sheet = bool(
        (h_prof is not None and h_prof.is_hairline_sheet) or (v_prof is not None and v_prof.is_hairline_sheet)
    )
    node = _BspNode(x0, y0, x1, y1, parent_conf, gutter_top, gutter_right, gutter_bottom, gutter_left)
    if choose_h:
        child_disable_gutter = disable_gutter or (any_hairline_sheet and not h_has_gutter)
        node.axis = "h"
        node.cuts = [(p.pos, p.score, p.coverage, p.is_gutter) for p in h_peaks]
        bounds = [y0] + [p.pos for p in h_peaks] + [y1]
        for i in range(len(bounds) - 1):
            p_idx = 0 if i == 0 else i - 1
            gt = h_peaks[i - 1].is_gutter if i > 0 else gutter_top
            gb = h_peaks[i].is_gutter if i < len(h_peaks) else gutter_bottom
            node.children.append(
                _build_bsp_tree(
                    img,
                    x0,
                    bounds[i],
                    x1,
                    bounds[i + 1],
                    depth + 1,
                    min_dim,
                    threshold,
                    bg_rgb,
                    bg_uniform,
                    is_trans,
                    bg_noise,
                    max(parent_conf, h_peaks[p_idx].score),
                    gt,
                    gutter_right,
                    gb,
                    gutter_left,
                    leaf_budget,
                    global_h_rails=global_h_rails,
                    global_v_rails=global_v_rails,
                    disable_gutter=child_disable_gutter,
                )
            )
    else:
        child_disable_gutter = disable_gutter or (any_hairline_sheet and not v_has_gutter)
        node.axis = "v"
        node.cuts = [(p.pos, p.score, p.coverage, p.is_gutter) for p in v_peaks]
        bounds = [x0] + [p.pos for p in v_peaks] + [x1]
        for i in range(len(bounds) - 1):
            p_idx = 0 if i == 0 else i - 1
            gl = v_peaks[i - 1].is_gutter if i > 0 else gutter_left
            gr = v_peaks[i].is_gutter if i < len(v_peaks) else gutter_right
            node.children.append(
                _build_bsp_tree(
                    img,
                    bounds[i],
                    y0,
                    bounds[i + 1],
                    y1,
                    depth + 1,
                    min_dim,
                    threshold,
                    bg_rgb,
                    bg_uniform,
                    is_trans,
                    bg_noise,
                    max(parent_conf, v_peaks[p_idx].score),
                    gutter_top,
                    gr,
                    gutter_bottom,
                    gl,
                    leaf_budget,
                    global_h_rails=global_h_rails,
                    global_v_rails=global_v_rails,
                    disable_gutter=child_disable_gutter,
                )
            )
    return node


def _sibling_propagate_h(
    img: np.ndarray,
    leaf: _BspNode,
    y_cut: int,
    min_dim: int,
    prop_threshold: float,
) -> Optional[Tuple[int, float, float]]:
    """
    沿邻居叶子节点的水平割线在当前区域内进行 ±12px 范围的 LUT 突显度对齐传播探测。

    使用 OpenCV SIMD 双通道色差 + `cv2.LUT` 韦伯压缩 + 局部基线突出度与 ±1px 邻域覆盖率，
    稳定召回白底/浅灰极弱对齐细线及渐变边沿面板边界，同时防止误切高纹理通高 Bento 卡片。

    :param img: 原始图像数组，要求为三维 uint8 数组（H, W, C）。
    :type img: numpy.ndarray
    :param leaf: 当前待尝试继续分割的叶子节点区域。
    :type leaf: _BspNode
    :param y_cut: 邻居叶子节点产生的水平割线候选纵坐标。
    :type y_cut: int
    :param min_dim: 最小允许子块高度或宽度跨度。
    :type min_dim: int
    :param prop_threshold: 邻居传播唤醒所需的最低缝隙评分阈值。
    :type prop_threshold: float
    :return: 若探测成功则返回 ``(best_y, score, coverage)``，否则返回 None。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：计算在当前叶子节点内的安全探测搜索区间及突出度上下文区间
    lo = max(leaf.y0 + min_dim, y_cut - 12)
    hi = min(leaf.y1 - min_dim, y_cut + 12)
    if hi < lo or leaf.x1 - leaf.x0 <= 10:
        return None
    clo = max(leaf.y0 + 1, lo - 4)
    chi = min(leaf.y1 - 1, hi + 4)

    # 第二步：采用 OpenCV INTER_NEAREST 沿横向提取带基线上下文的紧凑条带
    strip, _ = _sample_strip(img, leaf.x0 + 4, clo - 1, leaf.x1 - 4, chi + 2, 200, chi - clo + 3)
    if strip.shape[0] < 3 or strip.shape[1] < 2:
        return None

    # 第三步：计算中心行与相邻行的 SIMD 均衡色差、LUT 对比度及 1D 局部突出度
    c_mid = strip[1:-1]
    c_prev = strip[:-2]
    c_next = strip[2:]
    ds_u8 = _color_diff_u8(c_mid, c_next)
    dl_u8 = _color_diff_u8(c_mid, cv2.addWeighted(c_prev, 0.5, c_next, 0.5, 0.0))
    raw = 0.58 * _reduce_mean_axis1(cv2.LUT(ds_u8, _SEAM_LUT)) + 0.42 * _reduce_mean_axis1(
        cv2.LUT(dl_u8, _SEAM_LUT)
    )
    comb_u8 = cv2.add(ds_u8, dl_u8)

    # 第四步：评估 ±1px 覆盖度、突出度信噪比与三行最小平滑度
    _, act = cv2.threshold(comb_u8, 2, 1, cv2.THRESH_BINARY)
    cov = _reduce_mean_axis1(cv2.dilate(act, _KERNEL_3x1, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    prom = _prominence_1d(raw, half_win=4)

    luma = cv2.cvtColor(c_mid, cv2.COLOR_RGB2GRAY)
    tang = cv2.min(cv2.absdiff(luma[:, 1:], luma[:, :-1]), 40)
    sm = 3.0 * _reduce_mean_axis1(tang)
    sm_p = np.pad(sm, 1, mode="edge")
    min_sm = np.minimum(np.minimum(sm_p[:-2], sm_p[1:-1]), sm_p[2:])
    sc = (prom * np.power(cov, 1.8)) / (1.0 + min_sm * 0.12)

    off = lo - clo
    span_len = hi - lo + 1
    sc_w = sc[off : off + span_len]
    cov_w = cov[off : off + span_len]
    prom_w = prom[off : off + span_len]

    valid = (cov_w >= 0.78) & (prom_w >= 6.0) & (sc_w >= prop_threshold)
    if not np.any(valid):
        return None
    best_idx = int(np.argmax(np.where(valid, sc_w, -1.0)))
    return lo + best_idx, float(sc_w[best_idx]), float(cov_w[best_idx])


def _sibling_propagate_v(
    img: np.ndarray,
    leaf: _BspNode,
    x_cut: int,
    min_dim: int,
    prop_threshold: float,
) -> Optional[Tuple[int, float, float]]:
    """
    沿邻居叶子节点的垂直割线在当前区域内进行 ±12px 范围的 LUT 突显度对齐传播探测。

    :param img: 原始图像数组，要求为三维 uint8 数组（H, W, C）。
    :type img: numpy.ndarray
    :param leaf: 当前待尝试继续分割的叶子节点区域。
    :type leaf: _BspNode
    :param x_cut: 邻居叶子节点产生的垂直割线候选横坐标。
    :type x_cut: int
    :param min_dim: 最小允许子块高度或宽度跨度。
    :type min_dim: int
    :param prop_threshold: 邻居传播唤醒所需的最低缝隙评分阈值。
    :type prop_threshold: float
    :return: 若探测成功则返回 ``(best_x, score, coverage)``，否则返回 None。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：计算在当前叶子节点内的安全探测搜索区间及突出度上下文区间
    lo = max(leaf.x0 + min_dim, x_cut - 12)
    hi = min(leaf.x1 - min_dim, x_cut + 12)
    if hi < lo or leaf.y1 - leaf.y0 <= 10:
        return None
    clo = max(leaf.x0 + 1, lo - 4)
    chi = min(leaf.x1 - 1, hi + 4)

    # 第二步：采用 OpenCV INTER_NEAREST 沿纵向提取带基线上下文的紧凑条带
    strip, _ = _sample_strip(img, clo - 1, leaf.y0 + 4, chi + 2, leaf.y1 - 4, chi - clo + 3, 200)
    if strip.shape[1] < 3 or strip.shape[0] < 2:
        return None

    # 第三步：计算中心列与相邻列的 SIMD 均衡色差、LUT 对比度及 1D 局部突出度
    c_mid = strip[:, 1:-1]
    c_prev = strip[:, :-2]
    c_next = strip[:, 2:]
    ds_u8 = _color_diff_u8(c_mid, c_next)
    dl_u8 = _color_diff_u8(c_mid, cv2.addWeighted(c_prev, 0.5, c_next, 0.5, 0.0))
    raw = 0.58 * _reduce_mean_axis0(cv2.LUT(ds_u8, _SEAM_LUT)) + 0.42 * _reduce_mean_axis0(
        cv2.LUT(dl_u8, _SEAM_LUT)
    )
    comb_u8 = cv2.add(ds_u8, dl_u8)

    # 第四步：评估 ±1px 覆盖度、突出度信噪比与三列最小平滑度
    _, act = cv2.threshold(comb_u8, 2, 1, cv2.THRESH_BINARY)
    cov = _reduce_mean_axis0(cv2.dilate(act, _KERNEL_1x3, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    prom = _prominence_1d(raw, half_win=4)

    luma = cv2.cvtColor(c_mid, cv2.COLOR_RGB2GRAY)
    tang = cv2.min(cv2.absdiff(luma[1:, :], luma[:-1, :]), 40)
    sm = 3.0 * _reduce_mean_axis0(tang)
    sm_p = np.pad(sm, 1, mode="edge")
    min_sm = np.minimum(np.minimum(sm_p[:-2], sm_p[1:-1]), sm_p[2:])
    sc = (prom * np.power(cov, 1.8)) / (1.0 + min_sm * 0.12)

    off = lo - clo
    span_len = hi - lo + 1
    sc_w = sc[off : off + span_len]
    cov_w = cov[off : off + span_len]
    prom_w = prom[off : off + span_len]

    valid = (cov_w >= 0.78) & (prom_w >= 6.0) & (sc_w >= prop_threshold)
    if not np.any(valid):
        return None
    best_idx = int(np.argmax(np.where(valid, sc_w, -1.0)))
    return lo + best_idx, float(sc_w[best_idx]), float(cov_w[best_idx])


def _refine_and_trim(
    img: np.ndarray,
    raw: _RawRegion,
    bleed_trim: int,
    bg_rgb: np.ndarray,
    bg_uniform: bool,
    is_trans: bool,
    bg_noise: float = 0.0,
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    对初步切分出的区域执行按边吸附对齐、向量化背景剔除与纹理感知去渗色收边。

    算法分三个连续阶段处理：
    1. 按边独立的 ±8px 覆盖率门控 + 距离先验强梯度断裂面吸附（仅作用于非沟槽切割边）；
    2. 向量化四周平坦单色/透明背景边缘批量剔除（告别逐行 Python while 循环）；
    3. 纹理感知型切口内缘抗混叠/1px 灰线渗色（Bleed）逐层去污染收边。

    :param img: 原始图像数组，要求为三维 uint8 数组（H, W, C）。
    :type img: numpy.ndarray
    :param raw: 初步分割产出的原始矩形区域对象。
    :type raw: _RawRegion
    :param bleed_trim: 渗色收边剥离像素层数上限（通常为 0~8）。
    :type bleed_trim: int
    :param bg_rgb: 预估的主背景 RGB 色值数组，shape=(3,)。
    :type bg_rgb: numpy.ndarray
    :param bg_uniform: 背景是否平坦均匀。
    :type bg_uniform: bool
    :param is_trans: 是否为透明通道主导背景。
    :type is_trans: bool
    :param bg_noise: 背景单通道噪声估计值。
    :type bg_noise: float
    :return: 八元组 ``(x0, y0, x1, y1, trim_top, trim_right, trim_bottom, trim_left)``。
    :author: tongliga
    :date: 2026-11-18
    """
    h, w, ch = img.shape
    x0, y0, x1, y1 = raw.x0, raw.y0, raw.x1, raw.y1

    # 第一阶段：按边独立的 ±8px 覆盖率门控 + 距离先验断裂面吸附
    if not raw.gutter_top and y0 > 2 and y1 - y0 > 20 and x1 - x0 > 10:
        ya = max(1, y0 - 8)
        yb = min(y1 - 20, y0 + 8)
        if yb >= ya:
            s8, _ = _sample_strip(img, x0 + 4, ya - 1, x1 - 4, yb + 2, 120, yb - ya + 3)
            if s8.shape[0] >= 3 and s8.shape[1] >= 1:
                c_m = s8[1:-1]
                d_step = _color_diff_u8(c_m, s8[2:])
                d_lap = _color_diff_u8(c_m, cv2.addWeighted(s8[:-2], 0.5, s8[2:], 0.5, 0.0))
                comb = cv2.add(d_step, d_lap)
                e = 3.0 * (_reduce_mean_axis1(d_step) + _reduce_mean_axis1(d_lap))
                _, act = cv2.threshold(comb, 5, 1, cv2.THRESH_BINARY)
                cov = _reduce_mean_axis1(act)
                offsets = np.abs((ya + np.arange(len(e)) + 1) - y0).astype(np.float32)
                e_w = np.where(cov >= 0.55, e * (1.0 - 0.03 * offsets), -1.0)
                idx = int(np.argmax(e_w))
                if float(e_w[idx]) > 15.0:
                    y0 = ya + idx + 1

    if not raw.gutter_bottom and y1 < h - 2 and y1 - y0 > 20 and x1 - x0 > 10:
        ya = max(y0 + 20, y1 - 8)
        yb = min(h - 2, y1 + 8)
        if yb >= ya:
            s8, _ = _sample_strip(img, x0 + 4, ya - 1, x1 - 4, yb + 2, 120, yb - ya + 3)
            if s8.shape[0] >= 3 and s8.shape[1] >= 1:
                c_m = s8[1:-1]
                d_step = _color_diff_u8(s8[:-2], c_m)
                d_lap = _color_diff_u8(c_m, cv2.addWeighted(s8[:-2], 0.5, s8[2:], 0.5, 0.0))
                comb = cv2.add(d_step, d_lap)
                e = 3.0 * (_reduce_mean_axis1(d_step) + _reduce_mean_axis1(d_lap))
                _, act = cv2.threshold(comb, 5, 1, cv2.THRESH_BINARY)
                cov = _reduce_mean_axis1(act)
                offsets = np.abs((ya + np.arange(len(e))) - y1).astype(np.float32)
                e_w = np.where(cov >= 0.55, e * (1.0 - 0.03 * offsets), -1.0)
                idx = int(np.argmax(e_w))
                if float(e_w[idx]) > 15.0:
                    y1 = ya + idx

    if not raw.gutter_left and x0 > 2 and x1 - x0 > 20 and y1 - y0 > 10:
        xa = max(1, x0 - 8)
        xb = min(x1 - 20, x0 + 8)
        if xb >= xa:
            s8, _ = _sample_strip(img, xa - 1, y0 + 4, xb + 2, y1 - 4, xb - xa + 3, 120)
            if s8.shape[1] >= 3 and s8.shape[0] >= 1:
                c_m = s8[:, 1:-1]
                d_step = _color_diff_u8(c_m, s8[:, 2:])
                d_lap = _color_diff_u8(c_m, cv2.addWeighted(s8[:, :-2], 0.5, s8[:, 2:], 0.5, 0.0))
                comb = cv2.add(d_step, d_lap)
                e = 3.0 * (_reduce_mean_axis0(d_step) + _reduce_mean_axis0(d_lap))
                _, act = cv2.threshold(comb, 5, 1, cv2.THRESH_BINARY)
                cov = _reduce_mean_axis0(act)
                offsets = np.abs((xa + np.arange(len(e)) + 1) - x0).astype(np.float32)
                e_w = np.where(cov >= 0.55, e * (1.0 - 0.03 * offsets), -1.0)
                idx = int(np.argmax(e_w))
                if float(e_w[idx]) > 15.0:
                    x0 = xa + idx + 1

    if not raw.gutter_right and x1 < w - 2 and x1 - x0 > 20 and y1 - y0 > 10:
        xa = max(x0 + 20, x1 - 8)
        xb = min(w - 2, x1 + 8)
        if xb >= xa:
            s8, _ = _sample_strip(img, xa - 1, y0 + 4, xb + 2, y1 - 4, xb - xa + 3, 120)
            if s8.shape[1] >= 3 and s8.shape[0] >= 1:
                c_m = s8[:, 1:-1]
                d_step = _color_diff_u8(s8[:, :-2], c_m)
                d_lap = _color_diff_u8(c_m, cv2.addWeighted(s8[:, :-2], 0.5, s8[:, 2:], 0.5, 0.0))
                comb = cv2.add(d_step, d_lap)
                e = 3.0 * (_reduce_mean_axis0(d_step) + _reduce_mean_axis0(d_lap))
                _, act = cv2.threshold(comb, 5, 1, cv2.THRESH_BINARY)
                cov = _reduce_mean_axis0(act)
                offsets = np.abs((xa + np.arange(len(e))) - x1).astype(np.float32)
                e_w = np.where(cov >= 0.55, e * (1.0 - 0.03 * offsets), -1.0)
                idx = int(np.argmax(e_w))
                if float(e_w[idx]) > 15.0:
                    x1 = xa + idx

    # 第二阶段：仅当区块边界来自背景沟槽（raw.is_gutter）时剥离四周空白边缘，保留无缝白底幻灯片完整内边距
    tt = tr = tb = tl = 0
    if bg_uniform and raw.is_gutter and x1 - x0 > 24 and y1 - y0 > 24:
        bg_scalar = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 0)
        bg_tol = int(min(18, max(7, round(7.0 + bg_noise * 2.5))))
        need_alpha = is_trans and ch >= 4

        # 纵向批量扫描行背景占比
        r_rgb, r_alpha = _sample_strip(img, x0, y0, x1, y1, 64, y1 - y0, need_alpha=need_alpha)
        if r_alpha is not None:
            _, bg_bin = cv2.threshold(r_alpha, 23, 1, cv2.THRESH_BINARY_INV)
        else:
            _, bg_bin = cv2.threshold(_color_diff_u8(r_rgb, bg_scalar), bg_tol, 1, cv2.THRESH_BINARY_INV)
        row_bg_frac = _reduce_mean_axis1(bg_bin)
        fg_rows = np.flatnonzero(row_bg_frac <= 0.94)
        if fg_rows.size > 0:
            new_y0 = min(y1 - 24, y0 + int(fg_rows[0]))
            new_y1 = max(new_y0 + 24, y0 + int(fg_rows[-1]) + 1)
            tt += max(0, new_y0 - y0)
            tb += max(0, y1 - new_y1)
            y0, y1 = new_y0, new_y1

        # 横向批量扫描列背景占比
        c_rgb, c_alpha = _sample_strip(img, x0, y0, x1, y1, x1 - x0, 64, need_alpha=need_alpha)
        if c_alpha is not None:
            _, bg_bin = cv2.threshold(c_alpha, 23, 1, cv2.THRESH_BINARY_INV)
        else:
            _, bg_bin = cv2.threshold(_color_diff_u8(c_rgb, bg_scalar), bg_tol, 1, cv2.THRESH_BINARY_INV)
        col_bg_frac = _reduce_mean_axis0(bg_bin)
        fg_cols = np.flatnonzero(col_bg_frac <= 0.94)
        if fg_cols.size > 0:
            new_x0 = min(x1 - 24, x0 + int(fg_cols[0]))
            new_x1 = max(new_x0 + 24, x0 + int(fg_cols[-1]) + 1)
            tl += max(0, new_x0 - x0)
            tr += max(0, x1 - new_x1)
            x0, x1 = new_x0, new_x1

    # 第三阶段：向量化纹理感知型切口内缘抗混叠/1px 灰线渗色（Bleed）去污染收缩
    if bleed_trim > 0 and y1 - y0 > 32 and x1 - x0 > 32:
        band = bleed_trim + 3

        if raw.y0 > 2 and y0 + band <= y1:
            strip, _ = _sample_strip(img, x0 + 8, y0, x1 - 8, y0 + band, 80, band)
            if strip.shape[0] >= band and strip.shape[1] >= 1:
                ref_row = strip[-1:]
                interior_d = _color_diff_u8(strip[-2:-1], ref_row)
                thresh = max(20.0, float(interior_d.mean()) * 3.0 * 1.45)
                cand = strip[:bleed_trim]
                d_mat = _color_diff_u8(cand, np.broadcast_to(ref_row, cand.shape))
                diffs = 3.0 * _reduce_mean_axis1(d_mat)
                _, act = cv2.threshold(d_mat, 4, 1, cv2.THRESH_BINARY)
                covs = _reduce_mean_axis1(act)
                for k in range(bleed_trim):
                    if float(diffs[k]) > thresh and float(covs[k]) >= 0.45:
                        y0 += 1
                        tt += 1
                    else:
                        break

        if raw.y1 < h - 2 and y1 - band >= y0:
            strip, _ = _sample_strip(img, x0 + 8, y1 - band, x1 - 8, y1, 80, band)
            if strip.shape[0] >= band and strip.shape[1] >= 1:
                ref_row = strip[:1]
                interior_d = _color_diff_u8(strip[1:2], ref_row)
                thresh = max(20.0, float(interior_d.mean()) * 3.0 * 1.45)
                cand = strip[-bleed_trim:]
                d_mat = _color_diff_u8(cand, np.broadcast_to(ref_row, cand.shape))
                diffs = 3.0 * _reduce_mean_axis1(d_mat)
                _, act = cv2.threshold(d_mat, 4, 1, cv2.THRESH_BINARY)
                covs = _reduce_mean_axis1(act)
                for k in range(bleed_trim - 1, -1, -1):
                    if float(diffs[k]) > thresh and float(covs[k]) >= 0.45:
                        y1 -= 1
                        tb += 1
                    else:
                        break

        if raw.x0 > 2 and x0 + band <= x1:
            strip, _ = _sample_strip(img, x0, y0 + 8, x0 + band, y1 - 8, band, 80)
            if strip.shape[1] >= band and strip.shape[0] >= 1:
                ref_col = strip[:, -1:]
                interior_d = _color_diff_u8(strip[:, -2:-1], ref_col)
                thresh = max(20.0, float(interior_d.mean()) * 3.0 * 1.45)
                cand = strip[:, :bleed_trim]
                d_mat = _color_diff_u8(cand, np.broadcast_to(ref_col, cand.shape))
                diffs = 3.0 * _reduce_mean_axis0(d_mat)
                _, act = cv2.threshold(d_mat, 4, 1, cv2.THRESH_BINARY)
                covs = _reduce_mean_axis0(act)
                for k in range(bleed_trim):
                    if float(diffs[k]) > thresh and float(covs[k]) >= 0.45:
                        x0 += 1
                        tl += 1
                    else:
                        break

        if raw.x1 < w - 2 and x1 - band >= x0:
            strip, _ = _sample_strip(img, x1 - band, y0 + 8, x1, y1 - 8, band, 80)
            if strip.shape[1] >= band and strip.shape[0] >= 1:
                ref_col = strip[:, :1]
                interior_d = _color_diff_u8(strip[:, 1:2], ref_col)
                thresh = max(20.0, float(interior_d.mean()) * 3.0 * 1.45)
                cand = strip[:, -bleed_trim:]
                d_mat = _color_diff_u8(cand, np.broadcast_to(ref_col, cand.shape))
                diffs = 3.0 * _reduce_mean_axis0(d_mat)
                _, act = cv2.threshold(d_mat, 4, 1, cv2.THRESH_BINARY)
                covs = _reduce_mean_axis0(act)
                for k in range(bleed_trim - 1, -1, -1):
                    if float(diffs[k]) > thresh and float(covs[k]) >= 0.45:
                        x1 -= 1
                        tr += 1
                    else:
                        break

    return x0, y0, x1, y1, tt, tr, tb, tl


def _group_reading_order(
    items: List[Tuple[int, int, int, int, object]],
) -> List[List[Tuple[int, int, int, int, object]]]:
    """
    基于纵向重叠率与顶部对齐的稳定视觉行聚类与阅读顺序排序。

    相比仅按首元素 y0 比较，引入纵向重叠度（overlap_y / min_h >= 0.45）与动态行包络更新，
    保证高低错落卡片、图标组与 Bento 拼图始终维持稳定的从上至下、从左至右编号顺序。

    :param items: 元素列表，每项为 ``(x0, y0, x1, y1, payload)``。
    :type items: List[Tuple[int, int, int, int, object]]
    :return: 按视觉行分组且每行内按 x0 升序排列的二维列表。
    :author: tongliga
    :date: 2026-11-18
    """
    if not items:
        return []
    y_sorted = sorted(items, key=lambda it: (it[1], it[0]))
    groups: List[List[Tuple[int, int, int, int, object]]] = []
    # 记录每个行组的参考纵向区间 (min_y0, avg_y0, avg_y1)
    group_spans: List[Tuple[int, float, float]] = []

    for item in y_sorted:
        _, iy0, _, iy1, _ = item
        ih = max(1, iy1 - iy0)
        best_grp_idx = -1
        best_score = -1.0
        for g_idx, (_, gy0, gy1) in enumerate(group_spans):
            gh = max(1.0, gy1 - gy0)
            min_h = min(float(ih), gh)
            overlap = min(float(iy1), gy1) - max(float(iy0), gy0)
            overlap_ratio = overlap / min_h
            top_aligned = abs(float(iy0) - gy0) <= min_h * 0.35
            if overlap_ratio >= 0.45 or top_aligned:
                if overlap_ratio > best_score:
                    best_score = overlap_ratio
                    best_grp_idx = g_idx

        if best_grp_idx >= 0:
            grp = groups[best_grp_idx]
            grp.append(item)
            n = float(len(grp))
            min_y = min(group_spans[best_grp_idx][0], iy0)
            avg_y0 = sum(x[1] for x in grp) / n
            avg_y1 = sum(x[3] for x in grp) / n
            group_spans[best_grp_idx] = (min_y, avg_y0, avg_y1)
        else:
            groups.append([item])
            group_spans.append((iy0, float(iy0), float(iy1)))

    # 行组按顶部最小纵坐标排序，行内按左边界 x0 排序
    order = sorted(range(len(groups)), key=lambda idx: group_spans[idx][0])
    ordered_groups: List[List[Tuple[int, int, int, int, object]]] = []
    for idx in order:
        grp = groups[idx]
        grp.sort(key=lambda it: it[0])
        ordered_groups.append(grp)
    return ordered_groups


def _flatten_same_axis(node: _BspNode) -> None:
    """
    将 BSP 树中父子连续同轴的多叉拆分节点就地展平，消除邻居传播产生的同轴二级嵌套。

    :param node: 待处理的 BSP 树节点。
    :type node: _BspNode
    :author: tongliga
    :date: 2026-11-25
    """
    if not node.children:
        return
    for child in node.children:
        _flatten_same_axis(child)
    new_children: List[_BspNode] = []
    new_cuts: List[Tuple[int, float, float, bool]] = []
    for idx, ch in enumerate(node.children):
        if ch.axis == node.axis and ch.children:
            new_children.extend(ch.children)
            new_cuts.extend(ch.cuts)
        else:
            new_children.append(ch)
        if idx < len(node.cuts):
            new_cuts.append(node.cuts[idx])
    node.children = new_children
    node.cuts = new_cuts


def _dominant_tile_dims(peers: List[_BspNode]) -> Tuple[float, float, float, float]:
    """
    使用面积加权众数聚类计算同类切片的共识主导尺寸及集群一致性面积占比。

    面积加权保证即使个别内部幻灯片碎裂出多块小区域，其在全画面积中的权重占比依然很小，
    主干规律宫格/卡片尺寸始终能够稳定胜出。

    :param peers: 用于估算参考尺寸的同伴叶子节点列表。
    :type peers: List[_BspNode]
    :return: 四元组 ``(w_dom, h_dom, w_area_fraction, h_area_fraction)``。
    :author: tongliga
    :date: 2026-11-25
    """

    def _cluster_dom(vals: List[float], weights: List[float]) -> Tuple[float, float]:
        total_w = max(1e-6, sum(weights))
        best_score = -1.0
        best_val = vals[0]
        for v in vals:
            tol = max(14.0, v * 0.16)
            in_cl = [i for i, x in enumerate(vals) if abs(x - v) <= tol]
            w_sum = sum(weights[i] for i in in_cl)
            if w_sum > best_score:
                best_score = w_sum
                best_val = sum(vals[i] * weights[i] for i in in_cl) / max(1e-6, w_sum)
        return best_val, best_score / total_w

    ws = [float(p.x1 - p.x0) for p in peers]
    hs = [float(p.y1 - p.y0) for p in peers]
    areas = [w * h for w, h in zip(ws, hs)]
    w_dom, w_frac = _cluster_dom(ws, areas)
    h_dom, h_frac = _cluster_dom(hs, areas)
    return w_dom, h_dom, w_frac, h_frac


def _subtree_has_gutter(node: _BspNode) -> bool:
    """
    递归检查当前 BSP 子树内部是否包含任意纯净背景沟槽切分线。

    :param node: 待检查的 BSP 子树根节点。
    :type node: _BspNode
    :return: 若子树中任意层级包含背景沟槽割线则返回 True。
    :author: tongliga
    :date: 2026-11-25
    """
    if any(cut[3] for cut in node.cuts):
        return True
    return any(_subtree_has_gutter(ch) for ch in node.children)


def _regularize_bsp_tree(root: _BspNode) -> None:
    """
    执行自底向上同轴子树折叠与相邻兄弟异常切分合并，保障整版切片尺寸及长宽比协调一致。

    针对幻灯片套图、拼图网格内部常见的“分屏人物写真”、“深色表头横幅”或“三栏小卡片”等高对比度内部构图：
    1. 严格豁免所有背景沟槽切割与在邻近同伴间存在贯穿对齐支撑的导轨割线；
    2. 结合面积加权同伴尺寸共识 `(W_dom, H_dom)` 与对数长宽比例一致性，仅剪除导致局部尺寸显著萎缩（`< 0.72 * dom`）、
       且合并后可精确回归版面主导尺度的非沟槽内部误切（既包括整棵多级误切子树，也包括条带内的局部相邻兄弟分屏）。

    :param root: 经过初步构建与兄弟传播后的 BSP 空间树根节点。
    :type root: _BspNode
    :author: tongliga
    :date: 2026-11-25
    """
    changed = True
    while changed:
        changed = False
        _flatten_same_axis(root)
        all_leaves = root.collect_leaves()
        if len(all_leaves) < 5:
            break

        # 阶段 A：同轴非沟槽误切子树整体折叠（消除嵌套多级表格/图文区块误切）
        best_sub: Optional[_BspNode] = None
        best_sub_gain = 0.0

        def _walk_sub(node: _BspNode) -> None:
            nonlocal best_sub, best_sub_gain
            if not node.children:
                return
            for child in node.children:
                _walk_sub(child)
            has_outer_gutter = node.gutter_top or node.gutter_right or node.gutter_bottom or node.gutter_left
            if (_subtree_has_gutter(node) and (node is root or has_outer_gutter)) or node.axis is None:
                return

            desc_ids = {id(lf) for lf in node.collect_leaves()}
            peers = [lf for lf in all_leaves if id(lf) not in desc_ids]
            if len(peers) < 3:
                return

            w_dom, h_dom, w_frac, h_frac = _dominant_tile_dims(peers)
            dom_frac = h_frac if node.axis == "h" else w_frac
            if dom_frac < 0.50:
                return
            dom = h_dom if node.axis == "h" else w_dom
            n_dim = float((node.y1 - node.y0) if node.axis == "h" else (node.x1 - node.x0))
            n_err = abs(math.log(max(1.0, n_dim) / max(1.0, dom)))
            if n_err > 0.25:
                return

            # 检查该子树切线在常态维度同伴间是否存在跨区块对齐支撑
            align_tol = 8
            for cpos, _, _, _ in node.cuts:
                if node.axis == "v":
                    if any(
                        (p.y1 - p.y0) >= h_dom * 0.72 and (abs(p.x0 - cpos) <= align_tol or abs(p.x1 - cpos) <= align_tol)
                        for p in peers
                    ):
                        return
                else:
                    if any(
                        (p.x1 - p.x0) >= w_dom * 0.72 and (abs(p.y0 - cpos) <= align_tol or abs(p.y1 - cpos) <= align_tol)
                        for p in peers
                    ):
                        return

            sub_leaves = node.collect_leaves()
            for sl in sub_leaves:
                sw = sl.x1 - sl.x0
                sh = sl.y1 - sl.y0
                peer_matches = sum(
                    1
                    for p in peers
                    if abs((p.x1 - p.x0) - sw) <= max(14.0, 0.16 * sw)
                    and abs((p.y1 - p.y0) - sh) <= max(14.0, 0.16 * sh)
                )
                if peer_matches >= 2:
                    return

            leaf_dims = [float((lf.y1 - lf.y0) if node.axis == "h" else (lf.x1 - lf.x0)) for lf in sub_leaves]
            leaf_errs = [abs(math.log(max(1.0, d) / max(1.0, dom))) for d in leaf_dims]
            mean_leaf_err = sum(leaf_errs) / float(len(leaf_errs))
            if min(leaf_dims) / max(1.0, dom) < 0.72 and mean_leaf_err > 0.35 and n_err < mean_leaf_err * 0.45:
                gain = mean_leaf_err - n_err
                if gain > best_sub_gain:
                    best_sub_gain = gain
                    best_sub = node

        _walk_sub(root)
        if best_sub is not None:
            best_sub.children = []
            best_sub.cuts = []
            best_sub.axis = None
            changed = True
            continue

        # 阶段 B：连续相邻非沟槽叶子兄弟异常分块合并（支持 2~5 块连续局部条带如分屏/三栏内嵌图回归主导尺寸）
        best_span: Optional[Tuple[_BspNode, int, int]] = None
        best_span_gain = 0.0

        def _walk_span(node: _BspNode) -> None:
            nonlocal best_span, best_span_gain
            if not node.children:
                return
            for child in node.children:
                _walk_span(child)
            n_ch = len(node.children)
            if n_ch < 2 or node.axis is None:
                return

            is_v = node.axis == "v"
            max_span = min(5, n_ch)
            for span_len in range(2, max_span + 1):
                for start in range(0, n_ch - span_len + 1):
                    end = start + span_len
                    group = node.children[start:end]
                    if any(ch.children for ch in group):
                        continue
                    internal_cuts = node.cuts[start : end - 1]
                    has_outer_gutter = node.gutter_top or node.gutter_right or node.gutter_bottom or node.gutter_left
                    if any(cut[3] for cut in internal_cuts) and (node is root or has_outer_gutter):
                        continue

                    group_ids = {id(ch) for ch in group}
                    peers = [lf for lf in all_leaves if id(lf) not in group_ids]
                    if len(peers) < 3:
                        continue

                    w_dom, h_dom, w_frac, h_frac = _dominant_tile_dims(peers)
                    dom_frac = w_frac if is_v else h_frac
                    if dom_frac < 0.50:
                        continue

                    align_tol = 8
                    aligned_ext = False
                    for cut_pos, _, _, _ in internal_cuts:
                        if is_v:
                            if any(
                                (p.y1 - p.y0) >= h_dom * 0.72
                                and (abs(p.x0 - cut_pos) <= align_tol or abs(p.x1 - cut_pos) <= align_tol)
                                for p in peers
                            ):
                                aligned_ext = True
                                break
                        else:
                            if any(
                                (p.x1 - p.x0) >= w_dom * 0.72
                                and (abs(p.y0 - cut_pos) <= align_tol or abs(p.y1 - cut_pos) <= align_tol)
                                for p in peers
                            ):
                                aligned_ext = True
                                break
                    if aligned_ext:
                        continue

                    if any(
                        sum(
                            1
                            for p in peers
                            if abs((p.x1 - p.x0) - (ch.x1 - ch.x0)) <= max(14.0, 0.16 * (ch.x1 - ch.x0))
                            and abs((p.y1 - p.y0) - (ch.y1 - ch.y0)) <= max(14.0, 0.16 * (ch.y1 - ch.y0))
                        )
                        >= 2
                        for ch in group
                    ):
                        continue

                    dom = w_dom if is_v else h_dom
                    dims = [float((ch.x1 - ch.x0) if is_v else (ch.y1 - ch.y0)) for ch in group]
                    dp = sum(dims)
                    err_split = sum(abs(math.log(max(1.0, d) / max(1.0, dom))) for d in dims) / float(span_len)
                    err_merged = abs(math.log(max(1.0, dp) / max(1.0, dom)))
                    if min(dims) / max(1.0, dom) < 0.72 and err_merged <= 0.25 and err_merged < err_split * 0.48:
                        gain = err_split - err_merged
                        if gain > best_span_gain:
                            best_span_gain = gain
                            best_span = (node, start, end)

        _walk_span(root)
        if best_span is not None:
            pnode, start, end = best_span
            group = pnode.children[start:end]
            first, last = group[0], group[-1]
            max_conf = max(ch.conf for ch in group)
            if pnode.axis == "v":
                merged = _BspNode(
                    first.x0,
                    first.y0,
                    last.x1,
                    first.y1,
                    max_conf,
                    all(ch.gutter_top for ch in group),
                    last.gutter_right,
                    all(ch.gutter_bottom for ch in group),
                    first.gutter_left,
                )
            else:
                merged = _BspNode(
                    first.x0,
                    first.y0,
                    first.x1,
                    last.y1,
                    max_conf,
                    first.gutter_top,
                    all(ch.gutter_right for ch in group),
                    last.gutter_bottom,
                    all(ch.gutter_left for ch in group),
                )
            pnode.children[start:end] = [merged]
            del pnode.cuts[start : end - 1]
            if len(pnode.children) == 1:
                only = pnode.children[0]
                pnode.children = []
                pnode.cuts = []
                pnode.axis = None
                pnode.gutter_top = only.gutter_top
                pnode.gutter_right = only.gutter_right
                pnode.gutter_bottom = only.gutter_bottom
                pnode.gutter_left = only.gutter_left
            changed = True


def xy_cut_core(
    img: np.ndarray,
    sensitivity: float,
    min_region_ratio: float,
    bleed_trim: int,
    bg_rgb: np.ndarray,
    bg_uniform: bool,
    is_trans: bool,
    collect_curves: bool = False,
    bg_noise: float = 0.0,
) -> Tuple[List[RawSplitBox], Optional[Dict[str, List[float]]]]:
    """
    基于递归 XY-Cut（LUT 突出度投影）、全局导轨共识、兄弟对齐传播与 BSP 几何正则化的核心裁切算子。

    对输入图像执行四级流水线解构：
    1. 递归构建 BSP 空间划分树（采用韦伯对数 LUT + 1D 局部突出度识别全贯通弱边界与空白沟槽）；
    2. 完整多轮同级兄弟割线对齐传播（±12px 突出度召回遗漏网格边并挂载回 BSP 树）；
    3. 自底向上 BSP 几何尺度/长宽比正则化剪枝（结合全盘主导区块大小自动收敛卡片内部高反差误切）；
    4. 空背景滤除、视觉阅读顺序重排与纹理感知渗色收边净化。

    :param img: 待切分的原始图像数组，要求为三维 uint8 数组（H, W, C）。
    :type img: numpy.ndarray
    :param sensitivity: 切割灵敏度，取值范围 [0.1, 1.0]，值越高割线越容易被触发。
    :type sensitivity: float
    :param min_region_ratio: 单个切片尺寸占图像最小维度的比例下限，取值范围 [0.04, 0.35]。
    :type min_region_ratio: float
    :param bleed_trim: 渗色边缘收边剥离像素层数，取值范围通常为 0~8。
    :type bleed_trim: int
    :param bg_rgb: 背景色估计值 RGB 数组，shape=(3,)。
    :type bg_rgb: numpy.ndarray
    :param bg_uniform: 背景是否平坦单色。
    :type bg_uniform: bool
    :param is_trans: 背景是否为透明。
    :type is_trans: bool
    :param collect_curves: 是否采集并返回顶层 X/Y 轴投影能量曲线供前端诊断展示，默认为 False。
    :type collect_curves: bool
    :param bg_noise: 背景单通道噪声估计值，默认为 0.0。
    :type bg_noise: float
    :return: 包含切分框列表及可选诊断投影曲线字典的二元组：(boxes, curves)。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：根据最小尺寸比例与灵敏度推导最小块尺寸与 LUT 突出度阈值
    h, w, _ = img.shape
    min_dim = max(24, int(round(min(w, h) * _clamp(min_region_ratio, 0.04, 0.35))))
    threshold = 22.0 - _clamp(sensitivity, 0.1, 1.0) * 14.0
    prop_threshold = max(7.0, threshold * 0.65)

    # 第二步：构建递归 XY-Cut BSP 空间树
    top_curves: Optional[Dict[str, np.ndarray]] = {} if collect_curves else None
    root = _build_bsp_tree(
        img,
        0,
        0,
        w,
        h,
        0,
        min_dim,
        threshold,
        bg_rgb,
        bg_uniform,
        is_trans,
        bg_noise,
        36.0,
        False,
        False,
        False,
        False,
        [0],
        top_curves=top_curves,
    )

    # 第三步：完整多轮网格邻居割线对齐传播（最多 6 轮、上限 256 块）
    prop_iter = 0
    propagated = True
    while propagated and prop_iter < 6:
        leaves_cur = root.collect_leaves()
        if len(leaves_cur) >= 256:
            break
        propagated = False
        prop_iter += 1
        h_lines = sorted({y for lf in leaves_cur for y in (lf.y0, lf.y1) if min_dim < y < h - min_dim})
        v_lines = sorted({x for lf in leaves_cur for x in (lf.x0, lf.x1) if min_dim < x < w - min_dim})

        for leaf in leaves_cur:
            split_done = False
            for y_cut in h_lines:
                if leaf.y0 + min_dim <= y_cut <= leaf.y1 - min_dim:
                    res = _sibling_propagate_h(img, leaf, y_cut, min_dim, prop_threshold=prop_threshold)
                    if res is not None:
                        best_y, best_sc, best_cov = res
                        leaf.axis = "h"
                        leaf.cuts = [(best_y, best_sc, best_cov, False)]
                        leaf.children = [
                            _BspNode(
                                leaf.x0,
                                leaf.y0,
                                leaf.x1,
                                best_y,
                                max(leaf.conf, best_sc),
                                leaf.gutter_top,
                                leaf.gutter_right,
                                False,
                                leaf.gutter_left,
                            ),
                            _BspNode(
                                leaf.x0,
                                best_y,
                                leaf.x1,
                                leaf.y1,
                                max(leaf.conf, best_sc),
                                False,
                                leaf.gutter_right,
                                leaf.gutter_bottom,
                                leaf.gutter_left,
                            ),
                        ]
                        propagated = True
                        split_done = True
                        break
            if split_done:
                continue

            for x_cut in v_lines:
                if leaf.x0 + min_dim <= x_cut <= leaf.x1 - min_dim:
                    res = _sibling_propagate_v(img, leaf, x_cut, min_dim, prop_threshold=prop_threshold)
                    if res is not None:
                        best_x, best_sc, best_cov = res
                        leaf.axis = "v"
                        leaf.cuts = [(best_x, best_sc, best_cov, False)]
                        leaf.children = [
                            _BspNode(
                                leaf.x0,
                                leaf.y0,
                                best_x,
                                leaf.y1,
                                max(leaf.conf, best_sc),
                                leaf.gutter_top,
                                False,
                                leaf.gutter_bottom,
                                leaf.gutter_left,
                            ),
                            _BspNode(
                                best_x,
                                leaf.y0,
                                leaf.x1,
                                leaf.y1,
                                max(leaf.conf, best_sc),
                                leaf.gutter_top,
                                leaf.gutter_right,
                                leaf.gutter_bottom,
                                False,
                            ),
                        ]
                        propagated = True
                        break

    # 第四步：自底向上 BSP 几何尺寸与长宽比正则化，抑制卡片内部异常过切
    _regularize_bsp_tree(root)
    leaves = [node.to_raw_region() for node in root.collect_leaves()]

    # 第五步：若背景平坦则过滤掉由纯背景色构成的空白区域
    if bg_uniform:
        non_empty = [
            lf for lf in leaves if not _is_region_empty_bg(img, lf.x0, lf.y0, lf.x1, lf.y1, bg_rgb, is_trans, bg_noise)
        ]
        final_leaves = non_empty if non_empty else leaves
    else:
        final_leaves = list(leaves)

    # 第六步：采用视觉重叠行聚类按阅读顺序重排
    grouped = _group_reading_order([(lf.x0, lf.y0, lf.x1, lf.y1, lf) for lf in final_leaves])

    # 第七步：对每个区域执行精细化对齐收边，构建最终 RawSplitBox 数据包
    out: List[RawSplitBox] = []
    idx_counter = 1
    for cur_r, grp in enumerate(grouped):
        for cur_c, (_, _, _, _, payload) in enumerate(grp):
            leaf = payload
            assert isinstance(leaf, _RawRegion)
            rx0, ry0, rx1, ry1, tt, tr, tb, tl = _refine_and_trim(
                img, leaf, bleed_trim, bg_rgb, bg_uniform, is_trans, bg_noise=bg_noise
            )
            out.append(
                RawSplitBox(
                    index=idx_counter,
                    row=cur_r,
                    col=cur_c,
                    x=rx0,
                    y=ry0,
                    width=max(1, rx1 - rx0),
                    height=max(1, ry1 - ry0),
                    confidence=min(1.0, leaf.conf / 36.0),
                    strategy="gutter" if leaf.is_gutter else "xycut",
                    trim_top=tt,
                    trim_right=tr,
                    trim_bottom=tb,
                    trim_left=tl,
                )
            )
            idx_counter += 1

    # 第八步：若开启诊断曲线采集，提取下采样投影能量曲线供前端可视化
    curves_out: Optional[Dict[str, List[float]]] = None
    if collect_curves and top_curves is not None:
        curves_out = {}
        for axis_key in ("h", "v"):
            arr = top_curves.get(axis_key)
            if arr is not None and arr.size > 0:
                step = max(1, arr.size // 240)
                curves_out[axis_key] = [round(float(v), 2) for v in arr[::step]]
    return out, curves_out


def _suppress_contained_boxes(
    boxes: List[Tuple[int, int, int, int]],
    containment_thresh: float = 0.85,
) -> List[Tuple[int, int, int, int]]:
    """
    抑制被更大外接框几乎完全包含的内部重复嵌套框（例如镂空卡片/边框内部的孤立子元素）。

    :param boxes: 原始候选边界框列表 ``(x0, y0, x1, y1)``。
    :type boxes: List[Tuple[int, int, int, int]]
    :param containment_thresh: 判定嵌套包含的交集面积占比阈值，默认 0.85。
    :type containment_thresh: float
    :return: 去重后的顶层独立边界框列表。
    :author: tongliga
    :date: 2026-11-18
    """
    if len(boxes) <= 1:
        return boxes
    sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for cand in sorted_boxes:
        cx0, cy0, cx1, cy1 = cand
        c_area = max(1, (cx1 - cx0) * (cy1 - cy0))
        is_contained = False
        for kx0, ky0, kx1, ky1 in kept:
            ix0 = max(cx0, kx0)
            iy0 = max(cy0, ky0)
            ix1 = min(cx1, kx1)
            iy1 = min(cy1, ky1)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                if float(inter) / float(c_area) >= containment_thresh:
                    is_contained = True
                    break
        if not is_contained:
            kept.append(cand)
    return kept


def contours_core(
    img: np.ndarray,
    sensitivity: float,
    min_region_ratio: float,
    contour_gap_px: int,
    bg_rgb: np.ndarray,
    is_trans: bool,
    bg_noise: float = 0.0,
) -> List[RawSplitBox]:
    """
    基于 OpenCV 噪声自适应形态学闭合、8-连通域分析、防嵌套去重与向量化 1:1 边缘收紧的轮廓算子。

    关键稳定性增强：
    1. 精确间隙半径（radius = contour_gap_px / (2 * scale)），防止 2× 过度膨胀导致邻近物体误粘连；
    2. 背景噪声自适应阈值与孤立 1px 噪点剔除（保留细线条前景，消除 JPEG 背景尘点桥接）；
    3. 嵌套包含框抑制（_suppress_contained_boxes），避免镂空边框重复切出内部子块；
    4. 1:1 全分辨率向量化边界收紧（消除逐行 Python while 循环）。

    :param img: 原始图像数组，要求为三维 uint8 数组（H, W, C）。
    :type img: numpy.ndarray
    :param sensitivity: 颜色敏感度，取值范围 [0.1, 1.0]，值越高对微弱前景越敏感。
    :type sensitivity: float
    :param min_region_ratio: 过滤微小孤立噪点的有效前景面积比例下限，取值范围 [0.02, 0.30]。
    :type min_region_ratio: float
    :param contour_gap_px: 允许粘连合并的最大断裂间隙像素数。
    :type contour_gap_px: int
    :param bg_rgb: 背景色估计值 RGB 数组，shape=(3,)。
    :type bg_rgb: numpy.ndarray
    :param is_trans: 是否为透明背景模式。
    :type is_trans: bool
    :param bg_noise: 背景单通道噪声估计值，默认为 0.0。
    :type bg_noise: float
    :return: 提取出的独立前景目标边界框列表。
    :author: tongliga
    :date: 2026-11-18
    """
    h, w, ch = img.shape

    # 第一步：采用 OpenCV INTER_NEAREST 快速降采样至最长边 720px 以内加速形态学运算
    scale = max(1, (max(w, h) + 719) // 720)
    sw = max(1, (w + scale - 1) // scale)
    sh = max(1, (h + scale - 1) // scale)
    small_rgb, small_alpha = _sample_strip(img, 0, 0, w, h, sw, sh, need_alpha=(is_trans and ch >= 4))
    sh, sw = small_rgb.shape[:2]
    base_thresh_u8 = int(round((32.0 - _clamp(sensitivity, 0.1, 1.0) * 18.0) / 3.0))
    color_thresh_u8 = max(base_thresh_u8, int(round(bg_noise * 3.0 + 4.0)))

    # 第二步：依据透明通道或 SIMD 均衡背景色差提取前景二值掩模
    if small_alpha is not None:
        _, mask = cv2.threshold(small_alpha, 24, 1, cv2.THRESH_BINARY)
        del small_alpha
    else:
        bg_scalar = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 0)
        diff_u8 = _color_diff_u8(small_rgb, bg_scalar)
        _, mask = cv2.threshold(diff_u8, color_thresh_u8, 1, cv2.THRESH_BINARY)
        del diff_u8

        # 若背景存在噪声，仅移除 8 邻域孤立的 1px 离散尘点（保留所有 >=2px 连续细线条）
        if bg_noise >= 0.8:
            nbr_sum = cv2.boxFilter(mask, cv2.CV_8U, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
            mask = ((mask > 0) & (nbr_sum >= 2)).astype(np.uint8)
            del nbr_sum
    del small_rgb

    # 第三步：按精确间隙半径执行可分离形态学膨胀（两个物体各膨胀 gap/2 即可弥合 <= contour_gap_px 的间隙）
    gap_r = max(1, int(round(float(contour_gap_px) / (2.0 * float(scale)))))
    k_h = np.ones((1, 2 * gap_r + 1), dtype=np.uint8)
    k_v = np.ones((2 * gap_r + 1, 1), dtype=np.uint8)
    horiz = cv2.dilate(mask, k_h, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    dilated = cv2.dilate(horiz, k_v, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    del horiz

    # 第四步：计算 8-连通域标记矩阵
    num_labels, labels = cv2.connectedComponents(dilated, connectivity=8, ltype=cv2.CV_32S)
    del dilated
    if num_labels <= 1:
        return []

    # 第五步：利用 NumPy 向量化归约统计每个连通域的原始前景像素包围盒与面积
    fg_ys, fg_xs = np.nonzero(mask)
    if fg_ys.size == 0:
        return []
    fg_lbls = labels[fg_ys, fg_xs]
    valid = fg_lbls > 0
    if not np.any(valid):
        return []

    fg_lbls = fg_lbls[valid]
    fg_ys = fg_ys[valid]
    fg_xs = fg_xs[valid]

    fg_counts = np.bincount(fg_lbls, minlength=num_labels)
    min_area = max(16, int((sw * sh) * (_clamp(min_region_ratio, 0.02, 0.3) ** 2) * 0.45))

    min_x = np.full(num_labels, sw, dtype=np.int32)
    min_y = np.full(num_labels, sh, dtype=np.int32)
    max_x = np.full(num_labels, -1, dtype=np.int32)
    max_y = np.full(num_labels, -1, dtype=np.int32)

    np.minimum.at(min_x, fg_lbls, fg_xs)
    np.minimum.at(min_y, fg_lbls, fg_ys)
    np.maximum.at(max_x, fg_lbls, fg_xs)
    np.maximum.at(max_y, fg_lbls, fg_ys)

    active_labels = np.flatnonzero(fg_counts >= min_area)
    active_labels = active_labels[active_labels > 0]
    if active_labels.size == 0:
        return []

    bg_scalar = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 0)
    min_px_dim = max(8, int(min(w, h) * 0.01))
    rx_scale = float(w) / float(sw)
    ry_scale = float(h) / float(sh)
    need_alpha = is_trans and ch >= 4

    # 第六步：映射回全分辨率原图并执行单次向量化边缘紧缩与 1px 完整度校准
    raw_boxes: List[Tuple[int, int, int, int]] = []
    for lbl in active_labels:
        sx0 = int(min_x[lbl])
        sy0 = int(min_y[lbl])
        sx1 = int(max_x[lbl])
        sy1 = int(max_y[lbl])
        if sx1 < sx0 or sy1 < sy0:
            continue

        x0 = max(0, int(sx0 * rx_scale) - scale)
        y0 = max(0, int(sy0 * ry_scale) - scale)
        x1 = min(w, int((sx1 + 1) * rx_scale) + scale)
        y1 = min(h, int((sy1 + 1) * ry_scale) + scale)
        if x1 - x0 < min_px_dim or y1 - y0 < min_px_dim:
            continue

        # 向量化纵向边缘收敛
        r_rgb, r_alpha = _sample_strip(img, x0, y0, x1, y1, 160, y1 - y0, need_alpha=need_alpha)
        if r_alpha is not None:
            row_fg = np.any(r_alpha > 24, axis=1)
        else:
            row_fg = np.any(_color_diff_u8(r_rgb, bg_scalar) > color_thresh_u8, axis=1)
        fg_r_idx = np.flatnonzero(row_fg)
        if fg_r_idx.size > 0:
            y0_t = y0 + int(fg_r_idx[0])
            y1_t = y0 + int(fg_r_idx[-1]) + 1
            if y1_t - y0_t >= 8:
                y0, y1 = y0_t, y1_t

        # 向量化横向边缘收敛
        c_rgb, c_alpha = _sample_strip(img, x0, y0, x1, y1, x1 - x0, 160, need_alpha=need_alpha)
        if c_alpha is not None:
            col_fg = np.any(c_alpha > 24, axis=0)
        else:
            col_fg = np.any(_color_diff_u8(c_rgb, bg_scalar) > color_thresh_u8, axis=0)
        fg_c_idx = np.flatnonzero(col_fg)
        if fg_c_idx.size > 0:
            x0_t = x0 + int(fg_c_idx[0])
            x1_t = x0 + int(fg_c_idx[-1]) + 1
            if x1_t - x0_t >= 8:
                x0, x1 = x0_t, x1_t

        if x1 - x0 >= min_px_dim and y1 - y0 >= min_px_dim:
            raw_boxes.append((x0, y0, x1, y1))

    # 第七步：抑制被更大外接框包含的内部重复嵌套框，并按视觉阅读顺序输出
    dedup_boxes = _suppress_contained_boxes(raw_boxes, containment_thresh=0.85)
    grouped = _group_reading_order([(x0, y0, x1, y1, None) for (x0, y0, x1, y1) in dedup_boxes])

    boxes: List[RawSplitBox] = []
    idx = 1
    for r_idx, grp in enumerate(grouped):
        for c_idx, (x0, y0, x1, y1, _) in enumerate(grp):
            boxes.append(
                RawSplitBox(
                    index=idx,
                    row=r_idx,
                    col=c_idx,
                    x=x0,
                    y=y0,
                    width=max(1, x1 - x0),
                    height=max(1, y1 - y0),
                    confidence=0.98,
                    strategy="contour",
                )
            )
            idx += 1
    return boxes


def uniform_grid_core(w: int, h: int, rows: int, cols: int, gutter_px: int) -> List[RawSplitBox]:
    """
    按指定的固定行数与列数执行几何均匀九宫格/网格切分算子。

    严格按照指定的行列数对宽高区间进行浮点等比例坐标划分，并根据 gutter_px
    向内缩进对称消除单元格之间的机械间隔。

    :param w: 原始图像总宽度（像素）。
    :type w: int
    :param h: 原始图像总高度（像素）。
    :type h: int
    :param rows: 目标切分行数（限制在 [1, 20] 范围内）。
    :type rows: int
    :param cols: 目标切分列数（限制在 [1, 20] 范围内）。
    :type cols: int
    :param gutter_px: 单元格之间的间隔缝隙像素宽度（非负整数）。
    :type gutter_px: int
    :return: 均匀排布的切分框对象列表。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：校准行列数与间距缝隙上下限
    r_cnt = max(1, min(20, int(rows)))
    c_cnt = max(1, min(20, int(cols)))
    g = max(0, int(gutter_px))
    half_g = g // 2

    # 第二步：遍历行列生成等比矩形切片并执行 gutter 缝隙收缩
    boxes: List[RawSplitBox] = []
    idx = 1
    for r in range(r_cnt):
        y0 = int(round((r * h) / float(r_cnt))) + (half_g if r > 0 else 0)
        y1 = int(round(((r + 1) * h) / float(r_cnt))) - ((g - half_g) if r < r_cnt - 1 else 0)
        for c in range(c_cnt):
            x0 = int(round((c * w) / float(c_cnt))) + (half_g if c > 0 else 0)
            x1 = int(round(((c + 1) * w) / float(c_cnt))) - ((g - half_g) if c < c_cnt - 1 else 0)
            boxes.append(
                RawSplitBox(
                    index=idx,
                    row=r,
                    col=c,
                    x=x0,
                    y=y0,
                    width=max(1, x1 - x0),
                    height=max(1, y1 - y0),
                    confidence=1.0,
                    strategy="grid",
                )
            )
            idx += 1
    return boxes


def detect_auto_py(
    image: np.ndarray,
    mode: str = "auto",
    sensitivity: float = 0.65,
    min_region_ratio: float = 0.08,
    bleed_trim: int = 2,
    contour_gap_px: int = 8,
    rows: int = 3,
    cols: int = 3,
    gutter_px: int = 0,
    uniform_size: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    uniform_mode: str = "pad",
    collect_diagnostics: bool = False,
) -> Tuple[str, List[RawSplitBox], Optional[Dict[str, object]]]:
    """
    智能图像自动切分核心调度入口（支持 auto/xycut/contour/grid 策略）。

    根据配置或自动图像特征分析选择最优分割算子：
    - auto: 优先执行投影割线切分；若背景平坦且 xycut 未能紧密铺满画布，则结合轮廓算子择优切换；
    - xycut: 强制使用基于投影与沟槽分析的 XY-Cut 算子；
    - contour: 强制使用基于形态学连通域的独立主体轮廓提取算子；
    - grid: 强制使用固定行列的等比网格切分算子。

    :param image: 输入图像数组，要求为三维 uint8 数组（H, W, C）。
    :type image: numpy.ndarray
    :param mode: 切分模式，可选 'auto'、'xycut'、'contour' 或 'grid'，默认为 'auto'。
    :type mode: str
    :param sensitivity: 切分敏感度，取值范围 [0.1, 1.0]，默认为 0.65。
    :type sensitivity: float
    :param min_region_ratio: 过滤过小子块的比例下限，默认为 0.08。
    :type min_region_ratio: float
    :param bleed_trim: 渗色收边层数，默认为 2 像素。
    :type bleed_trim: int
    :param contour_gap_px: 轮廓模式下允许粘连的最大缝隙，默认为 8 像素。
    :type contour_gap_px: int
    :param rows: 网格模式行数，默认为 3。
    :type rows: int
    :param cols: 网格模式列数，默认为 3。
    :type cols: int
    :param gutter_px: 网格模式槽线宽度，默认为 0。
    :type gutter_px: int
    :param uniform_size: 是否启用输出尺寸统一对齐，默认为 False。
    :type uniform_size: bool
    :param target_size: 可选的目标统一切片尺寸 (width, height)。
    :type target_size: Optional[Tuple[int, int]]
    :param uniform_mode: 尺寸统一对齐模式，可选 'pad'、'resize' 或 'crop'，默认为 'pad'。
    :type uniform_mode: str
    :param collect_diagnostics: 是否收集背景诊断参数与投影曲线数据，默认为 False。
    :type collect_diagnostics: bool
    :return: 包含最终采用策略名称、检测出的切片框列表与诊断信息字典的三元组：(detected_mode, boxes, diag)。
    :raises ValueError: 当 uniform_mode 非法或 target_size 宽高不为正整数时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：校验参数合法性（统一切分模式与目标尺寸）
    u_mode = uniform_mode.lower()
    if u_mode not in ("pad", "resize", "crop"):
        raise ValueError(f"uniform_mode must be 'pad', 'resize', or 'crop', got {uniform_mode!r}")
    if target_size is not None:
        tw, th = target_size
        if tw <= 0 or th <= 0:
            raise ValueError(f"target_size width and height must be positive, got {target_size!r}")

    # 第二步：校验图像数组内存合法性并评估四周背景特性与噪声水平
    img = validate_image_array(image)
    h, w, _ = img.shape
    mode_lower = mode.lower()

    bg_rgb, is_trans, bg_unif, bg_noise = _estimate_border_bg_stats(img)
    diag: Optional[Dict[str, object]] = None
    curves: Optional[Dict[str, List[float]]] = None

    # 第三步：依据请求模式执行对应的分割算法内核
    if mode_lower == "grid":
        detected_mode = "grid"
        boxes = uniform_grid_core(w, h, rows, cols, gutter_px)
    elif mode_lower == "contour":
        detected_mode = "contour"
        boxes = contours_core(img, sensitivity, min_region_ratio, contour_gap_px, bg_rgb, is_trans, bg_noise=bg_noise)
    else:
        xy_boxes, curves = xy_cut_core(
            img,
            sensitivity,
            min_region_ratio,
            bleed_trim,
            bg_rgb,
            bg_unif > 0.72,
            is_trans,
            collect_curves=collect_diagnostics,
            bg_noise=bg_noise,
        )
        detected_mode = "xycut"
        boxes = xy_boxes

        # 在 auto 模式下，若背景平坦且尚未形成高密度规整网格，则运行轮廓算子对比择优
        if mode_lower != "xycut" and bg_unif > 0.72:
            total_area = w * h
            xy_area = sum(b.width * b.height for b in xy_boxes)
            # 若 xycut 已切出 >=4 个高置信度且紧密平铺 (>=74% 面积) 的卡片，无需重复扫描轮廓
            skip_contour = (
                len(xy_boxes) >= 4
                and (xy_area * 100) >= (total_area * 74)
                and all(b.confidence >= 0.45 for b in xy_boxes)
            )
            if not skip_contour:
                ct_boxes = contours_core(
                    img, sensitivity, min_region_ratio, contour_gap_px, bg_rgb, is_trans, bg_noise=bg_noise
                )
                ct_area = sum(b.width * b.height for b in ct_boxes)
                if len(ct_boxes) >= 2 and (
                    len(xy_boxes) <= 1 or ((ct_area * 100) < (total_area * 65) and len(ct_boxes) >= len(xy_boxes))
                ):
                    detected_mode = "contour"
                    boxes = ct_boxes

    # 第四步：若指定 crop 统一切割，重新计算各切分框使其保持统一目标外接矩形
    if u_mode == "crop" and (uniform_size or target_size is not None) and boxes:
        if target_size is not None:
            tw, th = target_size
        else:
            tw = max(b.width for b in boxes)
            th = max(b.height for b in boxes)
        cw = max(1, min(int(tw), w))
        ch_dim = max(1, min(int(th), h))
        for b in boxes:
            cx = b.x + b.width // 2
            cy = b.y + b.height // 2
            b.x = min(max(0, cx - cw // 2), max(0, w - cw))
            b.y = min(max(0, cy - ch_dim // 2), max(0, h - ch_dim))
            b.width = cw
            b.height = ch_dim

    # 第五步：组装诊断信息字典（背景色彩、平坦度得分、背景噪声与投影曲线）
    if collect_diagnostics:
        diag = {
            "bg_rgb": [int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2])],
            "is_transparent": bool(is_trans),
            "bg_uniformity": round(float(bg_unif), 4),
            "bg_noise_mad": round(float(bg_noise), 2),
            "seam_curves": curves or {},
        }

    return detected_mode, boxes, diag


def _normalize_tile_cv(
    tile_rgb_or_rgba: np.ndarray,
    target_size: Optional[Tuple[int, int]],
    uniform_mode: str,
    fill_rgba: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    对提取出的单个切片图像进行目标尺寸几何归一化（填充、等比缩放或拉伸）。

    支持三种统一对齐模式：
    - resize: 直接强制拉伸插值至目标宽高；
    - pad: 保持切片原始比例居中，不足部分使用指定背景色（fill_rgba）或透明边界进行 copyMakeBorder 填充；
    - crop: 前置已按目标比例裁切，此处若有微小偏离则等比补齐。

    :param tile_rgb_or_rgba: 切片图像像素矩阵（RGB 或 RGBA 通道）。
    :type tile_rgb_or_rgba: numpy.ndarray
    :param target_size: 可选的目标尺寸二元组 (width, height)。若为 None 则保持原样返回。
    :type target_size: Optional[Tuple[int, int]]
    :param uniform_mode: 尺寸统一对齐模式（'pad'、'resize' 或 'crop'）。
    :type uniform_mode: str
    :param fill_rgba: 填充空白边界时使用的 RGBA 颜色四元组。
    :type fill_rgba: Tuple[int, int, int, int]
    :return: 归一化后的切片图像数组。
    :author: tongliga
    :date: 2026-11-18
    """
    if target_size is None:
        return tile_rgb_or_rgba

    tw, th = int(target_size[0]), int(target_size[1])
    ch_h, cw = tile_rgb_or_rgba.shape[:2]
    if cw == tw and ch_h == th:
        return tile_rgb_or_rgba

    # 第一步：若为 resize 模式，直接执行全幅双线性/区域插值缩放
    if uniform_mode == "resize":
        interp = cv2.INTER_AREA if (cw > tw or ch_h > th) else cv2.INTER_LINEAR
        return cv2.resize(tile_rgb_or_rgba, (tw, th), interpolation=interp)

    # 第二步：若切片尺寸超过目标尺寸，等比缩小至目标边界内部
    if cw > tw or ch_h > th:
        scale = min(float(tw) / float(cw), float(th) / float(ch_h))
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch_h * scale)))
        fitted = cv2.resize(tile_rgb_or_rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        fitted = tile_rgb_or_rgba

    # 第三步：计算四周对称补白边距并调用 copyMakeBorder 执行常数填充
    fh, fw = fitted.shape[:2]
    top = max(0, (th - fh) // 2)
    bottom = max(0, th - fh - top)
    left = max(0, (tw - fw) // 2)
    right = max(0, tw - fw - left)

    channels = fitted.shape[2]
    border_val = fill_rgba[:channels]
    return cv2.copyMakeBorder(
        fitted,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=border_val,
    )


def _encode_tile_bytes(
    tile_rgb_or_rgba: np.ndarray,
    ext: str,
    quality: int = 95,
) -> bytes:
    """
    使用 OpenCV 内存编码器将 RGB/RGBA 切片图像矩阵编码为指定文件格式的二进制字节流。

    自动处理色彩空间转换（RGB/RGBA 转 BGR/BGRA）与压缩质量参数传递。

    :param tile_rgb_or_rgba: 切片图像像素矩阵（三通道 RGB 或四通道 RGBA）。
    :type tile_rgb_or_rgba: numpy.ndarray
    :param ext: 输出图像文件扩展名，支持 'png'、'jpg'、'jpeg' 或 'webp'。
    :type ext: str
    :param quality: JPEG 或 WebP 格式的压缩质量（1~100），默认为 95。
    :type quality: int
    :return: 编码完成的图像文件原始二进制字节流。
    :raises ValueError: 当 OpenCV 内存编码失败时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    ch = tile_rgb_or_rgba.shape[2]
    # 第一步：根据目标格式与输入通道转换色彩空间并调用 OpenCV 内存编码
    if ext in ("jpg", "jpeg"):
        bgr = cv2.cvtColor(tile_rgb_or_rgba, cv2.COLOR_RGBA2BGR if ch == 4 else cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif ext == "webp":
        bgr = cv2.cvtColor(tile_rgb_or_rgba, cv2.COLOR_RGBA2BGRA if ch == 4 else cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    else:
        bgr = cv2.cvtColor(tile_rgb_or_rgba, cv2.COLOR_RGBA2BGRA if ch == 4 else cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    # 第二步：检查编码结果状态，转换输出为原生 bytes 对象
    if not ok:
        raise ValueError(f"OpenCV image encoding failed for format '{ext}'")
    return buf.tobytes()


def batch_crop_to_memory_py(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    fmt: str = "png",
    target_size: Optional[Tuple[int, int]] = None,
    uniform_mode: str = "pad",
    max_workers: Optional[int] = None,
) -> List[bytes]:
    """
    基于内存视图与线程池并发将图像批量裁切为二进制字节流列表。

    严格执行零拷贝切片与边界安全约束（clamp_box + safe_roi_view），
    对每个切片根据 target_size 和 uniform_mode 进行尺寸归一化，
    并多线程并发编码为 PNG/JPEG/WebP 二进制数据流。

    :param image: 输入图像数据矩阵（RGB 或 RGBA 通道）。
    :type image: numpy.ndarray
    :param boxes: 待裁切矩形框坐标四元组列表，每个元素为 (x, y, width, height)。
    :type boxes: List[Tuple[int, int, int, int]]
    :param fmt: 目标编码图像文件格式（'png'、'jpg'、'jpeg' 或 'webp'），默认为 'png'。
    :type fmt: str
    :param target_size: 可选的目标切片归一化尺寸 (width, height)。
    :type target_size: Optional[Tuple[int, int]]
    :param uniform_mode: 尺寸统一对齐模式（'pad'、'resize' 或 'crop'），默认为 'pad'。
    :type uniform_mode: str
    :param max_workers: 线程池最大并发工作线程数。若为 None 则自适应系统 CPU 核心数。
    :type max_workers: Optional[int]
    :return: 编码后的各切片文件二进制字节流列表（与输入 boxes 顺序一一对应）。
    :raises ValueError: 当输入格式不支持或 uniform_mode 非法时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：校验图像内存、目标格式与尺寸对齐模式有效性
    img = validate_image_array(image)
    h, w, _ = img.shape
    ext = fmt.lower().strip(".")
    if ext not in ("png", "jpg", "jpeg", "webp"):
        raise ValueError(f"Unsupported output format '{fmt}', expected png/jpg/webp.")
    u_mode = uniform_mode.lower()
    if u_mode not in ("pad", "resize", "crop"):
        raise ValueError(f"uniform_mode must be 'pad', 'resize', or 'crop', got {uniform_mode!r}")

    # 第二步：确定填充背景色（优先继承原始图像边缘色或透明通道）
    if target_size is not None and u_mode == "pad":
        bg_rgb, is_trans, _ = estimate_border_bg(img)
        fill = (0, 0, 0, 0) if is_trans else (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 255)
    else:
        fill = (255, 255, 255, 255)

    # 第三步：定义单个切片的裁剪、尺寸归一化与内存编码处理闭包
    def _process_one(coords: Tuple[int, int, int, int]) -> bytes:
        x0, y0, cw, ch_box = clamp_box(coords[0], coords[1], coords[2], coords[3], w, h)
        tile_view = safe_roi_view(img, x0, y0, x0 + cw, y0 + ch_box)
        normalized = _normalize_tile_cv(tile_view, target_size, u_mode, fill)
        return _encode_tile_bytes(normalized, ext)

    # 第四步：若任务量微小（<=2）则单线程串行处理以消除线程上下文开销
    if len(boxes) <= 2:
        return [_process_one(b) for b in boxes]

    # 第五步：利用多线程并发池映射执行切片编码任务
    workers = max_workers or min(len(boxes), max(2, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_process_one, boxes))


def batch_crop_and_save_py(
    image_path: str | Path,
    boxes: List[Tuple[int, int, int, int]],
    output_dir: str | Path,
    prefix: str = "slice",
    fmt: str = "png",
    target_size: Optional[Tuple[int, int]] = None,
    uniform_mode: str = "pad",
    max_workers: Optional[int] = None,
) -> List[str]:
    """
    从本地图像文件读取并多线程批量裁切切片保存至指定目标目录。

    自动调用 decode_image_safe 安全解码图像，并创建输出目录；
    针对每个切片按索引序号格式化命名（例如 slice_01.png），
    利用线程池并发完成切片裁剪、尺寸归一化、图像编码与磁盘写入。

    :param image_path: 输入原始图像的文件路径。
    :type image_path: Union[str, pathlib.Path]
    :param boxes: 待裁切矩形框坐标四元组列表，每个元素为 (x, y, width, height)。
    :type boxes: List[Tuple[int, int, int, int]]
    :param output_dir: 保存切片文件的目标目录路径。
    :type output_dir: Union[str, pathlib.Path]
    :param prefix: 产出切片文件命名前缀，默认为 'slice'。
    :type prefix: str
    :param fmt: 目标保存文件格式（'png'、'jpg'、'jpeg' 或 'webp'），默认为 'png'。
    :type fmt: str
    :param target_size: 可选的目标切片归一化尺寸 (width, height)。
    :type target_size: Optional[Tuple[int, int]]
    :param uniform_mode: 尺寸统一对齐模式（'pad'、'resize' 或 'crop'），默认为 'pad'。
    :type uniform_mode: str
    :param max_workers: 线程池最大并发工作线程数。若为 None 则自适应系统 CPU 核心数。
    :type max_workers: Optional[int]
    :return: 保存成功的各切片文件绝对/相对路径字符串列表。
    :raises ValueError: 当输入格式不支持或 uniform_mode 非法时抛出。
    :raises FileNotFoundError: 当输入图像文件不存在时抛出。
    :author: tongliga
    :date: 2026-11-18
    """
    # 第一步：安全解码源图像文件并确保存储目录存在
    img = decode_image_safe(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 第二步：校验输出图像格式与尺寸归一化模式有效性
    ext = fmt.lower().strip(".")
    if ext not in ("png", "jpg", "jpeg", "webp"):
        raise ValueError(f"Unsupported output format '{fmt}', expected png/jpg/webp.")
    u_mode = uniform_mode.lower()
    if u_mode not in ("pad", "resize", "crop"):
        raise ValueError(f"uniform_mode must be 'pad', 'resize', or 'crop', got {uniform_mode!r}")
    h, w, _ = img.shape

    # 第三步：评估背景色彩以用于边界补齐
    bg_rgb, is_trans, _ = estimate_border_bg(img)
    fill = (0, 0, 0, 0) if is_trans else (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 255)

    # 第四步：定义单个切片的裁剪、尺寸归一化、编码与磁盘写入闭包
    def _save_one(item: Tuple[int, Tuple[int, int, int, int]]) -> str:
        i, coords = item
        x0, y0, cw, ch_box = clamp_box(coords[0], coords[1], coords[2], coords[3], w, h)
        tile_view = safe_roi_view(img, x0, y0, x0 + cw, y0 + ch_box)
        normalized = _normalize_tile_cv(tile_view, target_size, u_mode, fill)
        blob = _encode_tile_bytes(normalized, ext)
        out_path = out_dir / f"{prefix}_{i + 1:02d}.{ext}"
        out_path.write_bytes(blob)
        return str(out_path)

    # 第五步：若任务量微小（<=2）则单线程串行处理以消除线程上下文开销
    indexed = list(enumerate(boxes))
    if len(indexed) <= 2:
        return [_save_one(item) for item in indexed]

    # 第六步：利用多线程并发池映射执行切片保存任务
    workers = max_workers or min(len(indexed), max(2, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_save_one, indexed))
