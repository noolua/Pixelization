# -*- coding: utf-8 -*-
"""
背景色统一模块
聚类锚定 + Lab 容差扩展算法，将图像背景统一为指定颜色。
"""
import numpy as np
from collections import deque
from PIL import Image


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    纯 numpy 实现 sRGB → Lab 转换。

    参数:
        rgb: uint8 数组，形状 (..., 3)，范围 [0, 255]
    返回:
        float64 数组，形状 (..., 3)，Lab 值
    """
    # sRGB → 线性 RGB（gamma 解码）
    srgb = rgb.astype(np.float64) / 255.0
    mask = srgb <= 0.04045
    linear = np.where(mask, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

    # 线性 RGB → XYZ（D65 白点）
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ M.T

    # XYZ → Lab
    # D65 白点参考值
    xn, yn, zn = 0.95047, 1.0, 1.08883
    fx = _f(xyz[..., 0] / xn)
    fy = _f(xyz[..., 1] / yn)
    fz = _f(xyz[..., 2] / zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def _f(t: np.ndarray) -> np.ndarray:
    """Lab 转换的中间函数"""
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)


def hex_to_rgb(hex_color: str) -> tuple:
    """十六进制颜色转 (R, G, B) 元组"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _kmeans_anchor(pixels: np.ndarray, k: int = 3, max_iter: int = 20) -> np.ndarray:
    """
    对边缘像素做 K-Means 聚类，返回最大簇的中心（RGB float64）。

    参数:
        pixels: (N, 3) uint8 RGB 像素
        k: 聚类数
        max_iter: 最大迭代次数
    """
    # 唯一颜色 + 频率加权（与 api.py:kmeans_colors 相同策略）
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    n = len(unique_colors)
    k = min(k, n)

    rng = np.random.default_rng(42)
    probs = counts / counts.sum()
    initial_idx = rng.choice(n, size=k, replace=False, p=probs)
    centers = unique_colors[initial_idx].astype(np.float64)

    for _ in range(max_iter):
        diff = unique_colors[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.zeros_like(centers)
        for ki in range(k):
            m = labels == ki
            if m.any():
                new_centers[ki] = np.average(unique_colors[m], axis=0, weights=counts[m])
            else:
                new_centers[ki] = centers[ki]

        if np.allclose(centers, new_centers, atol=1.0):
            break
        centers = new_centers

    # 取最大簇（按像素数加权）的中心
    cluster_sizes = np.array([counts[labels == ki].sum() for ki in range(k)])
    largest = np.argmax(cluster_sizes)
    return centers[largest]


def unify_background(
    image: Image.Image,
    tolerance: float = 5.0,
    target_rgb: tuple = (128, 128, 128),
) -> Image.Image:
    """
    统一图像背景色。

    算法：边缘采样 → 锚定色 → Lab Delta-E 候选掩码 → BFS flood fill → 替换。

    参数:
        image: 输入 PIL 图像
        tolerance: Delta-E 容差（越大越激进）
        target_rgb: 目标背景色 (R, G, B)
    返回:
        背景统一后的 PIL 图像
    """
    img_array = np.array(image.convert("RGB"))
    H, W, _ = img_array.shape

    # 极小图片直接返回
    if H < 10 or W < 10:
        return Image.fromarray(img_array)

    # 1. 边缘采样（自适应宽度）
    border_w = max(3, min(H, W) // 50)
    top = img_array[:border_w, :, :]
    bottom = img_array[-border_w:, :, :]
    left = img_array[:, :border_w, :]
    right = img_array[:, -border_w:, :]
    edge_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ], axis=0)

    # 2. 锚定色：K-Means(K=3) 聚类取最大簇中心
    anchor_rgb = _kmeans_anchor(edge_pixels, k=3)
    anchor_lab = rgb_to_lab(anchor_rgb.reshape(1, 3)).reshape(3)

    # 3. 全图 Lab 转换 + Delta-E 距离
    all_lab = rgb_to_lab(img_array)
    delta_e = np.sqrt(np.sum((all_lab - anchor_lab) ** 2, axis=-1))

    # 4. 候选掩码（Delta-E ≤ tolerance）
    candidate = delta_e <= tolerance

    # 5. BFS flood fill 从四边出发（4-连通），仅保留边缘连通区域
    mask = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)
    queue = deque()

    # 种子：四边上的候选像素
    for c in range(W):
        if candidate[0, c]:
            queue.append((0, c))
        if candidate[H - 1, c]:
            queue.append((H - 1, c))
    for r in range(1, H - 1):
        if candidate[r, 0]:
            queue.append((r, 0))
        if candidate[r, W - 1]:
            queue.append((r, W - 1))

    while queue:
        r, c = queue.popleft()
        if r < 0 or r >= H or c < 0 or c >= W:
            continue
        if visited[r, c]:
            continue
        if not candidate[r, c]:
            continue
        visited[r, c] = True
        mask[r, c] = True
        queue.append((r - 1, c))
        queue.append((r + 1, c))
        queue.append((r, c - 1))
        queue.append((r, c + 1))

    # 6. 替换掩码区域为目标色
    if not mask.any():
        return Image.fromarray(img_array)

    result = img_array.copy()
    result[mask] = target_rgb

    return Image.fromarray(result)
