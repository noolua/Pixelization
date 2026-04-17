# -*- coding: utf-8 -*-
"""
K-Means 颜色优化模块
基于唯一颜色 + 频率加权聚类，减少图像颜色数。
"""
import numpy as np
from PIL import Image


def kmeans_colors(image: Image.Image, target_colors: int) -> Image.Image:
    """
    使用加权 K-Means 减少图像颜色数。
    基于唯一颜色 + 频率加权聚类，避免大图全像素运算。

    参数:
        image: 输入 PIL 图像
        target_colors: 目标颜色数
    返回:
        颜色减少后的 PIL 图像
    """
    img_array = np.array(image.convert("RGB"))
    pixels = img_array.reshape(-1, 3)

    # 获取唯一颜色及其频率
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    num_unique = len(unique_colors)

    if num_unique <= target_colors:
        return image

    # 加权 K-Means：用唯一颜色作为样本，counts 作为权重
    # 初始化：按频率加权随机选取初始中心
    rng = np.random.default_rng(42)
    probs = counts / counts.sum()
    initial_idx = rng.choice(num_unique, size=target_colors, replace=False, p=probs)
    centers = unique_colors[initial_idx].astype(np.float64)

    # 迭代 K-Means
    for _ in range(20):
        # 计算每个唯一颜色到各中心的距离 (N_unique, K)
        diff = unique_colors[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        # 加权更新中心
        new_centers = np.zeros_like(centers)
        for k in range(target_colors):
            mask = labels == k
            if mask.any():
                new_centers[k] = np.average(unique_colors[mask], axis=0, weights=counts[mask])
            else:
                new_centers[k] = centers[k]

        if np.allclose(centers, new_centers, atol=1.0):
            break
        centers = new_centers

    # 四舍五入到整数
    centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)

    # 映射：所有像素 -> 最近聚类中心
    # 利用唯一颜色映射表加速
    label_map = labels  # unique_colors[i] -> centers[labels[i]]
    # 建立 原始像素 -> 新颜色 的映射
    color_to_label = {}
    for i, c in enumerate(unique_colors):
        color_to_label[tuple(c)] = tuple(centers[label_map[i]])

    # 用向量化方式替换像素
    flat = pixels
    result = np.array([color_to_label[tuple(p)] for p in flat], dtype=np.uint8)
    result = result.reshape(img_array.shape)

    return Image.fromarray(result)
