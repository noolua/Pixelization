# -*- coding: utf-8 -*-
"""
边缘加深模块
检测前景边界像素并向暗色偏移，增强像素画轮廓。
"""
import numpy as np
from PIL import Image

from pipes.bg_unify import detect_background_mask


def edge_darken(
    image: Image.Image,
    tolerance: float = 5.0,
    strength: float = 0.3,
) -> Image.Image:
    """
    边缘加深：对前景区域边界像素做暗色偏移。

    参数:
        image: 输入 PIL 图像
        tolerance: 背景检测 Delta-E 容差
        strength: 加深强度，越大越暗（0.1-0.7）
    返回:
        边缘加深后的 PIL RGB 图像
    """
    img_array, bg_mask = detect_background_mask(image, tolerance)
    H, W = img_array.shape[:2]

    if H < 10 or W < 10:
        return Image.fromarray(img_array)

    # 前景 mask
    fg_mask = ~bg_mask

    if not fg_mask.any():
        return Image.fromarray(img_array)

    # 腐蚀前景 mask（去掉一层边界像素 = 内部区域）
    eroded = np.zeros_like(fg_mask)
    eroded[1:-1, 1:-1] = (
        fg_mask[:-2, 1:-1] & fg_mask[2:, 1:-1] &
        fg_mask[1:-1, :-2] & fg_mask[1:-1, 2:]
    )

    # 边界 = 前景 XOR 腐蚀后
    edge_mask = fg_mask & ~eroded

    if not edge_mask.any():
        return Image.fromarray(img_array)

    # 暗色偏移
    result = img_array.astype(np.float64)
    result[edge_mask] = result[edge_mask] * (1.0 - strength)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)
