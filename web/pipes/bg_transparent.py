# -*- coding: utf-8 -*-
"""
背景透明化模块
检测背景色并将背景像素 alpha 设为 0，输出 RGBA PNG。
"""
import numpy as np
from PIL import Image

from pipes.bg_unify import detect_background_mask


def bg_transparent(
    image: Image.Image,
    tolerance: float = 5.0,
) -> Image.Image:
    """
    背景透明化：将背景区域设为透明。

    注意：输出 RGBA 图像，应放在管线末尾。

    参数:
        image: 输入 PIL 图像
        tolerance: 背景检测 Delta-E 容差
    返回:
        背景透明的 PIL RGBA 图像
    """
    img_array, mask = detect_background_mask(image, tolerance)
    H, W = img_array.shape[:2]

    if H < 10 or W < 10:
        return image.convert("RGBA")

    # 构建 RGBA
    result = np.dstack([img_array, np.full((H, W), 255, dtype=np.uint8)])
    result[mask, 3] = 0

    return Image.fromarray(result, mode="RGBA")
