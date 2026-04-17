# -*- coding: utf-8 -*-
"""
灰度处理模块
将图像转为量化灰度，支持指定灰阶级数。
"""
import numpy as np
from PIL import Image


def grayscale(image: Image.Image, gray_levels: int = 0) -> Image.Image:
  """
  灰度处理 pipe。

  参数:
      image: 输入 PIL 图像
      gray_levels: 灰阶级数，0=连续灰度，2-256=量化级数
  返回:
      灰度 PIL Image（RGB 三通道）
  """
  img_array = np.array(image.convert("RGB"))

  # BT.601 亮度
  gray = (0.299 * img_array[..., 0] +
          0.587 * img_array[..., 1] +
          0.114 * img_array[..., 2])

  if gray_levels > 0:
    levels = np.linspace(0, 255, gray_levels)
    idx = np.argmin(np.abs(gray[..., np.newaxis] - levels[np.newaxis, :]), axis=-1)
    gray = levels[idx]

  gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)

  # 复制为 3 通道 RGB，保证下游 pipe 可直接拼接
  result = np.stack([gray, gray, gray], axis=-1)
  return Image.fromarray(result)
