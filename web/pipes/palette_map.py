# -*- coding: utf-8 -*-
"""
调色板映射模块
将图像颜色限定到预设调色板，支持最近色映射和抖动模式。
"""
import numpy as np
from PIL import Image

from pipes.bg_unify import rgb_to_lab

# ── 内置调色板 ──────────────────────────────────────────────

BUILT_IN_PALETTES = {
  "pico-8": [
    (0,0,0), (29,43,83), (126,37,83), (128,63,64),
    (149,99,68), (215,143,64), (224,178,76), (95,87,79),
    (194,195,199), (255,241,232), (255,0,77), (255,163,0),
    (255,236,39), (0,135,81), (41,173,255), (131,118,156),
  ],
  "gameboy": [
    (15,56,15), (48,98,48), (139,172,15), (155,188,15),
  ],
  "nes": [
    (124,124,124),(0,0,252),(0,0,188),(68,40,188),
    (148,0,132),(168,0,32),(168,16,0),(136,20,0),
    (80,48,0),(0,120,0),(0,104,0),(0,88,0),
    (0,64,88),(0,0,0),(0,0,0),(0,0,0),
    (188,188,188),(0,120,248),(0,88,248),(104,68,252),
    (216,0,204),(228,0,88),(248,56,0),(228,92,16),
    (172,124,0),(0,184,0),(0,168,0),(0,168,68),
    (0,136,136),(0,0,0),(0,0,0),(0,0,0),
    (248,248,248),(60,188,252),(104,136,252),(152,120,248),
    (248,120,248),(248,88,152),(248,120,88),(252,160,68),
    (248,184,0),(184,248,24),(88,216,84),(88,248,152),
    (0,232,216),(120,120,120),(0,0,0),(0,0,0),
    (252,252,252),(164,228,252),(184,184,248),(216,184,248),
    (248,184,248),(248,164,192),(240,208,176),(252,224,168),
    (248,216,120),(216,248,120),(184,248,184),(184,248,216),
    (0,252,252),(248,216,248),(0,0,0),(0,0,0),
  ],
  "c64": [
    (0,0,0),(255,255,255),(136,0,0),(170,255,238),
    (204,68,204),(0,204,85),(0,0,170),(238,238,119),
    (221,136,85),(102,68,0),(255,119,119),(51,51,51),
    (119,119,119),(170,255,102),(0,136,255),(187,187,187),
  ],
  "endesga-32": [
    (190,74,47),(215,118,67),(231,166,81),(249,207,120),
    (164,210,93),(92,183,91),(38,132,70),(15,85,60),
    (46,76,105),(52,124,169),(68,170,216),(109,204,227),
    (159,233,219),(218,240,218),(245,255,225),(249,237,139),
    (251,221,92),(246,169,57),(221,91,40),(189,55,31),
    (147,43,31),(107,29,22),(63,25,22),(150,62,42),
    (181,100,62),(199,140,95),(227,185,143),(244,222,194),
    (156,174,167),(116,140,152),(80,108,131),(62,82,103),
  ],
}


def _resolve_palette(palette: str) -> list:
  """
  解析调色板参数为 (R,G,B) 列表。

  palette 可以是内置名称（如 "pico-8"）或逗号分隔的 hex 列表（如 "#FF0000,#00FF00"）。
  """
  name = palette.lower().strip()
  if name in BUILT_IN_PALETTES:
    return BUILT_IN_PALETTES[name]
  # 尝试解析为 hex 列表
  try:
    colors = []
    for hex_str in palette.split(","):
      h = hex_str.strip().lstrip("#")
      colors.append(tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))
    if len(colors) < 2:
      raise ValueError("自定义调色板至少需要 2 种颜色")
    return colors
  except (ValueError, IndexError):
    raise ValueError(f"无效的调色板参数: {palette}")


def _nearest_color_map(img_array, palette_lab, palette_rgb):
  """Lab 空间最近色映射（无抖动），利用唯一颜色加速"""
  pixels = img_array.reshape(-1, 3)
  unique_colors, inverse = np.unique(pixels, axis=0, return_inverse=True)

  unique_lab = rgb_to_lab(unique_colors)

  # (N_unique, K) Delta-E²
  diff = unique_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
  distances = np.sum(diff ** 2, axis=2)
  nearest = np.argmin(distances, axis=1)

  result = palette_rgb[nearest[inverse]].reshape(img_array.shape)
  return result


def _ordered_dither(img_array, palette_lab, palette_rgb):
  """有序抖动（4×4 Bayer 矩阵），在 L 通道加阈值偏移"""
  H, W, _ = img_array.shape

  BAYER_4X4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
  ], dtype=np.float64) / 16.0

  # 构造与图像同尺寸的 Bayer 阈值矩阵
  bayer_h = (H + 3) // 4 * 4
  bayer_w = (W + 3) // 4 * 4
  bayer_tiled = np.tile(BAYER_4X4, (bayer_h // 4, bayer_w // 4))[:H, :W]

  img_lab = rgb_to_lab(img_array)
  img_lab_f = img_lab.copy()

  # 对 L 通道施加偏移（强度与调色板大小反相关）
  K = len(palette_lab)
  strength = max(10.0, 80.0 / K)
  img_lab_f[..., 0] += (bayer_tiled - 0.5) * strength

  # 逐像素最近色
  pixels_lab = img_lab_f.reshape(-1, 3)
  unique_lab, inverse = np.unique(pixels_lab, axis=0, return_inverse=True)

  diff = unique_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
  distances = np.sum(diff ** 2, axis=2)
  nearest = np.argmin(distances, axis=1)

  result = palette_rgb[nearest[inverse]].reshape(img_array.shape)
  return result


def _floyd_steinberg_dither(img_array, palette_lab, palette_rgb):
  """Floyd-Steinberg 误差扩散抖动"""
  H, W, _ = img_array.shape
  img_f = img_array.astype(np.float64)
  K = len(palette_lab)

  for y in range(H):
    for x in range(W):
      old = img_f[y, x].copy()
      # 找最近调色板色
      pixel_lab = rgb_to_lab(old.reshape(1, 3)).reshape(3)
      diff = pixel_lab - palette_lab
      dist = np.sum(diff ** 2, axis=1)
      nearest = np.argmin(dist)
      new = palette_rgb[nearest].astype(np.float64)
      img_f[y, x] = new

      err = old - new
      # 扩散误差到邻居
      if x + 1 < W:
        img_f[y, x + 1] += err * 7 / 16
      if y + 1 < H:
        if x - 1 >= 0:
          img_f[y + 1, x - 1] += err * 3 / 16
        img_f[y + 1, x] += err * 5 / 16
        if x + 1 < W:
          img_f[y + 1, x + 1] += err * 1 / 16

  return np.clip(np.round(img_f), 0, 255).astype(np.uint8)


def palette_map(image: Image.Image, palette: str = "pico-8", dither: str = "none") -> Image.Image:
  """
  调色板映射 pipe。

  参数:
      image: 输入 PIL 图像
      palette: 内置调色板名称或逗号分隔的 hex 色值列表
      dither: 抖动模式 (none / ordered / floyd-steinberg)
  返回:
      映射后的 PIL Image
  """
  colors = _resolve_palette(palette)
  palette_rgb = np.array(colors, dtype=np.uint8)
  palette_lab = rgb_to_lab(palette_rgb)

  img_array = np.array(image.convert("RGB"))

  if dither == "none":
    result = _nearest_color_map(img_array, palette_lab, palette_rgb)
  elif dither == "ordered":
    result = _ordered_dither(img_array, palette_lab, palette_rgb)
  elif dither == "floyd-steinberg":
    result = _floyd_steinberg_dither(img_array, palette_lab, palette_rgb)
  else:
    raise ValueError(f"不支持的抖动模式: {dither}")

  return Image.fromarray(result)
