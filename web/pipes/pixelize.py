# -*- coding: utf-8 -*-
"""
像素化处理模块
基于神经网络的图像像素化 pipe，包含模型定义、加载和推理。
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

from models.c2pGen import RGBEncoder, RGBDecoder, AliasNet

PROJECT_ROOT = Path(__file__).parent.parent.parent


class _PixelNet(nn.Module):
  """推理专用：只包含 RGBEnc + RGBDec，与 inference_net.pth 的 key 前缀对齐"""
  def __init__(self):
    super().__init__()
    self.RGBEnc = RGBEncoder(3, 64, 2, 4, "in", 'relu', 'reflect')
    self.RGBDec = RGBDecoder(256, 3, 2, 4, res_norm='adain', activ='relu', pad_type='reflect')


class Model():
  def __init__(self, model_name, device="cuda"):
    self.device = torch.device(device)
    self.net = None
    self.alias_net = None
    self.cell_size_code = None
    self.model_name = model_name

  def load(self):
    with torch.no_grad():
      # 加载推理专用网络（RGBEnc + RGBDec）
      self.net = _PixelNet()
      state = torch.load(str(PROJECT_ROOT / "downloads" / "inference_net.pth"), map_location='cpu')
      self.net.load_state_dict(state)
      self.net.to(self.device)
      self.net.eval()

      # 加载 AliasNet
      self.alias_net = AliasNet(3, 3, 64, 2, 3, activ='relu', pad_type='reflect')
      alias_state = torch.load(str(PROJECT_ROOT / "downloads" / "alias_net.pth"), map_location='cpu')
      if list(alias_state.keys())[0].startswith('module.'):
        alias_state = {k.replace('module.', ''): v for k, v in alias_state.items()}
      self.alias_net.load_state_dict(alias_state)
      self.alias_net.to(self.device)
      self.alias_net.eval()

      # 加载预计算的 cell_size_code
      self.cell_size_code = torch.load(str(PROJECT_ROOT / "downloads" / "cell_size_code.pt"),
                                        map_location=self.device)


def rescale(image, Rescale=True):
  if not Rescale:
    return image
  if Rescale:
    width, height = image.size
    while width > 4000 or height > 4000:
      image = image.resize((int(width // 2), int(height // 2)), Image.BICUBIC)
      width, height = image.size
    while width < 128 or height < 128:
      image = image.resize((int(width * 2), int(height * 2)), Image.BICUBIC)
      width, height = image.size
    return image


def _process(img):
  """图像裁齐 + 归一化，转为模型输入 tensor"""
  ow, oh = img.size
  nw = int(round(ow / 4) * 4)
  nh = int(round(oh / 4) * 4)
  left = (ow - nw) // 2
  top = (oh - nh) // 2
  right = left + nw
  bottom = top + nh
  img = img.crop((left, top, right, bottom))
  trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  return trans(img)[None, :, :, :]


def _to_image(tensor, target_w, target_h):
  """将推理输出 tensor 转为 PIL Image"""
  img = tensor.data[0].cpu().float().numpy()
  img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
  img = img.astype(np.uint8)
  img = Image.fromarray(img)
  img = img.resize((target_w, target_h), Image.NEAREST)
  return img


def pixelize(model, pil_img: Image.Image, cell_size: int) -> Image.Image:
  """
  像素化处理 pipe。

  参数:
      model: 已加载的 Model 实例
      pil_img: 输入 PIL 图像
      cell_size: 像素化程度 2-8
  返回:
      像素化后的 PIL Image
  """
  pil_img = pil_img.convert('RGB')
  with torch.no_grad():
    orig_width, orig_height = pil_img.size
    target_w = orig_width // cell_size
    target_h = orig_height // cell_size

    in_img = rescale(pil_img)
    width, height = in_img.size
    best_cell_size = 4
    in_img = in_img.resize(((width // cell_size) * best_cell_size, (height // cell_size) * best_cell_size),
                       Image.BICUBIC)
    in_t = _process(in_img).to(model.device)

    feature = model.net.RGBEnc(in_t)
    images = model.net.RGBDec(feature, model.cell_size_code)
    out_t = model.alias_net(images)
    return _to_image(out_t, target_w, target_h)
