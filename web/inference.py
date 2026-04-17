import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from models.c2pGen import RGBEncoder, RGBDecoder, AliasNet

PROJECT_ROOT = Path(__file__).parent.parent


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

  def pixelize(self, in_img, out_img, cell_size):
    with torch.no_grad():
      in_img = Image.open(in_img).convert('RGB')
      in_img = rescale(in_img)
      width, height = in_img.size
      best_cell_size = 4
      in_img = in_img.resize(((width // cell_size) * best_cell_size, (height // cell_size) * best_cell_size),
                         Image.BICUBIC)
      in_t = process(in_img).to(self.device)

      feature = self.net.RGBEnc(in_t)
      images = self.net.RGBDec(feature, self.cell_size_code)
      out_t = self.alias_net(images)
      save(out_t, out_img, cell_size, best_cell_size)

  def pixelize_original_size(self, in_img, out_img, cell_size):
    """像素化并返回原始像素网格尺寸（不放大）"""
    result = self.pixelize_image(Image.open(in_img).convert('RGB'), cell_size)
    result.save(out_img)

  def pixelize_image(self, pil_img: Image.Image, cell_size: int) -> Image.Image:
    """像素化，接收 PIL Image，返回 PIL Image（纯内存）"""
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
      in_t = process(in_img).to(self.device)

      feature = self.net.RGBEnc(in_t)
      images = self.net.RGBDec(feature, self.cell_size_code)
      out_t = self.alias_net(images)
      return to_image(out_t, target_w, target_h)


def process(img):
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


def save(tensor, file, cell_size, best_cell_size=4):
  img = tensor.data[0].cpu().float().numpy()
  img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
  img = img.astype(np.uint8)
  img = Image.fromarray(img)
  img = img.resize((img.size[0] // best_cell_size, img.size[1] // best_cell_size), Image.NEAREST)
  img = img.resize((img.size[0] * cell_size, img.size[1] * cell_size), Image.NEAREST)
  img.save(file)


def save_original_size(tensor, file, cell_size, best_cell_size=4, target_w=None, target_h=None):
  """保存为原始像素网格尺寸（不放大）"""
  img = to_image(tensor, target_w, target_h, best_cell_size)
  img.save(file)


def to_image(tensor, target_w=None, target_h=None, best_cell_size=4):
  """将推理输出 tensor 转为 PIL Image"""
  img = tensor.data[0].cpu().float().numpy()
  img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
  img = img.astype(np.uint8)
  img = Image.fromarray(img)
  if target_w and target_h:
    img = img.resize((target_w, target_h), Image.NEAREST)
  else:
    img = img.resize((img.size[0] // best_cell_size, img.size[1] // best_cell_size), Image.NEAREST)
  return img
