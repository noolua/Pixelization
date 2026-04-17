"""
诊断脚本：对比老模型和新模型的推理输出，定位差异点。
在远程机器上运行：python tools/verify_inference.py
"""
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "web"))
from models.networks import define_G
from models.c2pGen import RGBEncoder, RGBDecoder, AliasNet

PROJECT_ROOT = Path(__file__).parent.parent
DOWNLOADS = PROJECT_ROOT / "downloads"

# MLP_code 常量（与 inference.py 一致）
MLP_code = [
    233356.8125, -27387.5918, -32866.8008, 126575.0312, -181590.0156,
    -31543.1289, 50374.1289, 99631.4062, -188897.3750, 138322.7031,
    -107266.2266, 125778.5781, 42416.1836, 139710.8594, -39614.6250,
    -69972.6875, -21886.4141, 86938.4766, 31457.6270, -98892.2344,
    -1191.5887, -61662.1719, -180121.9062, -32931.0859, 43109.0391,
    21490.1328, -153485.3281, 94259.1797, 43103.1992, -231953.8125,
    52496.7422, 142697.4062, -34882.7852, -98740.0625, 34458.5078,
    -135436.3438, 11420.5488, -18895.8984, -71195.4141, 176947.2344,
    -52747.5742, 109054.6562, -28124.9473, -17736.6152, -41327.1562,
    69853.3906, 79046.2656, -3923.7344, -5644.5229, 96586.7578,
    -89315.2656, -146578.0156, -61862.1489, -83956.4375, 87574.5703,
    -75055.0469, 19571.8203, 79358.7891, -16501.5000, -147169.2188,
    -97861.6797, 60442.1797, 40156.9023, 223136.3906, -81118.0547,
    -221443.6406, 54911.6914, 54735.9258, -58805.7305, -168884.4844,
    40865.9609, -28627.9043, -18604.7227, 120274.6172, 49712.2383,
    164402.7031, -53165.0820, -60664.0469, -97956.1484, -121468.4062,
    -69926.1484, -4889.0151, 127367.7344, 200241.0781, -85817.7578,
    -143190.0625, -74049.5312, 137980.5781, -150788.7656, -115719.6719,
    -189250.1250, -153069.7344, -127429.7891, -187588.2500, 125264.7422,
    -79082.3438, -114144.5781, 36033.5039, -57502.2188, 80488.1562,
    36501.4570, -138817.5938, -22189.6523, -222146.9688, -73292.3984,
    127717.2422, -183836.3750, -105907.0859, 145422.8750, 66981.2031,
    -9596.6699, 78099.4922, 70226.3359, 35841.8789, -116117.6016,
    -150986.0156, 81622.4922, 113575.0625, 154419.4844, 53586.4142,
    118494.8750, 131625.4375, -19763.1094, 75581.1172, -42750.5039,
    97934.8281, 6706.7949, -101179.0078, 83519.6172, -83054.8359,
    -56749.2578, -30683.6992, 54615.9492, 84061.1406, -229136.7188,
    -60554.0000, 8120.2622, -106468.7891, -28316.3418, -166351.3125,
    47797.3984, 96013.4141, 71482.9453, -101429.9297, 209063.3594,
    -3033.6882, -38952.5352, -84920.6719, -5895.1543, -18641.8105,
    47884.3633, -14620.0273, -132898.6719, -40903.5859, 197217.3750,
    -128599.1328, -115397.8906, -22670.7676, -78569.9688, -54559.7070,
    -106855.2031, 40703.1484, 55568.3168, 60202.9844, -64757.9375,
    -32068.8652, 160663.3438, 72187.0703, -148519.5469, 162952.8906,
    -128048.2031, -136153.8906, -15270.3730, -52766.3281, -52517.4531,
    18652.1992, 195354.2188, -136657.3750, -8034.2622, -92699.6016,
    -129169.1406, 188479.9844, 46003.7500, -93383.0781, -67831.6484,
    -66710.5469, 104338.5234, 85878.8438, -73165.2031, 95857.3203,
    71213.1250, 94603.1094, -30359.8125, -107989.2578, 99822.1719,
    184626.3594, 79238.4531, -272978.9375, -137948.5781, -145245.8125,
    75359.2031, 26652.7930, 50421.4141, 60784.4102, -18286.3398,
    -182851.9531, -87178.7969, -13131.7539, 195674.8906, 59951.7852,
    124353.7422, -36709.1758, -54575.4766, 77822.6953, 43697.4102,
    -64394.3438, 113281.1797, -93987.0703, 221989.7188, 132902.5000,
    -9538.8574, -14594.1338, 65084.9453, -12501.7227, 130330.6875,
    -115123.4766, 20823.0898, 75512.4922, -75255.7422, -41936.7656,
    -186678.8281, -166799.9375, 138770.6250, -78969.9531, 124516.8047,
    -85558.5781, -69272.4375, -115539.1094, 228774.4844, -76529.3281,
    -107735.8906, -76798.8906, -194335.2812, 56530.5742, -9397.7529,
    132985.8281, 163929.8438, -188517.7969, -141155.6406, 45071.0391,
    207788.3125, -125826.1172, 8965.3320, -159584.8438, 95842.4609,
    -76929.4688
]


def load_old_model():
  """老方式：完整 C2PGen"""
  G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0])
  state = torch.load(str(DOWNLOADS / "160_net_G_A.pth"), map_location='cpu')
  if list(state.keys())[0].startswith('module.'):
    state = {k.replace('module.', ''): v for k, v in state.items()}
  G_A_net.load_state_dict(state)
  G_A_net.cpu()
  G_A_net.eval()
  return G_A_net


def load_new_model():
  """新方式：推理专用"""
  import torch.nn as nn
  class _PixelNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.RGBEnc = RGBEncoder(3, 64, 2, 4, "in", 'relu', 'reflect')
      self.RGBDec = RGBDecoder(256, 3, 2, 4, res_norm='adain', activ='relu', pad_type='reflect')

  net = _PixelNet()
  state = torch.load(str(DOWNLOADS / "inference_net.pth"), map_location='cpu')
  net.load_state_dict(state)
  net.eval()
  return net


def make_test_tensor():
  """创建一个简单的测试输入"""
  torch.manual_seed(42)
  return torch.randn(1, 3, 256, 256)


def compare_dict(name_old, dict_old, name_new, dict_new):
  """对比两个 state dict"""
  all_match = True
  for k in dict_old:
    if k in dict_new:
      if not torch.equal(dict_old[k], dict_new[k]):
        diff = (dict_old[k] - dict_new[k]).abs()
        print(f"  MISMATCH {k}: max_diff={diff.max().item():.8f}")
        all_match = False
    else:
      print(f"  MISSING in new: {k}")
      all_match = False
  if all_match:
    print(f"  All keys match!")


def main():
  print("=== 1. 对比权重 ===")
  old = load_old_model()
  new = load_new_model()

  print("RGBEnc weights:")
  old_enc = {k.replace('RGBEnc.', ''): v for k, v in old.state_dict().items() if k.startswith('RGBEnc.')}
  new_enc = {k.replace('RGBEnc.', ''): v for k, v in new.state_dict().items() if k.startswith('RGBEnc.')}
  compare_dict("old", old_enc, "new", new_enc)

  print("RGBDec weights:")
  old_dec = {k.replace('RGBDec.', ''): v for k, v in old.state_dict().items() if k.startswith('RGBDec.')}
  new_dec = {k.replace('RGBDec.', ''): v for k, v in new.state_dict().items() if k.startswith('RGBDec.')}
  compare_dict("old", old_dec, "new", new_dec)

  print("\n=== 2. 对比 cell_size_code ===")
  code = torch.tensor(MLP_code).reshape(1, 256, 1, 1)
  old_code = old.MLP(code)
  new_code = torch.load(str(DOWNLOADS / "cell_size_code.pt"), map_location='cpu')
  code_diff = (old_code - new_code).abs()
  print(f"  Max diff: {code_diff.max().item():.8f}")
  print(f"  Mean diff: {code_diff.mean().item():.8f}")
  print(f"  Equal: {torch.equal(old_code, new_code)}")

  print("\n=== 3. 对比推理中间结果 ===")
  x = make_test_tensor()
  with torch.no_grad():
    old_feat = old.RGBEnc(x)
    new_feat = new.RGBEnc(x)
    feat_diff = (old_feat - new_feat).abs()
    print(f"RGBEnc output diff: max={feat_diff.max().item():.8f}, mean={feat_diff.mean().item():.8f}")

    old_dec_out = old.RGBDec(old_feat, old_code)
    new_dec_out = new.RGBDec(new_feat, new_code)
    dec_diff = (old_dec_out - new_dec_out).abs()
    print(f"RGBDec output diff: max={dec_diff.max().item():.8f}, mean={dec_diff.mean().item():.8f}")


if __name__ == "__main__":
  main()
