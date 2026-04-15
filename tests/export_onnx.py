"""
Export Pixelization models to ONNX format.

Combines RGBEncoder + RGBDecoder + AliasNet into a single ONNX model,
with pre-computed MLP cell_size_code baked in as a constant.

The original ModulationConvBlock uses dynamic weight reshaping and
F.conv2d(groups=batch) which is incompatible with ONNX export.
This script provides an ONNX-friendly equivalent that is mathematically
identical for batch=1 inference.

Usage:
    python tests/export_onnx.py [--model-dir DIR] [--output-dir DIR]
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.networks import define_G

# Same MLP code from test_pro.py
MLP_CODE = [
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
    -89315.2656, -146578.0156, -61862.1484, -83956.4375, 87574.5703,
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
    -150986.0156, 81622.4922, 113575.0625, 154419.4844, 53586.4141,
    118494.8750, 131625.4375, -19763.1094, 75581.1172, -42750.5039,
    97934.8281, 6706.7949, -101179.0078, 83519.6172, -83054.8359,
    -56749.2578, -30683.6992, 54615.9492, 84061.1406, -229136.7188,
    -60554.0000, 8120.2622, -106468.7891, -28316.3418, -166351.3125,
    47797.3984, 96013.4141, 71482.9453, -101429.9297, 209063.3594,
    -3033.6882, -38952.5352, -84920.6715, -5895.1543, -18641.8105,
    47884.3633, -14620.0273, -132898.6719, -40903.5859, 197217.3750,
    -128599.1328, -115397.8906, -22670.7676, -78569.9688, -54559.7070,
    -106855.2031, 40703.1484, 55568.3164, 60202.9844, -64757.9375,
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


class ONNXModulationConvBlock(nn.Module):
    """
    ONNX-exportable equivalent of ModulationConvBlock for batch=1.

    The original uses dynamic weight reshaping and F.conv2d(groups=batch),
    which is incompatible with ONNX. For batch=1 this simplifies to a
    standard conv2d with modulated/demodulated weights — mathematically
    identical.
    """

    def __init__(self, orig):
        super().__init__()
        self.in_c = orig.in_c
        self.out_c = orig.out_c
        self.ksize = orig.ksize
        self.stride = orig.stride
        self.padding = orig.padding
        self.eps = orig.eps
        self.wscale = float(orig.wscale)
        self.activate_scale = float(orig.activate_scale)
        self.weight = nn.Parameter(orig.weight.data.clone())
        self.bias = nn.Parameter(orig.bias.data.clone())

    def forward(self, x, code):
        # code: (1, in_c) — a slice of cell_size_code
        weight = self.weight * self.wscale          # (out_c, in_c, k, k)

        # Replicate original 5D view modulation exactly.
        # view(1, k, k, in_c, out_c) changes the element grouping along the
        # in_c axis compared to the original 4D layout (out_c, in_c, k, k),
        # so a simple 4D broadcast would multiply *different* elements by code.
        _weight = weight.view(1, self.ksize, self.ksize, self.in_c, self.out_c)
        _weight = _weight * code.view(1, 1, 1, self.in_c, 1)

        # Demodulate (sum over dims 1,2,3 = k, k, in_c)
        _weight_norm = torch.sqrt(
            torch.sum(_weight ** 2, dim=[1, 2, 3]) + self.eps
        )
        _weight = _weight / _weight_norm.view(1, 1, 1, 1, self.out_c)

        # Reshape back to 4D for conv2d (batch=1 simplification)
        weight = _weight.permute(1, 2, 3, 0, 4).reshape(
            self.ksize, self.ksize, self.in_c, 1 * self.out_c)
        weight = weight.permute(3, 2, 0, 1)          # (out_c, in_c, k, k)

        x = F.conv2d(x, weight=weight, bias=None,
                      stride=self.stride, padding=self.padding)
        x = x + self.bias.view(1, -1, 1, 1)
        x = F.leaky_relu(x, negative_slope=0.2) * self.activate_scale
        return x


class ONNXRGBDecoder(nn.Module):
    """ONNX-exportable RGBDecoder using ONNXModulationConvBlock."""

    def __init__(self, orig):
        super().__init__()
        self.mod_conv_1 = ONNXModulationConvBlock(orig.mod_conv_1)
        self.mod_conv_2 = ONNXModulationConvBlock(orig.mod_conv_2)
        # mod_conv_3..8: original RGBDecoder.forward reuses mod_conv_2 for all
        # subsequent calls (mod_conv_3..8 exist in __init__ but are never used
        # in forward — they hold random untrained weights)
        self.mod_conv_3 = ONNXModulationConvBlock(orig.mod_conv_2)
        self.mod_conv_4 = ONNXModulationConvBlock(orig.mod_conv_2)
        self.mod_conv_5 = ONNXModulationConvBlock(orig.mod_conv_2)
        self.mod_conv_6 = ONNXModulationConvBlock(orig.mod_conv_2)
        self.mod_conv_7 = ONNXModulationConvBlock(orig.mod_conv_2)
        self.mod_conv_8 = ONNXModulationConvBlock(orig.mod_conv_2)

        self.upsample_block1 = orig.upsample_block1
        self.conv_1 = orig.conv_1
        self.upsample_block2 = orig.upsample_block2
        self.conv_2 = orig.conv_2
        self.conv_3 = orig.conv_3

    def forward(self, x, code):
        # Same structure as original RGBDecoder.forward
        residual = x
        x = self.mod_conv_1(x, code[:, :256])
        x = self.mod_conv_2(x, code[:, 256:512])
        x = x + residual
        residual = x
        x = self.mod_conv_3(x, code[:, 512:768])
        x = self.mod_conv_4(x, code[:, 768:1024])
        x = x + residual
        residual = x
        x = self.mod_conv_5(x, code[:, 1024:1280])
        x = self.mod_conv_6(x, code[:, 1280:1536])
        x = x + residual
        residual = x
        x = self.mod_conv_7(x, code[:, 1536:1792])
        x = self.mod_conv_8(x, code[:, 1792:2048])
        x = x + residual
        x = self.upsample_block1(x)
        x = self.conv_1(x)
        x = self.upsample_block2(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class PixelizationPipeline(nn.Module):
    """Combined pipeline: RGBEnc -> RGBDec(ONNX) -> AliasNet."""

    def __init__(self, G_A_net, alias_net, cell_size_code):
        super().__init__()
        self.rgb_enc = G_A_net.RGBEnc
        self.rgb_dec = ONNXRGBDecoder(G_A_net.RGBDec)
        self.alias_net = alias_net
        self.register_buffer("cell_size_code", cell_size_code)

    def forward(self, x):
        feature = self.rgb_enc(x)
        images = self.rgb_dec(feature, self.cell_size_code)
        out = self.alias_net(images)
        return out


def load_weights(net, state_dict):
    """Load state dict, stripping 'module.' prefix from DataParallel."""
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)


def export(args):
    print("=== ONNX Export for Pixelization ===\n")

    # 1. Build models
    print("[1/4] Building PyTorch models...")
    G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [])
    alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [])

    # 2. Load weights
    print("[2/4] Loading weights...")
    model_dir = args.model_dir

    ga_path = os.path.join(model_dir, "160_net_G_A.pth")
    alias_path = os.path.join(model_dir, "alias_net.pth")
    vgg_path = os.path.join(model_dir, "pixelart_vgg19.pth")

    for p in [ga_path, alias_path, vgg_path]:
        if not os.path.exists(p):
            print(f"  ERROR: Missing model file: {p}")
            sys.exit(1)

    load_weights(G_A_net, torch.load(ga_path, map_location="cpu"))
    load_weights(alias_net, torch.load(alias_path, map_location="cpu"))

    # The PixelBlockEncoder inside C2PGen loads VGG separately
    vgg_state = torch.load(vgg_path, map_location="cpu")
    G_A_net.PBEnc.vgg.load_state_dict(
        {k.replace("features.", ""): v for k, v in vgg_state.items() if k.startswith("features.")}
    )

    # 3. Pre-compute cell_size_code and build combined pipeline
    print("[3/4] Building ONNX-friendly pipeline...")
    with torch.no_grad():
        code = torch.tensor(MLP_CODE).reshape(1, 256, 1, 1)
        cell_size_code = G_A_net.MLP(code)

    pipeline = PixelizationPipeline(G_A_net, alias_net, cell_size_code)
    pipeline.eval()

    # 4. Export to ONNX
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, "pixelization.onnx")

    print(f"[4/4] Exporting to {onnx_path}...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 256)

        torch.onnx.export(
            pipeline,
            dummy_input,
            onnx_path,
            dynamo=False,
            opset_version=17,
            input_names=["image"],
            output_names=["output"],
            dynamic_axes={
                "image": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
        )

    # Verify the exported model
    import onnx

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"\n  Model size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")
    print(f"  Inputs:  {[inp.name for inp in model.graph.input]}")
    print(f"  Outputs: {[out.name for out in model.graph.output]}")
    print(f"\n  DONE: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export pixelization models to ONNX")
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "downloads"),
        help="Directory containing .pth model files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "onnx"),
        help="Directory to save ONNX model",
    )
    args = parser.parse_args()
    export(args)
