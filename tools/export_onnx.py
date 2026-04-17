"""
Export Pixelization models to ONNX format.

Combines RGBEncoder + RGBDecoder + AliasNet into a single ONNX model,
with pre-computed cell_size_code baked in as a constant.

The original ModulationConvBlock uses dynamic weight reshaping and
F.conv2d(groups=batch) which is incompatible with ONNX export.
This script provides an ONNX-friendly equivalent that is mathematically
identical for batch=1 inference.

Usage:
    python tools/export_onnx.py [--model-dir DIR] [--output-dir DIR]
"""

import sys
import os
import argparse

# Add web/ to path for models import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.c2pGen import RGBEncoder, RGBDecoder, AliasNet


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

    def __init__(self, rgb_enc, rgb_dec, alias_net, cell_size_code):
        super().__init__()
        self.rgb_enc = rgb_enc
        self.rgb_dec = ONNXRGBDecoder(rgb_dec)
        self.alias_net = alias_net
        self.register_buffer("cell_size_code", cell_size_code)

    def forward(self, x):
        feature = self.rgb_enc(x)
        images = self.rgb_dec(feature, self.cell_size_code)
        out = self.alias_net(images)
        return out


def export(args):
    print("=== ONNX Export for Pixelization ===\n")

    model_dir = args.model_dir

    # 1. Load inference weights
    print("[1/4] Loading inference weights...")
    import torch.nn as nn

    class _PixelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.RGBEnc = RGBEncoder(3, 64, 2, 4, "in", 'relu', 'reflect')
            self.RGBDec = RGBDecoder(256, 3, 2, 4, res_norm='adain', activ='relu', pad_type='reflect')

    net = _PixelNet()
    state = torch.load(os.path.join(model_dir, "inference_net.pth"), map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    alias_net = AliasNet(3, 3, 64, 2, 3, activ='relu', pad_type='reflect')
    alias_state = torch.load(os.path.join(model_dir, "alias_net.pth"), map_location="cpu")
    if list(alias_state.keys())[0].startswith('module.'):
        alias_state = {k.replace('module.', ''): v for k, v in alias_state.items()}
    alias_net.load_state_dict(alias_state)
    alias_net.eval()

    # 2. Load and normalize cell_size_code for FP16 safety
    print("[2/4] Loading cell_size_code...")
    cell_size_code = torch.load(os.path.join(model_dir, "cell_size_code.pt"), map_location="cpu")
    # Normalize each 256-element slice to unit norm.
    # The modulation path does: w_mod = w * code, then normalizes by |w_mod|.
    # Since only the *direction* of code matters (not magnitude), normalizing
    # code to unit norm is mathematically equivalent (eps difference < 1e-10).
    for i in range(0, cell_size_code.shape[1], 256):
        s = cell_size_code[:, i:i+256]
        cell_size_code[:, i:i+256] = s / torch.sqrt(torch.sum(s ** 2))

    # 3. Build combined pipeline
    print("[3/4] Building ONNX-friendly pipeline...")
    pipeline = PixelizationPipeline(net.RGBEnc, net.RGBDec, alias_net, cell_size_code)
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
