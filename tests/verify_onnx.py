"""
Verify ONNX model output matches PyTorch output.

Compares inference results between:
  1. PyTorch (ONNX-friendly pipeline — same model used for export)
  2. ONNX Runtime (CPU)

Usage:
    python tests/verify_onnx.py [--model-dir DIR] [--onnx-path PATH] [--image PATH]
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Re-use the ONNX-friendly pipeline from export_onnx
from export_onnx import PixelizationPipeline, load_weights, MLP_CODE
from models.networks import define_G


def preprocess(img, size=None):
    """Preprocess image: center-crop to multiple of 4, normalize to [-1, 1]."""
    if size:
        img = img.resize(size, Image.BICUBIC)

    ow, oh = img.size
    nw = int(round(ow / 4) * 4)
    nh = int(round(oh / 4) * 4)
    left = (ow - nw) // 2
    top = (oh - nh) // 2
    img = img.crop((left, top, left + nw, top + nh))

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return trans(img)[None, :, :, :]


def build_pipeline(model_dir):
    """Build the same ONNX-friendly pipeline used for export."""
    G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [])
    alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [])

    load_weights(G_A_net, torch.load(os.path.join(model_dir, "160_net_G_A.pth"), map_location="cpu"))
    load_weights(alias_net, torch.load(os.path.join(model_dir, "alias_net.pth"), map_location="cpu"))

    vgg_state = torch.load(os.path.join(model_dir, "pixelart_vgg19.pth"), map_location="cpu")
    G_A_net.PBEnc.vgg.load_state_dict(
        {k.replace("features.", ""): v for k, v in vgg_state.items() if k.startswith("features.")}
    )

    with torch.no_grad():
        code = torch.tensor(MLP_CODE).reshape(1, 256, 1, 1)
        cell_size_code = G_A_net.MLP(code)

    pipeline = PixelizationPipeline(G_A_net, alias_net, cell_size_code)
    pipeline.eval()
    return pipeline


def run_pytorch(input_tensor, pipeline):
    with torch.no_grad():
        output = pipeline(input_tensor)
    return output.numpy()


def run_onnx(input_tensor, onnx_path):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    result = session.run(["output"], {"image": input_tensor.numpy()})
    return result[0]


def compare_outputs(pt_output, onnx_output, label=""):
    diff = np.abs(pt_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = max_diff / (np.max(np.abs(pt_output)) + 1e-8)

    print(f"\n  {label}")
    print(f"    Shape:     PyTorch={pt_output.shape}, ONNX={onnx_output.shape}")
    print(f"    Max diff:  {max_diff:.6e}")
    print(f"    Mean diff: {mean_diff:.6e}")
    print(f"    Rel diff:  {rel_diff:.6e}")

    if max_diff < 1e-4:
        print(f"    PASS (diff < 1e-4)")
    elif max_diff < 1e-3:
        print(f"    ACCEPTABLE (diff < 1e-3)")
    else:
        print(f"    WARNING: large difference!")

    return max_diff


def verify(args):
    print("=== ONNX Verification ===\n")

    print("Building PyTorch pipeline...")
    pipeline = build_pipeline(args.model_dir)

    test_configs = [
        ("Random 256x256", None),
    ]

    if args.image and os.path.exists(args.image):
        img = Image.open(args.image).convert("RGB")
        img = img.resize((256, 256), Image.BICUBIC)
        test_configs.append(("Real image 256x256", img))

    all_passed = True

    for label, img_or_none in test_configs:
        print(f"\n[Test] {label}")

        if img_or_none is not None:
            input_tensor = preprocess(img_or_none)
        else:
            input_tensor = torch.randn(1, 3, 256, 256)

        print(f"  Input shape: {input_tensor.shape}")

        print("  Running PyTorch inference...")
        pt_output = run_pytorch(input_tensor, pipeline)

        print("  Running ONNX Runtime inference...")
        onnx_output = run_onnx(input_tensor, args.onnx_path)

        max_diff = compare_outputs(pt_output, onnx_output, label=label)
        if max_diff >= 1e-3:
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - check warnings above")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ONNX model vs PyTorch")
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "downloads"),
        help="Directory containing .pth model files",
    )
    parser.add_argument(
        "--onnx-path",
        default=os.path.join(os.path.dirname(__file__), "onnx", "pixelization.onnx"),
        help="Path to exported ONNX model",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional test image path",
    )
    args = parser.parse_args()
    verify(args)
