"""
Task 2: Multi-size & Performance Benchmark for ONNX model.

Tests:
  1. Correctness across resolutions (128, 256, 512, 1024, non-square)
  2. PyTorch vs ONNX Runtime CPU speed comparison
  3. Available Execution Providers (CUDA, CoreML, etc.)

Usage:
    python tests/benchmark_onnx.py [--model-dir DIR] [--onnx-path PATH] [--warmup N] [--runs N]
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from export_onnx import PixelizationPipeline, load_weights, MLP_CODE
from models.networks import define_G


# ── Helpers ──────────────────────────────────────────────────────────

def build_pipeline(model_dir):
    """Build ONNX-friendly PyTorch pipeline."""
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


def make_input(h, w):
    """Create random input tensor, ensure H/W are multiples of 4."""
    h = (h // 4) * 4
    w = (w // 4) * 4
    return torch.randn(1, 3, h, w)


# ── Part 1: Multi-size correctness ──────────────────────────────────

def test_multisize(pipeline, onnx_path):
    import onnxruntime as ort

    print("=" * 60)
    print("Part 1: Multi-size Correctness (PyTorch vs ONNX Runtime)")
    print("=" * 60)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    sizes = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (320, 480),   # non-square
        (640, 360),   # 16:9 landscape
    ]

    all_passed = True

    for h, w in sizes:
        inp = make_input(h, w)
        h, w = inp.shape[2], inp.shape[3]
        label = f"{h}x{w}"

        # PyTorch
        with torch.no_grad():
            pt_out = pipeline(inp).numpy()

        # ONNX Runtime
        ort_out = session.run(["output"], {"image": inp.numpy()})[0]

        diff = np.abs(pt_out - ort_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        status = "PASS" if max_diff < 1e-3 else "FAIL"
        if max_diff >= 1e-3:
            all_passed = False

        print(f"  {label:>12s}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  {status}")

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# ── Part 2: Speed benchmark ─────────────────────────────────────────

def benchmark(pipeline, onnx_path, warmup, runs):
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Part 2: Speed Benchmark (PyTorch CPU vs ONNX Runtime CPU)")
    print("=" * 60)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    sizes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (640, 480),
    ]

    print(f"  Warmup: {warmup} runs, Benchmark: {runs} runs\n")
    print(f"  {'Size':>12s}  {'PyTorch':>10s}  {'ONNX CPU':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")

    for h, w in sizes:
        inp = make_input(h, w)
        np_input = inp.numpy()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = pipeline(inp)
            _ = session.run(["output"], {"image": np_input})

        # PyTorch benchmark
        t0 = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                _ = pipeline(inp)
        pt_time = (time.perf_counter() - t0) / runs * 1000

        # ONNX Runtime benchmark
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = session.run(["output"], {"image": np_input})
        ort_time = (time.perf_counter() - t0) / runs * 1000

        speedup = pt_time / ort_time
        print(f"  {f'{h}x{w}':>12s}  {pt_time:>8.1f}ms  {ort_time:>8.1f}ms  {speedup:>7.2f}x")


# ── Part 3: Execution Provider detection ─────────────────────────────

def test_providers(onnx_path):
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Part 3: Execution Provider Availability")
    print("=" * 60)

    available = ort.get_available_providers()
    print(f"  Installed providers: {available}")

    ep_list = [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "MPSExecutionProvider",
    ]

    for ep in ep_list:
        if ep in available:
            try:
                s = ort.InferenceSession(onnx_path, providers=[ep])
                providers_used = s.get_providers()
                print(f"  {ep:>28s}: OK  (session uses: {providers_used})")
            except Exception as e:
                print(f"  {ep:>28s}: FAILED  ({e})")
        else:
            print(f"  {ep:>28s}: not installed")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ONNX multi-size & performance benchmark")
    parser.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "..", "downloads"))
    parser.add_argument("--onnx-path", default=os.path.join(os.path.dirname(__file__), "onnx", "pixelization.onnx"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    print("Loading PyTorch pipeline...")
    pipeline = build_pipeline(args.model_dir)

    print(f"Loading ONNX model: {args.onnx_path}")
    print(f"  File size: {os.path.getsize(args.onnx_path) / 1024 / 1024:.1f} MB\n")

    test_multisize(pipeline, args.onnx_path)
    benchmark(pipeline, args.onnx_path, args.warmup, args.runs)
    test_providers(args.onnx_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
