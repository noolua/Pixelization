"""
Convert FP32 ONNX model to FP16.

Tries multiple strategies in order:
  1. onnxconverter_common (default)
  2. onnxconverter_common + disable_shape_infer
  3. onnxconverter_common + block InstanceNorm/Resize
  4. onnxconverter_common + block InstanceNorm/Resize + disable_shape_infer
  5. Manual weight-only FP16 (initializers → FP16, Cast → FP32 for each use)

Strategy 5 is a fallback: model runs FP32 ops but weights are stored as FP16,
so model file is ~50% smaller. No compute acceleration, but smaller distribution.

Usage:
    pip install onnxconverter-common
    python tools/convert_fp16.py [--input PATH] [--output PATH]
"""

import sys
import os
import argparse
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def validate_model(path):
    """Validate model by loading with ONNX Runtime. Returns True if OK."""
    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        # Quick smoke test with fixed-size input (dynamic dims may have 0)
        info = sess.get_inputs()[0]
        # Use a small fixed size for smoke test
        dummy = np.random.randn(1, 3, 64, 64).astype(np.float32)
        sess.run(None, {info.name: dummy})
        del sess
        return True
    except Exception as e:
        print(f"    Validation failed: {e}")
        return False


def print_stats(path):
    """Print model statistics."""
    size_mb = os.path.getsize(path) / 1024 / 1024
    model = onnx.load(path)
    op_counts = Counter(node.op_type for node in model.graph.node)
    cast_count = op_counts.get("Cast", 0)
    print(f"    Size: {size_mb:.1f} MB, Nodes: {len(model.graph.node)}, Cast nodes: {cast_count}")
    print(f"    Top ops: {op_counts.most_common(5)}")


def try_onnxconverter(model, output_path, **kwargs):
    """Try conversion with onnxconverter_common."""
    from onnxconverter_common import float16

    desc = kwargs.pop("desc", "default")
    print(f"\n  Strategy: onnxconverter_common ({desc})...")

    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True, **kwargs)
    onnx.save(model_fp16, output_path)
    print_stats(output_path)

    if validate_model(output_path):
        print("    ✓ Model validates OK")
        return True
    return False


def convert_manual(model_path, output_path):
    """
    Manual weight-only FP16 conversion.

    Converts all float initializers to FP16, then inserts Cast(FP16→FP32)
    before each use so ops still receive FP32 data. Graph I/O stays FP32.

    Result: model file is ~50% smaller, but ops run in FP32 (no compute speedup).
    """
    print("\n  Strategy: manual weight-only FP16...")
    model = onnx.load(model_path)
    graph = model.graph

    # Collect initializer names
    init_names = {init.name for init in graph.initializer}

    # Convert float initializers to FP16, skipping those that overflow
    FP16_MAX = 65504.0
    converted = 0
    skipped = 0
    fp16_init_names = set()  # names of initializers actually converted to FP16

    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if np.any(np.abs(arr) > FP16_MAX):
                skipped += 1
                continue  # keep in FP32
            fp16_arr = arr.astype(np.float16)
            new_init = numpy_helper.from_array(fp16_arr, name=init.name)
            init.CopyFrom(new_init)
            fp16_init_names.add(init.name)
            converted += 1

    print(f"    Converted {converted} initializers to FP16, {skipped} kept in FP32 (overflow)")
    # Only FP16 initializers need Cast nodes
    init_names = fp16_init_names

    # For each unique initializer reference, create ONE Cast(FP16→FP32) node.
    # Use dict to deduplicate: init_name → cast_output_name
    cast_map = {}
    cast_nodes = []
    existing_names = {n.name for n in graph.node}

    for node in graph.node:
        for inp_name in node.input:
            if inp_name in init_names and inp_name not in cast_map and inp_name != "":
                cast_output = f"{inp_name}_fp32"
                cast_name = f"Cast_{inp_name}_to_fp32"
                # Ensure unique
                suffix = 0
                base_name = cast_name
                while cast_name in existing_names:
                    suffix += 1
                    cast_name = f"{base_name}_{suffix}"
                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[inp_name],
                    outputs=[cast_output],
                    to=TensorProto.FLOAT,
                    name=cast_name,
                )
                cast_nodes.append(cast_node)
                existing_names.add(cast_name)
                cast_map[inp_name] = cast_output

    # Update all node inputs: replace initializer references with cast outputs
    for node in graph.node:
        new_inputs = []
        for inp_name in node.input:
            if inp_name in cast_map:
                new_inputs.append(cast_map[inp_name])
            else:
                new_inputs.append(inp_name)
        del node.input[:]
        node.input.extend(new_inputs)

    # Prepend Cast nodes, then original nodes
    all_nodes = cast_nodes + list(graph.node)
    del graph.node[:]
    graph.node.extend(all_nodes)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(model, output_path)
    print_stats(output_path)

    if validate_model(output_path):
        print("    ✓ Model validates OK")
        return True
    return False


def convert(args):
    print("=== FP16 Conversion ===\n")

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"  ERROR: {input_path} not found")
        sys.exit(1)

    fp32_size = os.path.getsize(input_path) / 1024 / 1024
    print(f"  FP32 model: {input_path} ({fp32_size:.1f} MB)")

    # Analyze FP16 compatibility
    model = onnx.load(input_path)
    large_values = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if np.any(np.abs(arr) > 65504):
                large_values += 1
    if large_values:
        print(f"  WARNING: {large_values} initializers have values > FP16 max (65504)")
        print(f"           These will be clipped — model quality may degrade")

    # Try onnxconverter_common strategies (reload fresh model each time)
    strategies = [
        ("default", {}),
        ("no shape inference", {"disable_shape_infer": True}),
        ("block InstanceNorm+Resize", {
            "op_block_list": {"InstanceNormalization", "Resize"},
            "check_fp16_ready": False,
        }),
        ("block+no_shape_infer", {
            "op_block_list": {"InstanceNormalization", "Resize"},
            "disable_shape_infer": True,
            "check_fp16_ready": False,
        }),
    ]

    for desc, kwargs in strategies:
        try:
            fresh_model = onnx.load(input_path)  # reload to avoid cross-contamination
            if try_onnxconverter(fresh_model, output_path, **kwargs, desc=desc):
                print(f"\n  SUCCESS: {output_path}")
                return
        except Exception as e:
            print(f"    Exception: {e}")

    # Strategy 5: manual weight-only (always works, size reduction only)
    if convert_manual(input_path, output_path):
        print(f"\n  SUCCESS: {output_path}")
        print("  Note: weight-only FP16 — ops still FP32, but model is smaller")
        return

    print(f"\n  FAILED: All strategies failed")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FP32 ONNX model to FP16")
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "onnx", "pixelization.onnx"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "onnx", "pixelization_fp16.onnx"))
    args = parser.parse_args()
    convert(args)
