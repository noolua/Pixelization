# ONNX FP16 导出指南

## 背景

原始论文提供的预训练模型为 PyTorch `.pth` 格式，推理依赖 PyTorch 运行时。
我们的目标是获得一个独立的 **FP16 ONNX 模型**，便于在各类推理引擎（onnxruntime、CoreML 等）上部署，同时将模型体积减半。

直接转换遇到三个障碍：

1. **ModulationConvBlock 不兼容 ONNX** — 原始实现使用动态权重 + `F.conv2d(groups=batch)`，ONNX 无法追踪
2. **FP16 数值溢出** — demodulation 路径的 `eps=1e-8` 低于 FP16 精度下限（~6e-8），导致下溢
3. **Upsample/Resize 不接受 FP16** — `nn.Upsample` 的 scale 常量转 FP16 后，ONNX Resize 算子报类型错误

## 解决方案

`tools/export_onnx.py` 通过以下方式逐一解决：

| 问题 | 解决方式 |
|------|---------|
| ModulationConvBlock | `ONNXModulationConvBlock`：batch=1 简化，标准 conv2d 替代 grouped conv2d |
| eps 下溢 | eps 提升为 `max(orig.eps, 1e-3)`，用 `+ eps` 替代 `torch.clamp` |
| Upsample FP16 | `ONNNUpsample`：确保 scale 常量类型一致 |
| FP16 权重转换 | `--fp16` 参数：`pipeline.half()` 后直接导出，避免后处理转换的类型冲突 |

## 使用方法

### 前置条件

模型文件（从项目 README 的 Google Drive 下载）：

```
downloads/
  ├── inference_net.pth    # RGBEncoder + RGBDecoder 推理权重
  ├── alias_net.pth        # AliasNet 抗锯齿网络
  └── cell_size_code.pt    # 预计算的 cell size 编码
```

安装依赖：

```bash
pip install torch torchvision onnx onnxruntime
```

### 导出 FP32 ONNX

```bash
python tools/export_onnx.py --model-dir downloads
# 输出: tools/onnx/pixelization.onnx (~76 MB)
```

### 导出 FP16 ONNX

```bash
python tools/export_onnx.py --fp16 --model-dir downloads
# 输出: tools/onnx/pixelization_fp16.onnx (~38 MB)
```

### 验证推理

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("tools/onnx/pixelization_fp16.onnx",
                            providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
dummy = np.random.randn(1, 3, 256, 256).astype(np.float16)
out = sess.run(None, {inp.name: dummy})
print(f"Output: shape={out[0].shape}, dtype={out[0].dtype}")
```

## 精度对比

使用相同输入对比 FP32 和 FP16 输出：

| 指标 | 值 |
|------|------|
| Max diff | 0.124 |
| Mean diff | 0.006 |
| MSE | 0.000052 |

差异在可接受范围内，无需重新训练。如需进一步优化精度，可使用 AMP 混合精度重新训练后再导出。

## 可选：后处理 FP16 转换

如果需要从 FP32 ONNX 转换（而非直接导出），可使用 `tools/convert_fp16.py`：

```bash
python tools/convert_fp16.py --input tools/onnx/pixelization.onnx
```

该脚本尝试多种转换策略（onnxconverter_common），若全部失败则回退到 weight-only FP16（权重 FP16 存储但运算仍为 FP32）。**推荐优先使用 `--fp16` 直接导出。**

## 两条部署路径

```
路径 A（推荐）：预训练模型 → export_onnx.py --fp16 → FP16 ONNX
路径 B（可选）：自定义训练 → export_onnx.py --fp16 → FP16 ONNX
```

两条路径共用同一个导出脚本，`--fp16` 标志与训练方式无关。
