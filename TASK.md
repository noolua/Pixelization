# TASK.md

## FP16 ONNX 精度问题（未解决）

### 现象

FP16 ONNX 模型（`pixelization_fp16.onnx`）推理结果与 FP32 模型有明显色差，demodulation 混合精度优化未改善。

### 已排除

- **demodulation 路径**：已改为 FP32 计算 modulation/demodulation，与纯 FP16 版本输出几乎一致，说明该路径数值稳定，不是色差来源。

### 待排查方向

1. **InstanceNorm 方差计算**：`E[x²] - E[x]²` 在 FP16 下灾难性抵消。RGBEncoder 和 AliasNet 大量使用 InstanceNorm，颜色编码/解码都经过这些层，是最可能的色差来源。
2. **AdaIN (AdaptiveInstanceNorm2d)**：RGBDecoder 使用 `res_norm='adain'`，涉及 FP16 下的 batch_norm 计算。
3. **ActivationScale 乘法**：`activate_scale = sqrt(2) ≈ 1.414`，FP16 尾数精度对常量乘法影响较小，但累积 8 层可能放大误差。

### 可能的解决方案（按优先级）

1. **逐层定位色差源头**：在 `export_onnx.py` 中导出各阶段子模型（rgb_enc only / rgb_dec only / alias_net only），用同一张图对比 FP32 和 FP16 各阶段输出，找到色差首次出现的层。这是最优先的一步，避免盲目优化。
2. **InstanceNorm FP32 包装层**：在 `export_onnx.py` 中创建 `ONNXInstanceNorm2d`，输入 cast FP32 → InstanceNorm → cast 回 FP16。RGBEncoder 和 AliasNet 中有大量 InstanceNorm。
3. **AdaIN FP32 包装层**：类似思路包装 `AdaptiveInstanceNorm2d`（在 `basic_layer.py` 中）。
4. **weight-only FP16 保底方案**：用 `convert_fp16.py` 的策略 6（权重 FP16 存储 + Cast FP32 计算），体积减半但精度完全一致。牺牲推理加速，换取零精度损失。

### 相关文件

- `tools/export_onnx.py`：导出脚本，含 `ONNXModulationConvBlock`、`PixelizationPipeline`
- `web/models/basic_layer.py`：`AdaptiveInstanceNorm2d`、`InstanceNorm2d` 实现
- `web/models/c2pGen.py`：`RGBEncoder`、`RGBDecoder`、`AliasNet` 模型定义
- `FOR_ONNX.md`：ONNX 导出文档
