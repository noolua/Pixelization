# Pixelization ONNX 迁移备忘

## 项目概述

像素化（Pixelization）是一个基于 CycleGAN 架构的图像风格迁移项目，将普通图片转换为像素画风格。

**推理管线**：`输入图片 → preprocess → RGBEnc → RGBDec(code) → AliasNet → 后处理 → 输出`

## 模型架构

### 三个核心子网络

1. **RGBEncoder** (`G_A_net.RGBEnc`)
   - 标准 Conv 下采样：3ch → 64 → 128 → 256，2次 stride=2 下采样
   - 4个 ResBlock（256ch）
   - 输出：256 通道 feature map

2. **RGBDecoder** (`G_A_net.RGBDec`)
   - 8个 `ModulationConvBlock`（StyleGAN2 风格的动态权重调制）
   - 接受 `cell_size_code`（shape: [1, 2048]）控制风格
   - 2次 Upsample + Conv 上采样：256 → 128 → 64 → 3ch
   - 输出 tanh 激活（值域 [-1, 1]）

3. **AliasNet** (`alias_net`)
   - 抗锯齿后处理网络
   - 结构与 RGBEnc/Dec 类似但更简单，标准 Conv + ResBlock
   - 输入输出都是 3ch

### 不参与推理的模块（仅训练用）

- `PBEnc`（PixelBlockEncoder）：VGG19 特征提取 + 卷积，生成 style code
- `MLP`：将 PBEnc 输出的 style code 转换为 cell_size_code
- `G_B_net`（P2C Generator）、判别器等

### cell_size_code 的来源

MLP_code 是一组硬编码的常量（256个浮点数），通过 MLP 网络映射为 [1, 2048] 的 code。
MLP_code 不随输入变化，在模型加载时预计算一次，已烘焙到 ONNX 模型中。

## ONNX 导出的关键技术点

### ModulationConvBlock 原始实现的问题

原始实现使用 StyleGAN2 的 fused modulation 技巧：
```python
# 动态 reshape weight 为 (batch, k, k, in_c, out_c)
# 用 code 调制后 demodulate
# 再 reshape 回 (batch*out_c, in_c, k, k)
# 最后 F.conv2d(groups=batch) 实现 per-sample 不同权重
```

ONNX 不支持：
- `groups` 作为动态值（必须是常量属性）
- 权重 shape 在 trace 时不可静态推导

### ONNXModulationConvBlock 等价简化（batch=1）

batch=1 时 `groups=1`，等效为标准卷积：
```python
weight = self.weight * self.wscale          # (out_c, in_c, k, k)
weight = weight * code.view(1, in_c, 1, 1)  # 按通道调制
weight = weight / sqrt(sum(weight^2, dim=[1,2,3], keepdim=True) + eps)  # demodulate
x = F.conv2d(x, weight, groups=1)           # 标准卷积
```

验证精度：Max diff 6.3e-4，来源于 demodulate 步骤中求和轴顺序的浮点差异。

### RGBDecoder 中 mod_conv 的复用

原代码中 `mod_conv_1` 用一次，`mod_conv_2` 重复使用 7 次（共享权重但不同 code）。
ONNX 版本拆为 `mod_conv_1` 到 `mod_conv_8` 共 8 个独立模块，各自持有权重副本。
实际上 `mod_conv_3..8` 的权重与 `mod_conv_2` 相同。

## 图像预处理/后处理（Go 重写需要）

### 预处理（`test_pro.py:process`）
```
1. resize: (w, h) → ((w//cell_size)*4, (h//cell_size)*4)，BICUBIC
2. center crop: 确保宽高为 4 的倍数
3. ToTensor: [0,255] → [0,1]
4. Normalize: (x - 0.5) / 0.5 → [-1, 1]
5. 添加 batch 维度: (3, H, W) → (1, 3, H, W)
```

### 后处理（`test_pro.py:save`）
```
1. 去归一化: (x + 1) / 2 * 255 → [0, 255] uint8
2. 缩小到像素网格: resize(H/4, W/4), NEAREST
3. 放大到目标尺寸: resize(H/4*cell_size, W/4*cell_size), NEAREST
4. 保存为 PNG
```

### 后处理-原始尺寸版（`test_pro.py:save_original_size`）
```
1. 去归一化同上
2. 只缩小: resize(H/4, W/4), NEAREST
3. 不放大，直接保存
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `tests/export_onnx.py` | ONNX 导出脚本，含 ONNXModulationConvBlock |
| `tests/verify_onnx.py` | PyTorch vs ONNX Runtime 一致性验证 |
| `tests/onnx/pixelization.onnx` | 导出的 ONNX 模型 |
| `do_task.md` | 任务进度跟踪 |
| `models/basic_layer.py` | ModulationConvBlock 原始实现 |
| `models/c2pGen.py` | C2PGen, RGBEncoder, RGBDecoder, PixelBlockEncoder |
| `models/networks.py` | define_G / define_D 模型工厂 |
| `test_pro.py` | 原始 Python 推理管线 + Model 类 |
| `api.py` | FastAPI Web 服务（当前 Python 版） |

## Go 重写注意事项

- ONNX 模型输入 shape: `(1, 3, H, W)` float32，H/W 必须是 4 的倍数
- ONNX 模型输出 shape: `(1, 3, H, W)` float32，值域 [-1, 1]
- batch 固定为 1
- 推荐库：`github.com/yalue/onnxruntime_go`（纯 Go 绑定，活跃维护）
- GPU 加速：CUDA EP（Linux）、CoreML EP（macOS）
