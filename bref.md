# Pixelization ONNX 迁移备忘

## 项目概述

像素化（Pixelization）是一个基于 CycleGAN 架构的图像风格迁移项目，将普通图片转换为像素画风格。

**推理管线**：`输入图片 → preprocess → RGBEnc → RGBDec(code) → AliasNet → 后处理 → 输出`

**当前方案**：PyTorch → ONNX → Go + ONNX Runtime，支持 MPS/CPU/GPU。

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

### cell_size_code

MLP_code 是一组硬编码的常量（256个浮点数），通过 MLP 网络映射为 [1, 2048] 的 code。
MLP_code 不随输入变化，在模型加载时预计算一次，已烘焙到 ONNX 模型中。

**重要**：导出时对 cell_size_code 做了按 256 元素切片的 L2 归一化（详见 FP16 章节）。

## ONNX 导出的关键技术点

### Bug 修复 1：权重共享

原始 `RGBDecoder.forward` 只使用 `mod_conv_1` 和 `mod_conv_2`，其中 `mod_conv_2` 被复用 7 次。
`mod_conv_3..8` 在 `__init__` 中定义但从未在 `forward` 中使用，权重是随机未训练的。

ONNX 导出时 `mod_conv_3..8` 全部从 `orig.mod_conv_2` 拷贝权重。

### Bug 修复 2：5D view 调制等价

原始 `ModulationConvBlock` 使用 5D view 进行调制：
```python
_weight = weight.view(1, k, k, in_c, out_c)
_weight = _weight * code.view(1, 1, 1, in_c, 1)
```

5D view 改变了 `in_c` 维度在内存中的元素分组，**不能**简化为 4D `(out_c, in_c, k, k)` 上的简单广播。
ONNX 版本完整复刻原始 5D view 逻辑。

### ONNXModulationConvBlock（batch=1）

```python
weight = self.weight * self.wscale                    # (out_c, in_c, k, k)
_weight = weight.view(1, k, k, in_c, out_c)           # 5D view
_weight = _weight * code.view(1, 1, 1, in_c, 1)       # 5D 调制
_norm = sqrt(sum(_weight**2, dim=[1,2,3]) + eps)       # demodulate
_weight = _weight / _norm.view(1,1,1,1,out_c)
weight = _weight.permute(1,2,3,0,4).reshape(out_c,in_c,k,k)  # 回 4D
x = F.conv2d(x, weight, bias=None, groups=1)
```

验证精度：FP32 ONNX vs PyTorch，Max diff 6.3e-4。

### cell_size_code L2 归一化

导出时对 cell_size_code 每 256 个元素做 L2 归一化：
```python
for i in range(0, 2048, 256):
    s = cell_size_code[:, i:i+256]
    cell_size_code[:, i:i+256] = s / torch.sqrt(torch.sum(s ** 2))
```

**原因**：原始 cell_size_code 值高达 ~17M，超出 FP16 上限（65504）。

**数学等价性**：demodulation 做归一化 `w / |w|`，code 的方向不变、幅度对最终结果无实质影响。
eps 差异：原始 `eps / sum((w*code)^2)` ≈ 1e-22 vs 归一化后 `eps / sum((w*code_hat)^2)` ≈ 4e-10，均远小于任何有意义阈值。

## FP16 精度优化分析

### 目标

将 FP32 ONNX 模型（76 MB）转为 FP16，预期：
- 模型体积减半（~38 MB）
- Apple Silicon NPU/GPU 原生 FP16 加速

### 根因 1：cell_size_code 值域超出 FP16 范围

| 问题 | 详情 |
|------|------|
| cell_size_code 原始值 | ~10M（256个通道各自） |
| FP16 可表示范围 | ±65504 |
| 调制中间值 | weight × code ≈ 1M（仍超限） |

**解决方案**：导出时对 cell_size_code 做 L2 归一化（见上文），归一化后值 ~O(0.1)，所有中间值均在 FP16 安全范围内。

### 根因 2：onnxconverter_common v1.16.0 的 Cast 节点 bug

`onnxconverter_common.float16.convert_float_to_float16` 在处理 InstanceNorm 和 Resize 时有类型推导错误：

```
Type Error: Type (tensor(float16)) of output arg (.../Cast_output_0)
of node (.../Cast) does not match expected type (tensor(float)).
```

**尝试过的策略**：

| 策略 | 结果 |
|------|------|
| 默认转换 | ❌ InstanceNorm 附近 Cast 节点类型不匹配 |
| disable_shape_infer | ❌ Resize Constant 被转为 float16（Resize 不接受） |
| op_block_list InstanceNorm+Resize | ❌ 仍有 Cast 类型错误（block 只阻止 op 转换，不修正在途 Cast） |
| 全部组合 + check_fp16_ready=False | ❌ 同上 |

**结论**：这是 `onnxconverter_common` 库本身的 bug，不是我们模型的问题。ONNX 规范中 InstanceNorm 和 Resize 都声明支持 float16，但该库在插入 Cast 边界节点时类型推导出错。

### 当前最优方案：Weight-only FP16

手动将所有 float32 initializer（权重）转为 float16，并在每个使用处插入 Cast(FP16→FP32) 节点。

| 指标 | FP32 | Weight-only FP16 |
|------|------|-----------------|
| 文件大小 | 76.2 MB | 38.2 MB（减半） |
| 256x256 max diff vs PyTorch | 6.5e-4 | 2.2e-3 |
| 512x512 CPU 推理 | 2388ms | 2383ms（无变化） |
| CoreML EP | 148/230 节点，746ms | 同左 |
| Go 端兼容 | ✅ | ✅（输入输出仍 FP32） |

**适用场景**：需要减小分发体积（如嵌入 App bundle），但不带来推理加速。

### 未来方向：真混合精度 FP16

如需获得 FP16 计算加速，需要**手写 ONNX 图遍历转换器**：
1. 遍历所有节点，按 op type 决定 FP16/FP32
2. Conv、Add、Mul、LeakyRelu → FP16
3. InstanceNorm、Resize → 保持 FP32
4. 在 FP16↔FP32 边界正确插入 Cast 节点
5. 需要完整的 ONNX type inference 支持

预估工作量较大，ROI 取决于目标硬件（Apple Silicon 上 CoreML EP 已有 3x 加速）。

### FP8 可行性

- FP8（E4M3: ±448, E5M2: ±57344）精度仅 2-3 位有效数字
- 仅 NVIDIA Hopper/Ada GPU 原生支持
- ONNX Runtime 和 CoreML EP 均不支持
- **对本项目（目标 Apple Silicon）不适用**

## 图像预处理/后处理

### 预处理
```
1. resize: (w, h) → ((w//cell_size)*4, (h//cell_size)*4)，BICUBIC
2. center crop: 确保宽高为 4 的倍数
3. ToTensor: [0,255] → [0,1]
4. Normalize: (x - 0.5) / 0.5 → [-1, 1]
5. 添加 batch 维度: (3, H, W) → (1, 3, H, W)
```

### 后处理
```
1. 去归一化: (x + 1) / 2 * 255 → [0, 255] uint8
2. 缩小到像素网格: resize(H/4, W/4), NEAREST
3. 放大到目标尺寸: resize(H/4*cell_size, W/4*cell_size), NEAREST
```

### 原始尺寸版后处理
```
1. 去归一化同上
2. 只缩小: resize(H/4, W/4), NEAREST
3. 不放大，直接保存
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `tests/export_onnx.py` | ONNX 导出脚本（含归一化 code、ONNXModulationConvBlock） |
| `tests/verify_onnx.py` | PyTorch vs ONNX Runtime 一致性验证 |
| `tests/benchmark_onnx.py` | 多尺寸正确性 + 速度基准测试 |
| `tests/convert_fp16.py` | FP16 转换（多策略自动尝试） |
| `tests/onnx/pixelization.onnx` | 导出的 FP32 ONNX 模型（76 MB） |
| `server/` | Go + ONNX Runtime 推理服务 |
| `models/basic_layer.py` | ModulationConvBlock 原始实现 |
| `models/c2pGen.py` | C2PGen, RGBEncoder, RGBDecoder, PixelBlockEncoder |
| `test_pro.py` | 原始 Python 推理管线 |
| `api.py` | FastAPI Web 服务 |

## Go 重写注意事项

- ONNX 模型输入 shape: `(1, 3, H, W)` float32，H/W 必须是 4 的倍数
- ONNX 模型输出 shape: `(1, 3, H, W)` float32，值域 [-1, 1]
- batch 固定为 1
- 库：`github.com/yalue/onnxruntime_go` v1.27.0
- Session 类型：`DynamicAdvancedSession`（H/W 动态维度），`sync.Mutex` 保护
- GPU 加速：CoreML EP（macOS ARM64）、CUDA EP（Linux）
- 交叉编译：`CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build`
- FP16 weight-only 模型：ONNX Runtime 自动在 graph 边界处理 Cast，Go 端无需修改

## 性能参考（M4 Mac，512x512）

| 运行时 | 耗时 | 备注 |
|--------|------|------|
| PyTorch CPU | 2361ms | 基线 |
| ONNX Runtime CPU | 2388ms | 无加速 |
| ONNX Runtime CoreML | **746ms** | **3.13x 加速**，148/230 节点 |

---

最后更新：2026-04-15
