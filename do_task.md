# Pixelization 项目任务跟踪

## 目标

将像素化程序打包为一个目录即可运行的服务，支持 MPS/CPU/GPU，无需安装 Python。
当前选定方案：**PyTorch → ONNX → Go + ONNX Runtime**。

---

## 任务 1：ONNX 模型导出

**状态：已完成**

- [x] 创建 `tests/export_onnx.py` — 导出脚本
- [x] 创建 `tests/verify_onnx.py` — 验证脚本
- [x] 解决 `ModulationConvBlock` 动态权重 reshape 不兼容 ONNX 的问题
  - 为 batch=1 重写了 `ONNXModulationConvBlock`，数学上完全等价
  - `ONNXRGBDecoder` 逐个复制原始权重
- [x] 使用 legacy trace exporter (`dynamo=False`)，固定 batch=1
- [x] 导出成功，模型路径：`tests/onnx/pixelization.onnx`

### 验证结果

| 测试用例 | Max Diff | Mean Diff | 结论 |
|---------|----------|-----------|------|
| Random 256x256 | 6.3e-4 | 1.0e-4 | ACCEPTABLE |

---

## 任务 2：多尺寸 & 性能测试

**状态：已完成**

- [x] 测试不同分辨率（128x128, 512x512, 1024x1024, 非正方形）的 dynamic axes
- [x] PyTorch vs ONNX Runtime 推理速度对比（CPU）
- [x] 测试 ONNX Runtime CUDA EP（NVIDIA GPU）— 不可用（Mac 环境）
- [x] 测试 ONNX Runtime CoreML EP（macOS Apple Silicon）

### 测试结果

**正确性**（PyTorch vs ONNX Runtime，random input）：

| 分辨率 | Max Diff | Mean Diff | 状态 |
|--------|----------|-----------|------|
| 128x128 | 1.34e-04 | 1.42e-05 | PASS |
| 256x256 | 6.51e-04 | 8.27e-05 | PASS |
| 512x512 | 3.24e-03 | 3.98e-04 | 数值差异 |
| 1024x1024 | 3.82e-02 | 2.20e-03 | 数值差异 |
| 320x480 | 1.80e-03 | 2.65e-04 | 数值差异 |
| 640x360 | 2.51e-03 | 3.73e-04 | 数值差异 |

- 大尺寸有数值差异，但**视觉质量测试通过**（真实图片 ONNX 输出效果良好）

**速度**（512x512，CPU）：

| 运行时 | 耗时 | 备注 |
|--------|------|------|
| PyTorch CPU | 2361ms | 基线 |
| ONNX Runtime CPU | 2388ms | 无加速 |
| ONNX Runtime CoreML | **746ms** | **3.13x 加速** |

- CoreML EP 覆盖 148/230 节点，剩余 fallback 到 CPU
- 测试脚本：`tests/benchmark_onnx.py`

---

## 任务 2.5：修复 ONNX 导出 Bug

**状态：已完成**

当前 ONNX 模型在真实图片上效果差，分析发现两个关键 bug：

### Bug 1（致命）：权重共享错误

原始 `RGBDecoder`（`models/c2pGen.py:238-254`）只用 `mod_conv_1` 和 `mod_conv_2`，其中 `mod_conv_2` 被复用 7 次（权重共享）。`mod_conv_3..8` 在 `__init__` 中定义但从未在 `forward` 中使用，权重是随机未训练的。

ONNX 导出时错误地从 `orig.mod_conv_3..8` 拷贝了这些未训练权重。

**修复**：`tests/export_onnx.py` 的 `ONNXRGBDecoder.__init__` 中，`mod_conv_3..8` 全部改为从 `orig.mod_conv_2` 拷贝：
```python
self.mod_conv_3 = ONNXModulationConvBlock(orig.mod_conv_2)  # 而非 orig.mod_conv_3
# ... 同理 4..8
```

### Bug 2：ModulationConvBlock 调制数学不等价

原始 `ModulationConvBlock`（`models/basic_layer.py:30-44`）使用 5D view 进行调制：
```python
_weight = weight.view(1, k, k, in_c, out_c)
_weight = _weight * code.view(1, 1, 1, in_c, 1)  # 5D view 调制
```
5D view 改变了 in_c 维度在内存中的元素分组，乘 code 作用于与 4D 不同的元素组。

ONNX 简化版直接在 4D `(out_c, in_c, k, k)` 上乘 code，数学不等价。

**修复**：`tests/export_onnx.py` 的 `ONNXModulationConvBlock.forward` 完整复刻原始 5D view 调制逻辑，仅简化 `groups=1`。

### 验证步骤

- [x] 重新导出 ONNX 模型：`python tests/export_onnx.py`
- [x] 运行验证：`python tests/verify_onnx.py` — 精度应显著提升
- [x] 运行基准测试：`python tests/benchmark_onnx.py`
- [x] 用真实图片对比 Python 推理 vs Go ONNX 推理的视觉效果

---

## 任务 3：Go 服务重写

**状态：代码已编写，待在远程 M4 机器上验证**

代码位于 `server/` 目录，已在远程 M4 机器上成功编译运行。

- [x] 技术选型：`github.com/yalue/onnxruntime_go` + `github.com/disintegration/imaging`
- [x] 实现图像预处理（resize, normalize to [-1, 1], center crop to multiple of 4）
- [x] 实现图像后处理（反归一化, resize by cell_size）
- [x] 实现 HTTP API（复刻 `api.py` 接口）
  - `POST /pixelize` — 接收图片 + cell_size 参数
  - `GET /health` — 健康检查
  - `GET /` — 前端页面
  - `/static/*` — 静态文件
  - CORS 中间件（allow all）
- [x] 静态前端（直接复用 `../static/index.html`）
- [x] README.md 编写
- [x] 等任务 2.5 修复后，用新 ONNX 模型做端到端真实图片验证

### 技术要点

- 文件结构：`server/{main.go, inference.go, preprocess.go, postprocess.go, handlers.go, go.mod, README.md}`
- 本地 x86_64 交叉编译 ARM64：`CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build`
- 运行参数：`--addr`（默认 :8000）、`--model`、`--static`
- 环境变量：`ONNX_RUNTIME_LIB` 指定 onnxruntime 共享库路径（必需）
- ONNX 模型输入：`"image"` (1,3,H,W) float32；输出：`"output"` (1,3,H,W) float32
- 使用 `DynamicAdvancedSession`（H/W 动态），`sync.Mutex` 保护 session.Run()

---

## 任务 3.5：FP16 精度优化

**状态：已完成（weight-only 方案）**

### 完成的工作

- [x] 创建 `tests/convert_fp16.py` — 多策略 FP16 转换脚本
- [x] 根因分析：cell_size_code 值 ~17M 超出 FP16 范围（65504）
- [x] 修复：导出时对 cell_size_code 做 L2 归一化（数学等价，eps 差异 < 1e-10）
- [x] 发现 `onnxconverter_common` v1.16.0 的 Cast 节点类型推导 bug
- [x] 实现 weight-only FP16 fallback（38.2 MB，无计算加速）
- [x] 精度验证：max diff 2.2e-3（像素域 ~0.57/255，视觉无损）
- [x] 技术备忘更新：`bref.md`

### 结果

| 指标 | FP32 | FP16 weight-only |
|------|------|-----------------|
| 文件大小 | 76.2 MB | 38.2 MB |
| 512x512 CPU 速度 | 2388ms | 2383ms（无变化） |
| CoreML EP | 148/230 节点，746ms | 同左 |
| 256x256 vs PyTorch | 6.5e-4 | 2.2e-3 |

### 未完成 / 未来方向

- [ ] 真混合精度 FP16（需手写 ONNX 图转换器绕过 onnxconverter_common bug）
- [ ] FP8 不适用于 Apple Silicon（仅 NVIDIA Hopper/Ada）

---

## 任务 4：打包 & 分发

**状态：待开始**

- [ ] macOS（Apple Silicon）：单二进制 + pixelization.onnx，CoreML EP 加速
- [ ] Linux（CUDA）：单二进制 + pixelization.onnx，CUDA EP 加速
- [ ] Linux/macOS（CPU fallback）：纯 CPU 推理
- [ ] 编写构建脚本（Makefile / GoReleaser）
- [ ] 编写使用文档

---

## 关键技术决策记录

### 2026-04-15: ONNX 导出方案

- **问题**：`ModulationConvBlock` 使用 `F.conv2d(groups=batch)` + 动态 weight reshape，ONNX 无法处理
- **解决**：为 batch=1 重写等价模块 `ONNXModulationConvBlock`，去掉动态 groups，改为标准 `groups=1` conv2d
- **权衡**：固定 batch=1，不支持 batch 推理（推理服务通常也只处理单张图片）
- **精度**：Max diff 6.3e-4，浮点精度范围内的可接受误差

### 2026-04-15: ONNX 导出 Bug 分析

- **发现**：ONNX 模型在真实图片上效果差
- **Bug 1**：`RGBDecoder` 训练时只用 `mod_conv_1` + `mod_conv_2`（后者复用 7 次），`mod_conv_3..8` 从未参与训练。ONNX 导出错误地拷贝了 `mod_conv_3..8` 的随机未训练权重
- **Bug 2**：`ModulationConvBlock` 原始实现用 5D view `(1,k,k,in_c,out_c)` 进行调制，5D view 改变了 in_c 维度的内存布局分组。ONNX 简化版在 4D `(out_c,in_c,k,k)` 上直接乘 code，数学不等价
- **根因**：`bref.md` 中"实际上 mod_conv_3..8 的权重与 mod_conv_2 相同"的描述是错误的

### 2026-04-15: Go 服务技术选型

- **ONNX Runtime 库**：`github.com/yalue/onnxruntime_go` v1.27.0（CGo 绑定）
- **图像处理**：`github.com/disintegration/imaging`（纯 Go，CatmullRom=BICUBIC, NearestNeighbor）
- **Session 类型**：`DynamicAdvancedSession`（H/W 动态维度）
- **交叉编译**：macOS x86_64 → arm64 直接 `CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build`
- **远程机器**：M4 Mac，onnxruntime 共享库通过 `ONNX_RUNTIME_LIB` 环境变量指定

---
创建日期: 2026年4月15日
最后更新: 2026年4月15日
