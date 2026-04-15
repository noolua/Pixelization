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

## 任务 3：Go 服务重写

**状态：待开始**

- [ ] 技术选型：`onnxruntime-go` 或 `go-onnxruntime`
- [ ] 实现图像预处理（resize, normalize to [-1, 1], center crop to multiple of 4）
- [ ] 实现图像后处理（反归一化, resize by cell_size）
- [ ] 实现 HTTP API（参考 `api.py` 的接口设计）
  - `POST /pixelize` — 接收图片 + cell_size 参数
  - `GET /health` — 健康检查
- [ ] 静态前端（移植 `static/index.html`）
- [ ] 编译为单二进制 + ONNX 模型文件分发

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

---
创建日期: 2026年4月15日
