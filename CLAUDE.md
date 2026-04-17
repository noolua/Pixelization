# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

SIGGRAPH Asia 2022 论文 "Make Your Own Sprites: Aliasing-Aware and Cell-Controllable Pixelization" 的官方实现。将普通图像转换为像素风格艺术画，支持 2× 到 N× 的 cell size 控制。

## 目录结构

```
Pixelization/
├── web/                        # Python 应用
│   ├── api.py                  # FastAPI 入口
│   ├── batch_process.py        # 批量处理脚本
│   ├── inference.py            # 模型推理封装（Model 类）
│   ├── pipes/                  # 图像处理 pipe 包
│   │   ├── __init__.py
│   │   └── bg_unify.py
│   ├── models/                 # 神经网络定义
│   │   ├── __init__.py
│   │   ├── basic_layer.py
│   │   ├── c2pGen.py
│   │   └── networks.py
│   └── static/                 # 前端
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── server/                     # Go 生产服务
├── tests/                      # ONNX 验证 + 临时测试代码
├── tools/                      # 模型转换工具脚本
│   ├── export_onnx.py
│   └── convert_fp16.py
├── docs/                       # 文档
├── downloads/                  # 模型权重
├── build/                      # 构建输出
│
├── requirements.txt
├── CLAUDE.md / bref.md / do_task.md / README.md / LICENSE.md
```

## 常用命令

### python3执行环境
```bash
# 首先进入python3的环境
source ~/py39/bin/activate
```

### Python 后端（FastAPI 开发服务器）
```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API 服务（默认 127.0.0.1:8000）
cd web && python api.py

# 通过环境变量配置
API_HOST=0.0.0.0 API_PORT=8080 python web/api.py
```

### Go 后端（ONNX Runtime 生产服务）
```bash
cd server
# 需要设置 ONNX Runtime 库路径
export ONNX_RUNTIME_LIB=/path/to/libonnxruntime.dylib
go run . -addr :8000 -model ../tests/onnx/pixelization.onnx -static ../web/static
```

### ONNX 模型导出与验证
```bash
python tools/export_onnx.py    # 导出 PyTorch 模型为 ONNX
python tests/verify_onnx.py    # 验证 ONNX 输出与 PyTorch 一致性
python tools/convert_fp16.py   # FP16 权重转换
```

## 架构

### 双后端架构

项目有两套功能等价的后端服务，API 接口完全一致：

| | Python (web/api.py) | Go (server/) |
|---|---|---|
| 用途 | 开发/调试 | 生产部署（无 Python 依赖） |
| 推理引擎 | PyTorch 直接加载 .pth | ONNX Runtime |
| 前端 | `web/static/` | `web/static/` |

### HTTP API 端点

- `GET /` — 前端页面
- `POST /pixelize` — 像素化（参数: image 文件, cell_size 2-8）
- `POST /optimize-colors` — K-Means 颜色优化（参数: image 文件, target_colors 2-256）
- `POST /unify-background` — 背景色统一（参数: image 文件, tolerance 1.0-20.0, target_color）
- `GET /health` — 健康检查

### 推理流水线

```
输入图像 → rescale(128~4000px) → 尺寸对齐到 cell_size 倍数
    → RGBEncoder → RGBDec(feature, cell_size_code) → AliasNet → 缩放输出
```

关键模块（`web/`）：
- **inference.py** — Model 类，封装模型加载和推理流程
- **pipes/bg_unify.py** — 背景色统一 pipe
- **models/c2pGen.py** — C2P Generator（含 RGBEncoder、RGBDecoder、AliasNet）
- **models/networks.py** — 网络工厂函数 `define_G()`
- **models/basic_layer.py** — 基础网络层

### 模型权重

所有权重文件位于 `downloads/` 目录：
- `downloads/alias_net.pth` — AliasNet 反锯齿网络
- `downloads/pixelart_vgg19.pth` — VGG-19 结构提取器
- `downloads/160_net_G_A.pth` — I2P Generator 权重

### 设备支持

自动检测优先级：MPS (Apple Silicon) > CUDA > CPU。Go 版本使用 ONNX Runtime 的 CPU 后端。

## 代码规范

- 所有语言使用 **2 空格缩进**，禁止 Tab
- Git 提交格式：`类型(范围): 描述`（≤50 字符），类型用 feat/fix/refactor/docs/test/chore
- Co-Authored-By: Claude Sonnet 4.5 \<noreply@anthropic.com\>
- 交流用中文，通用技术术语保留英文
