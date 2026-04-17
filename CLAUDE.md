# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

可配置管线的像素画生产工具，不是论文复现仓库。基于 SIGGRAPH Asia 2022 论文实现。

## 核心架构原则

- **pipe 模式**：每个图像处理能力是 `web/pipes/` 下的无状态函数，接收 PIL Image + 参数，返回 PIL Image。新增功能先加 pipe，再接 API。
- **api.py 只做 HTTP**：参数校验、图像读写（`_read_image` / `_image_response`）、调用 pipe。不包含任何图像处理逻辑。
- **模型管理与推理分离**：`pipes/pixelize.py` 是唯一持有模型状态的 pipe（Model 类）。其他 pipe 都是纯算法。
- **双后端对等**：Python (FastAPI) 开发，Go (ONNX Runtime) 生产，API 接口一致。新增 pipe 需考虑两端同步。
- **前端三文件**：index.html / style.css / app.js，不拆更多文件。新 pipe 在 app.js 的 PIPE_REGISTRY 注册。

## 设计原则

- **最小入侵**：通过配置实现功能，最小化代码修改范围。优先复用已有函数（如 `rgb_to_lab`），不为两个消费者新建模块。
- **骨架优先**：先端到端跑通最简实现，再逐步增强。新 pipe 先写最近色映射验证全链路，再补抖动模式。
- **远虑近做**：设计阶段完整分析扩展场景并记录，实现阶段选最简单方案，不为"未来可能"提前写代码。

## 代码规范

- 所有语言 **2 空格缩进**，禁止 Tab
- Git 提交：`类型(范围): 描述`（≤50 字符），类型用 feat/fix/refactor/docs/test/chore
- Co-Authored-By: Claude Sonnet 4.5 \<noreply@anthropic.com\>
- 交流用中文，通用技术术语保留英文

## Python 环境

```bash
source ~/py39/bin/activate
```

## 关键路径

- 权重文件：`downloads/`（原始）+ `tools/extract_inference_weights.py`（推理用提取）
- 批量处理：`tools/batch_process.py`（HTTP 模式，读 JSON 配置调 API）
- 设备优先级：MPS (Apple Silicon) > CUDA > CPU
