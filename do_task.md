# 像素画工具 — 开发任务

像素化模型是本应用的一个 Pipe 组件，而非核心。应用定位：**可配置管线的像素画生产工具**。

---

## 模块一：代码瘦身与重组

**目标：** 删除训练代码，按应用职责重新组织目录。

### 目标结构

```
Pixelization/
├── web/                        # Python 应用
│   ├── api.py                  # FastAPI 入口
│   ├── batch_process.py        # 批量处理脚本
│   ├── pipes/                  # 图像处理 pipe 包
│   │   ├── __init__.py
│   │   └── bg_unify.py
│   ├── models/                 # 神经网络定义（精简后）
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
├── tests/                      # ONNX 导出/验证 + 临时测试代码
├── tools/                      # 模型转换等工具脚本（export_onnx.py, convert_fp16.py 等）
├── docs/                       # 文档
├── downloads/                  # 模型权重
├── build/                      # 构建输出
│
├── requirements.txt
├── CLAUDE.md / bref.md / do_task.md / README.md / LICENSE.md
```

### 任务

- [ ] 1.1 创建 `web/` 目录，迁移 Python 代码（api.py, batch_process.py, pipes/, models/, static/），更新 import 路径
- [ ] 1.2 创建 `tools/` 目录，迁移模型转换脚本（tests/export_onnx.py → tools/，tests/convert_fp16.py → tools/）
- [ ] 1.3 删除 `test_pro.py`（本地测试在 `tests/` 下写临时代码）
- [ ] 1.4 删除训练代码和废弃目录：`data/`、`options/`、`util/`、`datasets/`、`results/`、`prepare_data.py`、`test.py`、`checkpoints/`、`examples/`
- [ ] 1.5 删除训练模型：`models/base_model.py`、`models/pixelization_model.py`、`models/p2cGen.py`、`models/c2pDis.py`、`models/test_model.py`
- [ ] 1.6 精简 `models/networks.py`（删除 GANLoss、define_D、get_scheduler 等训练专用代码）
- [ ] 1.7 删除论文图片：`teaser.jpg`、`feedback.jpg`
- [ ] 1.8 更新 `CLAUDE.md` 反映新目录结构
- [ ] 1.9 验证：api.py 启动正常、前端正常

---

## 模块 1.5：推理模型瘦身

**目标：** 去掉训练专用子模块，只加载推理所需的权重，减少内存占用和启动时间。

### 现状分析

推理实际调用链：
```
G_A_net.RGBEnc(in_t)           ← 用到
G_A_net.RGBDec(feature, code)  ← 用到
alias_net(images)              ← 用到
```

C2PGen（`160_net_G_A.pth`）内部子模块使用情况：

| 子模块 | 推理需要？ | 说明 |
|--------|-----------|------|
| `RGBEnc` | 是 | 编码输入图像特征 |
| `RGBDec` | 是 | 解码为像素化结果 |
| `MLP` | 仅 load 时 | 预计算 `cell_size_code`，之后不再调用 |
| `PBEnc` | **否** | 含完整 VGG-19（~548MB），训练专用，推理从未调用 |

额外依赖的 `pixelart_vgg19.pth` 仅供 PBEnc 使用，推理不需要。

### 任务

- [x] 1.5.1 创建推理专用模型类 `_PixelNet`，只包含 `RGBEnc` + `RGBDec`
- [x] 1.5.2 预计算 `cell_size_code` 并缓存为文件，彻底移除 MLP 运行时依赖
- [x] 1.5.3 编写工具脚本 `tools/extract_inference_weights.py`，导出轻量 checkpoint（`downloads/inference_net.pth`）
- [x] 1.5.4 更新 `inference.py` 使用新模型和轻量 checkpoint，移除 PBEnc / MLP / VGG-19 加载
- [x] 1.5.5 验证：推理结果与原始模型完全一致（MD5 相同），API 正常工作
- [ ] 1.5.6（可选）清理：确认无误后，`pixelart_vgg19.pth` 可不再随应用分发

### 预期收益

- 内存：省掉 VGG-19 ~548MB + PBEnc 参数 + MLP 参数
- 启动速度：不再加载和初始化无用网络
- 依赖：不再需要 `pixelart_vgg19.pth` 文件
- 代码简化：`models/c2pGen.py` 中可移除 `PixelBlockEncoder`、`MLP` 等训练专用类

---

## 模块二：Pipe 组件

**目标：** 每个图像处理能力封装为独立 Pipe，后端无状态 API，前端可自由组合。

### 已完成

| Pipe | API 端点 | 状态 |
|------|----------|------|
| 像素化 | `POST /pixelize` | 已完成 |
| 颜色优化 | `POST /optimize-colors` | 已完成 |
| 背景色统一 | `POST /unify-background` | 已完成 |

### 待开发（详见 bref.md "新增 Pipe 规划"）

- [ ] 2.1 **调色板映射 `palette-map`** — Lab 空间最近色替换 + 抖动模式，内置 PICO-8/GameBoy/NES 等复古调色板
- [ ] 2.2 **蒙版轮廓 `silhouette`** — 前景检测 → 纯色填充，生成受击闪白/阴影剪影
- [ ] 2.3 **外描边 `outline`** — 前景边缘外扩描边，提升角色可读性
- [ ] 2.4 **灰度 `grayscale`** — 可量化灰阶级数，可组合调色板映射做单色版

### 共享基础

- [ ] 2.5 前景检测工具函数（alpha 通道或背景色容差），供 silhouette/outline 复用

---

## 模块三：前端管线架构

**目标：** 用户可动态增删排列 Pipe，实时预览中间结果。

### 已完成

- 可配置管线 UI（增删排列 Pipe 节点）
- 管线配置导入/导出
- 前端模块化拆分（index.html / style.css / app.js）

### 待改进

- [ ] 3.1 新 Pipe 的前端节点注册（随 2.x 开发同步）
- [ ] 3.2 中间结果对比预览增强

---

## 模块四：Batch 批量处理

**目标：** Web 配置设计 → 导出 JSON → Python 脚本批量执行。

### 已完成

- Batch 数据模型 + 持久化
- Batch 面板 UI
- 预览执行 + 瓦片拼图
- Python HTTP 模式批处理脚本

### 待改进

- [ ] 4.1 新 Pipe 的 Batch 支持（palette-map、silhouette、outline、grayscale）
- [ ] 4.2 Batch 进度反馈与断点续跑

---

## 模块五：Go 生产服务

**目标：** Go + ONNX Runtime 无 Python 依赖的生产部署。

### 已完成

- Go 像素化 + 颜色优化 API
- ONNX 模型导出与验证

### 待同步

- [ ] 5.1 Go 端同步新增 API（unify-background、palette-map 等）
- [ ] 5.2 Go Batch 执行能力（可选）

---

## 模块六：未来探索

### 48×48 像素画底稿管线（详见 `docs/idea/new_pixelization.md`）

三阶段概念验证，目标是将 AI 高清图自动转为像素画"清洁底稿"：

- Stage 1：语义边缘过滤（SAM + 结构线/纹理分类）
- Stage 2：降维坍缩与拓扑修复（48×48 闭合线稿）
- Stage 3：受控色彩平涂（全局 32 色调色板映射）

**当前状态：概念阶段，技术验证待启动。**

---

创建日期: 2026-04-17
