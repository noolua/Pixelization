# Pixelization — 像素画生产工具

基于 SIGGRAPH Asia 2022 论文 [*Make Your Own Sprites: Aliasing-Aware and Cell-Controllable Pixelization*](https://dl.acm.org/doi/pdf/10.1145/3550454.3555482) 的像素画生产工具。

将普通图像转换为像素风格艺术画，提供可配置管线和批量处理能力。

## 功能

- **像素化** — 神经网络驱动的像素化，支持 2× 到 8× cell size
- **颜色优化** — K-Means 聚类减少颜色数
- **背景统一** — Lab 空间 Delta-E 容差的背景色替换
- **灰度** — 可量化灰阶（GameBoy 4 级、8 级阴影等）
- **调色板映射** — 内置 PICO-8 / GameBoy / NES / C64 / Endesga 32 等复古调色板，支持有序抖动和 Floyd-Steinberg 抖动
- **批量处理** — JSON 配置驱动，批量处理图像目录并生成拼图
- **管线组合** — 前端可自由排列 pipe 顺序，实时预览中间结果

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
cd web && python api.py
```

浏览器打开 `http://127.0.0.1:8000` 即可使用。

### 批量处理

```bash
# 在 Web 界面设计管线 → 导出 batch 配置 JSON
python tools/batch_process.py --config batch.json --input ./images --output ./output --server http://127.0.0.1:8000
```

## HTTP API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/pixelize` | POST | 像素化（cell_size 2-8） |
| `/optimize-colors` | POST | K-Means 颜色优化（target_colors 2-256） |
| `/unify-background` | POST | 背景色统一（tolerance, target_color） |
| `/grayscale` | POST | 灰度（gray_levels: 0=连续, 2-256=量化） |
| `/palette-map` | POST | 调色板映射（palette, dither） |
| `/health` | GET | 健康检查 |

所有 POST 端点接收 `image` 文件上传 + 表单参数，返回 PNG 图像。

## 项目结构

```
web/
├── api.py              # FastAPI 入口
├── pipes/              # 图像处理 pipe
│   ├── pixelize.py     #   像素化（含模型加载）
│   ├── optimize_colors.py  # 颜色优化
│   ├── bg_unify.py     #   背景统一
│   ├── grayscale.py    #   灰度
│   └── palette_map.py  #   调色板映射
├── models/             # 神经网络层定义
└── static/             # 前端（HTML/CSS/JS）

server/                 # Go 生产服务（ONNX Runtime）
tools/                  # 工具脚本
downloads/              # 模型权重
```

## 双后端

Python（FastAPI + PyTorch）用于开发调试，Go（ONNX Runtime）用于生产部署，API 接口一致。

## License

Software Copyright License for non-commercial scientific research purposes. See [LICENSE.md](LICENSE.md).
