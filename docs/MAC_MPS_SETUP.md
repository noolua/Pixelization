# Mac MPS 像素化服务部署备忘

> 创建日期: 2026-04-12
> 系统: macOS (Apple Silicon)
> 加速: MPS (Metal Performance Shaders)

## 1. 环境要求

- macOS 12.0+ (支持 MPS 的最低版本)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.8+
- 至少 8GB RAM

## 2. 依赖安装

### 2.1 安装 PyTorch (支持 MPS)

```bash
# 检查 Python 版本
python --version

# 安装 PyTorch (支持 MPS)
pip install torch>=1.12.0 torchvision>=0.13.0

# 验证 MPS 可用
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**预期输出**: `MPS available: True`

### 2.2 安装项目依赖

```bash
cd /path/to/Pixelization
pip install -r requirements.txt
```

依赖包括：
- `torch` - PyTorch 核心库
- `torchvision` - 图像处理
- `Pillow` - 图像 I/O
- `numpy` - 数值计算
- `fastapi` - Web 框架
- `uvicorn[standard]` - ASGI 服务器
- `python-multipart` - 文件上传支持

## 3. 模型文件准备

### 3.1 创建模型目录

```bash
mkdir -p checkpoints/demo
```

### 3.2 下载模型文件

从 [Google Drive](https://drive.google.com/drive/folders/1VRYKQOsNlE1w1LXje3yTRU5THN2GTmxL) 下载以下文件：

| 文件 | 位置 | 说明 |
|------|------|------|
| `alias_net.pth` | 项目根目录 `./` | 抗锯齿网络 |
| `pixelart_vgg19.pth` | 项目根目录 `./` | 特征提取器 |
| `160_net_G_A.pth` | `./checkpoints/demo/` | 生成器 A |
| `160_net_G_B.pth` | `./checkpoints/demo/` | 生成器 B |

### 3.3 验证目录结构

```bash
tree -L 2 -I '__pycache__|*.pyc'
```

预期结构：
```
Pixelization/
├── api.py
├── test_pro.py
├── alias_net.pth              # ← 根目录
├── pixelart_vgg19.pth         # ← 根目录
├── checkpoints/
│   └── demo/
│       ├── 160_net_G_A.pth    # ← demo 目录下
│       └── 160_net_G_B.pth    # ← demo 目录下
├── static/
│   └── index.html
└── requirements.txt
```

## 4. 启动 Web 服务

### 4.1 开发模式（自动重载）

```bash
# 方式一：直接运行
python api.py

# 方式二：使用 uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4.2 生产模式

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

**注意**: MPS 不支持多 workers，保持 `workers=1`

### 4.3 验证服务启动

启动成功后应看到：
```
Using device: mps
Model loaded successfully
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 5. 测试验证

### 5.1 健康检查

```bash
curl http://localhost:8000/health
```

预期响应：
```json
{"status":"ok","model_loaded":true}
```

### 5.2 Web 界面测试

1. 浏览器访问: http://localhost:8000
2. 上传一张 PNG 图片
3. 选择 cell_size (2-8)
4. 点击"处理图像"
5. 查看像素化结果

### 5.3 API 测试

```bash
# 测试像素化接口
curl -X POST http://localhost:8000/pixelize \
  -F "image=@test.png" \
  -F "cell_size=4" \
  --output result.png
```

## 6. 命令行模式（可选）

如果不使用 Web 服务，可以直接用命令行：

```bash
python test_pro.py \
  --input ./data/test.png \
  --output ./data/result.png \
  --cell_size 4 \
  --model_name demo
```

参数说明：
- `--input`: 输入图片路径（文件或目录）
- `--output`: 输出路径（可选）
- `--cell_size`: 像素化程度 (2-8)
- `--model_name`: 模型名称（对应 checkpoints/ 下的目录名）
- `--cpu`: 强制使用 CPU（不加此参数则自动使用 MPS）

## 7. 常见问题

### Q1: MPS not available

```
RuntimeError: MPS is not available
```

**解决**: 检查 macOS 版本和 PyTorch 版本
```bash
# 更新 PyTorch
pip install --upgrade torch torchvision
```

### Q2: 模型加载失败

```
FileNotFoundError: ./checkpoints/demo/160_net_G_A.pth
```

**解决**: 确认模型文件路径正确（见 3.3 节）

### Q3: 图像处理失败

```
400: Only support PNG/JPG format
```

**解决**: 确保上传的是 PNG 或 JPG 格式图片

### Q4: 内存不足

```
MPS backend out of memory
```

**解决**:
1. 减小输入图片尺寸
2. 使用 CPU 模式: `python test_pro.py --cpu`

### Q5: 依赖冲突

```
ERROR: pip's dependency resolver does not currently take into account...
```

**解决**: 使用虚拟环境
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 8. 端口占用

如果 8000 端口被占用：

```bash
# 查找占用进程
lsof -i :8000

# 杀死进程
kill -9 <PID>

# 或使用其他端口
uvicorn api:app --port 8080
```

## 9. 后台运行（可选）

```bash
# 使用 nohup
nohup uvicorn api:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# 查看日志
tail -f server.log

# 停止服务
ps aux | grep uvicorn
kill <PID>
```

## 10. 性能参考

| 设备 | 处理时间 (512x512) |
|------|-------------------|
| M1 | ~2-3 秒 |
| M2 | ~1.5-2 秒 |
| M3 | ~1-1.5 秒 |

## 11. 文件清单

部署完成后，以下文件应存在：

```
必需文件:
├── api.py                  # Web 服务
├── test_pro.py             # 命令行工具
├── requirements.txt        # 依赖清单
├── models/                 # 网络定义
│   ├── networks.py
│   ├── c2pGen.py
│   └── ...
├── static/
│   └── index.html          # 前端页面
├── checkpoints/demo/
│   ├── 160_net_G_A.pth     # 模型权重
│   └── 160_net_G_B.pth
├── alias_net.pth           # 抗锯齿模型
└── pixelart_vgg19.pth      # VGG19 模型
```

---

**快速启动命令** (一键复制):

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python api.py

# 3. 访问 http://localhost:8000
```
