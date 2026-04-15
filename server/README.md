# Pixelization Server

基于 ONNX Runtime 的像素化图像处理服务，Go 实现。无需 Python 环境，单二进制 + ONNX 模型文件即可运行。

## 功能

- 将图片转换为像素画风格
- 支持 cell_size 参数（2-8）控制像素化程度
- 支持返回放大图（预览）或原始像素网格图（下载）
- 自动检测硬件加速：CoreML（macOS ARM）、CUDA（Linux）、CPU 回退
- 提供 Web UI 和 HTTP API

## 依赖

- Go 1.20+
- ONNX Runtime 共享库（[下载页](https://github.com/microsoft/onnxruntime/releases)）

## 编译

```bash
cd server/
go mod tidy

# 本机架构
go build -o pixelization-server .

# 从 x86_64 交叉编译 ARM64（macOS）
CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build -o pixelization-server .
```

## 运行

```bash
# 下载 ONNX Runtime（macOS ARM64 为例）
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-osx-arm64-1.27.0.tgz | tar xz

# 设置共享库路径并启动
export ONNX_RUNTIME_LIB=./onnxruntime-osx-arm64-1.27.0/lib/libonnxruntime.dylib
./pixelization-server --addr :8000 --model ../tests/onnx/pixelization.onnx
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `:8000` | 监听地址，局域网访问用 `0.0.0.0:8000` |
| `--model` | `../tests/onnx/pixelization.onnx` | ONNX 模型文件路径 |
| `--static` | `../static` | 静态文件目录 |

### 环境变量

| 变量 | 说明 |
|------|------|
| `ONNX_RUNTIME_LIB` | ONNX Runtime 共享库路径（必需） |

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web UI |
| `/pixelize` | POST | 像素化处理，multipart: `image`(文件) + `cell_size`(2-8) + `original_size`(bool) |
| `/health` | GET | 健康检查，返回 `{"status":"ok","model_loaded":true}` |
| `/static/*` | GET | 静态文件 |

### 示例

```bash
# 健康检查
curl http://localhost:8000/health

# 像素化
curl -X POST http://localhost:8000/pixelize \
  -F "image=@photo.png" \
  -F "cell_size=4" \
  -o result.png

# 下载原始像素网格
curl -X POST http://localhost:8000/pixelize \
  -F "image=@photo.png" \
  -F "cell_size=4" \
  -F "original_size=true" \
  -o pixel_grid.png
```
