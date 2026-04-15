"""
像素化 Web 服务
基于 test_pro.py 的 Model 类提供 HTTP 接口
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import io
import tempfile
import os
from pathlib import Path
from PIL import Image

from test_pro import Model

app = FastAPI(title="Pixelization API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 全局模型实例
model = None
MODEL_NAME = "demo"  # 默认模型名称，对应 checkpoints/demo/


def get_device():
    """自动检测设备：MPS > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model():
    """启动时加载模型"""
    global model
    device = get_device()
    print(f"Using device: {device}")
    model = Model(MODEL_NAME, device=device)
    model.load()
    print("Model loaded successfully")


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_model()


@app.get("/")
async def root():
    """根路径返回前端页面"""
    from fastapi.responses import FileResponse
    return FileResponse(str(static_dir / "index.html"))


@app.post("/pixelize")
async def pixelize(
    image: UploadFile = File(..., description="PNG 图像文件"),
    cell_size: int = Form(4, description="像素化程度，范围 2-8"),
):
    """
    像素化接口

    接收 PNG 图像和 cell_size 参数，返回原始像素网格尺寸的像素化图像。
    缩放预览由前端 CSS 处理。
    """
    # 验证参数
    if cell_size < 2 or cell_size > 8:
        raise HTTPException(status_code=400, detail="cell_size 必须在 2-8 之间")

    # 验证文件格式
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG 格式图像")

    # 创建临时文件
    temp_input = None
    temp_output = None

    try:
        # 读取上传的图像
        contents = await image.read()

        # 验证是否为有效图像
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="无效的图像文件")

        # 创建临时输入文件
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_input.write(contents)
        temp_input.close()

        # 创建临时输出文件
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_output.close()

        # 调用模型进行像素化（始终返回原始像素网格尺寸）
        model.pixelize_original_size(temp_input.name, temp_output.name, cell_size)

        # 读取结果图像
        with open(temp_output.name, "rb") as f:
            result_bytes = f.read()

        # 返回图像
        return Response(content=result_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_input and os.path.exists(temp_input.name):
            os.unlink(temp_input.name)
        if temp_output and os.path.exists(temp_output.name):
            os.unlink(temp_output.name)


def kmeans_colors(image: Image.Image, target_colors: int) -> Image.Image:
    """
    使用加权 K-Means 减少图像颜色数。
    基于唯一颜色 + 频率加权聚类，避免大图全像素运算。
    """
    img_array = np.array(image.convert("RGB"))
    pixels = img_array.reshape(-1, 3)

    # 获取唯一颜色及其频率
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    num_unique = len(unique_colors)

    if num_unique <= target_colors:
        return image

    # 加权 K-Means：用唯一颜色作为样本，counts 作为权重
    # 初始化：按频率加权随机选取初始中心
    rng = np.random.default_rng(42)
    probs = counts / counts.sum()
    initial_idx = rng.choice(num_unique, size=target_colors, replace=False, p=probs)
    centers = unique_colors[initial_idx].astype(np.float64)

    # 迭代 K-Means
    for _ in range(20):
        # 计算每个唯一颜色到各中心的距离 (N_unique, K)
        diff = unique_colors[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        # 加权更新中心
        new_centers = np.zeros_like(centers)
        for k in range(target_colors):
            mask = labels == k
            if mask.any():
                new_centers[k] = np.average(unique_colors[mask], axis=0, weights=counts[mask])
            else:
                new_centers[k] = centers[k]

        if np.allclose(centers, new_centers, atol=1.0):
            break
        centers = new_centers

    # 四舍五入到整数
    centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)

    # 映射：所有像素 -> 最近聚类中心
    # 利用唯一颜色映射表加速
    label_map = labels  # unique_colors[i] -> centers[labels[i]]
    # 建立 原始像素 -> 新颜色 的映射
    color_to_label = {}
    for i, c in enumerate(unique_colors):
        color_to_label[tuple(c)] = tuple(centers[label_map[i]])

    # 用向量化方式替换像素
    flat = pixels
    result = np.array([color_to_label[tuple(p)] for p in flat], dtype=np.uint8)
    result = result.reshape(img_array.shape)

    return Image.fromarray(result)


@app.post("/optimize-colors")
async def optimize_colors(
    image: UploadFile = File(..., description="PNG 图像文件"),
    target_colors: int = Form(32, description="目标颜色数，范围 2-256"),
):
    """
    K-Means 颜色优化接口

    接收图像和目标颜色数，返回颜色减少后的图像
    """
    if target_colors < 2 or target_colors > 256:
        raise HTTPException(status_code=400, detail="target_colors 必须在 2-256 之间")

    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG 格式图像")

    try:
        contents = await image.read()

        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="无效的图像文件")

        img = Image.open(io.BytesIO(contents))
        result_img = kmeans_colors(img, target_colors)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"颜色优化失败: {str(e)}")


@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    import os

    # 从环境变量读取配置，默认只监听本地
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
