"""
像素化 Web 服务
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import io
from pathlib import Path
from PIL import Image

from pipes.pixelize import Model, pixelize as pixelize_pipe
from pipes.bg_unify import unify_background, hex_to_rgb
from pipes.optimize_colors import kmeans_colors
from pipes.grayscale import grayscale
from pipes.palette_map import palette_map
from pipes.edge_darken import edge_darken
from pipes.bg_transparent import bg_transparent

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
MODEL_NAME = "demo"


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
    backend = os.environ.get("PIPELINE_BACKEND", "pytorch")
    print(f"Using device: {device}, backend: {backend}")
    model = Model(MODEL_NAME, device=device, backend=backend)
    model.load()
    print("Model loaded successfully")


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_model()


async def _read_image(image: UploadFile) -> Image.Image:
    """读取上传图像，验证格式和有效性"""
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPG 格式图像")
    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的图像文件")
    return Image.open(io.BytesIO(contents))


def _image_response(img: Image.Image) -> Response:
    """将 PIL Image 转为 PNG HTTP Response"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/")
async def root():
    """根路径返回前端页面"""
    return FileResponse(str(static_dir / "index.html"))


@app.post("/pixelize")
async def pixelize(
    image: UploadFile = File(..., description="PNG 图像文件"),
    cell_size: int = Form(4, description="像素化程度，范围 2-8"),
):
    """像素化接口"""
    if cell_size < 2 or cell_size > 8:
        raise HTTPException(status_code=400, detail="cell_size 必须在 2-8 之间")

    try:
        img = await _read_image(image)
        return _image_response(pixelize_pipe(model, img, cell_size))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/optimize-colors")
async def optimize_colors(
    image: UploadFile = File(..., description="PNG 图像文件"),
    target_colors: int = Form(32, description="目标颜色数，范围 2-256"),
):
    """K-Means 颜色优化接口"""
    if target_colors < 2 or target_colors > 256:
        raise HTTPException(status_code=400, detail="target_colors 必须在 2-256 之间")

    try:
        img = await _read_image(image)
        return _image_response(kmeans_colors(img, target_colors))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"颜色优化失败: {str(e)}")


@app.post("/unify-background")
async def unify_bg(
    image: UploadFile = File(..., description="PNG/JPG 图像文件"),
    tolerance: float = Form(5.0, description="Delta-E 容差，范围 1.0-20.0"),
    target_color: str = Form("#808080", description="目标背景色，十六进制"),
):
    """背景色统一接口"""
    if tolerance < 1.0 or tolerance > 20.0:
        raise HTTPException(status_code=400, detail="tolerance 必须在 1.0-20.0 之间")

    try:
        target_rgb = hex_to_rgb(target_color)
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="target_color 格式错误，示例: #808080")

    try:
        img = await _read_image(image)
        return _image_response(unify_background(img, tolerance, target_rgb))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"背景统一失败: {str(e)}")


@app.post("/grayscale")
async def grayscale_endpoint(
    image: UploadFile = File(..., description="PNG 图像文件"),
    gray_levels: int = Form(0, description="灰阶级数: 0=连续, 2-256=量化"),
):
    """灰度接口"""
    if gray_levels != 0 and (gray_levels < 2 or gray_levels > 256):
        raise HTTPException(status_code=400, detail="gray_levels 必须为 0 或 2-256")

    try:
        img = await _read_image(image)
        return _image_response(grayscale(img, gray_levels))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"灰度处理失败: {str(e)}")


@app.post("/palette-map")
async def palette_map_endpoint(
    image: UploadFile = File(..., description="PNG 图像文件"),
    palette: str = Form("pico-8", description="调色板名称或自定义色值"),
    dither: str = Form("none", description="抖动模式: none / ordered / floyd-steinberg"),
):
    """调色板映射接口"""
    if dither not in ("none", "ordered", "floyd-steinberg"):
        raise HTTPException(status_code=400, detail="dither 必须为 none / ordered / floyd-steinberg")

    try:
        img = await _read_image(image)
        return _image_response(palette_map(img, palette, dither))
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调色板映射失败: {str(e)}")


@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/edge-darken")
async def edge_darken_endpoint(
    image: UploadFile = File(..., description="PNG/JPG 图像文件"),
    tolerance: float = Form(5.0, description="Delta-E 容差，范围 1.0-20.0"),
    strength: float = Form(0.3, description="加深强度，范围 0.1-0.7"),
):
    """边缘加深接口"""
    if tolerance < 1.0 or tolerance > 20.0:
        raise HTTPException(status_code=400, detail="tolerance 必须在 1.0-20.0 之间")
    if strength < 0.1 or strength > 0.7:
        raise HTTPException(status_code=400, detail="strength 必须在 0.1-0.7 之间")

    try:
        img = await _read_image(image)
        return _image_response(edge_darken(img, tolerance, strength))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"边缘加深失败: {str(e)}")


@app.post("/bg-transparent")
async def bg_transparent_endpoint(
    image: UploadFile = File(..., description="PNG/JPG 图像文件"),
    tolerance: float = Form(5.0, description="Delta-E 容差，范围 1.0-20.0"),
):
    """背景透明化接口"""
    if tolerance < 1.0 or tolerance > 20.0:
        raise HTTPException(status_code=400, detail="tolerance 必须在 1.0-20.0 之间")

    try:
        img = await _read_image(image)
        return _image_response(bg_transparent(img, tolerance))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"背景透明化失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
