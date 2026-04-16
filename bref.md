# 背景色统一备忘

## 问题

AI 生成的 2D 图像背景色肉眼看似统一，实际有大量微小差异色值，且颜色不一定符合预期（指令遵循问题）。
当前工作流需要人工逐张处理背景色，是整个流水线的堵点。

## 算法：聚类锚定 + Lab 容差扩展

1. 采样四边像素 → 转 Lab 色彩空间
2. K-Means(K=3) 聚类，取最大簇中心作为「背景锚定色」
3. 全图 Lab 转换，计算每像素到锚定色的 Delta-E 距离
4. 生成候选掩码（Delta-E ≤ tolerance）
5. 从四边 BFS flood fill（4-连通），仅保留边缘连通的候选区域
6. 掩码区域替换为目标色 `#808080`

### 为什么 Lab 不用 RGB

Lab 是感知均匀空间，Delta-E 直接对应人眼色差。
RGB 中相同距离的两对颜色在视觉上差异可能完全不同。

### 为什么 K=3

边缘通常 1 个主背景色 + 1-2 个前景角色溢出到边缘，3 簇足以分离。

### 目标背景色 #808080

中性灰，模型归一化到 [-1,1] 后接近 0（中性），不产生色彩偏差，与大多数角色颜色对比度好。

## API 端点

`POST /unify-background` — 参数：image(file), tolerance(float, 默认5.0, 范围1-20), target_color(str, 默认"#808080")

## 文件

| 文件 | 用途 |
|------|------|
| `bg_unify.py` | 核心算法（Lab 转换、边缘聚类、BFS flood fill） |
| `api.py` | `/unify-background` 端点 |
| `do_task.md` | 任务拆分和进度跟踪 |

无新依赖，纯 numpy + PIL 实现 Lab 转换。

## 项目上下文

像素化（Pixelization）是基于 CycleGAN 的图像风格迁移项目，将普通图片转换为像素画。

**推理管线**：`输入图片 → preprocess → RGBEnc → RGBDec(code) → AliasNet → 后处理 → 输出`

双后端架构：Python FastAPI（开发）+ Go ONNX Runtime（生产），共享 `static/index.html` 前端。

---

最后更新：2026-04-16
