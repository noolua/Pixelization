# 项目备忘

## 像素化工具 — 整体架构

基于 SIGGRAPH Asia 2022 论文的像素化工具，将普通图像转换为像素风格艺术画。

**双后端**：Python FastAPI（开发）+ Go ONNX Runtime（生产），共享 `static/` 前端。

**前端**：三个文件 — `index.html`（HTML 结构）、`style.css`（样式）、`app.js`（逻辑）。可配置管线架构，用户可动态增删排列处理步骤。

**已有 API 端点**：
- `POST /pixelize` — 像素化（参数: cell_size 2-8）
- `POST /optimize-colors` — K-Means 颜色优化（参数: target_colors 2-256）
- `POST /unify-background` — 背景色统一（参数: tolerance, target_color）

---

## 背景色统一

### 问题

AI 生成的 2D 图像背景色肉眼看似统一，实际有大量微小差异色值，且颜色不一定符合预期（指令遵循问题）。
当前工作流需要人工逐张处理背景色，是整个流水线的堵点。

### 算法：聚类锚定 + Lab 容差扩展

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

---

## Batch 批量配置

### 业务背景

实际使用场景：有一批 512×512 的 AI 生成图标，需要转成多种规格的像素画（如 72×72、48×48、32×32）。
美术拿到拼图后在 Aseprite 中精修，所以拼图必须是像素级干净的 PNG。

### Web 页面的角色

Web 页面是**配置设计器和调参预览器**，不是最终执行工具。

**工作流**：
```
Web 上传样本图 → 调参预览 → 导出 batch 配置 JSON → Python 脚本批量跑大量图像
```

Batch 配置 JSON 是 Web 和 Python 之间的共享产物。

### 拼图设计

**目的**：给美术一张完整的工作稿，包含参考图 + 各规格结果，在 Aseprite 中打开后方便观察和精修。

**瓦片对齐规则**：
- 瓦片大小 16×16 像素
- 每个结果图像的**左上角**对齐到 16px 网格（坐标 = 16×gx, 16×gy）
- 图像数据原样绘制，**不填充不裁剪**
- 源图尺寸不一定是 16 的倍数，不需要补齐

**参考图**：不是特殊处理，就是 batch 配置中面积最大的那个输出结果。

**拼图形状**：尽量方正（近似正方形），使用瓦片网格的贪心摆放算法。

### 配置格式（Web 和 Python 共享）

```json
{
  "name": "图标多规格",
  "tile_size": 16,
  "configs": [
    { "name": "大图参考", "pipeline": [{"type":"pixelize","params":{"cell_size":4}}, {"type":"optimize-colors","params":{"target_colors":64}}] },
    { "name": "72x72", "pipeline": [{"type":"pixelize","params":{"cell_size":7}}, {"type":"optimize-colors","params":{"target_colors":32}}] },
    { "name": "48x48", "pipeline": [{"type":"pixelize","params":{"cell_size":8}}, {"type":"optimize-colors","params":{"target_colors":16}}] },
    { "name": "32x32", "pipeline": [{"type":"pixelize","params":{"cell_size":16}}, {"type":"optimize-colors","params":{"target_colors":8}}] }
  ]
}
```

### 任务拆分（详见 do_task.md）

| 任务 | 内容 |
|------|------|
| 任务 12 | Batch 数据模型 + 持久化 |
| 任务 13 | Batch 面板 UI |
| 任务 14 | 预览执行 + 瓦片拼图 |
| 任务 15 | Python 批量处理脚本 |

### 关键约束

- 管线执行引擎（`executePipe` / `runAll`）**不动**，复用
- batch 配置需持久化到 localStorage（可导出给 Python）
- 拼图输出为 PNG，像素级干净，用 Aseprite 打开无异常
- 最多三个前端文件（index.html、style.css、app.js）
- Python 脚本通过 HTTP API 调用远程服务，支持 `--server` 指定地址

---

## 新增 Pipe 规划

基于像素游戏美术工作流中的高频需求，规划四个新的原子图像处理 pipe。

### 共享基础：前景检测

蒙版轮廓和外描边都依赖同一个**前景检测**逻辑：
- 有 alpha 通道 → `alpha > 0` 即前景
- 无 alpha → 与背景色（如 `#808080`）不同的像素即前景

### 1. 调色板映射 `palette-map`

将图像颜色限定到预设的复古调色板，用于制作遵循特定风格的像素画素材。

**算法**：对每个像素，在 Lab 空间找目标调色板中 Delta-E 距离最近的颜色替换。复用 `bg_unify.py` 的 `rgb_to_lab()`。

**参数**：
- `palette` — 预设调色板名或自定义色值列表
- `dither` — 抖动模式（none / ordered / floyd-steinberg）

**内置调色板候选**：PICO-8(16色)、GameBoy(4色)、NES(54色)、Commodore 64(16色)、Endesga 32(32色)

### 2. 蒙版轮廓 `silhouette`

将前景像素替换为纯色，生成用于受击闪白、阴影投影等的剪影图。

**算法**：前景检测 → 填充为指定纯色。

**参数**：
- `mask_color` — 填充色，默认 `#FFFFFF`
- `threshold` — 背景色容差（可选，处理未完全统一的边缘残留）

**输出尺寸**：与输入一致

**用途**：游戏中角色受伤时交替显示原始 sprite 和纯色剪影。

### 3. 外描边 `outline`

在 sprite 外边缘绘制指定宽度的描边，提升角色在复杂背景上的可读性。

**算法**：前景检测 → 找到前景边缘 → 向外扩展 N 像素 → 填充描边色。结果为原图 + 外扩描边。

**参数**：
- `outline_color` — 描边颜色，默认 `#000000`
- `outline_width` — 描边宽度 1-2px（像素画通常 1px）

**输出尺寸**：可能比输入大（外扩像素）

**用途**：像素游戏角色的标准视觉处理。

### 4. 灰度 `grayscale`

将图像转为量化灰度，可指定灰阶级数。

**算法**：RGB → 亮度值 → 量化到指定级数。

**参数**：
- `gray_levels` — 灰阶级数（如 4=GameBoy 风格，8=阴影版，0=连续灰度）

**用途**：可独立使用，也可与调色板映射组合（灰度 → 调色板映射 = 任意单色版本）。

### 组合示例

```
# 受击动画素材
像素化 → 颜色优化 → 蒙版轮廓

# GameBoy 风格
像素化 → 灰度(gray_levels=4) → 调色板映射(palette=gameboy)

# 清晰角色卡
背景统一 → 像素化 → 颜色优化 → 外描边
```

---

最后更新：2026-04-17
