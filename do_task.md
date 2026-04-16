# 背景色统一功能 — 任务跟踪

## Context

当前工作流：AI 生成 2D 图像 → **人工处理背景色（堵点）** → 像素化 → 颜色优化。

AI 生成的图像背景色肉眼看似统一，实际有大量微小差异色值，且颜色不一定符合预期。当前靠人工逐张处理，非常耗时。

**目标**：新增 `/unify-background` API 端点，用"聚类锚定 + Lab 容差扩展"算法自动统一背景色为 `#808080`，消除人工堵点。

**范围限制**：本次只开发 Python 版本（`api.py` + `bg_unify.py`），不动 `server/` 目录下的 Go 代码。待 Python 版测试稳定后再考虑迁移到 Go。

---

## 算法：聚类锚定 + Lab 容差扩展

```
输入图像
  → 采样四边像素，转 Lab 色彩空间
  → K-Means(K=3) 聚类，取最大簇中心作为「背景锚定色」
  → 全图转 Lab，计算每像素到锚定色的 Delta-E 距离
  → 生成候选掩码（Delta-E ≤ tolerance）
  → 从四边 BFS flood fill，仅保留边缘连通的候选区域
  → 掩码区域替换为目标色 #808080
  → 输出图像
```

**为什么用 Lab 不用 RGB**：Lab 是感知均匀色彩空间，Delta-E 直接对应人眼色差。RGB 中距离=10 的两对颜色，在视觉上可能差异巨大。

**为什么 K=3**：边缘通常包含 1 个主背景色 + 1-2 个前景溢出到边缘的颜色，3 簇足以分离又不会过度分割。

---

## 任务 1：骨架实现

**状态：待开始**

- [ ] 创建 `bg_unify.py`
  - [ ] `rgb_to_lab()` — 纯 numpy 实现 sRGB→线性RGB→XYZ→Lab（~25行，标准公式）
  - [ ] `unify_background(image, tolerance, target_rgb)` — 主函数
    - [ ] 边缘采样（自适应宽度 `max(3, min(H,W)//50)`）
    - [ ] 边缘像素简单均值作为锚定色（先跳 K-Means）
    - [ ] 全图 Lab 转换 + Delta-E 预计算
    - [ ] `collections.deque` BFS flood fill（4-连通）
    - [ ] 掩码替换为目标色
- [ ] 在 `api.py` 添加 `POST /unify-background` 端点
  - 参数：image(file), tolerance(float, 默认5.0, 范围1-20), target_color(str, 默认"#808080")
  - 遵循 `/optimize-colors` 端点模式
- [ ] 端到端测试：用 AI 生成的 2D 图像验证效果

## 任务 2：K-Means 锚定 + 健壮性

**状态：待开始**

- [ ] 给边缘采样加 K-Means(K=3) 聚类，取最大簇中心为锚定色
  - 参照 `api.py:kmeans_colors()` 的加权 K-Means 模式
- [ ] 边界情况处理
  - [ ] 带 alpha 通道的图像先转 RGB
  - [ ] 极小图片（宽或高 < 10px）
  - [ ] 全图无合格候选像素的情况

## 任务 3：调试验证

**状态：待开始**

- [ ] 用实际 AI 生成 2D 图像测试不同 tolerance 值
- [ ] 确认默认 tolerance=5.0 是否合适，根据实测调整
- [ ] 确认背景统一后送入像素化流程效果正常

---

## API 契约

`POST /unify-background`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| image | file（必填） | — | PNG/JPG 图像 |
| tolerance | float | 5.0 | Delta-E 容差，范围 1.0-20.0，越大越激进 |
| target_color | str | "#808080" | 目标背景色，十六进制 |

响应：`image/png` 二进制，尺寸与输入一致。

## 文件变更

| 文件 | 变更 |
|------|------|
| `bg_unify.py`（新建） | 核心算法：Lab 转换、边缘采样聚类、BFS flood fill |
| `api.py`（修改） | 新增端点 + 1 行 import |

无新依赖 — numpy + PIL 已在 requirements.txt 中。

---
创建日期: 2026年4月16日
