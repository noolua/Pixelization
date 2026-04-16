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

**状态：已完成** ✅

- [x] 创建 `bg_unify.py`
  - [x] `rgb_to_lab()` — 纯 numpy 实现 sRGB→线性RGB→XYZ→Lab（~25行，标准公式）
  - [x] `unify_background(image, tolerance, target_rgb)` — 主函数
    - [x] 边缘采样（自适应宽度 `max(3, min(H,W)//50)`）
    - [x] 边缘像素简单均值作为锚定色（先跳 K-Means）
    - [x] 全图 Lab 转换 + Delta-E 预计算
    - [x] `collections.deque` BFS flood fill（4-连通）
    - [x] 掩码替换为目标色
- [x] 在 `api.py` 添加 `POST /unify-background` 端点
  - 参数：image(file), tolerance(float, 默认5.0, 范围1-20), target_color(str, 默认"#808080")
  - 遵循 `/optimize-colors` 端点模式
- [x] 端到端测试：用 AI 生成的 2D 图像验证效果

## 任务 2：K-Means 锚定 + 健壮性

**状态：已完成** ✅

- [x] 给边缘采样加 K-Means(K=3) 聚类，取最大簇中心为锚定色
  - 参照 `api.py:kmeans_colors()` 的加权 K-Means 模式
- [x] 边界情况处理
  - [x] 带 alpha 通道的图像先转 RGB
  - [x] 极小图片（宽或高 < 10px）
  - [x] 全图无合格候选像素的情况

## 任务 3：调试验证

**状态：已完成** ✅

- [x] 用实际 AI 生成 2D 图像测试不同 tolerance 值
- [x] 确认 tolerance 越高背景越纯净（符合预期）
- [x] 默认 tolerance=5.0 作为初始值

## 任务 4：前端集成

**状态：已完成** ✅

- [x] 修改 `static/index.html`，新增背景统一步骤
  - [x] 在上传区与像素化控制之间，添加「背景统一」控制区
    - checkbox 开关（默认关闭）
    - tolerance 滑块（范围 1-20，默认 5.0）
    - 目标色输入（默认 #808080）
    - "统一背景"按钮
  - [x] 点击"统一背景"后调用 `/unify-background`，生成 `unifiedBlob`
  - [x] 像素化流程：若开启了背景统一，用 `unifiedBlob` 作为输入；否则用原始文件
  - [x] 结果展示区：原图 → 背景统一（仅开启时显示）→ 像素化 → 颜色优化
  - [x] 缩放控制、下载按钮与现有列保持一致
- [x] 颜色优化后增加背景统一可选流程
  - [x] 优化结果列下方新增"背景色统一"控制
  - [x] 结果区新增"优化后背景统一"列和下载按钮

---

## 远期规划：可配置管线架构

**状态：已完成** ✅

### 设计理念

当前前端是硬编码的线性流程。实际使用中用户体会：图像处理本质上是一条**可配置的装配线**，每个 API 是独立的 **pipe**，用户可以像组装自行车/摩托车一样选择和排列 pipe。

### 核心概念

```
Pipe（管道）：一个独立的无状态图像处理单元
  - 每个 pipe 对应一个 API 端点
  - 输入：一张图像 + 参数
  - 输出：一张图像
  - 当前已有 pipe：
    - pixelize（像素化）
    - optimize-colors（颜色优化）
    - unify-background（背景统一）
```

### 自动化边界

**适合自动化的**：高分辨率预处理阶段的 pipe（背景统一、颜色优化），每个像素"权重"低，不影响语义。

**不适合自动化的**：32x32 以下像素画的精修操作（描边、特征刻画等）。该分辨率下每颗像素是语义单位（一个像素=一只眼睛），程序无法判断像素取舍，需要人类审美和意图。

### 技术要点

- 后端 pipe 接口保持无状态独立，不做改动
- 前端维护管线配置和中间 blob 传递逻辑
- 管线配置持久化到 localStorage，下次打开自动恢复

---

## 任务 5：Pipe 注册表 + 管线数据模型

**状态：已完成** ✅

建立数据层基础，此阶段不改动现有 UI，只新增 JS 数据结构和工具函数。

- [ ] 定义 Pipe 注册表 `PIPE_REGISTRY`
  ```
  每个注册项包含：
  {
    type: 'pixelize',              // 唯一标识
    label: '像素化',                // 显示名
    endpoint: '/pixelize',          // API 端点
    icon: '🔲',                     // 可选图标
    params: [                       // 参数 schema
      { key: 'cell_size', label: '像素化程度', type: 'select',
        options: [{value:2,label:'2 (轻微)'}, ...], default: 4 }
    ],
    resultType: 'pixelated'         // 结果类型（影响缩放渲染方式）
  }
  ```
  - [ ] 注册 `pixelize` pipe
  - [ ] 注册 `optimize-colors` pipe
  - [ ] 注册 `unify-background` pipe
- [ ] 定义管线数据模型 `PipelineState`
  ```js
  {
    pipes: [                        // 有序 pipe 列表
      { type: 'unify-background', id: 'pipe_0', params: { tolerance: 5.0, target_color: '#808080' } },
      { type: 'pixelize', id: 'pipe_1', params: { cell_size: 4 } },
      ...
    ],
    blobs: { 'pipe_0': Blob|null, 'pipe_1': Blob|null, ... },  // 中间结果
    executedUpTo: -1                 // 已执行到的 pipe 索引，-1 表示未执行
  }
  ```
- [ ] 实现管线操作函数
  - [ ] `addPipe(state, type, index?)` — 在指定位置插入 pipe
  - [ ] `removePipe(state, id)` — 删除 pipe 并清理其 blob
  - [ ] `movePipe(state, fromIndex, toIndex)` — 移动 pipe 顺序
  - [ ] `updateParam(state, id, key, value)` — 更新某个 pipe 的参数
  - [ ] `getInputBlob(state, pipeIndex)` — 获取某 pipe 的输入 blob（前一个 pipe 的输出，或原始文件）
  - [ ] `resetFrom(state, pipeIndex)` — 从某位置开始清除所有 blob（参数变更后调用）
- [ ] 单元验证：在浏览器控制台手动调用上述函数，确认数据操作正确

---

## 任务 6：动态 DOM 渲染引擎

**状态：已完成** ✅

用管线数据驱动 DOM 生成，替换当前硬编码的 HTML 结构。这是重构的核心。

- [ ] 设计 pipe 卡片 HTML 结构
  ```
  ┌─ pipe-card ──────────────────────┐
  │ ┌─ card-header ────────────────┐ │
  │ │ ⋮⋮  🔲 像素化        × 删除 │ │  ← 拖拽手柄 + 名称 + 删除
  │ └─────────────────────────────┘ │
  │ ┌─ card-params ───────────────┐ │  ← 点击 header 展开/折叠
  │ │ 像素化程度: [4 ▼]           │ │
  │ │          [执行此步骤]        │ │
  │ └─────────────────────────────┘ │
  │ ┌─ card-result ───────────────┐ │  ← 执行后显示
  │ │ [预览图]   [下载] [缩放]    │ │
  │ └─────────────────────────────┘ │
  └─────────────────────────────────┘
          │
          ▼  ← 管道连接线 (CSS ::after)
  ```
- [ ] 实现 `renderPipeline(state)` 函数
  - [ ] 遍历 `state.pipes`，为每个 pipe 生成卡片 DOM
  - [ ] 根据 `PIPE_REGISTRY[type].params` 动态生成参数控件（select/range/number/color）
  - [ ] 卡片之间渲染连接线（CSS 伪元素或 border）
- [ ] 实现 `renderResult(state, pipeId)` 函数
  - [ ] 检查 `state.blobs[pipeId]` 是否存在
  - [ ] 有 blob → 显示预览图 + 下载按钮 + 缩放控件
  - [ ] 无 blob → 显示"未执行"占位
- [ ] 参数控件事件绑定
  - [ ] 每个控件 change 事件 → `updateParam()` + `resetFrom()` + 重新渲染受影响的卡片
- [ ] 增量渲染优化（可选）
  - [ ] 仅更新变化的卡片，而非整体重渲染
  - [ ] 首期可先全量渲染，后续优化

---

## 任务 7：管线编排 UI（拖拽排序 + 增删 + 预设）

**状态：已完成** ✅

实现管线的可视化编排交互。

- [ ] 管线顶部工具栏
  - [ ] "添加步骤"下拉按钮 — 列出 PIPE_REGISTRY 中所有可用 pipe 类型
  - [ ] "预设模板"下拉按钮 — 内置常用管线组合
  - [ ] "重置默认"按钮 — 恢复为默认管线
- [ ] 拖拽排序
  - [ ] 使用 HTML5 Drag and Drop API（或引入 SortableJS ~10KB）
  - [ ] dragstart → 记录拖拽源 pipe id
  - [ ] dragover → 显示插入位置指示器
  - [ ] drop → 调用 `movePipe()` + 重新渲染 + 清除受影响的 blob
  - [ ] 拖拽过程中禁用"执行"按钮（防止状态不一致）
- [ ] 添加 pipe
  - [ ] 点击"添加步骤" → 选择 pipe 类型 → 调用 `addPipe()` → 渲染新卡片
  - [ ] 新 pipe 默认插入到管线末尾
  - [ ] 同一类型 pipe 可重复添加（如两次背景统一、参数不同）
- [ ] 删除 pipe
  - [ ] 卡片右上角 × 按钮 → 调用 `removePipe()` → 重新渲染
  - [ ] 管线最少保留 1 个 pipe（防止空管线）
- [ ] 预设模板定义
  ```js
  const PRESETS = {
    'default': ['pixelize', 'optimize-colors'],
    'with-bg': ['unify-background', 'pixelize', 'optimize-colors'],
    'bg-only': ['unify-background'],
  };
  ```
  - [ ] 选择预设 → 生成新 PipelineState → 渲染

---

## 任务 8：管线执行引擎 + 中间结果管理

**状态：已完成** ✅

实现管线的数据流执行和结果展示。

- [ ] 单步执行 `executePipe(state, pipeIndex)`
  - [ ] 获取输入 blob：`getInputBlob(state, pipeIndex)`
  - [ ] 构建 FormData（image + pipe.params）
  - [ ] POST 到 pipe 对应的 API 端点
  - [ ] 响应 blob 存入 `state.blobs[pipeId]`
  - [ ] 更新 `state.executedUpTo`
  - [ ] 重新渲染该 pipe 的结果区域
  - [ ] loading/error 状态管理
- [ ] 从头执行 `executePipeline(state)`
  - [ ] "处理图像"主按钮 → 依次执行所有 pipe
  - [ ] 逐个 pipe 串行调用 `executePipe()`
  - [ ] 每个 pipe 完成后立即渲染结果（用户可见中间进度）
  - [ ] 任一 pipe 失败则停止，显示错误信息
- [ ] 从中间节点重执行 `executeFrom(state, pipeIndex)`
  - [ ] 参数修改后，pipe 卡片显示"重新执行"提示
  - [ ] 点击 → 从该 pipe 开始执行到末尾
  - [ ] 清除该 pipe 及之后的所有 blob
- [ ] 缩放控制
  - [ ] 每个 pipe 结果独立的缩放状态
  - [ ] 像素化类结果使用 `image-rendering: pixelated`
- [ ] 颜色统计
  - [ ] pixelize 和 optimize-colors 的结果自动统计颜色数并显示
  - [ ] 复用现有 `countColors()` 函数

---

## 任务 9：localStorage 持久化 + 导出

**状态：已完成** ✅

- [ ] 管线配置持久化
  - [ ] `savePipeline(state)` — 将 pipes 数组和 params 序列化为 JSON 存入 localStorage
    - 存储键：`pixelization_pipeline`
    - 只存结构配置，不存 blob（localStorage 容量限制）
  - [ ] `loadPipeline()` — 页面加载时从 localStorage 读取配置
    - 无历史配置 → 使用默认预设
    - 有历史配置 → 恢复管线结构和参数（blob 不恢复，需重新执行）
  - [ ] 自动保存触发点：增删 pipe、移动顺序、修改参数时
  - [ ] 防抖处理：参数滑块拖动时 300ms 防抖后再保存
- [ ] "重置默认"按钮
  - [ ] 清除 localStorage 对应键
  - [ ] 恢复为默认管线配置
- [ ] 一键导出中间结果
  - [ ] 管线工具栏添加"导出所有结果"按钮
  - [ ] 将所有已执行的 blob 打包为独立文件下载
  - [ ] 文件命名规则：`01_unify-background.png`, `02_pixelize.png`, ...
  - [ ] 实现方式：逐个触发下载（无需引入 zip 库）

---

## 任务 10：样式重构 + UX 打磨

**状态：已完成** ✅

- [x] 布局重构
  - [x] 管线编排区：纵向卡片流（移动端友好），每张卡片宽度 100%
  - [x] 卡片展开/折叠动画（CSS transition max-height）
  - [x] 卡片之间的管道连接线（竖线 + 箭头，CSS 伪元素实现）
  - [x] 拖拽时的视觉反馈（虚线边框、阴影提升）
- [x] 状态指示
  - [x] 未执行：灰色边框
  - [x] 执行中：蓝色边框 + 旋转图标
  - [x] 已完成：绿色边框 + 缩略图
  - [x] 执行失败：红色边框 + 错误信息
  - [x] 需要重新执行（参数已变更）：橙色边框 + 提示文案
- [x] 响应式适配
  - [x] 桌面端（≥768px）：管线区最大宽度 800px 居中
  - [x] 移动端（<768px）：全宽，参数面板堆叠
- [x] 上传区保留不变
  - [x] 上传区与管线区分离，上传后管线区才出现
  - [x] 切换图片时清除所有 blob，保留管线结构
- [x] 清理遗留代码
  - [x] 删除硬编码的 image-box DOM
  - [x] 删除全局 blob 变量（selectedFile 除外）
  - [x] 删除旧的事件绑定代码

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
| `static/index.html`（修改） | 新增背景统一控制区 + 管线串联 |

无新依赖 — numpy + PIL 已在 requirements.txt 中。

---

## 任务 11：前端代码模块化重构

**状态：已完成** ✅

### 背景

当前 `static/index.html` 是 992 行的单文件，CSS/HTML/JS 全部混在一起。拆为三个文件，各司其职。

### 目标文件结构

```
static/
  index.html       ~50行（纯 HTML 结构）
  style.css        ~170行（全部样式）
  app.js           ~770行（全部逻辑，按注释分段组织）
```

### 实施步骤

- [x] 11.1 提取 CSS → `static/style.css`
  - [x] 将 `<style>` 块内全部内容移入 `static/style.css`
  - [x] `index.html` 中 `<style>...</style>` 替换为 `<link rel="stylesheet" href="/static/style.css">`
- [x] 11.2 提取 JS → `static/app.js`
  - [x] 将 `<script>` 块内全部内容移入 `static/app.js`
  - [x] `index.html` 中 `<script>...</script>` 替换为 `<script src="/static/app.js"></script>`
- [x] 11.3 清理 `index.html`
  - [x] 确认只剩下 DOCTYPE、HTML 结构、link、script 引用
  - [x] 确认所有 `id` 与 JS 中的引用一致
- [x] 11.4 JS 内部用注释分段组织（已有，确认清晰）
  - [ ] 数据层
  - [ ] 状态 & DOM 引用
  - [ ] 工具函数
  - [ ] 上传处理
  - [ ] 管线渲染
  - [ ] 拖拽排序
  - [ ] 管线执行
  - [ ] 工具栏
  - [ ] 保存/加载配置
  - [ ] 管线名称
  - [ ] 导入/导出
  - [ ] 初始化
- [x] 11.5 验证
  - [ ] `python api.py` 启动后访问页面，确认所有功能正常
  - [ ] 上传 → 执行 → 查看 → 添加/删除/拖拽 → 保存/加载 → 导入/导出
  - [ ] 浏览器控制台无报错
- [x] 11.6 后端路由确认
  - [x] Python (`api.py`)：确认 StaticFiles 中间件对 `.css` 和 `.js` 文件的 MIME 类型正确
  - [x] Go (`server/`)：确认静态文件路由能服务新文件

### 约束

- **最多三个文件**：`index.html`、`style.css`、`app.js`
- **功能不变**，纯拆分，不改逻辑
- **后端零改动**（仅需确认路由能服务 `.css` / `.js`）

---

## 任务 12：Batch 配置数据模型 + 持久化

**状态：已完成** ✅

建立 batch 的数据层基础，定义与 Web/Python 共享的配置格式。

- [x] 12.1 定义 batch 配置 JSON schema
  ```json
  {
    "name": "图标多规格",
    "tile_size": 16,
    "configs": [
      { "name": "大图参考", "pipeline": [{"type":"pixelize","params":{"cell_size":4}}, ...] },
      { "name": "72x72", "pipeline": [...] },
      ...
    ]
  }
  ```
  - `tile_size`：瓦片大小（像素），拼图时用于网格对齐
  - `configs[]`：有序配置列表，每个包含 name + pipeline 步骤数组
  - `pipeline[]` 中每个步骤包含 `type`（对应 PIPE_REGISTRY）和 `params`
- [x] 12.2 实现 batch 数据操作函数
  - `createBatch(name, tileSize)` — 创建空 batch
  - `addBatchConfig(batch, name, pipelineSteps)` — 添加一个配置
  - `removeBatchConfig(batch, index)` — 删除配置
  - `moveBatchConfig(batch, from, to)` — 调整配置顺序
  - `updateBatchConfig(batch, index, steps)` — 更新配置的 pipeline
- [x] 12.3 持久化
  - localStorage 键：`pixelization_batches`
  - 存储所有 batch 配置（不含执行结果）
  - `saveBatches(batches)` / `loadBatches()` 函数
  - 支持多个 batch（用户可能有不同用途的 batch）
- [x] 12.4 导入/导出
  - 导出：将单个 batch 配置序列化为 JSON 文件下载
  - 导入：读取 JSON 文件，验证格式，加入 batches 列表
  - 导出的 JSON 即为 Python `batch_process.py` 的输入格式

---

## 任务 13：Batch 面板 UI

**状态：已完成** ✅

在现有管线编辑区下方新增 Batch 面板，支持可视化管理 batch 配置。

- [x] 13.1 HTML 结构
  - 在 `pipeline-section` 下方新增 `batch-section`
  - 包含：batch 选择/新建、配置列表、预览画布、操作按钮
- [x] 13.2 Batch 选择器
  - 下拉选择已有 batch 或"新建 batch"
  - batch 名称编辑（同管线标题模式）
  - 删除 batch
- [x] 13.3 配置列表 UI
  - 每个配置显示为一行：序号 + 名称 + pipeline 概要（如"像素化(4) → 颜色优化(32)"）+ 编辑/删除按钮
  - 支持拖拽排序
  - "添加配置"按钮 — 可从"保存的配置"中选取，或手动创建
  - 点击配置行可展开编辑 pipeline 步骤和参数（复用现有 pipe-card 样式）
- [x] 13.4 样式
  - 与现有管线区风格统一（白底圆角卡片、蓝色主色调）
  - 配置列表紧凑排列（单行概要，展开编辑）
  - 响应式适配（复用现有 media query）

---

## 任务 14：Batch 预览执行 + 瓦片拼图

**状态：已完成** ✅

执行 batch 中所有配置，将结果按 16px 网格拼图展示。

- [x] 14.1 批量执行引擎
  - `executeBatch()` 函数
  - 遍历 `batch.configs`，对每个配置：
    - 按顺序执行 pipeline 中每一步（复用 `executePipe` 逻辑）
    - 收集最终输出 blob
  - 存储结果到 `batchResults[]` 数组
  - 逐个配置更新 UI 进度
  - 任一配置失败不阻塞其他，标记错误
- [x] 14.2 瓦片拼图算法
  - 输入：`batchResults[]`（每个含 name + blob + Image 对象）
  - 步骤：
    1. 将每个结果加载为 Image，获取 (w, h)
    2. 计算每个结果的瓦片占位 `(ceil(w/tile_size), ceil(h/tile_size))`
    3. 找到面积最大的结果 → 排在最前面（作为参考图）
    4. 其余按面积从大到小排序
    5. 贪心摆放：维护已占用区域，逐个找到第一个不重叠的 (gx, gy) 网格位置
    6. 目标：最终画布尽量方正
  - 简化策略（首期）：
    - 按面积排序后，固定列数 `cols = ceil(sqrt(总面积))`
    - 逐行逐列摆放，行高取当前行最大瓦片高度
- [x] 14.3 Canvas 渲染
  - 创建 Canvas，尺寸 = 最大 (gx*tile_size + w, gy*tile_size + h)
  - 透明/白底
  - 每个结果 `drawImage(img, gx*tile_size, gy*tile_size)` — 左上角对齐网格
  - 图像数据原样绘制，不填充不裁剪
- [x] 14.4 预览区 UI
  - 在 batch 配置列表下方显示拼图画布
  - 支持缩放查看（复用现有缩放控件模式）
  - "下载拼图"按钮 — 导出 Canvas 为 PNG
- [x] 14.5 执行状态
  - "执行预览"按钮（需已上传源图）
  - 执行过程中逐个配置显示进度（loading → done/error）
  - 全部完成后自动生成拼图

---

## 任务 15：Python 批量处理脚本（HTTP 模式）

**状态：待实施**

读取 batch 配置 JSON，通过 HTTP API 调用远程服务批量处理图像目录，生成拼图输出。

- [ ] 15.1 创建 `batch_process.py`
  - CLI 参数：`--config <batch.json> --input <图像目录> --output <输出目录> --server <API地址>`
  - `--server` 默认 `http://127.0.0.1:8000`，支持远程服务地址
  - 读取 batch 配置 JSON
  - 遍历输入目录中的所有图像文件
- [ ] 15.2 单图处理逻辑（HTTP）
  - 对每张图像，执行 batch.configs 中每个配置的 pipeline
  - 通过 `requests` 库 POST 到远程 API 端点（`/pixelize`、`/optimize-colors`、`/unify-background`）
  - 逐步串行执行 pipeline，上一步的输出作为下一步的输入
  - 收集每个配置的最终输出
- [ ] 15.3 拼图生成
  - 实现与 JS 端相同的瓦片拼图算法
  - 使用 PIL/Pillow 拼图
  - 输出 PNG 到输出目录，命名规则：`{原图名}_tiled.png`
- [ ] 15.4 容错与日志
  - 单张图像处理失败不阻塞，记录错误继续
  - 输出处理摘要（成功/失败/跳过数量）

---

创建日期: 2026年4月16日
更新日期: 2026年4月17日
