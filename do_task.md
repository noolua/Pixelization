# 像素化 Web 服务开发任务

## 目标

基于现有的 `test_pro.py`，创建一个简化的 Web 服务，支持通过 HTTP 接收前端上传的 PNG 图像，根据 cell 参数生成像素化图像并返回。

## 功能需求

### 后端
1. **HTTP 服务**：接收前端上传的 PNG 图像
2. **参数处理**：根据页面选择的 cell 参数（2-8）进行像素化
3. **图像返回**：返回生成的像素化图像

### 前端
1. **图像上传**：支持拖拽或点击上传 PNG 图像
2. **参数选择**：cell 参数选择器（2-8）
3. **结果展示**：原图与像素化后的图片对比显示

## 实现方案

### 骨架优先原则

**第一阶段：骨架（最小可行实现）**

#### 后端（FastAPI）
- 端点：`POST /pixelize`
- 接收：`multipart/form-data` (image + cell_size)
- 返回：`image/png`
- 复用：`test_pro.py` 中的 `Model` 类和 `pixelize` 方法

#### 前端（纯 HTML + JS）
- 单文件 `index.html`
- 原生 Fetch API
- 并排对比展示

**第二阶段：血肉**
- 错误处理（文件格式、大小限制）
- 进度提示
- 日志记录

**第三阶段：皮肤**
- 样式美化
- 性能优化（缓存模型）
- 部署配置

## 技术选型

| 组件 | 技术方案 | 理由 |
|------|----------|------|
| 后端框架 | FastAPI | 轻量、原生异步支持、自动文档 |
| 模型复用 | 直接导入 `test_pro.py` 的 `Model` 类 | 最小代码修改 |
| 前端 | HTML + Vanilla JS | 无构建依赖，快速验证 |
| 图像传输 | Base64 或 multipart | 兼容性好 |

## 目录结构

```
Pixelization/
├── api.py              # 新增：FastAPI 服务入口
├── static/
│   └── index.html      # 新增：前端页面
├── test_pro.py         # 复用：模型加载和推理
├── models/             # 现有模型
└── checkpoints/        # 现有权重
```

## 实现步骤

### Step 1: 后端服务骨架
1. 创建 `api.py`
2. 导入 `test_pro.py` 的 `Model` 类
3. 实现 `/pixelize` POST 端点
4. 本地测试：curl 验证

### Step 2: 前端页面骨架
1. 创建 `static/index.html`
2. 实现文件上传
3. 实现 fetch 调用 `/pixelize`
4. 实现图片对比展示

### Step 3: 联调
1. 本地启动服务
2. 浏览器访问测试
3. 修复问题

### Step 4: 增强功能
1. 错误处理
2. 加载状态
3. 参数范围验证

## 关键接口定义

### POST /pixelize

**请求**
```
Content-Type: multipart/form-data

image: file (PNG)
cell_size: int (2-8)
```

**响应**
```
Content-Type: image/png

<binary image data>
```

**错误**
```
Content-Type: application/json

{"error": "error message"}
```

## 注意事项

1. **模型加载优化**：服务启动时加载模型，避免每次请求重新加载
2. **设备选择**：沿用 `test_pro.py` 的 MPS/CPU 自动检测
3. **路径处理**：注意模型路径 `./checkpoints/demo/` 和 `./alias_net.pth`
4. **临时文件**：处理完成后清理临时图像文件

## 验收标准

- [ ] 上传 PNG 图片成功返回像素化结果
- [ ] cell 参数正确影响像素化程度
- [ ] 前端正确展示原图与结果对比
- [ ] 错误情况有友好提示

---
创建日期: 2026年4月12日
