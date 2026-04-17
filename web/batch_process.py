#!/usr/bin/env python3
"""
Batch 图像处理脚本（HTTP 模式）

读取 batch 配置 JSON，通过 HTTP API 调用远程服务批量处理图像目录，生成拼图输出。

用法:
  python batch_process.py --config batch.json --input ./images --output ./output --server http://127.0.0.1:8000
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO


# ============================================================
# API 端点参数映射
# ============================================================

PIPE_ENDPOINTS = {
  'pixelize': {
    'endpoint': '/pixelize',
    'param_types': {'cell_size': int}
  },
  'optimize-colors': {
    'endpoint': '/optimize-colors',
    'param_types': {'target_colors': int}
  },
  'unify-background': {
    'endpoint': '/unify-background',
    'param_types': {'tolerance': float, 'target_color': str}
  }
}

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


# ============================================================
# 单步执行
# ============================================================

def execute_step(server, step, image_bytes):
  """执行单个 pipeline 步骤，返回结果图像 bytes"""
  pipe_info = PIPE_ENDPOINTS.get(step['type'])
  if not pipe_info:
    raise ValueError(f"未知 pipe 类型: {step['type']}")

  url = server.rstrip('/') + pipe_info['endpoint']
  files = {'image': ('input.png', image_bytes, 'image/png')}
  data = {}
  for k, v in step.get('params', {}).items():
    type_fn = pipe_info['param_types'].get(k, str)
    data[k] = str(type_fn(v))

  resp = requests.post(url, files=files, data=data, timeout=120)
  if resp.status_code != 200:
    try:
      detail = resp.json().get('detail', resp.text)
    except Exception:
      detail = resp.text
    raise RuntimeError(f"API 错误 ({resp.status_code}): {detail}")

  return resp.content


def execute_pipeline(server, pipeline, image_bytes):
  """依次执行 pipeline 中每一步，返回最终图像 bytes"""
  current = image_bytes
  for step in pipeline:
    current = execute_step(server, step, current)
  return current


# ============================================================
# 瓦片拼图
# ============================================================

def generate_collage(results, tile_size):
  """
  生成瓦片拼图，算法与 JS 端一致。

  results: [(name, PIL.Image), ...]
  tile_size: 瓦片大小（像素）
  返回: PIL.Image
  """
  items = []
  for name, img in results:
    w, h = img.size
    gw = math.ceil(w / tile_size)
    gh = math.ceil(h / tile_size)
    items.append({'name': name, 'img': img, 'gw': gw, 'gh': gh})

  # 按面积从大到小排序（最大的排最前面）
  items.sort(key=lambda x: x['gw'] * x['gh'], reverse=True)

  if not items:
    return None

  # 贪心网格摆放
  occupied = set()
  total_cells = sum(it['gw'] * it['gh'] for it in items)
  target_cols = max(1, math.ceil(math.sqrt(total_cells)))

  positions = []

  def ok(x, y, gw, gh):
    for dy in range(gh):
      for dx in range(gw):
        if (x + dx, y + dy) in occupied:
          return False
    return True

  def mark(x, y, gw, gh):
    for dy in range(gh):
      for dx in range(gw):
        occupied.add((x + dx, y + dy))

  for item in items:
    placed = False
    gy = 0
    while not placed:
      for gx in range(target_cols + 1):
        if ok(gx, gy, item['gw'], item['gh']):
          mark(gx, gy, item['gw'], item['gh'])
          positions.append({
            'x': gx * tile_size,
            'y': gy * tile_size,
            'item': item
          })
          placed = True
          break
      if not placed:
        gy += 1

  # 计算画布大小
  cw = max(p['x'] + p['item']['img'].size[0] for p in positions)
  ch = max(p['y'] + p['item']['img'].size[1] for p in positions)

  canvas = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
  for p in positions:
    canvas.paste(p['item']['img'], (p['x'], p['y']))

  return canvas


# ============================================================
# 主流程
# ============================================================

def load_batch_config(path):
  """加载并验证 batch 配置 JSON"""
  with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

  if not data.get('name'):
    raise ValueError("配置缺少 'name' 字段")
  if not isinstance(data.get('tile_size'), int) or data['tile_size'] < 1:
    raise ValueError("配置 'tile_size' 必须为正整数")
  if not isinstance(data.get('configs'), list) or len(data['configs']) == 0:
    raise ValueError("配置 'configs' 必须为非空数组")

  for i, cfg in enumerate(data['configs']):
    if not cfg.get('name'):
      raise ValueError(f"配置 {i} 缺少 'name'")
    if not isinstance(cfg.get('pipeline'), list):
      raise ValueError(f"配置 {i} 缺少 'pipeline' 数组")
    for j, step in enumerate(cfg['pipeline']):
      if step.get('type') not in PIPE_ENDPOINTS:
        raise ValueError(f"配置 {i} 步骤 {j}: 未知类型 '{step.get('type')}'")

  return data


def collect_images(input_dir):
  """收集输入目录中所有图像文件"""
  paths = []
  for f in sorted(os.listdir(input_dir)):
    ext = os.path.splitext(f)[1].lower()
    if ext in IMAGE_EXTS:
      paths.append(os.path.join(input_dir, f))
  return paths


def process_image(server, batch_config, image_path):
  """
  对单张图像执行 batch 中所有配置，返回结果列表。

  返回: [(config_name, PIL.Image), ...]
  失败的配置返回 (config_name, error_string)
  """
  with open(image_path, 'rb') as f:
    image_bytes = f.read()

  results = []
  for cfg in batch_config['configs']:
    try:
      result_bytes = execute_pipeline(server, cfg['pipeline'], image_bytes)
      img = Image.open(BytesIO(result_bytes))
      results.append((cfg['name'], img))
    except Exception as e:
      results.append((cfg['name'], str(e)))

  return results


def main():
  parser = argparse.ArgumentParser(description='Batch 图像处理（HTTP 模式）')
  parser.add_argument('--config', required=True, help='Batch 配置 JSON 文件路径')
  parser.add_argument('--input', required=True, help='输入图像目录')
  parser.add_argument('--output', required=True, help='输出目录')
  parser.add_argument('--server', default='http://10.1.2.224:8080', help='API 服务地址 (默认 http://10.1.2.224:8080)')
  args = parser.parse_args()

  # 加载配置
  try:
    batch_config = load_batch_config(args.config)
  except Exception as e:
    print(f"配置加载失败: {e}", file=sys.stderr)
    sys.exit(1)

  print(f"Batch: {batch_config['name']}")
  print(f"配置数: {len(batch_config['configs'])}")
  for i, cfg in enumerate(batch_config['configs']):
    steps = ' → '.join(s['type'] for s in cfg['pipeline']) or '(空)'
    print(f"  [{i+1}] {cfg['name']}: {steps}")

  # 收集图像
  input_dir = args.input
  if not os.path.isdir(input_dir):
    print(f"输入目录不存在: {input_dir}", file=sys.stderr)
    sys.exit(1)

  images = collect_images(input_dir)
  if not images:
    print(f"输入目录中没有图像文件: {input_dir}", file=sys.stderr)
    sys.exit(1)

  print(f"\n找到 {len(images)} 张图像")

  # 创建输出目录
  output_dir = args.output
  os.makedirs(output_dir, exist_ok=True)

  # 检查服务可用性
  try:
    resp = requests.get(args.server.rstrip('/') + '/health', timeout=5)
    if resp.status_code != 200:
      raise ConnectionError()
    print(f"服务连接正常: {args.server}")
  except Exception:
    print(f"无法连接服务: {args.server}", file=sys.stderr)
    print("请先启动 API 服务: python api.py", file=sys.stderr)
    sys.exit(1)

  # 处理每张图像
  total = len(images)
  success_count = 0
  fail_count = 0
  skip_count = 0
  start_time = time.time()

  for idx, image_path in enumerate(images):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n[{idx+1}/{total}] {os.path.basename(image_path)}")

    try:
      results = process_image(args.server, batch_config, image_path)
    except Exception as e:
      print(f"  处理失败: {e}")
      fail_count += 1
      continue

    # 分离成功和失败
    ok_results = [(name, img) for name, img in results if not isinstance(img, str)]
    errors = [(name, err) for name, err in results if isinstance(err, str)]

    for name, err in errors:
      print(f"  配置「{name}」失败: {err}")

    if not ok_results:
      print(f"  所有配置均失败，跳过")
      skip_count += 1
      continue

    for name, img in ok_results:
      print(f"  -> {name}: {img.size[0]}x{img.size[1]}")

    # 生成拼图
    collage = generate_collage(ok_results, batch_config['tile_size'])
    if collage:
      collage_path = os.path.join(output_dir, f"{image_name}_tiled.png")
      collage.save(collage_path)
      print(f"  -> 拼图: {collage.size[0]}x{collage.size[1]}")

    success_count += 1

  elapsed = time.time() - start_time

  # 输出摘要
  print(f"\n{'='*40}")
  print(f"处理完成 ({elapsed:.1f}s)")
  print(f"  成功: {success_count}")
  print(f"  失败: {fail_count}")
  print(f"  跳过: {skip_count}")
  print(f"  输出: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
  main()
