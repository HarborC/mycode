#!/usr/bin/env python3
"""
诊断GPU利用率为0的原因：逐个测试每个数据集的加载速度。
用法: python diagnose_gpu_idle.py
"""
import sys, time, yaml
sys.path.insert(0, '/data1/zbf/my_dfrt')

import torch
from torch.utils.data import DataLoader
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn

CONFIG_PATH = '/data1/zbf/my_dfrt/configs/mixture_full_11datasets.yaml'
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_BATCHES = 5   # 每个数据集测几个batch
CLIP_LEN = 8     # 用短clip加快测试（正式训练是48）

with open(CONFIG_PATH) as f:
    base_config = yaml.safe_load(f)

print(f"{'Dataset':<20} {'Init(s)':>8} {'Batch1(s)':>10} {'Avg(s)':>8} {'判断':>10}")
print('-' * 65)

results = []
for ds in base_config['datasets']:
    cfg = {
        'mode': 'single',
        'name': ds['name'],
        'root': ds['root'],
        'clip_len': CLIP_LEN,
        'img_size': 256,
        'num_queries': 512,
        'use_augs': False,
        'seed': 42,
        'index_cache_dir': base_config.get('index_cache_dir'),
    }

    # 初始化时间
    t0 = time.time()
    try:
        dataset = create_training_dataset(cfg, split='train')
    except Exception as e:
        print(f"{ds['name']:<20} {'ERROR':>8}  {e}")
        continue
    init_time = time.time() - t0

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=d4rt_collate_fn,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
    )

    times = []
    try:
        for i, batch in enumerate(loader):
            if i == 0:
                t_batch0 = time.time()
                # 第一个batch单独计时（包含worker启动）
                first_batch_time = None
            if i >= 1:
                if first_batch_time is None:
                    first_batch_time = time.time() - t_batch0
                t_start = time.time()
                _ = batch  # 确保数据已加载
                times.append(time.time() - t_start)
            if i >= NUM_BATCHES:
                break
        # 重新计时：更准确的方式
        times = []
        t_all = time.time()
        for i, batch in enumerate(loader):
            if i >= NUM_BATCHES:
                break
            times.append(time.time() - t_all)
            t_all = time.time()
    except Exception as e:
        print(f"{ds['name']:<20} {init_time:>8.1f} {'ERROR':>10}  {e}")
        continue

    if not times:
        print(f"{ds['name']:<20} {init_time:>8.1f} {'no data':>10}")
        continue

    avg = sum(times) / len(times)
    first = times[0]
    status = '✓ 快' if avg < 2.0 else ('⚠ 中' if avg < 5.0 else '✗ 慢！')
    print(f"{ds['name']:<20} {init_time:>8.1f} {first:>10.2f} {avg:>8.2f} {status:>10}")
    results.append((ds['name'], avg))

print('-' * 65)
if results:
    slowest = max(results, key=lambda x: x[1])
    print(f"\n最慢数据集: {slowest[0]}  平均 {slowest[1]:.2f}s/batch")
    print("建议: 如果某数据集 >2s/batch，它就是GPU空转的主要原因")

# 额外：测试 torch.cuda.synchronize 的影响
print('\n--- 测试 torch.cuda.synchronize() 开销 ---')
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device='cuda')
    # 不同步
    t0 = time.time()
    for _ in range(100):
        y = x @ x
    t_nosync = time.time() - t0

    # 每次同步
    t0 = time.time()
    for _ in range(100):
        y = x @ x
        torch.cuda.synchronize()
    t_sync = time.time() - t0

    print(f"无 synchronize: {t_nosync*1000:.1f}ms / 100次矩阵乘")
    print(f"有 synchronize: {t_sync*1000:.1f}ms / 100次矩阵乘")
    print(f"synchronize 额外开销: {(t_sync-t_nosync)*1000:.1f}ms")
    if t_sync > t_nosync * 1.5:
        print("⚠ train_mixture.py:194 的 torch.cuda.synchronize() 有显著开销，建议删除！")
else:
    print("无CUDA，跳过")
