#!/usr/bin/env python3
"""详细检查MVSSynth数据集的深度分布"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.adapters.mvssynth import MVSSynthAdapter

print("=" * 60)
print("MVSSynth深度分布详细分析")
print("=" * 60)

# 加载adapter
adapter = MVSSynthAdapter(
    root="/data2/d4rt/datasets/MVS-Synth/GTAV_1080",
    split="train",
    verbose=False
)

print(f"\n数据集序列数: {len(adapter)}")

# 检查前3个序列
for i in range(min(3, len(adapter))):
    seq_name = adapter.get_sequence_name(i)
    print(f"\n{'='*60}")
    print(f"序列 {i}: {seq_name}")
    print(f"{'='*60}")

    # 加载clip
    clip = adapter.load_clip(seq_name, frame_indices=[0, 1, 2])

    if clip.depths is None:
        print("  无深度数据")
        continue

    # 分析每帧深度
    for fi, depth in enumerate(clip.depths):
        print(f"\n  帧 {fi}:")

        # 基本统计
        valid = depth[depth > 0]
        if len(valid) == 0:
            print("    无有效深度")
            continue

        print(f"    有效像素: {len(valid)}/{depth.size} ({100*len(valid)/depth.size:.1f}%)")
        print(f"    最小值: {valid.min():.2f}m")
        print(f"    最大值: {valid.max():.2f}m")
        print(f"    平均值: {valid.mean():.2f}m")
        print(f"    中位数: {np.median(valid):.2f}m")
        print(f"    标准差: {valid.std():.2f}m")

        # 分位��
        percentiles = [50, 75, 90, 95, 99, 99.9]
        print(f"    分位数:")
        for p in percentiles:
            val = np.percentile(valid, p)
            print(f"      {p:5.1f}%: {val:8.2f}m")

        # 深度范围分布
        ranges = [
            (0, 10, "0-10m"),
            (10, 50, "10-50m"),
            (50, 100, "50-100m"),
            (100, 500, "100-500m"),
            (500, 1000, "500-1000m"),
            (1000, float('inf'), ">1000m")
        ]
        print(f"    深度范围分布:")
        for min_d, max_d, label in ranges:
            count = ((valid >= min_d) & (valid < max_d)).sum()
            pct = 100 * count / len(valid)
            print(f"      {label:12s}: {count:6d} ({pct:5.2f}%)")

print("\n" + "=" * 60)
