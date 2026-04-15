#!/usr/bin/env python3
"""统计MVSSynth所有序列的深度上限"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.adapters.mvssynth import MVSSynthAdapter

adapter = MVSSynthAdapter(
    root="/data2/d4rt/datasets/MVS-Synth/GTAV_1080",
    split="train",
    verbose=False
)

print(f"分析{len(adapter)}个序列...")

max_depths = []
p999_depths = []

for i in range(len(adapter)):
    seq_name = adapter.get_sequence_name(i)
    clip = adapter.load_clip(seq_name, frame_indices=[0])

    if clip.depths is not None:
        depth = clip.depths[0]
        valid = depth[depth > 0]
        if len(valid) > 0:
            max_depths.append(valid.max())
            p999_depths.append(np.percentile(valid, 99.9))

    if (i + 1) % 20 == 0:
        print(f"  已处理 {i+1}/{len(adapter)}")

max_depths = np.array(max_depths)
p999_depths = np.array(p999_depths)

print("\n" + "="*60)
print("深度统计（基于每个序列的第一帧）")
print("="*60)
print(f"\n最大深度分布:")
print(f"  min:  {max_depths.min():.1f}m")
print(f"  p50:  {np.percentile(max_depths, 50):.1f}m")
print(f"  p90:  {np.percentile(max_depths, 90):.1f}m")
print(f"  p95:  {np.percentile(max_depths, 95):.1f}m")
print(f"  p99:  {np.percentile(max_depths, 99):.1f}m")
print(f"  max:  {max_depths.max():.1f}m")

print(f"\n99.9%分位数分布:")
print(f"  min:  {p999_depths.min():.1f}m")
print(f"  p50:  {np.percentile(p999_depths, 50):.1f}m")
print(f"  p90:  {np.percentile(p999_depths, 90):.1f}m")
print(f"  p95:  {np.percentile(p999_depths, 95):.1f}m")
print(f"  max:  {p999_depths.max():.1f}m")

print(f"\n建议的深度上限:")
extreme_count = (max_depths > 10000).sum()
print(f"  >10km的序列: {extreme_count}/{len(max_depths)} ({100*extreme_count/len(max_depths):.1f}%)")
print(f"  建议上限: {np.percentile(p999_depths, 99):.0f}m (覆盖99%序列的99.9%像素)")
