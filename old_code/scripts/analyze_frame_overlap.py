#!/usr/bin/env python3
"""
帧间重合度分析脚本 - 测试不同stride下的帧间overlap

用法:
    python scripts/analyze_frame_overlap.py --config configs/mixture_train.yaml --num-sequences 20
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.factory import create_training_dataset


def compute_frame_overlap(depths1, depths2, threshold=0.1):
    """计算两帧之间的深度重合度"""
    if depths1 is None or depths2 is None:
        return 0.0

    valid1 = np.isfinite(depths1) & (depths1 > 0)
    valid2 = np.isfinite(depths2) & (depths2 > 0)

    if valid1.sum() == 0 or valid2.sum() == 0:
        return 0.0

    # 计算有效像素的交集
    both_valid = valid1 & valid2
    overlap_ratio = both_valid.sum() / max(valid1.sum(), valid2.sum())

    return float(overlap_ratio)


def analyze_stride_overlap(dataset, num_sequences=20, max_stride=8):
    """分析不同stride下的帧间重合度"""
    stats = defaultdict(lambda: defaultdict(list))

    print(f"分析 {num_sequences} 个序列的帧间重合度...")

    for i in range(min(num_sequences, len(dataset))):
        try:
            sample = dataset[i]
            dataset_name = sample.get('dataset_name', 'unknown')

            # 获取视频帧
            if 'video' not in sample:
                continue

            video = sample['video']  # [T, 3, H, W]
            T = video.shape[0]

            # 如果有深度信息更好，否则用RGB相似度
            # 这里简化处理，实际应该用深度或特征匹配

            # 测试不同stride
            for stride in range(1, min(max_stride + 1, T // 2)):
                overlaps = []
                for t in range(0, T - stride, stride):
                    # 简化：用RGB差异估计重合度
                    frame1 = video[t].numpy()
                    frame2 = video[t + stride].numpy()

                    # 计算像素差异
                    diff = np.abs(frame1 - frame2).mean()
                    similarity = 1.0 - min(diff, 1.0)
                    overlaps.append(similarity)

                if overlaps:
                    stats[dataset_name][stride].extend(overlaps)

        except Exception as e:
            print(f"序列 {i} 分析失败: {e}")
            continue

        if (i + 1) % 5 == 0:
            print(f"  已处理 {i + 1}/{num_sequences}")

    return stats


def print_overlap_stats(stats):
    """打印重合度统计"""
    print("\n" + "="*80)
    print("帧间重合度统计")
    print("="*80)

    for dataset_name, stride_data in stats.items():
        print(f"\n【{dataset_name}】")
        print(f"  Stride | 平均重合度 | 标准差 | 建议")
        print(f"  " + "-"*60)

        for stride in sorted(stride_data.keys()):
            overlaps = np.array(stride_data[stride])
            mean_overlap = overlaps.mean()
            std_overlap = overlaps.std()

            # 建议：重合度>0.7为好，0.5-0.7为中等，<0.5为差
            if mean_overlap > 0.7:
                suggestion = "✓ 推荐"
            elif mean_overlap > 0.5:
                suggestion = "○ 可用"
            else:
                suggestion = "✗ 重合度低"

            print(f"  {stride:6d} | {mean_overlap:11.3f} | {std_overlap:6.3f} | {suggestion}")


def main():
    parser = argparse.ArgumentParser(description='分析帧间重合度')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num-sequences', type=int, default=20)
    parser.add_argument('--max-stride', type=int, default=8)
    args = parser.parse_args()

    dataset = create_training_dataset(args.config, split='train')
    stats = analyze_stride_overlap(dataset, args.num_sequences, args.max_stride)
    print_overlap_stats(stats)

    print("\n建议：根据实际重合度调整sampling.py中的stride权重")


if __name__ == '__main__':
    main()
