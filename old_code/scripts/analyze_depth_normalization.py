#!/usr/bin/env python3
"""
分析训练数据的深度分布，检查归一化是否合理
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.factory import create_training_dataset


def analyze_depth_distribution(dataset, num_samples=100):
    """分析深度分布"""
    all_depths = []
    dataset_depths = {}

    print(f"分析 {num_samples} 个样本的深度分布...")

    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            dataset_name = sample.dataset_name

            if 'pos_3d' not in sample.targets:
                continue

            pos_3d = sample.targets['pos_3d']
            mask_3d = sample.targets.get('mask_3d', None)

            depths = pos_3d[:, 2].numpy()

            if mask_3d is not None:
                valid_depths = depths[mask_3d.numpy()]
            else:
                valid_depths = depths[np.isfinite(depths)]

            if len(valid_depths) > 0:
                all_depths.extend(valid_depths.tolist())
                if dataset_name not in dataset_depths:
                    dataset_depths[dataset_name] = []
                dataset_depths[dataset_name].extend(valid_depths.tolist())

        except Exception as e:
            continue

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{num_samples}")

    return np.array(all_depths), dataset_depths


def print_statistics(all_depths, dataset_depths):
    """打印统计信息"""
    print("\n" + "="*80)
    print("整体深度分布")
    print("="*80)

    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]
    print(f"\n样本数: {len(all_depths)}")
    print(f"最小值: {all_depths.min():.4f}m")
    print(f"最大值: {all_depths.max():.4f}m")
    print(f"平均值: {all_depths.mean():.4f}m")
    print(f"中位数: {np.median(all_depths):.4f}m")
    print(f"标准差: {all_depths.std():.4f}m")

    print(f"\n百分位数:")
    for p in percentiles:
        val = np.percentile(all_depths, p)
        print(f"  P{p:5.2f}: {val:10.4f}m")

    # 检查异常值
    p99 = np.percentile(all_depths, 99)
    p999 = np.percentile(all_depths, 99.9)

    print(f"\n异常值分析:")
    print(f"  P99.9 / P99 比值: {p999/p99:.2f}")
    if p999 / p99 > 2:
        print(f"  ⚠️  P99.9远大于P99，存在极端异常值")
        outliers = all_depths[all_depths > p99 * 2]
        print(f"  ⚠️  >2×P99的点: {len(outliers)} ({len(outliers)/len(all_depths)*100:.3f}%)")
        print(f"      范围: {outliers.min():.2f}m - {outliers.max():.2f}m")

    print(f"\n各数据集深度范围:")
    for name, depths in sorted(dataset_depths.items()):
        d = np.array(depths)
        print(f"  {name:20s}: [{d.min():8.2f}, {d.max():8.2f}]m, P99={np.percentile(d, 99):8.2f}m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=100)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset = create_training_dataset(config, split='train')
    all_depths, dataset_depths = analyze_depth_distribution(dataset, args.num_samples)
    print_statistics(all_depths, dataset_depths)

    print("\n建议:")
    p99 = np.percentile(all_depths, 99)
    if all_depths.max() > p99 * 3:
        print(f"  ⚠️  最大值({all_depths.max():.1f}m) > 3×P99({p99:.1f}m)")
        print(f"  建议将far阈值从10000降低到 {int(p99 * 2)}m")


if __name__ == '__main__':
    main()
