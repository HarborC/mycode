#!/usr/bin/env python3
"""检查数据集过滤机制的实际效果"""

import numpy as np
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets.factory import create_training_dataset


def check_single_sample(dataset, idx):
    """检查单个样本的过滤情况"""
    try:
        sample = dataset[idx]
    except Exception as e:
        return {"error": str(e), "dataset": getattr(sample, 'dataset_name', 'unknown') if hasattr(dataset, '__getitem__') else 'unknown'}

    stats = {"dataset": sample.dataset_name}

    # 统计mask比例
    targets = sample.targets
    stats["mask_3d_ratio"] = float(targets["mask_3d"].float().mean())
    stats["mask_2d_ratio"] = float(targets["mask_2d"].float().mean())

    # 检查深度值分布
    if sample.depths is not None:
        depths = sample.depths.numpy()
        valid_depths = depths[depths > 0]
        if len(valid_depths) > 0:
            stats["depth_max"] = float(valid_depths.max())
            stats["depth_p95"] = float(np.percentile(valid_depths, 95))

    # 检查3D位置深度
    pos_3d = targets["pos_3d"].numpy()
    valid_3d = targets["mask_3d"].numpy()
    if valid_3d.any():
        depths_3d = pos_3d[valid_3d][:, 2]
        stats["pos_3d_depth_max"] = float(depths_3d.max())
        stats["pos_3d_depth_mean"] = float(depths_3d.mean())

    return stats


def main():
    print("=" * 60)
    print("数据集过滤机制检查")
    print("=" * 60)

    # 加载配置
    print("\n加载配置...")
    config_path = Path("configs/mixture_full_11datasets.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 加载数据集
    print("加载数据集...")
    dataset = create_training_dataset(config=config, split="train")
    print(f"数据集大小: {len(dataset)}")

    # 检查前10个样本
    print("\n" + "=" * 60)
    print("检查前10个样本")
    print("=" * 60)

    for i in range(min(10, len(dataset))):
        print(f"\n样本 {i}:")
        stats = check_single_sample(dataset, i)

        if "error" in stats:
            print(f"  ❌ 错误: {stats['error']}")
            continue

        print(f"  数据集: {stats['dataset']}")
        print(f"  Mask比例: mask_3d={stats['mask_3d_ratio']:.3f}, mask_2d={stats['mask_2d_ratio']:.3f}")

        if "depth_max" in stats:
            print(f"  深度图: max={stats['depth_max']:.1f}m, p95={stats['depth_p95']:.1f}m")

        if "pos_3d_depth_max" in stats:
            print(f"  3D深度: max={stats['pos_3d_depth_max']:.1f}m, mean={stats['pos_3d_depth_mean']:.1f}m")
            if stats['pos_3d_depth_max'] > 100:
                print(f"    ⚠️  极大深度值: {stats['pos_3d_depth_max']:.1f}m")


if __name__ == "__main__":
    main()
