#!/usr/bin/env python3
"""
测试数据加载修复是否生效

验证：
1. 深度值是否在 [1e-3, 10000] 范围内
2. 无负深度值
3. 无极端异常值
4. mask_3d正确标记有效点
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.factory import create_training_dataset


def test_depth_filtering(dataset, num_samples=20):
    """测试深度过滤是否正确"""
    print(f"测试 {num_samples} 个样本...")

    issues = []
    stats = {
        'total_samples': 0,
        'total_queries': 0,
        'valid_3d_queries': 0,
        'depth_min': float('inf'),
        'depth_max': float('-inf'),
        'negative_depths': 0,
        'extreme_depths': 0,
    }

    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            dataset_name = sample.dataset_name

            if 'pos_3d' not in sample.targets:
                continue

            stats['total_samples'] += 1
            stats['total_queries'] += len(sample.targets['pos_3d'])

            pos_3d = sample.targets['pos_3d']  # [Q, 3]
            mask_3d = sample.targets.get('mask_3d', None)

            # 提取深度值
            depths = pos_3d[:, 2].numpy()

            if mask_3d is not None:
                valid_depths = depths[mask_3d.numpy()]
                stats['valid_3d_queries'] += mask_3d.sum().item()
            else:
                valid_depths = depths[np.isfinite(depths)]

            if len(valid_depths) == 0:
                continue

            # 检查深度范围
            min_d = valid_depths.min()
            max_d = valid_depths.max()
            stats['depth_min'] = min(stats['depth_min'], min_d)
            stats['depth_max'] = max(stats['depth_max'], max_d)

            # 检查负值
            neg_count = (valid_depths < 0).sum()
            if neg_count > 0:
                stats['negative_depths'] += neg_count
                issues.append(f"样本{i} ({dataset_name}): {neg_count}个负深度值, 最小={min_d:.4f}m")

            # 检查极端值
            extreme_count = (valid_depths > 10000).sum()
            if extreme_count > 0:
                stats['extreme_depths'] += extreme_count
                issues.append(f"样本{i} ({dataset_name}): {extreme_count}个极端深度值, 最大={max_d:.4f}m")

            # 检查near阈值
            too_close = ((valid_depths > 0) & (valid_depths < 1e-3)).sum()
            if too_close > 0:
                issues.append(f"样本{i} ({dataset_name}): {too_close}个过近深度值 (<1e-3)")

        except Exception as e:
            issues.append(f"样本{i}加载失败: {e}")
            continue

        if (i + 1) % 5 == 0:
            print(f"  已测试 {i + 1}/{num_samples}")

    return stats, issues


def print_test_results(stats, issues):
    """打印测试结果"""
    print("\n" + "="*80)
    print("测试结果")
    print("="*80)

    print(f"\n样本统计:")
    print(f"  测试样本数: {stats['total_samples']}")
    print(f"  总查询点数: {stats['total_queries']}")
    print(f"  有效3D点数: {stats['valid_3d_queries']}")
    print(f"  有效率: {stats['valid_3d_queries']/stats['total_queries']*100:.1f}%")

    print(f"\n深度范围:")
    print(f"  最小值: {stats['depth_min']:.6f}m")
    print(f"  最大值: {stats['depth_max']:.4f}m")

    print(f"\n异常检测:")
    print(f"  负深度值: {stats['negative_depths']}")
    print(f"  极端深度值(>10000m): {stats['extreme_depths']}")

    if len(issues) > 0:
        print(f"\n⚠️  发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前10个
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues)-10} 个问题")
        print("\n❌ 测试失败！")
        return False
    else:
        print("\n✅ 所有测试通过！")
        print("\n验证结果:")
        print(f"  ✓ 深度值在合理范围 [{stats['depth_min']:.6f}, {stats['depth_max']:.4f}]m")
        print(f"  ✓ 无负深度值")
        print(f"  ✓ 无极端异常值")
        return True


def main():
    parser = argparse.ArgumentParser(description='测试数据加载修复')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=20)
    args = parser.parse_args()

    print(f"加载配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset = create_training_dataset(config, split='train')
    print(f"数据集大小: {len(dataset)}\n")

    stats, issues = test_depth_filtering(dataset, args.num_samples)
    success = print_test_results(stats, issues)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
