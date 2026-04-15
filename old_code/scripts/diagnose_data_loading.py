#!/usr/bin/env python3
"""
数据加载诊断脚本 - 分析实际数据特征而非凭经验设置阈值

用法:
    python scripts/diagnose_data_loading.py --config configs/mixture_train.yaml --num-samples 100
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import yaml

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.factory import create_training_dataset


def analyze_depth_distribution(dataset, num_samples=100):
    """分析深度分布，找出实际的异常值"""
    stats = defaultdict(lambda: {
        'depths': [],
        'valid_points_per_frame': [],
        'has_mask': False,
        'depth_range': [],
    })

    print(f"正在采样 {num_samples} 个样本...")

    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            dataset_name = sample.dataset_name

            # 检查是否有深度数据
            if 'pos_3d' not in sample.targets:
                continue

            pos_3d = sample.targets['pos_3d']  # [Q, 3]
            mask_3d = sample.targets.get('mask_3d', None)

            # 提取深度值（z坐标）
            depths = pos_3d[:, 2].numpy()

            if mask_3d is not None:
                stats[dataset_name]['has_mask'] = True
                valid_depths = depths[mask_3d.numpy()]
            else:
                valid_depths = depths[np.isfinite(depths)]

            if len(valid_depths) > 0:
                stats[dataset_name]['depths'].extend(valid_depths.tolist())
                stats[dataset_name]['depth_range'].append((valid_depths.min(), valid_depths.max()))

            # 统计每帧有效点数（如果有的话）
            if mask_3d is not None:
                stats[dataset_name]['valid_points_per_frame'].append(mask_3d.sum().item())

        except Exception as e:
            print(f"样本 {i} 加载失败: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{num_samples}")

    return stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "="*80)
    print("数据集深度分布统计")
    print("="*80)

    for dataset_name, data in stats.items():
        print(f"\n【{dataset_name}】")
        print(f"  是否有mask: {data['has_mask']}")

        if len(data['depths']) > 0:
            depths = np.array(data['depths'])
            print(f"  深度统计:")
            print(f"    样本数: {len(depths)}")
            print(f"    最小值: {depths.min():.4f}m")
            print(f"    最大值: {depths.max():.4f}m")
            print(f"    中位数: {np.median(depths):.4f}m")
            print(f"    平均值: {depths.mean():.4f}m")
            print(f"    标准差: {depths.std():.4f}m")
            print(f"    95分位: {np.percentile(depths, 95):.4f}m")
            print(f"    99分位: {np.percentile(depths, 99):.4f}m")
            print(f"    99.9分位: {np.percentile(depths, 99.9):.4f}m")

            # 检测异常值
            p99 = np.percentile(depths, 99)
            outliers = depths[depths > p99 * 2]
            if len(outliers) > 0:
                print(f"  ⚠️  潜在异常值 (>2×P99): {len(outliers)} 个 ({len(outliers)/len(depths)*100:.2f}%)")
                print(f"      范围: {outliers.min():.2f}m - {outliers.max():.2f}m")

        if len(data['valid_points_per_frame']) > 0:
            valid_pts = np.array(data['valid_points_per_frame'])
            print(f"  每帧有效点数:")
            print(f"    平均: {valid_pts.mean():.1f}")
            print(f"    最小: {valid_pts.min()}")
            print(f"    最大: {valid_pts.max()}")
            if valid_pts.min() < 2048:
                ratio = 1 - valid_pts.mean() / 2048
                print(f"  ⚠️  平均有效点 < 2048, 预计重复采样率: {ratio*100:.1f}%")


def plot_depth_histograms(stats, output_dir):
    """绘制深度分布直方图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, data in stats.items():
        if len(data['depths']) == 0:
            continue

        depths = np.array(data['depths'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 线性尺度
        axes[0].hist(depths, bins=100, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Depth (m)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{dataset_name} - Linear Scale')
        axes[0].grid(True, alpha=0.3)

        # 对数尺度
        axes[1].hist(depths, bins=100, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Depth (m)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'{dataset_name} - Log Scale')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name}_depth_dist.png', dpi=150)
        plt.close()

        print(f"  保存图表: {output_dir / f'{dataset_name}_depth_dist.png'}")


def save_report(stats, output_path):
    """保存JSON报告"""
    report = {}
    for dataset_name, data in stats.items():
        if len(data['depths']) > 0:
            depths = np.array(data['depths'])
            report[dataset_name] = {
                'has_mask': data['has_mask'],
                'num_samples': len(depths),
                'depth_stats': {
                    'min': float(depths.min()),
                    'max': float(depths.max()),
                    'mean': float(depths.mean()),
                    'median': float(np.median(depths)),
                    'std': float(depths.std()),
                    'p95': float(np.percentile(depths, 95)),
                    'p99': float(np.percentile(depths, 99)),
                    'p99.9': float(np.percentile(depths, 99.9)),
                },
            }

            if len(data['valid_points_per_frame']) > 0:
                valid_pts = np.array(data['valid_points_per_frame'])
                report[dataset_name]['valid_points_stats'] = {
                    'mean': float(valid_pts.mean()),
                    'min': int(valid_pts.min()),
                    'max': int(valid_pts.max()),
                }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='诊断数据加载管线')
    parser.add_argument('--config', type=str, required=True, help='数据集配置文件')
    parser.add_argument('--num-samples', type=int, default=100, help='采样数量')
    parser.add_argument('--output-dir', type=str, default='./diagnosis_output', help='输出目录')
    args = parser.parse_args()

    print(f"加载数据集配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    dataset = create_training_dataset(config, split='train')

    print(f"数据集大小: {len(dataset)}")

    # 分析深度分布
    stats = analyze_depth_distribution(dataset, args.num_samples)

    # 打印统计
    print_statistics(stats)

    # 绘制图表
    print(f"\n生成可视化图表...")
    plot_depth_histograms(stats, args.output_dir)

    # 保存报告
    report_path = Path(args.output_dir) / 'diagnosis_report.json'
    save_report(stats, report_path)

    print("\n" + "="*80)
    print("诊断完成！")
    print("="*80)
    print("\n建议:")
    print("1. 检查各数据集的P99.9值，如果远大于P99，说明存在极端异常值")
    print("2. 如果数据集has_mask=True，说明已有质量控制，可能不需要额外far阈值")
    print("3. 如果有效点数<2048，检查重复采样率是否可接受")
    print("4. 根据实际深度分布决定是否需要数据集特定的阈值")


if __name__ == '__main__':
    main()
