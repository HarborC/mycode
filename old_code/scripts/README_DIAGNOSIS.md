# 数据加载管线诊断工具

## 问题背景

数据加载管线需要验证以下问题：
1. 极远值过滤是否合理（考虑数据集自带mask和尺度差异）
2. 采样点策略是否适配静态/动态场景
3. 帧间隔stride是否匹配实际重合度

## 诊断步骤

### 1. 深度分布诊断

运行此脚本分析实际深度分布，而非凭经验设置阈值：

```bash
python scripts/diagnose_data_loading.py \
    --config configs/mixture_train.yaml \
    --num-samples 100 \
    --output-dir ./diagnosis_output
```

**输出**：
- 各数据集的深度统计（min/max/mean/P95/P99/P99.9）
- 是否有自带mask
- 潜在异常值检测
- 深度分布直方图
- JSON报告

**如何判断**：
- 如果 `has_mask=True`，说明数据集已有质量控制
- 如果 P99.9 >> P99，说明存在极端异常值
- 根据实际分布决定是否需要far阈值

### 2. 帧间重合度分析

运行此脚本测试不同stride下的实际重合度：

```bash
python scripts/analyze_frame_overlap.py \
    --config configs/mixture_train.yaml \
    --num-sequences 20 \
    --max-stride 8
```

**输出**：
- 各数据集在不同stride下的平均重合度
- 推荐的stride设置

**如何判断**：
- 重合度 > 0.7：推荐使用
- 重合度 0.5-0.7：可用
- 重合度 < 0.5：不推荐（帧间变化过大）

## 下一步

根据诊断结果决定：
1. 是否需要添加far阈值（如果数据集无mask且有异常值）
2. 是否需要调整boundary_ratio（静态场景可能需要降低）
3. 是否需要调整stride权重（根据实际重合度）
