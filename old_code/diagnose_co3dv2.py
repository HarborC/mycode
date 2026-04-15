#!/usr/bin/env python3
"""
Diagnose Co3Dv2 precomputed data issues.
"""
import sys
sys.path.insert(0, '/data1/zbf/my_dfrt')

import numpy as np
from pathlib import Path
from datasets.adapters.co3dv2 import Co3Dv2Adapter

def diagnose_sequence(adapter, seq):
    """Diagnose a single Co3D sequence."""
    print(f"\n{'='*80}")
    print(f"Sequence: {seq}")
    print('='*80)

    # Load precomputed metadata
    npz = adapter.precompute_root / seq / 'precomputed.npz'
    if not npz.exists():
        print("No precomputed.npz found")
        return

    d = np.load(npz, allow_pickle=True)
    ref = int(d['ref_frame'])
    num_frames = int(d['num_frames'])

    print(f"Precomputed metadata:")
    print(f"  ref_frame: {ref}")
    print(f"  num_frames: {num_frames}")
    print(f"  num_points: {d['num_points']}")

    # Check valids/visibs distribution
    valids = d['valids']
    visibs = d['visibs']
    print(f"\nValids/visibs per frame:")
    for t in range(num_frames):
        v = valids[t].sum()
        vis = visibs[t].sum()
        marker = " <-- ref_frame" if t == ref else ""
        print(f"  Frame {t:3d}: valids={v:5d}, visibs={vis:5d}{marker}")

    # Load clip at ref_frame
    clip = adapter.load_clip(seq, [ref])

    # Check 3D point cloud quality
    valid_mask = clip.valids[0].astype(bool)
    pts3d = clip.trajs_3d_world[0, valid_mask]
    pts2d = clip.trajs_2d[0, valid_mask]

    print(f"\n3D point cloud at ref_frame:")
    print(f"  Valid points: {len(pts3d)}")
    print(f"  3D range: x=[{pts3d[:,0].min():.2f}, {pts3d[:,0].max():.2f}], "
          f"y=[{pts3d[:,1].min():.2f}, {pts3d[:,1].max():.2f}], "
          f"z=[{pts3d[:,2].min():.2f}, {pts3d[:,2].max():.2f}]")

    # Check planarity
    if len(pts3d) > 3:
        centered = pts3d - pts3d.mean(0)
        _, s, _ = np.linalg.svd(centered)
        planarity = s[2] / s[0]
        print(f"  SVD singular values: [{s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}]")
        print(f"  Planarity ratio (s[2]/s[0]): {planarity:.6f}")
        if planarity < 0.1:
            print(f"  ⚠️  WARNING: Points are highly planar (ratio < 0.1)")

    # Check 2D coordinates
    H, W = clip.image_size
    print(f"\n2D trajectories at ref_frame:")
    print(f"  Image size: {W}x{H}")
    print(f"  2D range: x=[{pts2d[:,0].min():.1f}, {pts2d[:,0].max():.1f}], "
          f"y=[{pts2d[:,1].min():.1f}, {pts2d[:,1].max():.1f}]")

    # Check if 2D points are within image bounds
    in_bounds = (pts2d[:,0] >= 0) & (pts2d[:,0] < W) & (pts2d[:,1] >= 0) & (pts2d[:,1] < H)
    print(f"  Points in bounds: {in_bounds.sum()} / {len(pts2d)} ({100*in_bounds.mean():.1f}%)")

    # Check depth backprojection
    if clip.depths[0] is not None:
        depth = clip.depths[0]
        K = clip.intrinsics[0]
        E = clip.extrinsics[0]

        # Sample some 2D points and backproject
        sample_idx = np.random.choice(len(pts2d), min(100, len(pts2d)), replace=False)
        pts2d_sample = pts2d[sample_idx]
        pts3d_sample = pts3d[sample_idx]

        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        xs = np.clip(np.round(pts2d_sample[:,0]).astype(int), 0, W-1)
        ys = np.clip(np.round(pts2d_sample[:,1]).astype(int), 0, H-1)
        z = depth[ys, xs].astype(np.float32)

        depth_valid = np.isfinite(z) & (z > 1e-3)
        print(f"\nDepth backprojection check:")
        print(f"  Sampled points: {len(pts2d_sample)}")
        print(f"  Valid depth values: {depth_valid.sum()} / {len(z)} ({100*depth_valid.mean():.1f}%)")

        if depth_valid.any():
            # Backproject to 3D
            x_cam = (xs[depth_valid] - cx) / fx * z[depth_valid]
            y_cam = (ys[depth_valid] - cy) / fy * z[depth_valid]
            pts_cam = np.stack([x_cam, y_cam, z[depth_valid]], axis=1)
            E_inv = np.linalg.inv(E.astype(np.float64))
            pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam),1), dtype=np.float32)], axis=1)
            pts3d_from_depth = (E_inv @ pts_h.T).T[:, :3].astype(np.float32)

            # Compare with GT 3D
            pts3d_gt = pts3d_sample[depth_valid]
            dist = np.linalg.norm(pts3d_from_depth - pts3d_gt, axis=1)
            print(f"  3D distance (depth vs GT): mean={dist.mean():.4f}m, max={dist.max():.4f}m")
            if dist.mean() > 0.1:
                print(f"  ⚠️  WARNING: Large 3D discrepancy (mean > 0.1m)")


def main():
    print("Co3Dv2 Precomputed Data Diagnosis")
    print("="*80)

    # Test a few sequences from different categories
    test_cases = [
        ('apple', 'apple/110_13048_23163'),
        ('toaster', None),  # Will pick first sequence
        ('orange', None),
    ]

    for category, seq in test_cases:
        adapter = Co3Dv2Adapter('/data2/d4rt/datasets/Co3Dv2',
                               categories=[category], verbose=False)
        if seq is None:
            seq = adapter.list_sequences()[0]

        diagnose_sequence(adapter, seq)


if __name__ == '__main__':
    main()
