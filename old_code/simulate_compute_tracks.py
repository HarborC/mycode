#!/usr/bin/env python3
"""Simulate compute_tracks_co3d logic."""
import sys
sys.path.insert(0, '/data1/zbf/my_dfrt')

import numpy as np
from datasets.adapters.co3dv2 import Co3Dv2Adapter

adapter = Co3Dv2Adapter(root="/data2/d4rt/datasets/Co3Dv2", split="train")
seq = "vase/380_44868_89574"
clip = adapter.load_clip(seq, list(range(31)))

depths = clip.depths
intrinsics = clip.intrinsics
extrinsics = clip.extrinsics
T = len(depths)
H, W = depths[0].shape

# Find ref frame
depth_max = 10000.0
valid_counts = [int(((d > 0) & np.isfinite(d) & (d < depth_max)).sum()) for d in depths]
ref = int(np.argmax(valid_counts))
print(f"Ref frame: {ref}, valid counts: {valid_counts}")

depth_ref = depths[ref]
K_ref = intrinsics[ref]
E_ref = extrinsics[ref]
valid_ref = (depth_ref > 0) & np.isfinite(depth_ref) & (depth_ref < depth_max)

# Sample points
uni_ys, uni_xs = np.where(valid_ref)
rng = np.random.default_rng(42)
n = min(8000, len(uni_ys))
idx = rng.choice(len(uni_ys), n, replace=False)
src_y, src_x = uni_ys[idx], uni_xs[idx]
src_uv = np.stack([src_x, src_y], axis=-1).astype(np.float32)
src_d = depth_ref[src_y, src_x].astype(np.float32)

print(f"Sampled {n} points from ref frame")

# Unproject to world
fx, fy, cx, cy = K_ref[0,0], K_ref[1,1], K_ref[0,2], K_ref[1,2]
X = (src_uv[:, 0] - cx) * src_d / fx
Y = (src_uv[:, 1] - cy) * src_d / fy
Z = src_d
P_cam_ref = np.stack([X, Y, Z], axis=-1).astype(np.float32)

E_inv = np.linalg.inv(E_ref)
ones = np.ones((n, 1), dtype=np.float32)
P_world = (E_inv @ np.concatenate([P_cam_ref, ones], axis=-1).T).T[:, :3]

print(f"World coords: x=[{P_world[:,0].min():.2f}, {P_world[:,0].max():.2f}], "
      f"y=[{P_world[:,1].min():.2f}, {P_world[:,1].max():.2f}], "
      f"z=[{P_world[:,2].min():.2f}, {P_world[:,2].max():.2f}]")

# Project to each frame
depth_consistency_thresh = 0.10
for t in [0, 7, 15, 30]:
    E_t = extrinsics[t]
    K_t = intrinsics[t]
    P_hom_t = np.concatenate([P_world, ones], axis=-1)
    P_cam_t = (E_t @ P_hom_t.T).T[:, :3]

    fx_t, fy_t, cx_t, cy_t = K_t[0,0], K_t[1,1], K_t[0,2], K_t[1,2]
    z_t = P_cam_t[:, 2]
    uv_t = np.zeros((n, 2), dtype=np.float32)
    uv_t[:, 0] = P_cam_t[:, 0] / z_t * fx_t + cx_t
    uv_t[:, 1] = P_cam_t[:, 1] / z_t * fy_t + cy_t

    in_bounds = (
        (uv_t[:, 0] >= 0) & (uv_t[:, 0] < W) &
        (uv_t[:, 1] >= 0) & (uv_t[:, 1] < H) &
        (z_t > 0)
    )

    depth_t = depths[t]
    px = np.clip(np.round(uv_t[:, 0]).astype(np.int32), 0, W - 1)
    py = np.clip(np.round(uv_t[:, 1]).astype(np.int32), 0, H - 1)
    sampled_d = depth_t[py, px]
    has_depth = (sampled_d > 0) & np.isfinite(sampled_d) & (sampled_d < depth_max)

    # Key logic
    depth_consistent = np.abs(sampled_d - z_t) / np.maximum(z_t, 1e-6) < depth_consistency_thresh
    depth_ok = ~has_depth | depth_consistent

    valid_t = in_bounds & depth_ok

    print(f"\nFrame {t}:")
    print(f"  in_bounds: {in_bounds.sum()}/{n}")
    print(f"  has_depth: {has_depth.sum()}/{in_bounds.sum()}")
    print(f"  depth_consistent: {depth_consistent[has_depth].sum()}/{has_depth.sum()}")
    print(f"  depth_ok: {depth_ok.sum()}/{n}")
    print(f"  valid_t: {valid_t.sum()}/{n}")

    if has_depth.sum() > 0:
        rel_err = np.abs(sampled_d[has_depth] - z_t[has_depth]) / z_t[has_depth]
        print(f"  rel_err: mean={rel_err.mean():.4f}, median={np.median(rel_err):.4f}, max={rel_err.max():.4f}")
