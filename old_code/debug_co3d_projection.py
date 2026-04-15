#!/usr/bin/env python3
"""Debug Co3D depth scale by checking projection."""
import sys
sys.path.insert(0, '/data1/zbf/my_dfrt')

import numpy as np
from datasets.adapters.co3dv2 import Co3Dv2Adapter

adapter = Co3Dv2Adapter(root="/data2/d4rt/datasets/Co3Dv2", split="train")

seq = "vase/380_44868_89574"
clip = adapter.load_clip(seq, list(range(31)))

print(f"Sequence: {seq}")
print(f"Depth scale cache: {adapter._depth_scale_cache.get(seq, 'NOT FOUND')}")

# Pick ref frame (frame 7 based on previous output)
ref = 7
depth_ref = clip.depths[ref]
K_ref = clip.intrinsics[ref]
E_ref = clip.extrinsics[ref]

# Sample 100 valid depth pixels from ref frame
valid_mask = (depth_ref > 0) & np.isfinite(depth_ref)
ys, xs = np.where(valid_mask)
rng = np.random.default_rng(42)
idx = rng.choice(len(ys), min(100, len(ys)), replace=False)
sample_y, sample_x = ys[idx], xs[idx]
sample_d = depth_ref[sample_y, sample_x]

print(f"\nRef frame {ref}: sampled {len(sample_d)} points")
print(f"  Depth range: [{sample_d.min():.3f}, {sample_d.max():.3f}]m")

# Unproject to camera space
fx, fy, cx, cy = K_ref[0,0], K_ref[1,1], K_ref[0,2], K_ref[1,2]
x_cam = (sample_x - cx) / fx * sample_d
y_cam = (sample_y - cy) / fy * sample_d
z_cam = sample_d
pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

# To world
E_inv = np.linalg.inv(E_ref)
pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
pts_world = (E_inv @ pts_h.T).T[:, :3]

print(f"  World coords range: x=[{pts_world[:,0].min():.3f}, {pts_world[:,0].max():.3f}], "
      f"y=[{pts_world[:,1].min():.3f}, {pts_world[:,1].max():.3f}], "
      f"z=[{pts_world[:,2].min():.3f}, {pts_world[:,2].max():.3f}]")

# Project to another frame (frame 0)
t = 0
E_t = clip.extrinsics[t]
K_t = clip.intrinsics[t]
depth_t = clip.depths[t]

pts_h_t = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)
pts_cam_t = (E_t @ pts_h_t.T).T[:, :3]
z_proj = pts_cam_t[:, 2]

fx_t, fy_t, cx_t, cy_t = K_t[0,0], K_t[1,1], K_t[0,2], K_t[1,2]
u_proj = pts_cam_t[:, 0] / z_proj * fx_t + cx_t
v_proj = pts_cam_t[:, 1] / z_proj * fy_t + cy_t

H, W = depth_t.shape
in_bounds = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H) & (z_proj > 0)

print(f"\nFrame {t}: {in_bounds.sum()}/{len(pts_world)} points in bounds")

if in_bounds.sum() > 0:
    u_in = u_proj[in_bounds]
    v_in = v_proj[in_bounds]
    z_in = z_proj[in_bounds]

    px = np.clip(np.round(u_in).astype(int), 0, W-1)
    py = np.clip(np.round(v_in).astype(int), 0, H-1)
    sampled_d = depth_t[py, px]

    has_depth = (sampled_d > 0) & np.isfinite(sampled_d)
    print(f"  {has_depth.sum()}/{len(z_in)} in-bounds points have depth")

    if has_depth.sum() > 0:
        z_with_d = z_in[has_depth]
        d_with_d = sampled_d[has_depth]
        rel_err = np.abs(z_with_d - d_with_d) / z_with_d

        print(f"  z_proj:  {z_with_d[:10]}")
        print(f"  sampled: {d_with_d[:10]}")
        print(f"  rel_err: {rel_err[:10]}")
        print(f"  rel_err stats: mean={rel_err.mean():.4f}, median={np.median(rel_err):.4f}, max={rel_err.max():.4f}")
        print(f"  Points passing 10% thresh: {(rel_err < 0.10).sum()}/{len(rel_err)}")
