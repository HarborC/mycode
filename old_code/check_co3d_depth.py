#!/usr/bin/env python3
"""Check Co3D depth coverage and scale."""
import sys
sys.path.insert(0, '/data1/zbf/my_dfrt')

import numpy as np
from datasets.adapters.co3dv2 import Co3Dv2Adapter

adapter = Co3Dv2Adapter(root="/data2/d4rt/datasets/Co3Dv2", split="train")

# Test the vase sequence from result.json
seq = "vase/380_44868_89574"
clip = adapter.load_clip(seq, list(range(31)))

print(f"Sequence: {seq}")
print(f"Frames: {len(clip.depths)}")
print(f"Image size: {clip.image_size}")

for t, depth in enumerate(clip.depths):
    valid = (depth > 0) & np.isfinite(depth)
    coverage = valid.sum() / depth.size * 100
    if valid.any():
        print(f"  Frame {t:2d}: coverage={coverage:5.2f}%, depth range=[{depth[valid].min():.2f}, {depth[valid].max():.2f}]m")
    else:
        print(f"  Frame {t:2d}: coverage=0.00% (no valid depth)")

# Check trajs_3d vs depth backprojection
if clip.trajs_3d_world is not None:
    print(f"\nTrajs_3d_world shape: {clip.trajs_3d_world.shape}")
    print(f"Valids shape: {clip.valids.shape}")
    print(f"Valid points per frame: {clip.valids.sum(axis=1)}")

    # Check depth consistency at valid track positions
    for t in range(min(5, len(clip.depths))):
        valid_t = clip.valids[t]
        if not valid_t.any():
            continue

        K = clip.intrinsics[t]
        E = clip.extrinsics[t]
        pts3d = clip.trajs_3d_world[t, valid_t]
        pts2d = clip.trajs_2d[t, valid_t]

        # Backproject depth at pts2d positions
        depth = clip.depths[t]
        H, W = depth.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        xs = np.clip(np.round(pts2d[:, 0]).astype(int), 0, W-1)
        ys = np.clip(np.round(pts2d[:, 1]).astype(int), 0, H-1)
        z_depth = depth[ys, xs]

        # Project pts3d to get z_proj
        pts3d_h = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
        pts_cam = (E @ pts3d_h.T).T[:, :3]
        z_proj = pts_cam[:, 2]

        depth_valid = (z_depth > 0) & np.isfinite(z_depth)
        if depth_valid.any():
            print(f"\n  Frame {t}: {depth_valid.sum()}/{len(pts3d)} tracks have depth")
            print(f"    z_depth: {z_depth[depth_valid][:5]}")
            print(f"    z_proj:  {z_proj[depth_valid][:5]}")
            print(f"    diff:    {np.abs(z_depth[depth_valid] - z_proj[depth_valid])[:5]}")
