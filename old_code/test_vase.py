#!/usr/bin/env python3
"""Test vase specifically."""
import sys
sys.path.insert(0, '/data1/zbf/my_dfrt')

import numpy as np
from datasets.adapters.co3dv2 import Co3Dv2Adapter

adapter = Co3Dv2Adapter(root="/data2/d4rt/datasets/Co3Dv2", split="train")
seq = "vase/380_44868_89574"
clip = adapter.load_clip(seq, list(range(31)))

print(f"Sequence: {seq}")
print(f"Depth scale: {adapter._depth_scale_cache.get(seq, 'N/A')}")
print(f"has_tracks: {clip.metadata.get('has_tracks')}")
print(f"valids per frame: {clip.valids.sum(axis=1) if clip.valids is not None else 'N/A'}")

if clip.valids is not None:
    # Compute 2D reprojection error
    errs = []
    d3d_errs = []
    for t in range(len(clip.depths)):
        valid = clip.valids[t].astype(bool) & clip.visibs[t].astype(bool)
        if not valid.any():
            continue
        K = clip.intrinsics[t]
        E = clip.extrinsics[t]
        pts3d = clip.trajs_3d_world[t, valid]
        pts2d_gt = clip.trajs_2d[t, valid]

        # Project 3D to 2D
        pts3d_h = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
        pts_cam = (E @ pts3d_h.T).T[:, :3]
        z = pts_cam[:, 2]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u = pts_cam[:, 0] / z * fx + cx
        v = pts_cam[:, 1] / z * fy + cy
        pts2d_proj = np.stack([u, v], axis=1)

        err = np.linalg.norm(pts2d_proj - pts2d_gt, axis=1)
        errs.append(float(err.mean()))

        # Check depth consistency
        depth = clip.depths[t]
        H, W = depth.shape
        xs = np.clip(np.round(pts2d_gt[:, 0]).astype(int), 0, W-1)
        ys = np.clip(np.round(pts2d_gt[:, 1]).astype(int), 0, H-1)
        z_sampled = depth[ys, xs]
        depth_valid = (z_sampled > 0) & np.isfinite(z_sampled)
        if depth_valid.any():
            z_proj = z[depth_valid]
            z_samp = z_sampled[depth_valid]
            d3d_errs.append(float(np.abs(z_proj - z_samp).mean()))

    if errs:
        print(f"2D reproj error: mean={np.mean(errs):.4f}px")
    else:
        print("2D reproj error: N/A (no valid frames)")
    if d3d_errs:
        print(f"Depth consistency error: mean={np.mean(d3d_errs):.4f}m")
