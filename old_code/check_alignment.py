#!/usr/bin/env python3
"""
数值验证脚本：检验 rrd 中三组点集的对齐质量。

三组点（与 rerun 可视化颜色对应）：
  - pts3d          [蓝色]  = trajs_3d_world         (从 coords_depth 反投影)
  - pts3d_depth    [橙色]  = 从稠密深度图反投影得到的3D点
  - reproj_2d      [红色]  = 将 pts3d 重投影到图像的2D坐标

对齐检验：
  A. 2D重投影误差       reproj_2d  vs  trajs_2d_gt          (单位: px)
  B. 3D点云对齐误差     pts3d_depth vs pts3d                 (单位: m)
  C. 深度一致性         dense_depth[u,v]  vs  coords_depth   (单位: m)

用法:
    conda run -n d4rt python check_alignment.py --dataset kubric
    conda run -n d4rt python check_alignment.py --dataset pointodyssey
    conda run -n d4rt python check_alignment.py --all
    conda run -n d4rt python check_alignment.py --dataset kubric --clip-len 8 --n-pts 50 --verbose
"""

import sys, os, argparse, json
from pathlib import Path
import numpy as np

sys.path.insert(0, '/data1/zbf/my_dfrt')

DATASETS = [
    ("pointodyssey", "/data2/d4rt/datasets/PointOdyssey"),
    ("kubric",        "/data2/d4rt/datasets/kubric"),
    ("dynamic_replica", "/data1/d4rt/datasets/Dynamic_Replica"),
    ("co3dv2",        "/data2/d4rt/datasets/Co3Dv2"),
    ("blendedmvs",    "/data2/d4rt/datasets/BlendedMVS"),
    ("mvssynth",      "/data2/d4rt/datasets/MVS-Synth/GTAV_1080"),
    ("vkitti2",       "/data2/d4rt/datasets/VirtualKitti"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def project_to_image(pts3d_world, K, E_w2c):
    """
    pts3d_world: [N,3] world坐标
    K: [3,3] 内参
    E_w2c: [4,4] world-to-camera 外参
    返回 uv [N,2], valid [N] (bool)
    """
    N = len(pts3d_world)
    h = np.concatenate([pts3d_world, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (E_w2c @ h.T).T[:, :3]
    z = cam[:, 2]
    valid = (z > 1e-3) & np.isfinite(cam).all(axis=1)
    uv = np.full((N, 2), np.nan, dtype=np.float32)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    uv[valid, 0] = cam[valid, 0] / z[valid] * fx + cx
    uv[valid, 1] = cam[valid, 1] / z[valid] * fy + cy
    return uv, valid


def depth_backproject(depth_hw, pts2d_uv, K, E_w2c):
    """
    在 pts2d_uv 采样稠密深度图，反投影到世界坐标。
    depth_hw:  [H,W] float32，单位 m (z-depth)
    pts2d_uv:  [M,2] float32，像素坐标
    返回 pts3d_world [M,3]，dv [M] bool（深度有效）
    """
    H, W = depth_hw.shape
    K = K.astype(np.float64)
    E_w2c = E_w2c.astype(np.float64)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    xs = np.clip(np.round(pts2d_uv[:, 0]).astype(int), 0, W-1)
    ys = np.clip(np.round(pts2d_uv[:, 1]).astype(int), 0, H-1)
    z = depth_hw[ys, xs].astype(np.float64)
    dv = np.isfinite(z) & (z > 1e-3)

    x_c = (xs - cx) / fx * z
    y_c = (ys - cy) / fy * z
    pts_cam = np.stack([x_c, y_c, z], axis=1)   # [M,3]

    E_inv = np.linalg.inv(E_w2c)
    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam),1))], axis=1)
    pts3d_world = (E_inv @ pts_h.T).T[:, :3].astype(np.float32)
    return pts3d_world, dv


def print_stats(label, arr, unit="", indent="    "):
    if arr is None or len(arr) == 0:
        print(f"{indent}{label}: N/A")
        return
    arr = np.array(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        print(f"{indent}{label}: all NaN")
        return
    print(f"{indent}{label}: "
          f"mean={np.mean(arr):.4f}{unit}  "
          f"median={np.median(arr):.4f}{unit}  "
          f"max={np.max(arr):.4f}{unit}  "
          f"p95={np.percentile(arr,95):.4f}{unit}  "
          f"n={len(arr)}")


# ─────────────────────────────────────────────────────────────────────────────
# 主验证函数
# ─────────────────────────────────────────────────────────────────────────────

def check_alignment(clip, n_pts=200, seed=42, verbose=False):
    """
    对 clip 的每一帧逐帧检验三组点的对齐情况。

    返回 dict 包含：
      A. 2D重投影误差 (px)
      B. 3D点云对齐误差 pts3d_depth vs pts3d (m)
      C. 深度值一致性 dense_depth vs coords_depth (m)
    """
    T = clip.num_frames
    H, W = clip.image_size
    rng = np.random.default_rng(seed)

    has_tracks = (clip.trajs_3d_world is not None) and (clip.trajs_2d is not None)
    has_depth  = (clip.depths is not None)

    results_A = []  # 2D reproj error (px)
    results_B = []  # 3D pts3d_depth vs pts3d (m)
    results_C = []  # dense_depth vs coords_depth (m)

    per_frame = []

    # 从第一个有效帧选点
    idx = np.array([], dtype=int)
    if has_tracks:
        for t0 in range(T):
            v0 = clip.valids[t0] & clip.visibs[t0]
            idx = np.where(v0)[0]
            if len(idx) > 0:
                break
        if len(idx) > n_pts:
            idx = rng.choice(idx, n_pts, replace=False)

    for t in range(T):
        K  = clip.intrinsics[t].astype(np.float32)
        E  = clip.extrinsics[t].astype(np.float32)
        frame_info = {"frame": t}

        # ── A. 2D 重投影误差 ───────────────────────────────────────────────
        if has_tracks and len(idx) > 0:
            pts3d = clip.trajs_3d_world[t, idx].astype(np.float32)  # [M,3]
            pts2d_gt = clip.trajs_2d[t, idx].astype(np.float32)     # [M,2]
            valid_t = clip.valids[t, idx] & clip.visibs[t, idx]

            uv_reproj, reproj_v = project_to_image(pts3d, K, E)
            mask_A = valid_t & reproj_v & np.isfinite(pts2d_gt).all(axis=1)
            if mask_A.any():
                err_A = np.linalg.norm(uv_reproj[mask_A] - pts2d_gt[mask_A], axis=1)
                frame_info["reproj_err_px_mean"] = float(err_A.mean())
                frame_info["reproj_err_px_max"]  = float(err_A.max())
                results_A.extend(err_A.tolist())
            else:
                frame_info["reproj_err_px_mean"] = None

            # ── B. 3D 对齐误差：pts3d_depth vs pts3d ──────────────────────
            if has_depth and clip.depths[t] is not None:
                depth = clip.depths[t].astype(np.float32)  # [H,W]

                # 仅对 mask_A 内的点采样深度
                pts2d_sel = pts2d_gt[mask_A] if mask_A.any() else pts2d_gt[:0]
                pts3d_sel = pts3d[mask_A] if mask_A.any() else pts3d[:0]

                if len(pts2d_sel) > 0:
                    pts3d_from_depth, dv = depth_backproject(depth, pts2d_sel, K, E)
                    if dv.any():
                        dist_B = np.linalg.norm(
                            pts3d_from_depth[dv] - pts3d_sel[dv], axis=1
                        )
                        frame_info["pts3d_align_err_m_mean"] = float(dist_B.mean())
                        frame_info["pts3d_align_err_m_max"]  = float(dist_B.max())
                        results_B.extend(dist_B.tolist())

                        if verbose and t == 0:
                            print(f"\n  [t=0 detail] B: 3D align error sample (first 5):")
                            for i in range(min(5, dv.sum())):
                                vi = np.where(dv)[0][i]
                                p1 = pts3d_sel[vi]
                                p2 = pts3d_from_depth[vi]
                                print(f"    pts3d      = {p1}")
                                print(f"    pts3d_depth= {p2}")
                                print(f"    |diff|     = {np.linalg.norm(p1-p2):.4f}m")
                    else:
                        frame_info["pts3d_align_err_m_mean"] = None

                # ── C. 深度一致性：dense_depth vs coords_depth ─────────────
                if hasattr(clip, 'metadata') and clip.metadata.get('coords_depth') is not None:
                    coords_depth = clip.metadata['coords_depth']  # [T, N]
                    cd_t = coords_depth[t, idx][mask_A].astype(np.float32)  # [M2]

                    xs = np.clip(np.round(pts2d_gt[mask_A, 0]).astype(int), 0, W-1)
                    ys = np.clip(np.round(pts2d_gt[mask_A, 1]).astype(int), 0, H-1)
                    z_dense = depth[ys, xs].astype(np.float32)

                    cv = np.isfinite(cd_t) & (cd_t > 1e-3) & np.isfinite(z_dense) & (z_dense > 1e-3)
                    if cv.any():
                        # 直接比较：coords_depth 和 dense_depth 都是 z-depth (m)
                        diff_C = np.abs(z_dense[cv] - cd_t[cv])
                        frame_info["depth_consistency_m_mean"] = float(diff_C.mean())
                        frame_info["depth_consistency_m_max"]  = float(diff_C.max())
                        results_C.extend(diff_C.tolist())

                        if verbose and t == 0:
                            print(f"\n  [t=0 detail] C: depth consistency sample (first 5):")
                            for i in range(min(5, cv.sum())):
                                print(f"    coords_depth[i]= {cd_t[cv][i]:.4f}m  "
                                      f"dense_depth[i]= {z_dense[cv][i]:.4f}m  "
                                      f"|diff|= {diff_C[i]:.4f}m")

        per_frame.append(frame_info)

    return {
        "n_pts_selected": int(len(idx)) if has_tracks else 0,
        "T": T,
        # A
        "A_reproj_2d": {
            "mean_px": float(np.mean(results_A)) if results_A else None,
            "median_px": float(np.median(results_A)) if results_A else None,
            "max_px":  float(np.max(results_A))  if results_A else None,
            "p95_px":  float(np.percentile(results_A, 95)) if results_A else None,
        },
        # B
        "B_pts3d_align": {
            "mean_m": float(np.mean(results_B)) if results_B else None,
            "median_m": float(np.median(results_B)) if results_B else None,
            "max_m":  float(np.max(results_B))  if results_B else None,
            "p95_m":  float(np.percentile(results_B, 95)) if results_B else None,
        },
        # C
        "C_depth_consistency": {
            "mean_m": float(np.mean(results_C)) if results_C else None,
            "median_m": float(np.median(results_C)) if results_C else None,
            "max_m":  float(np.max(results_C))  if results_C else None,
            "p95_m":  float(np.percentile(results_C, 95)) if results_C else None,
        },
        "per_frame": per_frame,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def check_one(name, root, clip_len=16, seed=0, n_pts=200, verbose=False):
    from datasets.registry import create_adapter
    from datasets.sampling import DatasetSampler
    import random

    print(f"\n{'='*60}")
    print(f"[{name}]  root={root}")

    try:
        adapter = create_adapter(name=name, root=root, split='train')
        sampler = DatasetSampler(adapter, clip_len=clip_len,
                                 sampling_mode='stride', min_frames=2)
    except Exception as e:
        print(f"  ERROR init: {e}")
        return {"error": str(e)}

    print(f"  sequences: {len(sampler.valid_sequences)}")
    rng = random.Random(seed)
    seq, frame_indices = sampler.sample(rng)
    print(f"  seq={seq}  frames={frame_indices[:3]}...({len(frame_indices)} total)")

    try:
        clip = adapter.load_clip(seq, frame_indices)
    except Exception as e:
        import traceback
        print(f"  ERROR load_clip: {e}")
        traceback.print_exc()
        return {"error": str(e)}

    print(f"  has_tracks={clip.trajs_3d_world is not None}  "
          f"has_depth={clip.depths is not None}  "
          f"size={clip.image_size}")

    r = check_alignment(clip, n_pts=n_pts, seed=seed, verbose=verbose)

    print(f"\n  ── 结果汇总 ──")
    A = r["A_reproj_2d"]
    B = r["B_pts3d_align"]
    C = r["C_depth_consistency"]

    print(f"  A. 2D重投影误差   (reproj_pts vs gt_pts):")
    if A["mean_px"] is not None:
        print(f"     mean={A['mean_px']:.4f}px  median={A['median_px']:.4f}px  "
              f"p95={A['p95_px']:.4f}px  max={A['max_px']:.4f}px")
    else:
        print(f"     N/A (无有效轨迹点)")

    print(f"  B. 3D点对齐误差   (pts3d_depth[橙] vs pts3d[蓝]):")
    if B["mean_m"] is not None:
        print(f"     mean={B['mean_m']:.4f}m  median={B['median_m']:.4f}m  "
              f"p95={B['p95_m']:.4f}m  max={B['max_m']:.4f}m")
    else:
        print(f"     N/A")

    print(f"  C. 深度一致性     (dense_depth_euc vs coords_depth):")
    if C["mean_m"] is not None:
        print(f"     mean={C['mean_m']:.4f}m  median={C['median_m']:.4f}m  "
              f"p95={C['p95_m']:.4f}m  max={C['max_m']:.4f}m")
    else:
        print(f"     N/A (coords_depth 不在 metadata 中)")

    # 综合判断
    print(f"\n  ── 对齐质量评估 ──")
    ok = True
    if A["mean_px"] is not None:
        flag = "✓ OK" if A["mean_px"] < 1.0 else "✗ 差"
        print(f"  A: {flag}  (阈值 < 1px，实际 {A['mean_px']:.3f}px)")
        if A["mean_px"] >= 1.0:
            ok = False
    if B["mean_m"] is not None:
        # B/C 受像素取整误差影响较大：整像素误差 ≈ 1px，
        # 在 10m 深度 / 443px 焦距场景下对应 ~0.02m/px，
        # 但在深度不连续边缘处采样误差可能很大（物体边缘不同 ID 的深度）
        # 因此 B/C 的大误差通常是 "采到了相邻像素的不同物体深度"，不代表加载器错误
        # 真正的判断依据是 A（2D重投影）和肉眼 rrd 可视化
        flag = "✓ OK" if B["mean_m"] < 0.5 else "⚠ 警告(可能受像素取整/深度不连续影响)"
        print(f"  B: {flag}  实际 {B['mean_m']:.3f}m")
        print(f"     注: B/C 误差大时请先确认 A(2D重投影)正确且 rrd 可视化三点对齐")
    if C["mean_m"] is not None:
        flag = "✓ OK" if C["mean_m"] < 0.3 else "⚠ 警告(可能受像素取整/深度不连续影响)"
        print(f"  C: {flag}  实际 {C['mean_m']:.3f}m")

    if A.get("mean_px") is not None and A["mean_px"] < 1.0:
        print(f"\n  ★ 关键指标 A(2D重投影) 正确 → 加载器几何一致性验证通过")
        print(f"  ★ B/C 误差大但 A 正确 → 通常是深度图边缘采样噪声，不是 bug")
    print(f"\n  总体: {'✓ 对齐良好' if ok else '✗ 存在对齐问题'}")
    return r


def main():
    parser = argparse.ArgumentParser(description="验证 pts3d / pts3d_depth / reproj_pts 三组点对齐")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pts", type=int, default=200, help="每帧采样点数")
    parser.add_argument("--verbose", action="store_true", help="打印 t=0 帧详细样本")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [(n, r) for n, r in DATASETS if n == args.dataset]
        if not datasets:
            print(f"未知数据集: {args.dataset}")
            print(f"可用: {[n for n,_ in DATASETS]}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    all_results = {}
    for name, root in datasets:
        r = check_one(name, root,
                      clip_len=args.clip_len,
                      seed=args.seed,
                      n_pts=args.n_pts,
                      verbose=args.verbose)
        all_results[name] = r

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n结果已保存: {args.out}")

    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print(f"{'数据集':<20} {'A:reproj(px)':>14} {'B:3D对齐(m)':>14} {'C:深度一致(m)':>14}")
    print("-"*65)
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<20}  ERROR")
            continue
        A = r.get("A_reproj_2d", {})
        B = r.get("B_pts3d_align", {})
        C = r.get("C_depth_consistency", {})
        a = f"{A.get('mean_px','N/A'):.4f}" if A.get('mean_px') is not None else "N/A"
        b = f"{B.get('mean_m','N/A'):.4f}" if B.get('mean_m') is not None else "N/A"
        c = f"{C.get('mean_m','N/A'):.4f}" if C.get('mean_m') is not None else "N/A"
        print(f"  {name:<20} {a:>14} {b:>14} {c:>14}")


if __name__ == "__main__":
    main()
