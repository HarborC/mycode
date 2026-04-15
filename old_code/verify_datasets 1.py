#!/usr/bin/env python3
"""
Per-dataset GT verification: checks 2D/3D consistency via reprojection.
For has_tracks datasets: reprojects trajs_3d_world -> 2D and compares with trajs_2d.
For no-tracks datasets: reprojects depth -> 3D -> 2D (round-trip) and checks error.
Saves visual outputs to vis_gt/verify/<dataset>/

Usage:
    python verify_datasets.py --dataset pointodyssey
    python verify_datasets.py --all
    python verify_datasets.py --dataset kubric --rerun  # saves .rrd for rerun viewer
"""
import sys, os, json, argparse, traceback
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

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


def project_world_to_image(pts_world, K, E):
    """pts_world: [N,3], K: [3,3], E: [4,4] w2c -> returns [N,2] pixel coords"""
    N = pts_world.shape[0]
    h = np.concatenate([pts_world, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N,4]
    cam = (E @ h.T).T[:, :3]  # [N,3]
    z = cam[:, 2:3]
    valid = (z[:, 0] > 1e-3) & np.isfinite(cam).all(1)
    uv = np.full((N, 2), np.nan, dtype=np.float32)
    if valid.any():
        uv[valid, 0] = cam[valid, 0] / z[valid, 0] * K[0, 0] + K[0, 2]
        uv[valid, 1] = cam[valid, 1] / z[valid, 0] * K[1, 1] + K[1, 2]
    return uv, valid


def save_gif(frames, path, fps=8):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000/fps), loop=0)


def draw_tracks_frame(img_rgb, pts_gt, pts_reproj, valid_mask, t):
    """Draw GT (green) vs reprojected (red) points on frame."""
    frame = img_rgb.copy()
    H, W = frame.shape[:2]
    for i in range(len(pts_gt)):
        if not valid_mask[i]:
            continue
        gx, gy = int(np.clip(pts_gt[i, 0], 0, W-1)), int(np.clip(pts_gt[i, 1], 0, H-1))
        cv2.circle(frame, (gx, gy), 6, (0, 220, 0), -1)
        if np.isfinite(pts_reproj[i]).all():
            rx, ry = int(np.clip(pts_reproj[i, 0], 0, W-1)), int(np.clip(pts_reproj[i, 1], 0, H-1))
            cv2.circle(frame, (rx, ry), 5, (220, 0, 0), -1)
            cv2.line(frame, (gx, gy), (rx, ry), (255, 200, 0), 1)
    cv2.putText(frame, f"t={t} green=GT red=reproj", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return frame


def verify_has_tracks(clip, out_dir, n_pts=200, seed=42):
    """Verify 3D->2D reprojection matches GT trajs_2d."""
    T = clip.num_frames
    H, W = clip.image_size
    rng = np.random.default_rng(seed)

    # Pick valid points from the first frame that has valid+visible points
    idx = np.array([], dtype=int)
    for t0 in range(T):
        v0 = clip.valids[t0] & clip.visibs[t0]
        idx = np.where(v0)[0]
        if len(idx) > 0:
            break
    if len(idx) == 0:
        return {"error": "no valid points in any frame"}
    if len(idx) > n_pts:
        idx = rng.choice(idx, n_pts, replace=False)

    errors_per_frame = []
    depth_3d_errors_per_frame = []
    frames_vis = []

    for t in range(T):
        K = clip.intrinsics[t]
        E = clip.extrinsics[t]
        pts3d = clip.trajs_3d_world[t, idx]   # [M,3]
        pts2d_gt = clip.trajs_2d[t, idx]       # [M,2]
        valid_t = clip.valids[t, idx] & clip.visibs[t, idx]

        pts2d_reproj, reproj_valid = project_world_to_image(pts3d, K, E)

        # Compute reprojection error only for valid+visible points
        mask = valid_t & reproj_valid
        if mask.any():
            err = np.linalg.norm(pts2d_reproj[mask] - pts2d_gt[mask], axis=1)
            errors_per_frame.append(float(err.mean()))
        else:
            errors_per_frame.append(np.nan)

        # Depth backprojection vs trajs_3d_world
        if clip.depths is not None and clip.depths[t] is not None and mask.any():
            depth = clip.depths[t]  # [H,W]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            uv = pts2d_gt[mask]  # [M2,2]
            xs = np.clip(np.round(uv[:, 0]).astype(int), 0, W-1)
            ys = np.clip(np.round(uv[:, 1]).astype(int), 0, H-1)
            z = depth[ys, xs].astype(np.float32)
            depth_valid = np.isfinite(z) & (z > 1e-3)
            if depth_valid.any():
                x_cam = (xs[depth_valid] - cx) / fx * z[depth_valid]
                y_cam = (ys[depth_valid] - cy) / fy * z[depth_valid]
                pts_cam = np.stack([x_cam, y_cam, z[depth_valid]], axis=1)
                E_inv = np.linalg.inv(E)
                pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam),1), dtype=np.float32)], axis=1)
                pts3d_from_depth = (E_inv @ pts_h.T).T[:, :3]
                pts3d_gt = pts3d[mask][depth_valid]
                dist = np.linalg.norm(pts3d_from_depth - pts3d_gt, axis=1)
                depth_3d_errors_per_frame.append(float(dist.mean()))
            else:
                depth_3d_errors_per_frame.append(np.nan)
        else:
            depth_3d_errors_per_frame.append(np.nan)

        # Visualize
        img = clip.images[t]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 255)).astype(np.uint8)
        frame = draw_tracks_frame(img, pts2d_gt[valid_t], pts2d_reproj[valid_t],
                                   np.ones(valid_t.sum(), dtype=bool), t)
        frames_vis.append(frame)

    # Save GIF
    out_dir.mkdir(parents=True, exist_ok=True)
    save_gif(frames_vis, str(out_dir / "reproj_check.gif"))

    valid_errs = [e for e in errors_per_frame if not np.isnan(e)]
    valid_depth_errs = [e for e in depth_3d_errors_per_frame if not np.isnan(e)]
    result = {
        "mean_reproj_error_px": float(np.mean(valid_errs)) if valid_errs else None,
        "max_reproj_error_px":  float(np.max(valid_errs)) if valid_errs else None,
        "mean_depth3d_error_m": float(np.mean(valid_depth_errs)) if valid_depth_errs else None,
        "max_depth3d_error_m":  float(np.max(valid_depth_errs)) if valid_depth_errs else None,
        "per_frame_errors": errors_per_frame,
        "per_frame_depth3d_errors": depth_3d_errors_per_frame,
        "n_pts": int(len(idx)),
        "image_size": [H, W],
    }
    return result


def verify_no_tracks(clip, out_dir, n_pts=500, seed=42):
    """Verify depth round-trip: pixel->3D->pixel."""
    T = clip.num_frames
    H, W = clip.image_size
    rng = np.random.default_rng(seed)

    if clip.depths is None:
        return {"error": "no depth available"}

    errors_per_frame = []
    frames_vis = []

    for t in range(T):
        depth = clip.depths[t]  # [H,W]
        K = clip.intrinsics[t]
        E = clip.extrinsics[t]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        # Sample valid depth pixels
        valid_mask = np.isfinite(depth) & (depth > 1e-3) & (depth < 1500)
        ys, xs = np.where(valid_mask)
        if len(ys) == 0:
            errors_per_frame.append(np.nan)
            frames_vis.append(np.zeros((H, W, 3), dtype=np.uint8))
            continue
        if len(ys) > n_pts:
            sel = rng.choice(len(ys), n_pts, replace=False)
            ys, xs = ys[sel], xs[sel]

        # Backproject to camera coords
        z = depth[ys, xs].astype(np.float32)
        x_c = (xs - cx) * z / fx
        y_c = (ys - cy) * z / fy
        pts_cam = np.stack([x_c, y_c, z], axis=1)  # [N,3]

        # cam -> world
        E_inv = np.linalg.inv(E)
        pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)], axis=1)
        pts_world = (E_inv @ pts_h.T).T[:, :3]

        # world -> cam (same frame, should be identity round-trip)
        pts2d_reproj, reproj_valid = project_world_to_image(pts_world.astype(np.float32), K, E)

        pts2d_gt = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        mask = reproj_valid
        if mask.any():
            err = np.linalg.norm(pts2d_reproj[mask] - pts2d_gt[mask], axis=1)
            errors_per_frame.append(float(err.mean()))
        else:
            errors_per_frame.append(np.nan)

        # Visualize depth map
        d_vis = np.clip(depth / np.percentile(depth[valid_mask], 95) * 255, 0, 255).astype(np.uint8)
        d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_PLASMA)
        d_vis = cv2.cvtColor(d_vis, cv2.COLOR_BGR2RGB)
        err_val = errors_per_frame[-1]
        cv2.putText(d_vis, f"t={t} roundtrip_err={err_val:.3f}px", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        frames_vis.append(d_vis)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_gif(frames_vis, str(out_dir / "depth_roundtrip.gif"))

    valid_errs = [e for e in errors_per_frame if not np.isnan(e)]
    return {
        "mean_roundtrip_error_px": float(np.mean(valid_errs)) if valid_errs else None,
        "max_roundtrip_error_px":  float(np.max(valid_errs)) if valid_errs else None,
        "per_frame_errors": errors_per_frame,
        "n_pts": n_pts,
        "image_size": [H, W],
    }


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert [H,W,2] optical flow to an HSV-based RGB image [H,W,3] uint8."""
    import cv2
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2          # hue  → direction
    hsv[..., 1] = 255                              # saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def log_clip_to_rerun(clip, name, seq, rrd_path: Path):
    """Log clip to a rerun .rrd file for offline inspection using Blueprint layout."""
    import rerun as rr
    import rerun.blueprint as rrb

    H, W = clip.image_size
    has_depth   = clip.depths  is not None
    has_normals = clip.normals is not None
    has_flow    = clip.flows   is not None
    has_trajs3d = clip.trajs_3d_world is not None

    # ── Blueprint ─────────────────────────────────────────────────────
    top_views = [
        rrb.Spatial2DView(
            name="RGB & Reprojection",
            origin="world/camera/image",
            contents=["+ $origin/**"],
            background=[30, 30, 30],
        ),
    ]
    if has_depth:
        top_views.append(rrb.Spatial2DView(
            name="Depth",
            origin="world/camera/image",
            contents=["+ world/camera/image/depth"],
            background=[30, 30, 30],
        ))
    if has_normals:
        top_views.append(rrb.Spatial2DView(
            name="Normals",
            origin="vis/normals",
            background=[30, 30, 30],
        ))
    if has_flow:
        top_views.append(rrb.Spatial2DView(
            name="Flow",
            origin="vis/flow",
            background=[30, 30, 30],
        ))

    view_3d = rrb.Spatial3DView(
        name="3D Scene",
        origin="/",
        contents=["+ world/**"],
        background=[20, 20, 20],
    )

    info_view = rrb.TextDocumentView(name="Info", origin="info")

    layout = rrb.Vertical(
        rrb.Horizontal(*top_views),
        rrb.Horizontal(view_3d, info_view, column_shares=[4, 1]),
        row_shares=[1, 2],
    )
    blueprint = rrb.Blueprint(layout, collapse_panels=True)

    # ── Recording ─────────────────────────────────────────────────────
    with rr.RecordingStream(application_id=f"{name}/{seq}", make_default=False) as rec:
        rec.save(str(rrd_path), default_blueprint=blueprint)

        # ── Metadata ──────────────────────────────────────────────────────
        info_lines = [
            f"# {name} / {seq}",
            f"- **Frames**: {clip.num_frames}",
            f"- **Resolution**: {W} x {H}",
            f"- **Depth**: {'yes' if has_depth else 'no'}",
            f"- **Normals**: {'yes' if has_normals else 'no'}",
            f"- **Flow**: {'yes' if has_flow else 'no'}",
            f"- **2D tracks**: {'yes' if clip.trajs_2d is not None else 'no'}",
            f"- **3D tracks**: {'yes' if has_trajs3d else 'no'}",
            "\n### Visualization",
            "- **Green points**: Ground truth 2D trajectories",
            "- **Red points**: Reprojected 3D trajectories"
        ]
        rec.log("info", rr.TextDocument("\n".join(info_lines), media_type=rr.MediaType.MARKDOWN))

        # ── Per-frame data ────────────────────────────────────────────────
        # Estimate frustum_scale from world-space scene extent (camera positions or trajs)
        frustum_scale = 0.3
        if has_trajs3d:
            # Use median distance between consecutive camera centers as scale reference
            centers = np.stack([np.linalg.inv(clip.extrinsics[t])[:3, 3] for t in range(clip.num_frames)])
            scene_extent = float(np.linalg.norm(clip.trajs_3d_world[0].max(0) - clip.trajs_3d_world[0].min(0)))
            frustum_scale = max(0.05, scene_extent * 0.05)
        elif has_depth:
            median_depths = [float(np.median(d[d > 0])) for d in clip.depths if np.any(d > 0)]
            if median_depths:
                frustum_scale = float(np.median(median_depths)) * 0.1

        for t in range(clip.num_frames):
            rec.set_time("frame", sequence=t)

            K = clip.intrinsics[t]
            E = clip.extrinsics[t]

            E_c2w = np.linalg.inv(E)
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(E_c2w[:3, :3]).as_quat()  # [x,y,z,w]
            rec.log("world/camera", rr.Transform3D(
                translation=E_c2w[:3, 3],
                quaternion=rr.Quaternion(xyzw=quat),
            ))
            rec.log("world/camera/image", rr.Pinhole(
                image_from_camera=K, width=W, height=H,
                image_plane_distance=frustum_scale,
            ))

            img = clip.images[t]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            rec.log("world/camera/image", rr.Image(img))

            if has_depth and clip.depths[t] is not None:
                rec.log("world/camera/image/depth", rr.DepthImage(
                    clip.depths[t], meter=1.0, colormap="Turbo", point_fill_ratio=1.0,
                ))
                # Explicitly backproject full depth map -> dense 3D point cloud
                depth = clip.depths[t]
                fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
                ys_all, xs_all = np.where(depth > 1e-3)
                z_all = depth[ys_all, xs_all].astype(np.float32)
                fin = np.isfinite(z_all)
                xs_v, ys_v, z_v = xs_all[fin], ys_all[fin], z_all[fin]
                if len(z_v) > 0:
                    x_c = (xs_v - cx) / fx * z_v
                    y_c = (ys_v - cy) / fy * z_v
                    pts_cam = np.stack([x_c, y_c, z_v], axis=1).astype(np.float64)
                    E_inv = np.linalg.inv(E.astype(np.float64))
                    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
                    pts3d_dense = (E_inv @ pts_h.T).T[:, :3].astype(np.float32)
                    # Subsample to ~100k points for performance
                    step = max(1, len(pts3d_dense) // 100000)
                    pts3d_dense = pts3d_dense[::step]
                    img = clip.images[t]
                    rgb = img[ys_v[::step], xs_v[::step]]
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    rec.log("world/pts3d_dense", rr.Points3D(pts3d_dense, colors=rgb, radii=0.3))

            if has_normals and clip.normals[t] is not None:
                normal_vis = ((clip.normals[t] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                rec.log("vis/normals", rr.Image(normal_vis))

            if has_flow and clip.flows[t] is not None:
                rec.log("vis/flow", rr.Image(_flow_to_rgb(clip.flows[t])))

            if has_trajs3d and clip.trajs_2d is not None and clip.valids is not None:
                valid = clip.valids[t].astype(bool)
                if valid.any():
                    pts2d_gt = clip.trajs_2d[t][valid].astype(np.float32)
                    pts3d_v = clip.trajs_3d_world[t][valid].astype(np.float32)
                    uv_reproj, reproj_valid = project_world_to_image(pts3d_v, K.astype(np.float32), E.astype(np.float32))
                    rec.log("world/camera/image/gt_pts",
                            rr.Points2D(pts2d_gt[reproj_valid], colors=[0, 255, 0], radii=6))
                    rec.log("world/camera/image/reproj_pts",
                            rr.Points2D(uv_reproj[reproj_valid], colors=[255, 0, 0], radii=5))

                    # Backproject depth at pts2d_gt positions -> orange points for comparison
                    # Use dv mask first, then log pts3d with same mask so both point sets align
                    if has_depth and clip.depths[t] is not None:
                        depth = clip.depths[t]
                        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
                        xs = np.clip(np.round(pts2d_gt[:,0]).astype(int), 0, W-1)
                        ys = np.clip(np.round(pts2d_gt[:,1]).astype(int), 0, H-1)
                        z = depth[ys, xs].astype(np.float32)
                        dv = np.isfinite(z) & (z > 1e-3)
                        if dv.any():
                            x_c = (xs[dv] - cx) / fx * z[dv]
                            y_c = (ys[dv] - cy) / fy * z[dv]
                            pts_cam = np.stack([x_c, y_c, z[dv]], axis=1)
                            E_inv = np.linalg.inv(E.astype(np.float64))
                            pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam),1), dtype=np.float32)], axis=1)
                            pts3d_from_depth = (E_inv @ pts_h.T).T[:, :3].astype(np.float32)
                            # Log both with the same dv-filtered subset so they are point-to-point comparable
                            rec.log("world/pts3d", rr.Points3D(pts3d_v[dv], colors=[0, 180, 255], radii=0.5))
                            rec.log("world/pts3d_from_depth", rr.Points3D(pts3d_from_depth, colors=[255, 140, 0], radii=0.5))
                        else:
                            rec.log("world/pts3d", rr.Points3D(pts3d_v, colors=[0, 180, 255], radii=0.5))
                    else:
                        rec.log("world/pts3d", rr.Points3D(pts3d_v, colors=[0, 180, 255], radii=0.5))

        # # ── 3D trajectories ───────────────────────────────────────────────
        # if has_trajs3d:
        #     N = clip.trajs_3d_world.shape[1]
        #     strips, strip_colors = [], []
        #     palette = np.random.RandomState(42).randint(60, 255, size=(N, 3), dtype=np.uint8)
        #     for n in range(N):
        #         track = clip.trajs_3d_world[:, n, :]
        #         if clip.valids is not None:
        #             track = track[clip.valids[:, n].astype(bool)]
        #         if len(track) >= 2:
        #             strips.append(track)
        #             strip_colors.append(palette[n])
        #     if strips:
        #         rec.log("world/tracks", rr.LineStrips3D(strips, colors=strip_colors, radii=0.005))

    print(f"  -> rerun saved: {rrd_path}")


def verify_one_dataset(name, root, out_base, clip_len=16, seed=0, use_rerun=False, num_clips=1):
    from datasets.registry import create_adapter
    from datasets.sampling import DatasetSampler
    import random

    out_dir = Path(out_base) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{name}] root={root}")

    try:
        adapter = create_adapter(name=name, root=root, split='train')
    except Exception as e:
        result = {"error": f"adapter init failed: {e}"}
        print(f"  ERROR: {e}")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    try:
        sampler = DatasetSampler(adapter, clip_len=clip_len, sampling_mode='stride', min_frames=2)
    except Exception as e:
        result = {"error": f"sampler init failed: {e}"}
        print(f"  ERROR: {e}")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    print(f"  sequences: {len(sampler.valid_sequences)}")

    clips_results = []
    for clip_i in range(num_clips):
        clip_seed = seed + clip_i
        rng = random.Random(clip_seed)
        seq, frame_indices = sampler.sample(rng)

        # For datasets with precomputed tracks, resample around ref_frame where valids are non-zero
        if name in ("co3dv2", "vkitti2"):
            import numpy as _np
            npz = adapter.precompute_root / seq / "precomputed.npz"
            h5 = npz.with_suffix(".h5")
            cache_path = h5 if h5.exists() else (npz if npz.exists() else None)
            if cache_path is not None:
                if str(cache_path).endswith(".h5"):
                    import h5py as _h5
                    with _h5.File(cache_path, "r") as _f:
                        ref = int(_f["ref_frame"][()])
                else:
                    ref = int(_np.load(cache_path, allow_pickle=True)["ref_frame"])
                half = clip_len // 2
                frame_indices = list(range(max(0, ref - half), ref + half))[:clip_len]

        print(f"  [clip {clip_i}] seq={seq}  frames={frame_indices[:3]}...({len(frame_indices)} total)")

        try:
            clip = adapter.load_clip(seq, frame_indices)
        except Exception as e:
            print(f"  ERROR loading clip {clip_i}: {e}")
            clips_results.append({"error": f"load_clip failed: {e}"})
            continue

        has_tracks = clip.metadata.get("has_tracks", False) and clip.trajs_3d_world is not None
        print(f"  has_tracks={has_tracks}, frames={len(clip.images)}, size={clip.image_size}")

        metrics = {}
        try:
            if has_tracks:
                metrics = verify_has_tracks(clip, out_dir)
                me = metrics.get('mean_reproj_error_px')
                mx = metrics.get('max_reproj_error_px')
                print(f"  reproj_error: mean={me:.3f}px  max={mx:.3f}px" if me is not None
                      else f"  reproj_error: N/A (no valid points)")
            else:
                metrics = verify_no_tracks(clip, out_dir)
                me = metrics.get('mean_roundtrip_error_px')
                mx = metrics.get('max_roundtrip_error_px')
                print(f"  roundtrip_error: mean={me:.4f}px  max={mx:.4f}px" if me is not None
                      else f"  roundtrip_error: N/A")
        except Exception as e:
            metrics = {"error": f"verify failed: {e}", "traceback": traceback.format_exc()}
            print(f"  ERROR in verify: {e}")

        if use_rerun:
            rrd_name = f"clip_{clip_i}.rrd" if num_clips > 1 else "clip.rrd"
            try:
                log_clip_to_rerun(clip, name, seq, out_dir / rrd_name)
            except Exception as e:
                print(f"  rerun ERROR: {e}")

        clips_results.append({
            "clip_index": clip_i,
            "sequence": seq,
            "frame_indices": frame_indices,
            "has_tracks": has_tracks,
            "num_frames": len(clip.images),
            "image_size": list(clip.image_size),
            "metrics": metrics,
        })

    result = {
        "dataset": name,
        "clips": clips_results,
        # keep top-level fields from first clip for backward compat
        **(clips_results[0] if clips_results else {}),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"  -> saved to {out_dir}/")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="single dataset name to test")
    parser.add_argument("--all", action="store_true", help="test all datasets")
    parser.add_argument("--out", default="vis_gt/verify")
    parser.add_argument("--clip-len", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clips", type=int, default=1, help="number of clips to render per dataset")
    parser.add_argument("--no-rerun", action="store_true", help="disable saving .rrd for rerun viewer")
    args = parser.parse_args()

    use_rerun = not args.no_rerun

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [(n, r) for n, r in DATASETS if n == args.dataset]
        if not datasets:
            print(f"Unknown dataset: {args.dataset}. Available: {[n for n,_ in DATASETS]}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    all_results = {}
    for name, root in datasets:
        result = verify_one_dataset(name, root, args.out, clip_len=args.clip_len,
                                    seed=args.seed, use_rerun=use_rerun,
                                    num_clips=args.num_clips)
        all_results[name] = result

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, r in all_results.items():
        m = r.get("metrics", {})
        if "error" in r:
            print(f"  {name:20s}  ERROR: {r['error']}")
        elif "error" in m:
            print(f"  {name:20s}  METRICS ERROR: {m['error']}")
        elif r.get("has_tracks"):
            print(f"  {name:20s}  reproj_err={m.get('mean_reproj_error_px', 'N/A'):.3f}px")
        else:
            print(f"  {name:20s}  roundtrip_err={m.get('mean_roundtrip_error_px', 'N/A'):.4f}px")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(
        json.dumps(all_results, indent=2, default=str))
    print(f"\nFull results: {args.out}/summary.json")


if __name__ == "__main__":
    main()
