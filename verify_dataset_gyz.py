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
    python verify_datasets.py \
        --dataset scannet \
        --out /mnt/ccw_1/d4rt/rerun/scannet_1 \
        --sequence scene0030_00 \
        --clip-len 100 \
        --pt2d-radius 1.0 \
        --pt3d-radius 0.008

    python verify_datasets.py \
        --dataset blendedmvs \
        --out /mnt/ccw_1/d4rt/rerun/blendedmvs \
        --sequence 5a3ca9cb270f0e3f14d0eddb \
        --clip-len 30 \
        --pt2d-radius 1.0 \
        --pt3d-radius 0.008

    python verify_datasets.py \
        --dataset scannetpp \
        --out /mnt/ccw_1/d4rt/rerun/scannetpp_0c5385e84b_3 \
        --sequence 0c5385e84b \
        --start-frame 500 \
        --clip-len 30 \
        --pt2d-radius 1.0 \
        --pt3d-radius 0.008
"""
import sys, os, json, argparse, traceback
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, '/data2/mycode')

DATASETS = [
     ("mvssynth",  "/data2/d4rt/datasets/MVS-Synth/GTAV_1080"),
     ("tartanair", "/data2/d4rt/datasets/TartanAir"),
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


def safe_remap(img, u, v):
    """Safely apply cv2.remap to 1D point arrays exceeding OpenCV's SHRT_MAX limit."""
    N = len(u)
    MAX_DIM = 30000
    out = np.zeros(N, dtype=np.float32)
    for i in range(0, N, MAX_DIM):
        end = min(i + MAX_DIM, N)
        map_x = u[i:end].astype(np.float32).reshape(1, -1)
        map_y = v[i:end].astype(np.float32).reshape(1, -1)
        out[i:end] = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).flatten()
    return out


def scale_intrinsics_single(K, src_w, src_h, dst_w, dst_h):
    """Scale one 3x3 intrinsic matrix from (src_w,src_h) to (dst_w,dst_h)."""
    K_out = K.astype(np.float32).copy()
    sx = dst_w / src_w
    sy = dst_h / src_h
    K_out[0, 0] *= sx
    K_out[0, 2] *= sx
    K_out[1, 1] *= sy
    K_out[1, 2] *= sy
    return K_out


def compute_depth_edge_mask(depth, rel_thresh=0.03, abs_thresh=0.03):
    """Compute a conservative edge mask on a single depth map."""
    kernel = np.ones((3, 3), dtype=np.float32)
    d_max = cv2.dilate(depth, kernel)
    d_min = cv2.erode(depth, kernel)
    local_range = d_max - d_min
    safe_d = np.maximum(depth, 1e-6)
    edge = ((local_range / safe_d) > rel_thresh) | (local_range > abs_thresh) | (depth <= 0)
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    return edge.astype(bool)


def load_scannetpp_raw_depth(scene_dir: Path, frame_path: str):
    """Load the original ScanNet++ depth PNG for a frame, in metres."""
    frame_name = Path(frame_path).stem
    depth_path = scene_dir / "depths" / f"{frame_name}.png"
    d = np.asarray(Image.open(depth_path))
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32) / 1000.0


def backproject_depth_samples(depth, uv, K, E, edge_mask=None):
    """Backproject sampled depth values at floating-point pixel coordinates."""
    if len(uv) == 0:
        return None, np.zeros((0,), dtype=bool)

    H, W = depth.shape[:2]
    u = uv[:, 0].astype(np.float32)
    v = uv[:, 1].astype(np.float32)
    z = safe_remap(depth, u, v)

    valid = np.isfinite(z) & (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if edge_mask is not None:
        ui = np.clip(np.round(u).astype(np.int32), 0, W - 1)
        vi = np.clip(np.round(v).astype(np.int32), 0, H - 1)
        valid &= ~edge_mask[vi, ui]

    if not valid.any():
        return None, valid

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u[valid] - cx) / fx * z[valid]
    y_cam = (v[valid] - cy) / fy * z[valid]
    pts_cam = np.stack([x_cam, y_cam, z[valid]], axis=1)

    E_inv = np.linalg.inv(E.astype(np.float64))
    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)], axis=1)
    pts3d = (E_inv @ pts_h.T).T[:, :3].astype(np.float32)
    return pts3d, valid


def draw_tracks_frame(img_rgb, pts_gt, pts_reproj, valid_mask, t):
    """Draw GT (green) vs reprojected (red) points on frame."""
    frame = img_rgb.copy()
    H, W = frame.shape[:2]
    for i in range(len(pts_gt)):
        if not valid_mask[i]:
            continue
        gx, gy = int(np.clip(pts_gt[i, 0], 0, W-1)), int(np.clip(pts_gt[i, 1], 0, H-1))
        cv2.circle(frame, (gx, gy), 3, (0, 220, 0), -1)
        if np.isfinite(pts_reproj[i]).all():
            rx, ry = int(np.clip(pts_reproj[i, 0], 0, W-1)), int(np.clip(pts_reproj[i, 1], 0, H-1))
            cv2.circle(frame, (rx, ry), 2, (220, 0, 0), -1)
            cv2.line(frame, (gx, gy), (rx, ry), (255, 200, 0), 1)
    cv2.putText(frame, f"t={t} green=GT red=reproj", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return frame


def verify_has_tracks(clip, out_dir, n_pts=200, seed=42, scene_dir=None):
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
        if mask.any():
            uv = pts2d_gt[mask]  # [M2,2]
            if clip.metadata.get("dataset_name") == "scannetpp" and scene_dir is not None:
                depth = load_scannetpp_raw_depth(scene_dir, clip.frame_paths[t])
                H_d, W_d = depth.shape[:2]
                K_d = scale_intrinsics_single(K, W, H, W_d, H_d)
                uv_d = uv.astype(np.float32).copy()
                uv_d[:, 0] *= W_d / W
                uv_d[:, 1] *= H_d / H
                edge_mask = compute_depth_edge_mask(depth)
                pts3d_from_depth, depth_valid = backproject_depth_samples(depth, uv_d, K_d, E, edge_mask=edge_mask)
            elif clip.depths is not None and clip.depths[t] is not None:
                depth = clip.depths[t]
                pts3d_from_depth, depth_valid = backproject_depth_samples(depth, uv, K, E)
            else:
                pts3d_from_depth, depth_valid = None, np.zeros((len(uv),), dtype=bool)

            if pts3d_from_depth is not None and depth_valid.any():
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

def log_clip_to_rerun(clip, name, seq, rrd_path: Path, pt2d_radius=1.5, pt3d_radius=0.01, scene_dir=None):
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

        # ── Pre-compute union of all valid 3D world points (time-independent) ──
        # Log once outside the time loop so all points are visible regardless
        # of which frame is selected in the viewer timeline.
        if has_trajs3d and clip.valids is not None:
            all_valid_pts = []
            for _t in range(clip.num_frames):
                _valid = clip.valids[_t].astype(bool)
                if clip.visibs is not None:
                    _valid &= clip.visibs[_t].astype(bool)
                if _valid.any():
                    all_valid_pts.append(clip.trajs_3d_world[_t][_valid].astype(np.float32))
            if all_valid_pts:
                union_pts = np.concatenate(all_valid_pts, axis=0)
                # Deduplicate (world coords are identical across frames for the same point)
                union_pts = np.unique(union_pts, axis=0)
                rec.log("world/pts3d_all",
                        rr.Points3D(union_pts, colors=[0, 200, 255], radii=pt3d_radius),
                        static=True)

        # ── Per-frame data ────────────────────────────────────────────────
        frustum_scale = 0.3
        if has_depth:
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

            if has_normals and clip.normals[t] is not None:
                normal_vis = ((clip.normals[t] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                rec.log("vis/normals", rr.Image(normal_vis))

            if has_flow and clip.flows[t] is not None:
                rec.log("vis/flow", rr.Image(_flow_to_rgb(clip.flows[t])))

            if has_trajs3d and clip.trajs_2d is not None and clip.valids is not None:
                valid = clip.valids[t].astype(bool)
                if clip.visibs is not None:
                    valid &= clip.visibs[t].astype(bool)
                if valid.any():
                    pts2d_gt = clip.trajs_2d[t][valid].astype(np.float32)
                    pts3d_v = clip.trajs_3d_world[t][valid].astype(np.float32)
                    uv_reproj, reproj_valid = project_world_to_image(pts3d_v, K.astype(np.float32), E.astype(np.float32))
                    rec.log("world/camera/image/gt_pts",
                            rr.Points2D(pts2d_gt[reproj_valid], colors=[0, 255, 0], radii=pt2d_radius))
                    rec.log("world/camera/image/reproj_pts",
                            rr.Points2D(uv_reproj[reproj_valid], colors=[255, 0, 0], radii=pt2d_radius * 0.67))
                    # Per-frame colored pts: highlight currently-valid subset over the static union cloud
                    rec.log("world/pts3d_cur",
                            rr.Points3D(pts3d_v, colors=[255, 220, 0], radii=pt3d_radius * 1.2))

                    # Backproject depth at pts2d_gt positions -> orange points for comparison
                    if clip.metadata.get("dataset_name") == "scannetpp" and scene_dir is not None:
                        depth = load_scannetpp_raw_depth(scene_dir, clip.frame_paths[t])
                        H_d, W_d = depth.shape[:2]
                        K_d = scale_intrinsics_single(K, W, H, W_d, H_d)
                        uv_d = pts2d_gt.copy()
                        uv_d[:, 0] *= W_d / W
                        uv_d[:, 1] *= H_d / H
                        edge_mask = compute_depth_edge_mask(depth)
                        pts3d_from_depth, dv = backproject_depth_samples(depth, uv_d, K_d, E, edge_mask=edge_mask)
                    elif has_depth and clip.depths[t] is not None:
                        depth = clip.depths[t]
                        pts3d_from_depth, dv = backproject_depth_samples(depth, pts2d_gt, K, E)
                    else:
                        pts3d_from_depth, dv = None, None

                    if pts3d_from_depth is not None and len(pts3d_from_depth) > 0:
                        rec.log("world/pts3d_from_depth", rr.Points3D(pts3d_from_depth, colors=[255, 140, 0], radii=pt3d_radius))

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


def verify_one_dataset(name, root, out_base, clip_len=16, seed=0, use_rerun=False, sequence=None, pt2d_radius=1.5, pt3d_radius=0.01, all_frames=False, start_frame=0):
    from datasets.registry import create_adapter
    from datasets.sampling import DatasetSampler
    import random

    out_dir = Path(out_base) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{name}] root={root}")

    try:
        adapter = create_adapter(name=name, root=root, split='train', strict=False)
    except Exception as e:
        result = {"error": f"adapter init failed: {e}"}
        print(f"  ERROR: {e}")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    print(f"  sequences: {len(adapter.list_sequences())}")
    rng = random.Random(seed)

    if sequence is not None:
        # Manually specified sequence — bypass sampler (its clip_len filter may reject short sequences)
        all_sequences = adapter.list_sequences()
        if sequence not in all_sequences:
            print(f"  ERROR: sequence '{sequence}' not found. Available: {all_sequences[:10]}")
            return {"error": f"sequence not found: {sequence}"}
        seq = sequence
        seq_info = adapter.get_sequence_info(seq)
        total_frames = seq_info['num_frames']
        # Use all frames if all_frames flag is set, otherwise use clip_len from start_frame
        if all_frames:
            frame_indices = list(range(start_frame, total_frames))
            if start_frame > 0:
                print(f"  sequence has {total_frames} frames, starting from frame {start_frame}, using all {len(frame_indices)} remaining frames")
        else:
            end_frame = start_frame + clip_len
            if end_frame > total_frames:
                print(f"  WARNING: requested frames [{start_frame}, {end_frame}) exceeds total {total_frames} frames. "
                      f"Using remaining {total_frames - start_frame} frames from frame {start_frame}.")
                end_frame = total_frames
            frame_indices = list(range(start_frame, end_frame))
        print(f"  sequence has {total_frames} frames, using {len(frame_indices)} frames (start={start_frame})")
    else:
        try:
            sampler = DatasetSampler(adapter, clip_len=clip_len, sampling_mode='stride', min_frames=2)
        except Exception as e:
            result = {"error": f"sampler init failed: {e}"}
            print(f"  ERROR: {e}")
            (out_dir / "result.json").write_text(json.dumps(result, indent=2))
            return result
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
            num_frames = adapter.get_sequence_info(seq)['num_frames']
            frame_indices = list(range(max(0, ref - half), min(num_frames, ref + half)))[:clip_len]

    print(f"  testing sequence: {seq}  frames={frame_indices[:3]}...({len(frame_indices)} total)")

    try:
        clip = adapter.load_clip(seq, frame_indices)
    except Exception as e:
        result = {"error": f"load_clip failed: {e}", "traceback": traceback.format_exc()}
        print(f"  ERROR loading clip: {e}")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    has_tracks = clip.metadata.get("has_tracks", False) and clip.trajs_3d_world is not None

    print(f"  has_tracks={has_tracks}, frames={len(clip.images)}, size={clip.image_size}")
    print(f"  has_depth={clip.depths is not None}, has_normals={clip.normals is not None}")

    metrics = {}
    try:
        if has_tracks:
            scene_dir = Path(root) / seq if name == "scannetpp" else None
            metrics = verify_has_tracks(clip, out_dir, scene_dir=scene_dir)
            me = metrics.get('mean_reproj_error_px')
            mx = metrics.get('max_reproj_error_px')
            print(f"  reproj_error: mean={me:.3f}px  max={mx:.3f}px" if me is not None
                  else f"  reproj_error: N/A (no valid points)")
            d3d = metrics.get('mean_depth3d_error_m')
            if d3d is not None:
                print(f"  depth3d_error: mean={d3d:.4f}m  max={metrics.get('max_depth3d_error_m'):.4f}m")
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
        try:
            scene_dir = Path(root) / seq if name == "scannetpp" else None
            log_clip_to_rerun(clip, name, seq, out_dir / "clip.rrd", pt2d_radius, pt3d_radius, scene_dir=scene_dir)
        except Exception as e:
            print(f"  rerun ERROR: {e}")

    result = {
        "dataset": name,
        "sequence": seq,
        "frame_indices": frame_indices,
        "has_tracks": has_tracks,
        "num_frames": len(clip.images),
        "image_size": list(clip.image_size),
        "metrics": metrics,
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
    parser.add_argument("--no-rerun", action="store_true", help="disable saving .rrd for rerun viewer")
    parser.add_argument("--sequence", default=None, help="manually specify a sequence, e.g. apple/106_12648_23157")
    parser.add_argument("--start-frame", type=int, default=0, help="starting frame index (only with --sequence)")
    parser.add_argument("--all-frames", action="store_true", help="use all frames in the sequence (only with --sequence)")
    parser.add_argument("--pt2d-radius", type=float, default=1.5, help="2D point radius in rerun viewer")
    parser.add_argument("--pt3d-radius", type=float, default=0.01, help="3D point radius in rerun viewer")
    args = parser.parse_args()

    use_rerun = not args.no_rerun

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [(n, r) for n, r, *_ in DATASETS if n == args.dataset]
        if not datasets:
            print(f"Unknown dataset: {args.dataset}. Available: {[n for n,*_ in DATASETS]}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    all_results = {}
    for name, root, *_ in datasets:
        result = verify_one_dataset(name, root, args.out, clip_len=args.clip_len,
                                    seed=args.seed, use_rerun=use_rerun, sequence=args.sequence,
                                    pt2d_radius=args.pt2d_radius, pt3d_radius=args.pt3d_radius,
                                    all_frames=args.all_frames, start_frame=args.start_frame)
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
