#!/usr/bin/env python3
"""
Verify the coordinate transform logic in mvssynth_dataset.py
(datasets/adapters/mvssynth_dataset.py) independently of its framework.

Replicates exactly what _get_clip() does for extrinsics and trajs_3d_world,
then checks:
  1. depth round-trip reprojection error  (single-frame, always passes if E is self-consistent)
  2. trajs_3d_world -> trajs_2d reprojection error  (cross-validates 3D points vs GT 2D)
  3. saves a clip.rrd for visual inspection via rerun

Usage:
    python verify_mvssynth_dataset.py
    python verify_mvssynth_dataset.py --sequence 0042 --frames 0 2 4 6 8
    python verify_mvssynth_dataset.py --out /data1/zbf/my_dfrt/vis_gyz2/mvssynth_dataset
"""

import sys, os, json, argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/data1/zbf/my_dfrt')
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

DATA_ROOT   = Path("/data2/d4rt/datasets/MVS-Synth/GTAV_1080")
DEFAULT_OUT = Path("/data1/zbf/my_dfrt/vis_gyz2/mvssynth_dataset")


# ---------------------------------------------------------------------------
# Helpers copied from verify_datasets_ccw.py
# ---------------------------------------------------------------------------

def project_world_to_image(pts_world, K, E):
    """pts_world: [N,3], K: [3,3], E: [4,4] w2c -> returns [N,2] pixel coords"""
    N = pts_world.shape[0]
    h = np.concatenate([pts_world, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (E @ h.T).T[:, :3]
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


def _flow_to_rgb(flow):
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# Replicate mvssynth_dataset.py _get_clip() transform logic exactly
# ---------------------------------------------------------------------------

def load_clip_mvssynth_dataset(seq, frame_idxs, data_root=DATA_ROOT):
    """
    Replicates the coordinate transform logic from mvssynth_dataset.py _get_clip().
    Returns a dict with keys: images, depths, intrinsics, extrinsics_w2c,
    trajs_2d, trajs_3d_world, valids, visibs.

    extrinsics_w2c: [T,4,4] w2c (same convention as mvssynth_dataset.py camera_pose's inverse)
    """
    from datasets.adapters.base import load_precomputed_fast

    seq_dir = data_root / seq

    # ---- Load precomputed tracks ----
    pc = load_precomputed_fast(data_root / seq / "precomputed.npz", list(frame_idxs))
    trajs_2d = trajs_3d_world = valids = visibs = None
    has_tracks = False
    if pc is not None and 'trajs_2d' in pc:
        trajs_2d = pc['trajs_2d'].astype(np.float32)
        trajs_3d_world = pc.get('trajs_3d_world')
        if trajs_3d_world is not None:
            trajs_3d_world = trajs_3d_world.astype(np.float32)
            # Unit and coordinate transform applied after c0 is known (see below)
        valids  = pc.get('valids')
        visibs  = pc.get('visibs')
        has_tracks = True

    images, depths, intrinsics, extrinsics_w2c = [], [], [], []
    c0 = None

    for idx in frame_idxs:
        img_path   = seq_dir / "images" / f"{idx:04d}.png"
        depth_path = seq_dir / "depths" / f"{idx:04d}.exr"
        pose_path  = seq_dir / "poses"  / f"{idx:04d}.json"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        dep = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        dep[~np.isfinite(dep)] = 0.0
        dep /= 100.0                     # cm -> m
        depths.append(dep)

        with open(pose_path) as f:
            pose = json.load(f)

        K = np.array([
            [pose["f_x"], 0.0,         pose["c_x"]],
            [0.0,         pose["f_y"], pose["c_y"]],
            [0.0,         0.0,         1.0        ],
        ], dtype=np.float32)
        intrinsics.append(K)

        w2c = np.array(pose["extrinsic"], dtype=np.float32)  # [4,4]
        w2c[:3, 3] /= 100.0              # cm -> m

        # GTA V left-handed -> right-handed: flip world X axis
        w2c[:, 0] *= -1

        # World centering: shift origin to first selected camera position
        if c0 is None:
            c0 = np.linalg.inv(w2c.astype(np.float64))[:3, 3].astype(np.float32)
        w2c[:3, 3] += w2c[:3, :3] @ c0

        extrinsics_w2c.append(w2c)

    # Apply coordinate transform to trajs_3d_world to match the extrinsic convention:
    #   1. /= 100.0       : cm -> m  (extrinsic translation was also divided by 100)
    #   2. [..., 0] *= -1 : flip X   (matches w2c[:, 0] *= -1 applied to extrinsics)
    #   3. -= c0          : world centering (matches the w2c[:3,3] += R @ c0 shift)
    if trajs_3d_world is not None and c0 is not None:
        trajs_3d_world /= 100.0
        trajs_3d_world[..., 0] *= -1
        trajs_3d_world -= c0

    return dict(
        images=images,
        depths=depths,
        intrinsics=np.stack(intrinsics, axis=0),     # [T,3,3]
        extrinsics_w2c=np.stack(extrinsics_w2c, axis=0),  # [T,4,4] w2c
        trajs_2d=trajs_2d,
        trajs_3d_world=trajs_3d_world,
        valids=valids,
        visibs=visibs,
        has_tracks=has_tracks,
        c0=c0,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_depth_roundtrip(clip, n_pts=500, seed=42):
    """Depth round-trip: pixel -> 3D -> pixel. Checks single-frame self-consistency."""
    rng = np.random.default_rng(seed)
    errors = []
    for t in range(len(clip['images'])):
        dep = clip['depths'][t]
        K   = clip['intrinsics'][t]
        E   = clip['extrinsics_w2c'][t]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        valid_mask = np.isfinite(dep) & (dep > 1e-3) & (dep < 1500)
        ys, xs = np.where(valid_mask)
        if len(ys) == 0:
            errors.append(np.nan)
            continue
        if len(ys) > n_pts:
            sel = rng.choice(len(ys), n_pts, replace=False)
            ys, xs = ys[sel], xs[sel]

        zs = dep[ys, xs].astype(np.float64)
        xc = (xs - cx) / fx * zs
        yc = (ys - cy) / fy * zs
        pts_cam = np.stack([xc, yc, zs, np.ones_like(zs)], axis=1)

        E64 = E.astype(np.float64)
        pts_world = (np.linalg.inv(E64) @ pts_cam.T).T
        pts_cam2  = (E64 @ pts_world.T).T[:, :3]
        uv = K.astype(np.float64) @ pts_cam2.T
        u2 = uv[0] / uv[2]; v2 = uv[1] / uv[2]
        err = np.hypot(u2 - xs, v2 - ys)
        errors.append(float(err.mean()))

    return errors


def verify_trajs3d_reproj(clip, n_pts=200, seed=42):
    """trajs_3d_world -> trajs_2d reprojection error."""
    if not clip['has_tracks'] or clip['trajs_3d_world'] is None:
        return None, []

    rng = np.random.default_rng(seed)
    T = len(clip['images'])
    t3d = clip['trajs_3d_world']  # [T,N,3]
    t2d = clip['trajs_2d']        # [T,N,2]
    valids  = clip['valids']
    visibs  = clip['visibs']

    # Pick point subset from first frame that has finite 2D GT
    finite_2d = np.isfinite(t2d[0]).all(-1)
    mask0 = finite_2d
    if valids  is not None: mask0 = mask0 & valids[0].astype(bool)
    if visibs  is not None: mask0 = mask0 & visibs[0].astype(bool)
    idx = np.where(mask0)[0]
    if len(idx) == 0:
        return None, []
    if len(idx) > n_pts:
        idx = rng.choice(idx, n_pts, replace=False)

    errors, frames_vis = [], []
    for t in range(T):
        K = clip['intrinsics'][t]
        E = clip['extrinsics_w2c'][t]
        pts3d   = t3d[t, idx]       # [M,3]
        pts2d_gt = t2d[t, idx]      # [M,2]

        finite_gt = np.isfinite(pts2d_gt).all(-1)
        mask_t = finite_gt
        if valids  is not None: mask_t = mask_t & valids[t,  idx].astype(bool)
        if visibs  is not None: mask_t = mask_t & visibs[t,  idx].astype(bool)

        pts2d_reproj, reproj_valid = project_world_to_image(
            pts3d.astype(np.float32), K, E)

        combined = mask_t & reproj_valid
        if combined.any():
            err = np.linalg.norm(pts2d_reproj[combined] - pts2d_gt[combined], axis=1)
            errors.append(float(err.mean()))
        else:
            errors.append(np.nan)

        # Visualize
        img = clip['images'][t].copy()
        H, W = img.shape[:2]
        for i in range(len(idx)):
            if not (mask_t[i] and reproj_valid[i]):
                continue
            gx = int(np.clip(pts2d_gt[i, 0], 0, W-1))
            gy = int(np.clip(pts2d_gt[i, 1], 0, H-1))
            rx = int(np.clip(pts2d_reproj[i, 0], 0, W-1))
            ry = int(np.clip(pts2d_reproj[i, 1], 0, H-1))
            cv2.circle(img, (gx, gy), 3, (0, 220, 0), -1)
            cv2.circle(img, (rx, ry), 2, (220, 0, 0), -1)
            cv2.line(img, (gx, gy), (rx, ry), (255, 200, 0), 1)
        err_val = errors[-1]
        label = f"t={t} err={err_val:.3f}px" if not np.isnan(err_val) else f"t={t} no valid pts"
        cv2.putText(img, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
        frames_vis.append(img)

    return errors, frames_vis


def save_rrd(clip, seq, out_path: Path):
    """Save rerun .rrd for visual inspection."""
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("  rerun not available, skipping .rrd")
        return

    T = len(clip['images'])
    H, W = clip['images'][0].shape[:2]
    has_depth    = True
    has_trajs3d  = clip['has_tracks'] and clip['trajs_3d_world'] is not None

    top_views = [rrb.Spatial2DView(name="RGB & Reprojection",
                                   origin="world/camera/image",
                                   contents=["+ $origin/**"],
                                   background=[30,30,30]),
                 rrb.Spatial2DView(name="Depth",
                                   origin="world/camera/image",
                                   contents=["+ world/camera/image/depth"],
                                   background=[30,30,30])]
    view_3d = rrb.Spatial3DView(name="3D Scene", origin="/",
                                contents=["+ world/**"], background=[20,20,20])
    info_view = rrb.TextDocumentView(name="Info", origin="info")
    layout = rrb.Vertical(
        rrb.Horizontal(*top_views),
        rrb.Horizontal(view_3d, info_view, column_shares=[4,1]),
        row_shares=[1,2],
    )
    blueprint = rrb.Blueprint(layout, collapse_panels=True)

    median_depths = [float(np.median(d[d > 0])) for d in clip['depths'] if np.any(d > 0)]
    frustum_scale = float(np.median(median_depths)) * 0.1 if median_depths else 0.3

    with rr.RecordingStream(application_id=f"mvssynth_dataset/{seq}", make_default=False) as rec:
        rec.save(str(out_path), default_blueprint=blueprint)

        info_lines = [
            f"# mvssynth_dataset / {seq}",
            f"- **Frames**: {T}",
            f"- **Resolution**: {W} x {H}",
            f"- **has_tracks**: {has_trajs3d}",
            "\n### Convention",
            "- Extrinsic: w2c, flip-X (non-standard right-handed)",
            "- Green: GT trajs_2d, Red: reprojected trajs_3d_world",
        ]
        rec.log("info", rr.TextDocument("\n".join(info_lines), media_type=rr.MediaType.MARKDOWN))

        # Static union of all 3D points
        if has_trajs3d:
            t3d = clip['trajs_3d_world']
            valids = clip['valids']; visibs = clip['visibs']
            all_pts = []
            for _t in range(T):
                mask = np.ones(t3d.shape[1], dtype=bool)
                if valids is not None: mask &= valids[_t].astype(bool)
                if visibs is not None: mask &= visibs[_t].astype(bool)
                if mask.any():
                    all_pts.append(t3d[_t][mask])
            if all_pts:
                union = np.unique(np.concatenate(all_pts, axis=0), axis=0)
                rec.log("world/pts3d_all", rr.Points3D(union, colors=[0,200,255], radii=0.01), static=True)

        for t in range(T):
            rec.set_time("frame", sequence=t)
            K = clip['intrinsics'][t]
            E = clip['extrinsics_w2c'][t]
            E_c2w = np.linalg.inv(E.astype(np.float64))
            quat = Rotation.from_matrix(E_c2w[:3,:3]).as_quat()
            rec.log("world/camera", rr.Transform3D(
                translation=E_c2w[:3,3],
                quaternion=rr.Quaternion(xyzw=quat)))
            rec.log("world/camera/image", rr.Pinhole(
                image_from_camera=K, width=W, height=H,
                image_plane_distance=frustum_scale))
            rec.log("world/camera/image", rr.Image(clip['images'][t]))
            rec.log("world/camera/image/depth", rr.DepthImage(
                clip['depths'][t], meter=1.0, colormap="Turbo", point_fill_ratio=1.0))

            if has_trajs3d:
                t3d = clip['trajs_3d_world']
                t2d = clip['trajs_2d']
                valids = clip['valids']; visibs = clip['visibs']
                mask = np.isfinite(t2d[t]).all(-1)
                if valids  is not None: mask &= valids[t].astype(bool)
                if visibs  is not None: mask &= visibs[t].astype(bool)
                if mask.any():
                    pts2d_gt = t2d[t][mask].astype(np.float32)
                    pts3d_v  = t3d[t][mask].astype(np.float32)
                    uv_reproj, rv = project_world_to_image(pts3d_v, K.astype(np.float32), E.astype(np.float32))
                    rec.log("world/camera/image/gt_pts",
                            rr.Points2D(pts2d_gt[rv], colors=[0,255,0], radii=1.5))
                    rec.log("world/camera/image/reproj_pts",
                            rr.Points2D(uv_reproj[rv], colors=[255,0,0], radii=1.0))
                    rec.log("world/pts3d_cur",
                            rr.Points3D(pts3d_v, colors=[255,220,0], radii=0.012))

    print(f"  -> rerun saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=None, help="sequence id, e.g. 0042 (default: random)")
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="frame indices (default: 0 2 4 6 8 10 12 14)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--no-rerun", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick sequence
    if args.sequence is not None:
        seq = args.sequence
    else:
        import random
        num_images_path = DATA_ROOT / "num_images.json"
        with open(num_images_path) as f:
            num_images_list = json.load(f)
        rng = random.Random(0)
        seqs = [f"{i:04d}" for i in range(len(num_images_list))]
        seq = rng.choice(seqs)

    num_images_path = DATA_ROOT / "num_images.json"
    with open(num_images_path) as f:
        num_images_list = json.load(f)
    T_total = num_images_list[int(seq)]

    frame_idxs = args.frames if args.frames is not None else list(range(0, min(16, T_total), 2))
    frame_idxs = [i for i in frame_idxs if i < T_total]

    print(f"\n{'='*60}")
    print(f"[mvssynth_dataset] seq={seq}  frames={frame_idxs}")

    clip = load_clip_mvssynth_dataset(seq, frame_idxs)
    T = len(clip['images'])
    H, W = clip['images'][0].shape[:2]
    print(f"  loaded {T} frames  size=({H},{W})")
    print(f"  has_tracks={clip['has_tracks']}")

    # --- Test 1: depth round-trip ---
    rt_errors = verify_depth_roundtrip(clip)
    valid_rt = [e for e in rt_errors if not np.isnan(e)]
    print(f"\n  [Test 1] Depth round-trip (single-frame self-consistency):")
    print(f"    mean={np.mean(valid_rt):.6f}px  max={np.max(valid_rt):.6f}px")
    print(f"    per-frame: {[f'{e:.6f}' for e in rt_errors]}")

    # --- Test 2: trajs_3d_world -> trajs_2d ---
    reproj_errors, frames_vis = verify_trajs3d_reproj(clip)
    if reproj_errors is not None:
        valid_rp = [e for e in reproj_errors if not np.isnan(e)]
        print(f"\n  [Test 2] trajs_3d_world -> trajs_2d reprojection:")
        if valid_rp:
            print(f"    mean={np.mean(valid_rp):.4f}px  max={np.max(valid_rp):.4f}px")
            print(f"    per-frame: {[f'{e:.4f}' for e in reproj_errors]}")
            # Save GIF
            if frames_vis:
                gif_path = out_dir / "reproj_check.gif"
                save_gif(frames_vis, str(gif_path))
                print(f"    -> saved {gif_path}")
        else:
            print("    all points behind camera (z < 0) -> FAIL")
    else:
        print("\n  [Test 2] trajs_3d_world: no tracks available, skipped")

    # --- Rerun ---
    if not args.no_rerun:
        save_rrd(clip, seq, out_dir / "clip.rrd")

    print(f"\n  -> results in {out_dir}/")


if __name__ == "__main__":
    main()
