#!/usr/bin/env python3
"""Visualize D4RT mixture training: 2D tracks + dense 3D point cloud."""

import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn
from models import create_d4rt
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from contextlib import nullcontext
from tqdm import tqdm


def inference_autocast_context(device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_d4rt(
        encoder="base", decoder_depth=6, img_size=256, num_frames=48,
        patch_size=(2, 16, 16), query_patch_size=9,
        videomae_model="/data1/zbf/pretrained/videomae-base"
    ).to(device)
    state = checkpoint.get('model', checkpoint.get('model_state_dict'))
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', -1)}")
    return model


def make_dense_queries(S, H, W, grid_size=128, device='cpu'):
    """Build dense grid queries: all points from frame 0, target all frames."""
    ys = np.linspace(0.01, 0.99, grid_size)
    xs = np.linspace(0.01, 0.99, grid_size)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # [N, 2]
    N = len(coords)

    # Each query: source=frame 0, target=each frame (repeat N queries per frame)
    # To keep memory manageable: query all N points with t_tgt spread across frames
    # Use t_src=0 for all, t_tgt cycles through frames
    t_src = np.zeros(N, dtype=np.int64)
    t_tgt = np.zeros(N, dtype=np.int64)  # will run per-frame
    t_cam = np.zeros(N, dtype=np.int64)

    coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)       # [1, N, 2]
    t_src_t = torch.from_numpy(t_src).unsqueeze(0).to(device)         # [1, N]
    t_tgt_t = torch.from_numpy(t_tgt).unsqueeze(0).to(device)         # [1, N]
    t_cam_t = torch.from_numpy(t_cam).unsqueeze(0).to(device)         # [1, N]
    return coords_t, t_src_t, t_tgt_t, t_cam_t, coords


def visualize_2d_tracks(video_rgb, pred_2d, gt_2d, pred_vis, gt_vis, t_tgt, output_path, num_vis=512):
    """Side-by-side: left=GT, right=Pred."""
    S, H, W, _ = video_rgb.shape
    N = min(num_vis, pred_2d.shape[0])
    cmap = matplotlib.colormaps.get_cmap("hsv")
    colors = (cmap(np.linspace(0, 0.9, N))[:, :3] * 255).astype(np.uint8)

    frames = []
    for t in range(S):
        left = video_rgb[t].copy()
        right = video_rgb[t].copy()

        mask_gt = (t_tgt[:N] == t) & (gt_vis[:N] > 0.5)
        mask_pred = (t_tgt[:N] == t) & (pred_vis[:N] > 0.5)

        for q in np.where(mask_gt)[0]:
            x, y = gt_2d[q]
            px, py = int(np.clip(x*(W-1), 0, W-1)), int(np.clip(y*(H-1), 0, H-1))
            cv2.rectangle(left, (px-4, py-4), (px+4, py+4), (0, 255, 0), 2)

        for q in np.where(mask_pred)[0]:
            x, y = pred_2d[q]
            px, py = int(np.clip(x*(W-1), 0, W-1)), int(np.clip(y*(H-1), 0, H-1))
            c = tuple(int(v) for v in colors[q])
            cv2.circle(right, (px, py), 3, c, -1)

        cv2.putText(left,  f"GT  t={t}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(right, f"Pred t={t}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        frames.append(np.concatenate([left, right], axis=1))

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (W*2, H))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved 2D tracks: {output_path}")


def visualize_3d_pointcloud(model, video_tensor, device, output_path, grid_size=100, S=48):
    """Dense 3D point cloud video: run inference per-frame, render with matplotlib."""
    H, W = 256, 256
    coords_t, t_src_t, _, t_cam_t, coords_np = make_dense_queries(S, H, W, grid_size, device)
    N = coords_np.shape[0]

    # Get source frame colors from video
    video_np = video_tensor[0].float().cpu().numpy()  # [S, 3, H, W]
    src_frame = (video_np[0].transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3]

    # Sample colors at query positions
    px = np.clip((coords_np[:, 0] * (W-1)).astype(int), 0, W-1)
    py = np.clip((coords_np[:, 1] * (H-1)).astype(int), 0, H-1)
    point_colors = src_frame[py, px] / 255.0  # [N, 3]

    # Run inference per target frame
    all_pos3d = []  # [S, N, 3]
    all_vis = []    # [S, N]

    chunk = 4096  # process in chunks to avoid OOM
    with torch.no_grad():
        with inference_autocast_context(device):
            encoder_features = model.encode(video_tensor)

    for t in tqdm(range(S), desc="Dense 3D inference"):
        t_tgt_t = torch.full_like(t_src_t, t)
        pos3d_frame = []
        vis_frame = []
        for i in range(0, N, chunk):
            c = coords_t[:, i:i+chunk]
            ts = t_src_t[:, i:i+chunk]
            tt = t_tgt_t[:, i:i+chunk]
            tc = t_cam_t[:, i:i+chunk]
            with torch.no_grad():
                with inference_autocast_context(device):
                    out = model.decode(encoder_features,
                                       video_tensor.permute(0,2,1,3,4) if video_tensor.dim()==5 and video_tensor.shape[1]==3 else video_tensor,
                                       c, ts, tt, tc)
            pos3d_frame.append(out['pos_3d'][0].float().cpu().numpy())
            vis_frame.append(torch.sigmoid(out['visibility'][0]).float().cpu().numpy().squeeze(-1) if out['visibility'].shape[-1]==1 else out['visibility'][0].float().cpu().numpy())
        all_pos3d.append(np.concatenate(pos3d_frame, axis=0))
        all_vis.append(np.concatenate(vis_frame, axis=0))

    # Render 3D point cloud video
    frames = []
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Compute global bounds from all frames
    pts_all = np.concatenate(all_pos3d, axis=0)
    vis_all = np.concatenate(all_vis, axis=0) > 0.5
    if vis_all.sum() > 0:
        pts_valid = pts_all[vis_all]
        x_lim = (np.percentile(pts_valid[:,0], 2), np.percentile(pts_valid[:,0], 98))
        y_lim = (np.percentile(pts_valid[:,1], 2), np.percentile(pts_valid[:,1], 98))
        z_lim = (np.percentile(pts_valid[:,2], 2), np.percentile(pts_valid[:,2], 98))
    else:
        x_lim = y_lim = z_lim = (-1, 1)

    for t in tqdm(range(S), desc="Rendering 3D frames"):
        ax.cla()
        pts = all_pos3d[t]
        vis = all_vis[t] > 0.5
        if vis.sum() > 100:
            ax.scatter(pts[vis, 0], pts[vis, 2], -pts[vis, 1],
                      c=point_colors[vis], s=1.5, alpha=0.7, linewidths=0)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*z_lim)
        ax.set_zlim(y_lim[0], y_lim[1])
        ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
        ax.set_title(f'3D Point Cloud  t={t}  vis={vis.sum()}')
        # Rotate view slowly
        ax.view_init(elev=20, azim=t * 360 / S)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)

    plt.close(fig)

    fh, fw = frames[0].shape[:2]
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (fw, fh))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved 3D point cloud: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mixture_full_11datasets.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="vis_outputs")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-size", type=int, default=100, help="Dense grid size (grid_size^2 points)")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = create_training_dataset(config, split=args.split)
    dataset = Subset(dataset, list(range(args.num_samples)))

    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=d4rt_collate_fn, shuffle=False)

    model = load_model(args.checkpoint, device)

    for idx, batch in enumerate(tqdm(loader, desc="Samples")):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch['targets'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch['targets'].items()}
        if batch['video'].dtype == torch.uint8:
            batch['video'] = batch['video'].float() / 255.0

        ds_name = batch['dataset_names'][0]
        seq_name = batch['sequence_names'][0].replace('/', '_')
        prefix = f"{idx:03d}_{ds_name}_{seq_name}"

        # --- 2D track visualization ---
        with torch.no_grad():
            with inference_autocast_context(device):
                outputs = model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'])

        pred_2d = outputs['pos_2d'][0].float().cpu().numpy()
        pred_vis = torch.sigmoid(outputs['visibility'][0]).float().cpu().numpy()
        if pred_vis.ndim == 2: pred_vis = pred_vis.squeeze(-1)
        gt_2d = batch['targets']['pos_2d'][0].float().cpu().numpy()
        gt_vis = batch['targets']['visibility'][0].float().cpu().numpy()
        t_tgt_np = batch['t_tgt'][0].cpu().numpy()

        video_np = batch['video'][0].float().cpu().numpy()
        video_rgb = (video_np.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

        visualize_2d_tracks(video_rgb, pred_2d, gt_2d, pred_vis, gt_vis, t_tgt_np,
                            output_dir / f"{prefix}_2d.mp4")

        # --- Dense 3D point cloud ---
        visualize_3d_pointcloud(model, batch['video'], device,
                                output_dir / f"{prefix}_3d.mp4",
                                grid_size=args.grid_size)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

