#!/usr/bin/env python3
"""Visualize GT data: 2D tracks (GIF+MP4) + dense 3D point cloud (GIF+MP4)."""

import argparse
import random as _rnd
import yaml
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def save_gif(frames_rgb, path, fps=10):
    imgs = [Image.fromarray(f) for f in frames_rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000/fps), loop=0)


def vis_2d_tracks(video_rgb, trajs_2d, valids, visibs, out_path, num_tracks=300):
    """Full trajectory with trail. trajs_2d: [T, N, 2] pixel coords."""
    S, H, W, _ = video_rgb.shape
    T, N, _ = trajs_2d.shape

    valid_first = valids[0] & visibs[0]
    idx = np.where(valid_first)[0]
    if len(idx) > num_tracks:
        np.random.seed(42)
        idx = np.random.choice(idx, num_tracks, replace=False)

    cmap = matplotlib.colormaps.get_cmap("hsv")
    colors = (cmap(np.linspace(0, 0.9, len(idx)))[:, :3] * 255).astype(np.uint8)
    trail_len = 12

    frames = []
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (W, H))
    for t in range(min(S, T)):
        frame = video_rgb[t].copy()
        for ci, q in enumerate(idx):
            c = tuple(int(v) for v in colors[ci])
            for dt in range(1, trail_len + 1):
                tp, tc_ = t - dt, t - dt + 1
                if tp < 0 or not (valids[tp, q] and valids[tc_, q]):
                    continue
                x0 = int(np.clip(trajs_2d[tp, q, 0], 0, W-1))
                y0 = int(np.clip(trajs_2d[tp, q, 1], 0, H-1))
                x1 = int(np.clip(trajs_2d[tc_, q, 0], 0, W-1))
                y1 = int(np.clip(trajs_2d[tc_, q, 1], 0, H-1))
                alpha = 1.0 - dt / (trail_len + 1)
                cv2.line(frame, (x0, y0), (x1, y1),
                         tuple(int(v * alpha) for v in colors[ci]), 1)
            if valids[t, q]:
                x = int(np.clip(trajs_2d[t, q, 0], 0, W-1))
                y = int(np.clip(trajs_2d[t, q, 1], 0, H-1))
                cv2.circle(frame, (x, y), 3, c if visibs[t, q] else (80, 80, 80), -1)
        cv2.putText(frame, f"t={t}  tracks={len(idx)}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(frame)
    writer.release()
    save_gif(frames, str(out_path).replace('.mp4', '.gif'))


def vis_3d_from_pos3d(video_rgb, gt_3d, coords, t_src, t_tgt, mask_3d, out_path):
    """Dense 3D point cloud, colored by source RGB, rotating view."""
    S, H, W, _ = video_rgb.shape
    valid_all = mask_3d
    if valid_all.sum() < 10:
        print(f"  Warning: only {valid_all.sum()} valid 3D points, skipping 3D vis")
        return

    pts_all = gt_3d[valid_all]
    xlim = (np.percentile(pts_all[:, 0], 1), np.percentile(pts_all[:, 0], 99))
    ylim = (np.percentile(pts_all[:, 1], 1), np.percentile(pts_all[:, 1], 99))
    zlim = (np.percentile(pts_all[:, 2], 1), np.percentile(pts_all[:, 2], 99))

    t_src_v = t_src[valid_all]
    coords_v = coords[valid_all]
    px = np.clip((coords_v[:, 0] * (W-1)).astype(int), 0, W-1)
    py = np.clip((coords_v[:, 1] * (H-1)).astype(int), 0, H-1)
    colors_all = np.array([video_rgb[t_src_v[i], py[i], px[i]] / 255.0
                            for i in range(len(pts_all))])

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    frames = []

    for t in tqdm(range(S), desc="  3D frames", leave=False):
        ax.cla()
        ax.scatter(pts_all[:, 0], pts_all[:, 2], -pts_all[:, 1],
                   c=colors_all, s=0.5, alpha=0.6, linewidths=0)
        cur = mask_3d & (t_tgt == t)
        if cur.sum() > 0:
            p = gt_3d[cur]
            ax.scatter(p[:, 0], p[:, 2], -p[:, 1], c='white', s=4, alpha=1.0, linewidths=0)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_zlim(*ylim)
        ax.set_title(f'GT 3D  t={t}  n={valid_all.sum()}')
        ax.view_init(elev=20, azim=t * 360 / S)
        ax.set_axis_off()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy())

    plt.close(fig)
    fh, fw = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (fw, fh))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    save_gif(frames, str(out_path).replace('.mp4', '.gif'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mixture_full_11datasets.yaml")
    parser.add_argument("--output-dir", default="vis_gt")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=65536)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = create_training_dataset(config, split=args.split)
    dataset.query_builder.num_queries = args.num_queries
    dataset.set_epoch(args.epoch)
    dataset = Subset(dataset, list(range(args.num_samples)))
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=d4rt_collate_fn, shuffle=False)
    raw_ds = dataset.dataset

    for idx, batch in enumerate(tqdm(loader, desc="Samples")):
        ds_name = batch['dataset_names'][0]
        seq_name = batch['sequence_names'][0].replace('/', '_')
        prefix = f"{idx:03d}_{ds_name}_{seq_name}"
        print(f"\n[{idx}] {ds_name}/{seq_name}")

        video = batch['video'][0].float().numpy()
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)
        video_rgb = video.transpose(0, 2, 3, 1)

        gt_3d   = batch['targets']['pos_3d'][0].numpy()
        mask_3d = batch['targets']['mask_3d'][0].numpy().astype(bool)
        mask_2d = batch['targets']['mask_2d'][0].numpy().astype(bool)
        t_src   = batch['t_src'][0].numpy()
        t_tgt   = batch['t_tgt'][0].numpy()
        coords_np = batch['coords'][0].numpy()

        print(f"  2D valid={mask_2d.sum()}, 3D valid={mask_3d.sum()}")

        # Load raw clip and apply same transform for aligned trajs_2d
        rng = _rnd.Random(raw_ds.seed + idx)
        ds_idx, seq_name_raw, frame_indices = raw_ds.mixture_sampler.sample(rng)
        adapter = raw_ds.adapters[ds_idx]
        clip = adapter.load_clip(seq_name_raw, frame_indices)
        result = raw_ds.transform(clip, rng=rng)
        if result.trajs_2d is not None:
            H_, W_ = video_rgb.shape[1], video_rgb.shape[2]
            cw, ch = result.crop.crop_w, result.crop.crop_h
            trajs = result.trajs_2d.copy()
            trajs[..., 0] = trajs[..., 0] * W_ / cw
            trajs[..., 1] = trajs[..., 1] * H_ / ch
            vis_2d_tracks(video_rgb, trajs, result.valids, result.visibs,
                          output_dir / f"{prefix}_2d.mp4")
            print(f"  2D tracks={trajs.shape[1]}")
        else:
            print(f"  No 2D tracks for this dataset")

        vis_3d_from_pos3d(video_rgb, gt_3d, coords_np, t_src, t_tgt, mask_3d,
                          output_dir / f"{prefix}_3d.mp4")
        print(f"  Saved: {prefix}_2d/3d .mp4 + .gif")

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
