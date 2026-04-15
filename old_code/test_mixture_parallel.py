#!/usr/bin/env python3
"""Test and visualize D4RT mixture dataset training with parallel processing."""

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
from tqdm import tqdm
import json
from contextlib import nullcontext
import multiprocessing as mp
from functools import partial

def get_inference_autocast_dtype(device):
    if device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def inference_autocast_context(device):
    dtype = get_inference_autocast_dtype(device)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)

def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from mixture training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = create_d4rt(
        encoder="base",
        decoder_depth=6,
        img_size=256,
        num_frames=48,
        patch_size=(2, 16, 16),
        query_patch_size=9,
        videomae_model="/data1/zbf/pretrained/videomae-base"
    ).to(device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        raise ValueError("Checkpoint missing 'model' or 'model_state_dict' key")

    model.eval()
    epoch = checkpoint.get('epoch', -1)
    print(f"Loaded checkpoint from epoch {epoch}")
    return model

def visualize_tracks(video_rgb, coords_2d_pred, coords_2d_gt, visibility_pred, visibility_gt,
                     t_src, t_tgt, output_path, num_vis=256):
    """Visualize predicted vs ground truth tracks.

    Visualization:
    - GT: Large GREEN squares (easy to see)
    - Pred: Small colored circles with white center
    """
    S, H, W, _ = video_rgb.shape
    num_queries = min(num_vis, coords_2d_pred.shape[0])

    # Create color map for predictions
    cmap = matplotlib.colormaps.get_cmap("hsv")
    colors = (cmap(np.linspace(0, 0.9, num_queries))[:, :3] * 255).astype(np.uint8)

    frames_vis = []
    for t in range(S):
        frame = video_rgb[t].copy()

        # Count visible points
        num_pred_vis = ((t_tgt[:num_queries] == t) & (visibility_pred[:num_queries] > 0.5)).sum()
        num_gt_vis = ((t_tgt[:num_queries] == t) & (visibility_gt[:num_queries] > 0.5)).sum()

        # Draw GT first (GREEN SQUARES - very visible)
        mask_t_gt = (t_tgt[:num_queries] == t) & (visibility_gt[:num_queries] > 0.5)
        for q in np.where(mask_t_gt)[0]:
            x, y = coords_2d_gt[q]
            x_px = int(np.clip(x * (W - 1), 0, W - 1))
            y_px = int(np.clip(y * (H - 1), 0, H - 1))
            # Draw large green square
            cv2.rectangle(frame, (x_px-5, y_px-5), (x_px+5, y_px+5), (0, 255, 0), 2)

        # Draw predictions (colored circles)
        mask_t = (t_tgt[:num_queries] == t) & (visibility_pred[:num_queries] > 0.5)
        for q in np.where(mask_t)[0]:
            x, y = coords_2d_pred[q]
            x_px = int(np.clip(x * (W - 1), 0, W - 1))
            y_px = int(np.clip(y * (H - 1), 0, H - 1))
            color = tuple(int(c) for c in colors[q])
            cv2.circle(frame, (x_px, y_px), 4, color, -1)
            cv2.circle(frame, (x_px, y_px), 2, (255, 255, 255), -1)

        # Add legend and stats
        cv2.putText(frame, f"Frame {t}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"GT: {num_gt_vis} (green squares)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Pred: {num_pred_vis} (colored circles)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frames_vis.append(frame)

    # Save as video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (W, H))
    for frame in frames_vis:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def compute_metrics(pred_2d, gt_2d, pred_3d, gt_3d, pred_vis, gt_vis, mask_2d, mask_3d, mask_vis):
    """Compute evaluation metrics."""
    metrics = {}

    if mask_2d.sum() > 0:
        error_2d = np.linalg.norm((pred_2d - gt_2d) * 256, axis=-1)
        metrics['2d_mean'] = float(error_2d[mask_2d].mean())
        metrics['2d_median'] = float(np.median(error_2d[mask_2d]))

    if mask_3d.sum() > 0:
        error_3d = np.linalg.norm(pred_3d - gt_3d, axis=-1)
        metrics['3d_mean'] = float(error_3d[mask_3d].mean())
        metrics['3d_median'] = float(np.median(error_3d[mask_3d]))

    if mask_vis.sum() > 0:
        vis_pred_binary = (pred_vis > 0.5).astype(float)
        vis_gt_binary = (gt_vis > 0.5).astype(float)
        metrics['vis_acc'] = float((vis_pred_binary[mask_vis] == vis_gt_binary[mask_vis]).mean())

    return metrics

def test_sample(model, batch, device, output_dir, sample_idx, save_video=True):
    """Test on a single batch and save visualization."""
    with torch.no_grad():
        with inference_autocast_context(device):
            outputs = model(
                batch['video'],
                batch['coords'],
                batch['t_src'],
                batch['t_tgt'],
                batch['t_cam']
            )

    # Move to CPU and convert to numpy (handle BFloat16)
    pred_2d = outputs['pos_2d'][0].float().cpu().numpy()
    pred_3d = outputs['pos_3d'][0].float().cpu().numpy()
    pred_vis = outputs['visibility'][0].float().cpu().numpy()

    gt_2d = batch['targets']['pos_2d'][0].float().cpu().numpy()
    gt_3d = batch['targets']['pos_3d'][0].float().cpu().numpy()
    gt_vis = batch['targets']['visibility'][0].float().cpu().numpy()

    mask_2d = batch['targets']['mask_2d'][0].cpu().numpy().astype(bool)
    mask_3d = batch['targets']['mask_3d'][0].cpu().numpy().astype(bool)
    mask_vis = batch['targets']['mask_vis'][0].cpu().numpy().astype(bool)

    t_src = batch['t_src'][0].cpu().numpy()
    t_tgt = batch['t_tgt'][0].cpu().numpy()

    # Compute metrics
    metrics = compute_metrics(pred_2d, gt_2d, pred_3d, gt_3d, pred_vis, gt_vis, mask_2d, mask_3d, mask_vis)

    dataset_name = batch['dataset_names'][0]
    seq_name = batch['sequence_names'][0]

    # Save visualization
    if save_video:
        video = batch['video'][0].float().cpu().numpy()
        if video.dtype == np.float32 or video.dtype == np.float64:
            video = (video * 255).astype(np.uint8)
        video_rgb = video.transpose(0, 2, 3, 1)

        vis_path = output_dir / f"sample_{sample_idx:04d}_{dataset_name}_{seq_name}.mp4"
        visualize_tracks(video_rgb, pred_2d, gt_2d, pred_vis, gt_vis, t_src, t_tgt, vis_path)

    metrics_data = {
        'sample_idx': sample_idx,
        'dataset': dataset_name,
        'sequence': seq_name,
        **metrics
    }

    return metrics_data

def worker_process(gpu_id, indices, config_path, checkpoint_path, output_dir, save_videos):
    """Worker process for parallel testing."""
    device = torch.device(f"cuda:{gpu_id}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create dataset
    dataset = create_training_dataset(config, split='val')
    dataset = Subset(dataset, indices)

    # DataLoader - use num_workers=0 to avoid nested multiprocessing
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,  # Avoid nested multiprocessing
        collate_fn=d4rt_collate_fn,
        shuffle=False,
        pin_memory=False,
    )

    # Load model
    model = load_model(Path(checkpoint_path), device)

    # Test
    results = []
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch['targets'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch['targets'].items()}

        if batch['video'].dtype == torch.uint8:
            batch['video'] = batch['video'].float() / 255.0

        metrics = test_sample(model, batch, device, output_dir, indices[batch_idx], save_videos)
        results.append(metrics)
        print(f"[GPU {gpu_id}] Sample {indices[batch_idx]}: {metrics['dataset']}/{metrics['sequence']} - "
              f"2D: {metrics.get('2d_mean', 0):.2f}px, 3D: {metrics.get('3d_mean', 0):.4f}m")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mixture_full_11datasets.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="test_outputs")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--no-videos", action="store_true", help="Skip video generation for speed")
    args = parser.parse_args()

    # Set multiprocessing start method to spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse GPU IDs - these are logical IDs (0, 1, 2...) after CUDA_VISIBLE_DEVICES mapping
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)

    # After CUDA_VISIBLE_DEVICES is set, use logical IDs 0, 1, 2...
    logical_gpu_ids = list(range(num_gpus))

    print(f"Using {num_gpus} GPUs (logical IDs: {logical_gpu_ids})")
    print(f"Testing {args.num_samples} samples")

    # Split work across GPUs
    indices = list(range(args.num_samples))
    chunks = [indices[i::num_gpus] for i in range(num_gpus)]

    # Run parallel workers
    if num_gpus > 1:
        with mp.Pool(num_gpus) as pool:
            worker_fn = partial(
                worker_process,
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                output_dir=output_dir,
                save_videos=not args.no_videos
            )
            results_list = pool.starmap(worker_fn, [(logical_gpu_ids[i], chunks[i]) for i in range(num_gpus)])

        # Flatten results
        all_metrics = [m for results in results_list for m in results]
    else:
        # Single GPU
        all_metrics = worker_process(
            0, indices, args.config, args.checkpoint, output_dir, not args.no_videos
        )

    # Save metrics
    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, 'w') as f:
        for m in all_metrics:
            f.write(json.dumps(m) + '\n')

    # Compute aggregate
    if all_metrics:
        avg_metrics = {}
        for key in ['2d_mean', '2d_median', '3d_mean', '3d_median', 'vis_acc']:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[key] = np.mean(values)

        print("\n=== Aggregate Metrics ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(avg_metrics, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
