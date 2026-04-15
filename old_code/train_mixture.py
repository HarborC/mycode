#!/usr/bin/env python3
"""Mixed dataset training for D4RT."""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn
from models import create_d4rt
from losses import D4RTLoss
import json
import time
import os
import contextlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Dataset config YAML")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=2500)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--num-queries", type=int, default=2048)
    parser.add_argument("--loss-w-3d", type=float, default=1.0)
    parser.add_argument("--loss-w-2d", type=float, default=0.1)
    parser.add_argument("--loss-w-vis", type=float, default=0.1)
    parser.add_argument("--loss-w-disp", type=float, default=0.1)
    parser.add_argument("--loss-w-conf", type=float, default=0.2)
    parser.add_argument("--loss-w-normal", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="outputs/mixture")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode with only 10 samples")
    parser.add_argument("--save-interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--val-interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--val-samples", type=int, default=200, help="Number of val samples per validation run")
    parser.add_argument("--keep-checkpoints", type=int, default=10, help="Keep last N checkpoints (except milestone)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    args = parser.parse_args()

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create dataset
    train_dataset = create_training_dataset(config, split='train')
    val_dataset = create_training_dataset(config, split='val')
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset, range(args.val_samples))
    if local_rank == 0:
        print(f"Dataset: {train_dataset.get_dataset_names()}")
        print(f"Train Length: {len(train_dataset)}, Val Length: {len(val_dataset)}")

    # Quick validation mode: use only first 10 samples
    if args.quick_test:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(10, len(train_dataset))))
        if local_rank == 0:
            print(f"Quick test mode: using {len(train_dataset)} samples")

    # DataLoader with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=d4rt_collate_fn,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=max(2, args.num_workers // 2),
        collate_fn=d4rt_collate_fn,
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    # Setup device and model
    device = torch.device(f"cuda:{local_rank}")
    model = create_d4rt(encoder="base", decoder_depth=6, img_size=args.resolution,
                        num_frames=args.num_frames, patch_size=(2, 16, 16),
                        query_patch_size=9, videomae_model="/data1/zbf/pretrained/videomae-base").to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # Load pretrained weights
    if args.pretrain:
        if local_rank == 0:
            print(f"Loading pretrained weights from {args.pretrain}")
        checkpoint = torch.load(args.pretrain, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = D4RTLoss(lambda_3d=args.loss_w_3d, lambda_2d=args.loss_w_2d,
                       lambda_vis=args.loss_w_vis, lambda_disp=args.loss_w_disp,
                       lambda_conf=args.loss_w_conf, lambda_normal=args.loss_w_normal)

    # LR scheduler: warmup + cosine annealing
    total_steps = args.epochs * len(train_loader)
    def lr_lambda(step):
        if step < args.lr_warmup_steps:
            return step / args.lr_warmup_steps
        progress = (step - args.lr_warmup_steps) / (total_steps - args.lr_warmup_steps)
        return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_list = []  # Track regular checkpoints for rotation

    global_step = 0
    if local_rank == 0:
        print(f"Starting training for {args.epochs} epochs")
        print(f"Grad accum steps: {args.grad_accum}, effective batch size: {args.batch_size * args.grad_accum * 2}")
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        t_data_start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            t_data_end = time.perf_counter()
            t_data = t_data_end - t_data_start

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['targets'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['targets'].items()}

            is_last_accum = (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader)

            t_fwd_start = time.perf_counter()
            # 梯度累积：只在最后一步才同步梯度，减少NCCL通信频次
            ctx = model.no_sync() if not is_last_accum else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'])
                    loss_dict = loss_fn(outputs, batch['targets'], normalize_groups=batch['dataset_id'])
                    loss = loss_dict['loss'] / args.grad_accum

                loss.backward()

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            torch.cuda.synchronize()
            t_fwd = time.perf_counter() - t_fwd_start

            real_loss = loss.item() * args.grad_accum  # 还原除法，得到真实loss值
            epoch_loss += real_loss
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                lr_info = f"LR: {current_lr:.2e} (warmup {global_step}/{args.lr_warmup_steps} → {args.lr:.2e})" \
                    if global_step < args.lr_warmup_steps else f"LR: {current_lr:.2e}"
                # Print from ALL ranks so we can compare data/compute time per rank
                print(f"[{time.strftime('%H:%M:%S')}][rank{local_rank}] Epoch {epoch}, Batch {batch_idx}, "
                      f"data={t_data*1000:.0f}ms fwd+bwd={t_fwd*1000:.0f}ms Loss: {real_loss:.4f}, {lr_info}", flush=True)

                if local_rank == 0:
                    # Save loss log
                    log_entry = {
                        'epoch': epoch,
                        'step': global_step,
                        'batch': batch_idx,
                        'loss': f"{real_loss:.4f}",
                        'loss_3d': f"{loss_dict.get('loss_3d', 0) * args.grad_accum:.4f}",
                        'loss_2d': f"{loss_dict.get('loss_2d', 0) * args.grad_accum:.4f}",
                        'loss_vis': f"{loss_dict.get('loss_vis', 0) * args.grad_accum:.4f}",
                        'loss_disp': f"{loss_dict.get('loss_disp', 0) * args.grad_accum:.4f}",
                        'loss_conf': f"{loss_dict.get('loss_conf', 0) * args.grad_accum:.4f}",
                        'loss_normal': f"{loss_dict.get('loss_normal', 0) * args.grad_accum:.4f}",
                        'lr': f"{current_lr:.6f}"
                    }
                    with open(output_dir / 'loss_log.jsonl', 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')

            t_data_start = time.perf_counter()

        avg_loss = epoch_loss / len(train_loader)
        if local_rank == 0:
            print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    batch['targets'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['targets'].items()}
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'])
                        loss_dict = loss_fn(outputs, batch['targets'], normalize_groups=batch['dataset_id'])
                    val_loss += loss_dict['loss'].item()
            val_loss /= len(val_loader)
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()
            dist.barrier()  # 等所有 rank 跑完 val，防止死锁
            if local_rank == 0:
                print(f"Validation Loss: {val_loss:.4f}")
                with open(output_dir / 'val_log.jsonl', 'a') as f:
                    f.write(json.dumps({'epoch': epoch + 1, 'val_loss': f"{val_loss:.4f}"}) + '\n')

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 and local_rank == 0:
            is_milestone = (epoch + 1) % 1000 == 0
            if is_milestone:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            else:
                checkpoint_path = output_dir / f"checkpoint_latest_{epoch+1}.pth"
                checkpoint_list.append(checkpoint_path)
                # Remove old checkpoints if exceeds limit
                if len(checkpoint_list) > args.keep_checkpoints:
                    old_ckpt = checkpoint_list.pop(0)
                    if old_ckpt.exists():
                        old_ckpt.unlink()

            torch.save({'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(),
                       'epoch': epoch}, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
