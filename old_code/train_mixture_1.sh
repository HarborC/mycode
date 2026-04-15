#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"


CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
CONFIG="${CONFIG:-configs/mixture_pointodyssey.yaml}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-32}"
EPOCHS="${EPOCHS:-500000}"
LR="${LR:-5e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mixture_pointodyssey}"
RESUME="${RESUME:-/data1/zbf/my_dfrt/outputs/mixture_pointodyssey/checkpoint_latest_40.pth}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /root/miniconda3/envs/d4rt/bin/torchrun --nproc_per_node=1 --master_port=29509 train_mixture.py \
  --config "$CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --output-dir "$OUTPUT_DIR" \
  --loss-w-conf 0.0 \
  ${RESUME:+--resume "$RESUME"} \
  "$@"
