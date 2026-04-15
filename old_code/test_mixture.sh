#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Configuration
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
CONFIG="${CONFIG:-configs/mixture_full_11datasets.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/mixture_5_3d/checkpoint_latest_126.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-test_outputs/mixture_test_$(date +%Y%m%d_%H%M%S)}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SPLIT="${SPLIT:-val}"

echo "=== D4RT Mixture Dataset Test ==="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "Split: $SPLIT"
echo "================================"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python test_mixture.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --split "$SPLIT" \
  "$@"

echo ""
echo "Test complete! Results saved to: $OUTPUT_DIR"
