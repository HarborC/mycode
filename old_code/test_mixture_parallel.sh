#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Configuration
GPUS="${GPUS:-0,6,7}"
CONFIG="${CONFIG:-configs/mixture_full_11datasets.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/mixture_5_3d/checkpoint_latest_126.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-test_outputs/mixture_parallel_$(date +%Y%m%d_%H%M%S)}"
NUM_SAMPLES="${NUM_SAMPLES:-30}"
NO_VIDEOS="${NO_VIDEOS:-}"

echo "=== D4RT Mixture Dataset Parallel Test ==="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "GPUs: $GPUS"
echo "=========================================="

CUDA_VISIBLE_DEVICES="$GPUS" python test_mixture_parallel.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --gpus "$GPUS" \
  ${NO_VIDEOS:+--no-videos} \
  "$@"

echo ""
echo "Test complete! Results saved to: $OUTPUT_DIR"
