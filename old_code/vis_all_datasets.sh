#!/usr/bin/env bash
# Visualize 3 samples from each dataset separately
set -euo pipefail
cd /data1/zbf/my_dfrt

DATASETS=(
  "pointodyssey:/data2/d4rt/datasets/PointOdyssey"
  "kubric:/data2/d4rt/datasets/kubric"
  "dynamic_replica:/data1/d4rt/datasets/Dynamic_Replica"
  "scannet:/data2/d4rt/datasets/scannet/scannet"
  "co3dv2:/data2/d4rt/datasets/Co3Dv2"
  "blendedmvs:/data2/d4rt/datasets/BlendedMVS"
  "mvssynth:/data2/d4rt/datasets/MVS-Synth/GTAV_1080"
  "vkitti2:/data2/d4rt/datasets/VirtualKitti"
)

for entry in "${DATASETS[@]}"; do
  name="${entry%%:*}"
  root="${entry##*:}"
  echo "=== $name ==="
  python visualize_gt.py \
    --config /dev/stdin \
    --output-dir vis_gt/per_dataset/$name \
    --num-samples 3 \
    --num-queries 32768 \
    --split train <<EOF
mode: single
name: $name
root: $root
clip_len: 48
img_size: 256
num_queries: 32768
EOF
done
echo "Done!"
