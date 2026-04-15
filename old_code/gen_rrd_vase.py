#!/usr/bin/env python3
"""Generate RRD for vase sequence."""
import sys
sys.path.insert(0, '/data1/zbf/my_dfrt')

from pathlib import Path
import verify_datasets

out_dir = Path("vis_gt/verify/co3dv2_vase")
out_dir.mkdir(parents=True, exist_ok=True)

from datasets.adapters.co3dv2 import Co3Dv2Adapter
from datasets.adapters.base import UnifiedClip

adapter = Co3Dv2Adapter(root="/data2/d4rt/datasets/Co3Dv2", split="train")
seq = "vase/380_44868_89574"
clip = adapter.load_clip(seq, list(range(31)))

import json
metrics = verify_datasets.verify_has_tracks(clip, out_dir)
print(f"Results: {json.dumps(metrics, indent=2, default=str)}")
verify_datasets.log_clip_to_rerun(clip, "co3dv2", seq, out_dir / "clip.rrd")
