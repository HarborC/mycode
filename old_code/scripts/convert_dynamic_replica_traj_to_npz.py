#!/usr/bin/env python3
"""Convert Dynamic_Replica trajectory .pth files to .npz for faster loading."""
import sys, glob, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

ROOT = '/data1/d4rt/datasets/Dynamic_Replica'

def convert_one(pth_path: str) -> str:
    p = Path(pth_path)
    npz_path = p.with_suffix('.npz')
    if npz_path.exists():
        return 'skip'
    import torch
    data = torch.load(p, map_location='cpu', weights_only=False)
    np.savez_compressed(npz_path,
        traj_3d_world=data['traj_3d_world'].numpy().astype(np.float32),
        traj_2d=data['traj_2d'].numpy().astype(np.float32),
        verts_inds_vis=data['verts_inds_vis'].numpy().astype(bool),
    )
    return 'done'

pth_files = glob.glob(f'{ROOT}/*/*/trajectories/*.pth')
print(f'Found {len(pth_files)} .pth files')

t0 = time.time()
done = skip = err = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(convert_one, p): p for p in pth_files}
    for i, fut in enumerate(as_completed(futures)):
        try:
            r = fut.result()
            if r == 'done': done += 1
            else: skip += 1
        except Exception as e:
            err += 1
            print(f'ERROR {futures[fut]}: {e}')
        if (i+1) % 500 == 0:
            print(f'  {i+1}/{len(pth_files)} done={done} skip={skip} err={err}', flush=True)

print(f'完成: done={done} skip={skip} err={err}, 耗时={time.time()-t0:.0f}s')
