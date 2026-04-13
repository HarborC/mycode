import sys
sys.path.insert(0, '.')

import re
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets.base.base_dataset import BaseDataset
from datasets.base.types import UnifiedClip
from datasets.base.transforms import *

_CAM_RE = re.compile(r"^(\d+)_cam\.txt$")

_SPLIT_LIST_FILES = {
    'train': 'BlendedMVS_training.txt',
    'val':   'validation_list.txt',
}


def _read_pfm(path):
    """Read a PFM (Portable Float Map) file and return a float32 2-D array.

    BlendedMVS depth maps are single-channel Pf files, stored bottom-up.
    """
    with open(path, "rb") as f:
        magic = f.readline().decode("latin-1").strip()
        width, height = map(int, f.readline().decode("latin-1").strip().split())
        scale_raw = float(f.readline().decode("latin-1").strip())
        little_endian = scale_raw < 0
        scale = abs(scale_raw)
        channels = 3 if magic == "PF" else 1
        dtype = np.dtype("<f4" if little_endian else ">f4")
        data = np.frombuffer(f.read(), dtype=dtype)

    data = data.reshape((height, width, channels) if channels > 1 else (height, width))
    data = np.flipud(data)   # PFM stored bottom-up
    if scale != 1.0:
        data = data * scale
    return data.astype(np.float32)


def _parse_cam_file(path):
    """Parse a BlendedMVS camera file.

    Returns dict with 'extrinsic' (4,4) w2c and 'intrinsic' (3,3).
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # lines[0]='extrinsic', lines[1-4]=4x4, lines[5]='intrinsic', lines[6-8]=3x3
    extrinsic = np.array(
        [[float(v) for v in lines[i].split()] for i in range(1, 5)],
        dtype=np.float32,
    )
    intrinsic = np.array(
        [[float(v) for v in lines[i].split()] for i in range(6, 9)],
        dtype=np.float32,
    )
    return extrinsic, intrinsic


class BlendedMVSDataset(BaseDataset):
    STRIDE_CANDIDATES = [1]
    STRIDE_WEIGHTS = [1.0]

    def __init__(
        self,
        data_root=None,
        verbose=False,
        use_masked=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'BlendedMVS'
        self.data_root = Path(data_root)
        self.use_masked = use_masked

        # mode -> split list file
        split_key = 'val' if self.mode in ('val', 'valid', 'validation') else 'train'
        list_path = self.data_root / _SPLIT_LIST_FILES[split_key]
        if not list_path.exists():
            raise FileNotFoundError(f"Split list not found: {list_path}")

        cache_path = f'data/dataset_cache/blendedmvs_{self.mode}_cache.npy'
        os.makedirs('data/dataset_cache', exist_ok=True)
        if not os.path.exists(cache_path):
            with open(list_path, "r") as f:
                scene_ids = [ln.strip() for ln in f if ln.strip()]

            self.sequences = []
            self.frame_ids = {}   # scene_id -> list of frame_id strings

            for scene_id in tqdm(scene_ids):
                scene_dir = self.data_root / scene_id
                cam_dir = scene_dir / "cams"
                if not cam_dir.is_dir():
                    continue
                cam_files = [p for p in cam_dir.iterdir()
                             if p.is_file() and _CAM_RE.match(p.name)]
                if not cam_files:
                    continue
                cam_files.sort(key=lambda p: int(_CAM_RE.match(p.name).group(1)))
                fids = [_CAM_RE.match(p.name).group(1) for p in cam_files]
                self.sequences.append(scene_id)
                self.frame_ids[scene_id] = fids

            np.save(cache_path, dict(sequences=self.sequences, frame_ids=self.frame_ids))
        else:
            npy = np.load(cache_path, allow_pickle=True).item()
            self.sequences = npy['sequences']
            self.frame_ids = npy['frame_ids']

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} sequences in {data_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    def _get_clip(self, index, resolution, rng):
        scene_id = self.sequences[index]
        scene_dir = self.data_root / scene_id
        fids = self.frame_ids[scene_id]
        T_total = len(fids)
        frame_idxs = self._sample_frame_indices(T_total, rng)

        images, depths, poses, intrinsics = [], [], [], []
        pts3d_list, valid_mask_list = [], []
        instances = []

        clip_label = scene_id

        for idx in frame_idxs:
            fid = fids[idx]
            img_suffix = f"{fid}_masked.jpg" if self.use_masked else f"{fid}.jpg"
            img_path   = scene_dir / "blended_images" / img_suffix
            depth_path = scene_dir / "rendered_depth_maps" / f"{fid}.pfm"
            cam_path   = scene_dir / "cams" / f"{fid}_cam.txt"

            rgb_image = np.asarray(Image.open(img_path).convert("RGB"))
            depthmap  = _read_pfm(depth_path)

            w2c, K = _parse_cam_file(cam_path)
            camera_pose = np.linalg.inv(w2c)   # c2w [4,4]

            rgb_image, depthmap, K, _, _, _, _ = self._crop_resize_if_necessary(
                rgb_image, depthmap, K, resolution, rng=rng, info=str(img_path))

            pts3d_i, valid_mask_i, depthmap = self._process_depth(
                depthmap, K, camera_pose,
                label=f'{self.dataset_label}/{clip_label}', frame_id=str(fid))

            images.append(self.transform(rgb_image))
            depths.append(depthmap.astype(np.float32))
            poses.append(camera_pose.astype(np.float32))
            intrinsics.append(K)
            instances.append(fid)
            pts3d_list.append(pts3d_i)
            valid_mask_list.append(valid_mask_i)

        clip = UnifiedClip(
            images=torch.stack(images, dim=0),
            depths=np.stack(depths, axis=0),
            camera_poses=np.stack(poses, axis=0),
            intrinsics=np.stack(intrinsics, axis=0),
            dataset=self.dataset_label,
            label=clip_label,
            instances=instances,
            metadata={},
        )
        clip.pts3d = np.stack(pts3d_list, axis=0)
        clip.valid_mask = np.stack(valid_mask_list, axis=0)
        return clip


if __name__ == '__main__':
    from datasets.utils.viser import visualize_dataset

    dataset = BlendedMVSDataset(
        data_root='/data2/d4rt/datasets/BlendedMVS',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride",
    )

    visualize_dataset(dataset, start_idx=0)
