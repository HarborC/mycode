import sys
sys.path.insert(0, '.')

import re
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets.base.base_dataset import BaseDataset
from datasets.base.types import UnifiedClip
from datasets.base.transforms import *


class PointOdysseyDataset(BaseDataset):
    STRIDE_CANDIDATES = [1, 2, 4]
    STRIDE_WEIGHTS = [0.45, 0.35, 0.2]

    def __init__(
        self,
        data_root=None,
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'PointOdyssey'
        self.data_root = Path(data_root)
        self.split_root = self.data_root / self.mode

        if not self.split_root.exists():
            raise FileNotFoundError(f"Split root not found: {self.split_root}")

        cache_path = f'data/dataset_cache/pointodyssey_{self.mode}_cache.npy'
        os.makedirs('data/dataset_cache', exist_ok=True)
        if not os.path.exists(cache_path):
            self.sequences = []
            self.num_frames = {}
            for scene_dir in tqdm(sorted(self.split_root.iterdir())):
                if not scene_dir.is_dir():
                    continue
                rgb_dir = scene_dir / "rgbs"
                if not rgb_dir.exists():
                    continue
                files = [p for p in rgb_dir.iterdir()
                         if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}]
                num_frame = len(files)
                if not num_frame:
                    continue
                self.sequences.append(scene_dir.name)
                self.num_frames[scene_dir.name] = num_frame
            np.save(cache_path, dict(sequences=self.sequences, num_frames=self.num_frames))
        else:
            npy = np.load(cache_path, allow_pickle=True).item()
            self.sequences = npy['sequences']
            self.num_frames = npy['num_frames']

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} sequences in {self.split_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    def _get_clip(self, idx, resolution, rng):
        seq = self.sequences[idx]
        scene_dir = self.split_root / seq
        T_total = self.num_frames[seq]
        frame_idxs = self._sample_frame_indices(T_total, rng)

        anno = self._load_anno(scene_dir)
        intrinsics_all = anno['intrinsics']   # [T_total,3,3]
        extrinsics_all = anno['extrinsics']   # [T_total,4,4], w2c

        rgb_files   = self._sorted_frame_files(scene_dir / "rgbs")
        depth_files = self._sorted_frame_files(scene_dir / "depths")

        images, depths, poses, intrinsics = [], [], [], []
        pts3d_list, valid_mask_list = [], []
        instances = []

        clip_label = seq

        for fi in frame_idxs:
            rgb_image = self._read_rgb(rgb_files[fi])

            if depth_files:
                depthmap = self._read_depth(depth_files[fi])
                if depthmap is None:
                    depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)
                else:
                    depthmap = depthmap.astype(np.float32) * 0.01
            else:
                depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            K   = intrinsics_all[fi].astype(np.float32)
            w2c = extrinsics_all[fi].astype(np.float32)
            camera_pose = np.linalg.inv(w2c)   # c2w [4,4]

            rgb_image, depthmap, K, _, _, _, _ = self._crop_resize_if_necessary(
                rgb_image, depthmap, K, resolution, rng=rng, info=str(rgb_files[fi]))

            pts3d_i, valid_mask_i, depthmap = self._process_depth(
                depthmap, K, camera_pose,
                label=f'{self.dataset_label}/{clip_label}', frame_id=str(int(fi)))

            images.append(self.transform(rgb_image))
            depths.append(depthmap.astype(np.float32))
            poses.append(camera_pose.astype(np.float32))
            intrinsics.append(K)
            instances.append(str(int(fi)))
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

    def _load_anno(self, scene_dir):
        h5_path = scene_dir / "anno.h5"
        if h5_path.exists():
            import h5py
            with h5py.File(h5_path, 'r') as f:
                return {k: f[k][()] for k in ('intrinsics', 'extrinsics')}
        z = np.load(scene_dir / "anno.npz", allow_pickle=True)
        return {k: z[k] for k in ('intrinsics', 'extrinsics')}

    def _sorted_frame_files(self, d):
        files = [p for p in d.iterdir()
                 if p.suffix.lower() in {'.jpg', '.png', '.jpeg', '.npy'}]
        def key(p):
            m = re.search(r'(\d+)$', p.stem)
            return int(m.group(1)) if m else p.stem
        return sorted(files, key=key)

    def _read_rgb(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_depth(self, path):
        if Path(path).suffix.lower() == '.npy':
            return np.load(path).astype(np.float32)
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return depth.astype(np.float32) if depth is not None else None


if __name__ == '__main__':
    from datasets.utils.viser import visualize_dataset

    dataset = PointOdysseyDataset(
        data_root='/data2/d4rt/datasets/PointOdyssey',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride"
    )

    visualize_dataset(dataset, start_idx=0)
