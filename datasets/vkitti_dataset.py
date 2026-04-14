import sys
sys.path.insert(0, '.')

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets.base.base_dataset import BaseDataset, load_precomputed_fast
from datasets.base.types import UnifiedClip
from datasets.base.transforms import *


class VKittiDataset(BaseDataset):
    STRIDE_CANDIDATES = [1, 2, 3]
    STRIDE_WEIGHTS = [0.5, 0.3, 0.2]

    def __init__(
        self,
        data_root=None,
        verbose=False,
        depth_max=80.0,
        precompute_root=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'VKitti'
        self.data_root = Path(data_root)
        self.depth_max = depth_max
        self.precompute_root = Path(precompute_root) if precompute_root else None

        cache_path = f'data/dataset_cache/vkitti_{self.mode}_cache.npy'
        os.makedirs('data/dataset_cache', exist_ok=True)
        if not os.path.exists(cache_path):
            self.sequences = []
            self.num_frames = {}

            for rgb_dir in sorted(self.data_root.rglob("frames/rgb/Camera_*")):
                if not rgb_dir.is_dir():
                    continue
                seq = str(rgb_dir.relative_to(self.data_root))
                frame_num = len(list(rgb_dir.glob("rgb_*.jpg")))
                if frame_num == 0:
                    continue
                self.sequences.append(seq)
                self.num_frames[seq] = frame_num

            np.save(cache_path, dict(sequences=self.sequences, num_frames=self.num_frames))
        else:
            npy = np.load(cache_path, allow_pickle=True).item()
            self.sequences = npy['sequences']
            self.num_frames = npy['num_frames']

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} sequences in {data_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    def _get_clip(self, index, resolution, rng):
        seq = self.sequences[index]
        T_total = self.num_frames[seq]
        frame_idxs = self._sample_frame_indices(T_total, rng)

        seq_parts = seq.split('/')
        scene_variant = '/'.join(seq_parts[:2])
        camera_id = int(seq_parts[-1].split('_')[-1])

        rgb_dir   = self.data_root / seq
        depth_dir = self.data_root / scene_variant / 'frames' / 'depth' / seq_parts[-1]

        extrinsic_all = np.loadtxt(
            self.data_root / scene_variant / 'extrinsic.txt',
            delimiter=' ', skiprows=1
        )
        intrinsic_all = np.loadtxt(
            self.data_root / scene_variant / 'intrinsic.txt',
            delimiter=' ', skiprows=1
        )
        extrinsic_all = extrinsic_all[extrinsic_all[:, 1] == camera_id]
        intrinsic_all = intrinsic_all[intrinsic_all[:, 1] == camera_id]

        images, depths, poses, intrinsics = [], [], [], []
        pts3d_list, valid_mask_list = [], []
        instances = []

        clip_label = seq

        # ---- Load precomputed tracks if available ----
        trajs_2d = trajs_3d_world = visibility = None
        has_tracks = False
        if self.precompute_root is not None:
            pc = load_precomputed_fast(
                self.precompute_root / seq / "precomputed.npz",
                frame_idxs.tolist(),
            )
            if pc is not None and 'trajs_2d' in pc:
                trajs_2d = pc['trajs_2d'].astype(np.float32)
                trajs_3d_world = pc.get('trajs_3d_world')
                if trajs_3d_world is not None:
                    trajs_3d_world = trajs_3d_world.astype(np.float32)
                visibility = pc.get('visibs')
                if visibility is not None:
                    visibility = visibility.astype(bool)
                has_tracks = True

        for t_i, idx in enumerate(frame_idxs):
            img_path   = rgb_dir   / f"rgb_{idx:05d}.jpg"
            depth_path = depth_dir / f"depth_{idx:05d}.png"

            rgb_image = np.asarray(Image.open(img_path).convert("RGB"))

            dep = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depthmap = dep.astype(np.float32) / 100.0
            depthmap[depthmap > self.depth_max] = 0.0

            w2c = extrinsic_all[idx, 2:].reshape(4, 4).astype(np.float32)
            camera_pose = np.linalg.inv(w2c)

            fx, fy, cx, cy = intrinsic_all[idx, 2:].astype(np.float32)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            frame_trajs = trajs_2d[t_i].copy() if has_tracks else None

            rgb_image, depthmap, K, _, _, frame_trajs, _ = self._crop_resize_if_necessary(
                rgb_image, depthmap, K, resolution, rng=rng, info=str(img_path),
                trajs_2d=frame_trajs)

            pts3d_i, valid_mask_i, depthmap = self._process_depth(
                depthmap, K, camera_pose,
                label=f'{self.dataset_label}/{clip_label}', frame_id=f"{idx:05d}")

            images.append(self.transform(rgb_image))
            depths.append(depthmap.astype(np.float32))
            poses.append(camera_pose)
            intrinsics.append(K)
            instances.append(f"{idx:05d}")
            pts3d_list.append(pts3d_i)
            valid_mask_list.append(valid_mask_i)

            if has_tracks:
                trajs_2d[t_i] = frame_trajs

        clip = UnifiedClip(
            images=torch.stack(images, dim=0),
            depths=np.stack(depths, axis=0),
            camera_poses=np.stack(poses, axis=0),
            intrinsics=np.stack(intrinsics, axis=0),
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            visibility=visibility,
            dataset=self.dataset_label,
            label=clip_label,
            instances=instances,
            metadata={
                'has_tracks': has_tracks,
                'has_visibility': visibility is not None,
                'has_trajs_3d_world': trajs_3d_world is not None,
            },
        )
        clip.pts3d = np.stack(pts3d_list, axis=0)
        clip.valid_mask = np.stack(valid_mask_list, axis=0)
        return clip


if __name__ == '__main__':
    from datasets.utils.viser import visualize_dataset

    dataset = VKittiDataset(
        data_root='/data2/d4rt/datasets/VirtualKitti',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride",
    )

    dataset[0].save_as_rrd('vkitti_clip.rrd')
    visualize_dataset(dataset, start_idx=0)
