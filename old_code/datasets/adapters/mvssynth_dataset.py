import sys
sys.path.insert(0, '.')

import json
import os
import cv2
import numpy as np
from pathlib import Path
from datasets.base.base_dataset import BaseDataset, load_precomputed_fast
from datasets.base.types import UnifiedClip
from datasets.base.transforms import *

# Enable OpenEXR support in OpenCV
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


class MVSSynthDataset(BaseDataset):
    STRIDE_CANDIDATES = [1, 2, 4]
    STRIDE_WEIGHTS = [0.55, 0.3, 0.15]

    def __init__(
        self,
        data_root=None,
        verbose=False,
        precompute_root=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'MVSSynth'
        self.data_root = Path(data_root)
        self.precompute_root = Path(precompute_root) if precompute_root else None

        cache_path = f'data/dataset_cache/mvssynth_{self.mode}_cache.npy'
        os.makedirs('data/dataset_cache', exist_ok=True)
        if not os.path.exists(cache_path):
            self.sequences = []
            self.num_frames = {}

            num_images_path = self.data_root / "num_images.json"
            if num_images_path.exists():
                with open(num_images_path, "r") as f:
                    num_images_list = json.load(f)
            else:
                num_images_list = None

            for seq_dir in sorted(self.data_root.iterdir()):
                if not seq_dir.is_dir() or not seq_dir.name.isdigit():
                    continue
                if not all((seq_dir / sub).is_dir() for sub in ("images", "depths", "poses")):
                    continue
                seq = seq_dir.name
                if num_images_list is not None:
                    T = num_images_list[int(seq)]
                else:
                    T = len(list((seq_dir / "poses").glob("*.json")))
                if not T:
                    continue
                self.sequences.append(seq)
                self.num_frames[seq] = T

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
        seq_dir = self.data_root / seq
        T_total = self.num_frames[seq]
        frame_idxs = self._sample_frame_indices(T_total, rng)

        images, depths, poses, intrinsics = [], [], [], []
        pts3d_list, valid_mask_list = [], []
        instances = []
        c0 = None

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
                    # Unit and coordinate transform applied after c0 is known (see below)
                visibility = pc.get('visibs')
                if visibility is not None:
                    visibility = visibility.astype(bool)
                has_tracks = True

        for t_i, idx in enumerate(frame_idxs):
            img_path   = seq_dir / "images" / f"{idx:04d}.png"
            depth_path = seq_dir / "depths" / f"{idx:04d}.exr"
            pose_path  = seq_dir / "poses"  / f"{idx:04d}.json"

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            dep = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
            dep[~np.isfinite(dep)] = 0.0
            dep /= 100.0

            with open(pose_path, "r") as f:
                pose = json.load(f)

            K = np.array([
                [pose["f_x"], 0.0,         pose["c_x"]],
                [0.0,         pose["f_y"], pose["c_y"]],
                [0.0,         0.0,         1.0        ],
            ], dtype=np.float32)

            w2c = np.array(pose["extrinsic"], dtype=np.float32)  # [4,4]
            w2c[:3, 3] /= 100.0  # cm -> m

            # GTA V left-handed -> right-handed: flip world X axis
            w2c[:, 0] *= -1

            # World centering: shift origin to first selected camera position
            if c0 is None:
                c0 = np.linalg.inv(w2c.astype(np.float64))[:3, 3].astype(np.float32)
            w2c[:3, 3] += w2c[:3, :3] @ c0

            camera_pose = np.linalg.inv(w2c)  # c2w [4,4]

            frame_trajs = trajs_2d[t_i].copy() if has_tracks else None

            rgb_image, dep, K, _, _, frame_trajs, _ = self._crop_resize_if_necessary(
                rgb_image, dep, K, resolution, rng=rng, info=str(img_path),
                trajs_2d=frame_trajs)

            pts3d_i, valid_mask_i, dep = self._process_depth(
                dep, K, camera_pose,
                label=f'{self.dataset_label}/{clip_label}', frame_id=f"{idx:04d}")

            images.append(self.transform(rgb_image))
            depths.append(dep.astype(np.float32))
            poses.append(camera_pose.astype(np.float32))
            intrinsics.append(K)
            instances.append(f"{idx:04d}")
            pts3d_list.append(pts3d_i)
            valid_mask_list.append(valid_mask_i)

            if has_tracks:
                trajs_2d[t_i] = frame_trajs

        # Apply coordinate transform to trajs_3d_world to match the extrinsic convention:
        #   1. /= 100.0       : cm -> m  (extrinsic translation was also divided by 100)
        #   2. [..., 0] *= -1 : flip X   (matches w2c[:, 0] *= -1 applied to extrinsics)
        #   3. -= c0          : world centering (matches the w2c[:3,3] += R @ c0 shift)
        if trajs_3d_world is not None and c0 is not None:
            trajs_3d_world /= 100.0
            trajs_3d_world[..., 0] *= -1
            trajs_3d_world -= c0

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

    dataset = MVSSynthDataset(
        data_root='/data2/d4rt/datasets/MVS-Synth/GTAV_1080',
        precompute_root='/data2/d4rt/datasets/MVS-Synth/GTAV_1080',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride"
    )

    dataset[0].save_as_rrd('MVSSynth_clip.rrd')
    visualize_dataset(dataset, start_idx=0)
