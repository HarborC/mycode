import sys
sys.path.append('.')

import json
import os
import cv2
import numpy as np
from pathlib import Path
from datasets.base.base_dataset import BaseDataset
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
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'MVSSynth'
        self.data_root = Path(data_root)

        cache_path = f'data/dataset_cache/mvssynth_{self.mode}_cache.npy'
        if not os.path.exists(cache_path):
            self.sequences = []
            self.num_frames = {}

            # num_images.json: [100, 100, ..., 100], one entry per sequence
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

    def _get_views(self, index, resolution, rng):
        seq = self.sequences[index]
        seq_dir = self.data_root / seq
        T_total = self.num_frames[seq]
        idxs = self._sample_frame_indices(T_total, rng)

        # Read the first frame's pose to get the first camera centre for world centering
        # (centering is done relative to the first *selected* frame below)

        views = []
        c0 = None  # first camera centre in world coords (for centering)
        for i, idx in enumerate(idxs):
            img_path   = seq_dir / "images" / f"{idx:04d}.png"
            depth_path = seq_dir / "depths" / f"{idx:04d}.exr"
            pose_path  = seq_dir / "poses"  / f"{idx:04d}.json"

            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load depth: EXR float32, unit cm -> m, inf -> 0
            dep = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
            dep[~np.isfinite(dep)] = 0.0
            dep /= 100.0

            # Load pose
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

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, dep, K.copy(), resolution, rng=rng, info=str(img_path))

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=seq,
                instance=f"{idx:04d}",
            ))
        return views


if __name__ == '__main__':
    from datasets.utils.viser import visualize_dataset

    dataset = MVSSynthDataset(
        data_root='/data2/d4rt/datasets/MVS-Synth/GTAV_1080',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride"
    )

    visualize_dataset(dataset, start_idx=0)
