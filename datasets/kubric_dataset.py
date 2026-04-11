import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
from pathlib import Path
from PIL import Image
from datasets.base.transforms import *


class KubricDataset(BaseDataset):
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
        self.dataset_label = 'Kubric'
        self.data_root = Path(data_root)

        cache_path = f'data/dataset_cache/kubric_{self.mode}_cache.npy'
        if not os.path.exists(cache_path):
            self.sequences = []
            for p in sorted(self.data_root.iterdir()):
                if not p.is_dir():
                    continue
                seq = p.name
                seq_dir = p / seq if (p / seq).is_dir() else p
                if (
                    (seq_dir / f"{seq}.npy").exists()
                    and (seq_dir / f"{seq}_with_rank.npz").exists()
                    and (seq_dir / "frames").exists()
                ):
                    self.sequences.append(seq)
            if not self.sequences:
                raise RuntimeError(f"No valid Kubric scenes found under: {self.data_root}")
            np.save(cache_path, dict(sequences=self.sequences))
        else:
            npy = np.load(cache_path, allow_pickle=True).item()
            self.sequences = npy['sequences']

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} sequences in {data_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        seq = self.sequences[index]
        scene_dir = self.data_root / seq
        # Support nested structure: root/0001/0001/
        if (scene_dir / seq).is_dir():
            scene_dir = scene_dir / seq

        rank = np.load(scene_dir / f"{seq}_with_rank.npz", allow_pickle=True)
        K = np.asarray(rank["shared_intrinsics"], dtype=np.float32)        # [3,3]
        extrinsics_t34 = np.asarray(rank["extrinsics"], dtype=np.float32)  # [T,3,4], w2c

        frame_files = sorted((scene_dir / "frames").glob("*.png"))
        T_total = len(frame_files)
        idxs = self._sample_frame_indices(T_total, rng)

        # Load dense depth maps — prefer h5 (depths/ dir), fall back to .npy
        h5_path = (scene_dir / f"{seq}.npy").with_suffix('.h5')
        if h5_path.exists():
            depth_dir = scene_dir / "depths"
            depth_files = sorted(depth_dir.glob("*.npy")) if depth_dir.exists() else []
            def get_depth(i):
                if depth_files:
                    return np.load(depth_files[i]).astype(np.float32).squeeze()
                return None
        else:
            ann = np.load(scene_dir / f"{seq}.npy", allow_pickle=True).item()
            dense_depth = np.asarray(ann["depth"], dtype=np.float32)  # [T,H,W,1]
            def get_depth(i):
                return dense_depth[i, :, :, 0]

        views = []
        for idx in idxs:
            rgb_image = np.asarray(Image.open(frame_files[idx]).convert("RGB"))

            depthmap = get_depth(int(idx))
            if depthmap is None:
                depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)

            # w2c [3,4] -> c2w [4,4]
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :4] = extrinsics_t34[idx]
            camera_pose = np.linalg.inv(w2c)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, K.copy(), resolution, rng=rng, info=str(frame_files[idx]))

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=seq,
                instance=frame_files[idx].name,
            ))
        return views


if __name__ == '__main__':
    from datasets.utils.viser import visualize_dataset

    dataset = KubricDataset(
        data_root='/data2/d4rt/datasets/kubric',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride"
    )

    visualize_dataset(dataset, start_idx=0)
