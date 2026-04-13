import sys
sys.path.insert(0, '.')

import gzip
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets.base.base_dataset import BaseDataset
from datasets.base.types import UnifiedClip
from datasets.base.transforms import *


# ---------------------------------------------------------------------------
# Camera conversion helpers (PyTorch3D → OpenCV, same as Co3Dv2)
# ---------------------------------------------------------------------------

def _ndc_to_pinhole(focal_length, principal_point, image_size):
    """Convert PyTorch3D NDC intrinsics to a standard 3x3 pinhole matrix."""
    H, W = float(image_size[0]), float(image_size[1])
    half_s = min(H, W) / 2.0
    fx = focal_length[0] * half_s
    fy = focal_length[1] * half_s
    cx = W / 2.0 - principal_point[0] * half_s
    cy = H / 2.0 - principal_point[1] * half_s
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _p3d_to_opencv_w2c(R_p3d, T_p3d):
    """Convert PyTorch3D R/T to a 4x4 OpenCV-style world-to-camera matrix."""
    R = np.array(R_p3d, dtype=np.float64)
    T = np.array(T_p3d, dtype=np.float64)
    D = np.diag([-1.0, -1.0, 1.0])
    R_cv = (D @ R.T).astype(np.float32)
    T_cv = (D @ T).astype(np.float32)
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R_cv
    E[:3, 3] = T_cv
    return E


# ---------------------------------------------------------------------------
# Depth loading
# ---------------------------------------------------------------------------

def _load_depth(path):
    """Load 16-bit PNG depth map stored as float16."""
    with Image.open(path) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DynamicReplicaDataset(BaseDataset):
    STRIDE_CANDIDATES = [1, 2, 3]
    STRIDE_WEIGHTS = [0.5, 0.3, 0.2]

    def __init__(
        self,
        data_root=None,
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'DynamicReplica'
        self.data_root = Path(data_root)
        self.split_root = self.data_root / self.mode

        if not self.split_root.exists():
            raise FileNotFoundError(f"Split root not found: {self.split_root}")

        anno_file = self.split_root / f"frame_annotations_{self.mode}.jgz"
        if not anno_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_file}")

        cache_path = f'data/dataset_cache/dynamic_replica_{self.mode}_cache.npy'
        os.makedirs('data/dataset_cache', exist_ok=True)
        if not os.path.exists(cache_path):
            with gzip.open(anno_file, "rb") as f:
                raw_anno = json.load(f)
            anno_index = {}
            for entry in raw_anno:
                key = (entry["sequence_name"], entry["camera_name"], int(entry["frame_number"]))
                anno_index[key] = entry

            self.sequences = []
            self.num_frames = {}
            self.frame_numbers = {}

            for seq_dir in tqdm(sorted(self.split_root.iterdir())):
                if not seq_dir.is_dir() or "_source_" not in seq_dir.name:
                    continue
                base_name, _, camera = seq_dir.name.rpartition("_source_")
                if camera != "left":
                    continue
                if not (seq_dir / "images").exists():
                    continue

                fns = sorted(
                    fn for (bsn, cam, fn) in anno_index
                    if bsn == base_name and cam == camera
                )
                if len(fns) < 2:
                    continue

                seq = seq_dir.name
                self.sequences.append(seq)
                self.num_frames[seq] = len(fns)
                self.frame_numbers[seq] = fns

            np.save(cache_path, dict(
                sequences=self.sequences,
                num_frames=self.num_frames,
                frame_numbers=self.frame_numbers,
            ))
        else:
            npy = np.load(cache_path, allow_pickle=True).item()
            self.sequences    = npy['sequences']
            self.num_frames   = npy['num_frames']
            self.frame_numbers = npy['frame_numbers']

        self._anno_file  = anno_file
        self._anno_index = None

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} sequences in {self.split_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    def _get_clip(self, index, resolution, rng):
        seq = self.sequences[index]
        seq_dir = self.split_root / seq
        base_name, _, camera = seq.rpartition("_source_")
        fns = self.frame_numbers[seq]
        T_total = len(fns)
        frame_idxs = self._sample_frame_indices(T_total, rng)

        self._ensure_anno_loaded()

        images, depths, poses, intrinsics = [], [], [], []
        pts3d_list, valid_mask_list = [], []
        instances = []

        clip_label = seq

        for idx in frame_idxs:
            fn = fns[idx]
            anno = self._anno_index[(base_name, camera, fn)]

            img_path   = self.split_root / anno["image"]["path"]
            depth_path = self.split_root / anno["depth"]["path"]

            rgb_image = np.asarray(Image.open(img_path).convert("RGB"))
            depthmap  = _load_depth(depth_path)

            K = _ndc_to_pinhole(
                anno["viewpoint"]["focal_length"],
                anno["viewpoint"]["principal_point"],
                anno["image"]["size"],
            )
            w2c = _p3d_to_opencv_w2c(
                anno["viewpoint"]["R"],
                anno["viewpoint"]["T"],
            )
            camera_pose = np.linalg.inv(w2c)   # c2w [4,4]

            rgb_image, depthmap, K, _, _, _, _ = self._crop_resize_if_necessary(
                rgb_image, depthmap, K, resolution, rng=rng, info=str(img_path))

            pts3d_i, valid_mask_i, depthmap = self._process_depth(
                depthmap, K, camera_pose,
                label=f'{self.dataset_label}/{clip_label}', frame_id=str(fn))

            images.append(self.transform(rgb_image))
            depths.append(depthmap.astype(np.float32))
            poses.append(camera_pose.astype(np.float32))
            intrinsics.append(K)
            instances.append(str(fn))
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

    def _ensure_anno_loaded(self):
        if self._anno_index is not None:
            return
        with gzip.open(self._anno_file, "rb") as f:
            raw = json.load(f)
        index = {}
        for entry in raw:
            key = (entry["sequence_name"], entry["camera_name"], int(entry["frame_number"]))
            index[key] = entry
        self._anno_index = index


if __name__ == '__main__':
    from datasets.utils.viser import visualize_dataset

    dataset = DynamicReplicaDataset(
        data_root='/data1/d4rt/datasets/Dynamic_Replica',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride"
    )

    visualize_dataset(dataset, start_idx=0)
