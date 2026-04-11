import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
import h5py
from utils.basic import seed_anything
from PIL import Image
from tqdm import tqdm
from datasets.base.transforms import *
from datasets.base.utils import view_name, tensor2numpy

def xyzqxqyqxqw_to_c2w(xyzqxqyqxqw):
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)
    #NOTE: we need to convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    c2w[:3, 3] = np.array([x, y, z])
    return c2w

class TarTanAirDataset(BaseDataset):
    STRIDE_CANDIDATES = [1, 2, 3]
    STRIDE_WEIGHTS = [0.5, 0.3, 0.2]

    def __init__(
        self,
        data_root='ssd:s3://TartanAir',
        verbose=False,
        max_distance=24,
        seq_num=-1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.verbose = verbose
        self.dataset_label = 'TarTanAir'
        self.max_distance = max_distance
        self.data_root = data_root

        cache_path = f'data/dataset_cache/tartanair_cache.npy'
        if not os.path.exists(cache_path):
            sequences = []
            for seq in os.listdir(data_root):
                if not os.path.isdir(os.path.join(data_root, seq)):
                    continue
                names = os.listdir(os.path.join(data_root, seq, 'Easy'))
                seq_ = [(seq, 'Easy', name) for name in names]
                sequences.extend(seq_)
                names = os.listdir(os.path.join(data_root, seq, 'Hard'))
                seq_ = [(seq, 'Hard', name) for name in names]
                sequences.extend(seq_)

            sequences = sorted(sequences)

            num_imgs = {}
            for seq in sequences:
                rgb_path = os.path.join(data_root, seq[0], seq[1], seq[2], 'image_left')
                num_imgs[seq] = len(os.listdir(rgb_path))

            os.makedirs('data/dataset_cache', exist_ok=True)
            np.save(cache_path, dict(sequences=sequences, num_imgs=num_imgs))
        else:
            npy = np.load(cache_path, allow_pickle=True).item()
            sequences = npy['sequences']
            num_imgs = npy['num_imgs']

        if seq_num > 0:
            sequences = sequences[:seq_num]

        self.sequences = sequences
        self.num_imgs = num_imgs

        fx = 320.0
        fy = 320.0
        cx = 320.0
        cy = 240.0
        width = 640                                                                                                                                                                         
        height = 480 
        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences:', self.sequences)
        print(f'[{self.dataset_label}] Found {len(self.sequences)} unique videos in {data_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        num_imgs = self.num_imgs[scene]

        # Use parent class stride sampling if sampling_mode is 'stride'
        if self.sampling_mode == 'stride':
            idxs = self._sample_frame_indices(num_imgs, rng)
        elif self.sampling_mode == 'random':
            idxs = [rng.integers(0, num_imgs)]

            # max_distance = 12 if scene in self.special_scenes else self.max_distance
            max_distance = int(self.max_distance / 8 * self.frame_num)
            start_idx = max(0, idxs[-1] - max_distance)
            end_idx = min(num_imgs-1, start_idx + 2*max_distance)
            start_idx = max(0, end_idx - 2*max_distance)
            valid_indices = np.arange(start_idx, end_idx + 1)

            should_replace = len(valid_indices) < self.frame_num - 1
            idxs.extend(list(rng.choice(valid_indices, self.frame_num-1, replace=should_replace)))

        self.this_views_info = dict(
            scene=scene,
            pairs=idxs,
        )

        cam_path = os.path.join(self.data_root, scene[0], scene[1], scene[2], 'pose_left.txt')
        caminfo = np.loadtxt(cam_path)

        views = []
        for idx in idxs:
            impath = os.path.join(self.data_root, scene[0], scene[1], scene[2], 'image_left', f'{idx:06d}_left.png')
            depthpath = os.path.join(self.data_root, scene[0], scene[1], scene[2], 'depth_left', f'{idx:06d}_left_depth.npy')

            # load camera params
            camera_pose = np.array(xyzqxqyqxqw_to_c2w(caminfo[idx]), dtype=np.float32)

            # load image and depth
            rgb_image = np.array(Image.open(impath))

            depthmap = np.load(depthpath)
            depthmap[depthmap > 80] = -1

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, self.intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=f'{scene[0]}_{scene[1]}_{scene[2]}',
                instance=str(idx),
            ))
        return views


if __name__ == '__main__':
    from datasets.utils.viser import visualize_dataset

    dataset = TarTanAirDataset(
        data_root='/data2/d4rt/datasets/TartanAir2',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride"
    )

    visualize_dataset(dataset, start_idx=0)