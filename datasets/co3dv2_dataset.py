import sys
sys.path.insert(0, '.')

from datasets.base.base_dataset import BaseDataset, load_precomputed_fast
from datasets.base.types import UnifiedClip
import os
import numpy as np
import os.path as osp
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets.base.transforms import *
import gzip
import json

def convert_ndc_to_pinhole(focal_length, principal_point, image_size):
    focal_length = np.array(focal_length)
    principal_point = np.array(principal_point)
    image_size_wh = np.array([image_size[1], image_size[0]])
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    fx, fy = focal_length_px[0], focal_length_px[1]
    cx, cy = principal_point_px[0], principal_point_px[1]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    R = R[None, :, :]
    T = T[None, :]
    focal = focal[None, :]
    p0 = p0[None, :]
    image_size = image_size[None, :]

    R_pytorch3d = R.copy()
    T_pytorch3d = T.copy()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.transpose(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size[:, ::-1]

    # NDC to screen conversion.
    scale = np.min(image_size_wh, axis=1, keepdims=True) / 2.0
    scale = np.repeat(scale, 2, axis=1)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = np.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]

class CO3DV2Dataset(BaseDataset):
    STRIDE_CANDIDATES = [1, 2, 4]
    STRIDE_WEIGHTS = [0.50, 0.35, 0.15]

    def __init__(
        self,
        data_root=None,
        verbose=False,
        mask_bg='rand',
        precompute_root=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'CO3DV2'
        self.data_root = data_root

        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.precompute_root = Path(precompute_root) if precompute_root else None

        os.makedirs('data/dataset_cache', exist_ok=True)
        if not os.path.exists(f'data/dataset_cache/co3dv2_{self.mode}_cache.npy'):
            self.sequences = []
            self.num_image = {}

            for seq in tqdm(os.listdir(data_root)):
                try:
                    annotation_path_train = osp.join(data_root, seq + '_train.jgz')
                    annotation_path_test = osp.join(data_root, seq + '_test.jgz')
                    if self.mode == 'train':
                        with gzip.open(annotation_path_train, 'rt', encoding='utf-8') as f:
                            annotation = json.load(f)
                    else:
                        with gzip.open(annotation_path_test, 'rt', encoding='utf-8') as f:
                            annotation = json.load(f)
                except Exception:
                    continue

                for sub_seq in annotation.keys():
                    self.num_image[(seq, sub_seq)] = len(annotation[sub_seq])
                    self.sequences.append((seq, sub_seq))

            np.save(f'data/dataset_cache/co3dv2_{self.mode}_cache', dict(sequences=self.sequences, num_image=self.num_image))

        else:
            npy = np.load(f'data/dataset_cache/co3dv2_{self.mode}_cache.npy', allow_pickle=True).item()
            self.sequences = npy['sequences']
            self.num_image = npy['num_image']

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences of {self.dataset_label} dataset:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} unique videos in {data_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    def _get_clip(self, index, resolution, rng):
        scene = tuple(self.sequences[index])

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        annotation_path_train = osp.join(self.data_root, scene[0] + '_train.jgz')
        annotation_path_test = osp.join(self.data_root, scene[0] + '_test.jgz')

        with gzip.open(annotation_path_train, 'rt', encoding='utf-8') as f:
            annotation_train = json.load(f)

        with gzip.open(annotation_path_test, 'rt', encoding='utf-8') as f:
            annotation_test = json.load(f)

        annotation = {**annotation_train, **annotation_test}[scene[1]]

        num_img = len(annotation)
        frame_idxs = self._sample_frame_indices(num_img, rng)

        self.this_views_info = dict(
            scene=scene,
            idxs=frame_idxs,
        )

        images, depths, poses, intrinsics = [], [], [], []
        pts3d_list, valid_mask_list = [], []
        instances = []

        clip_label = scene[0]+'-'+scene[1]

        # ---- Load precomputed tracks if available ----
        trajs_2d = trajs_3d_world = visibility = None
        has_tracks = False
        if self.precompute_root is not None:
            pc_path = self.precompute_root / scene[0] / scene[1] / "precomputed.npz"
            pc = load_precomputed_fast(pc_path, frame_idxs.tolist())
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
            anno = annotation[idx]
            filepath = anno['filepath']

            impath = osp.join(self.data_root, filepath)
            depthpath = osp.join(self.data_root, filepath.replace('images', 'depths')+'.geometric.png')

            # load image and depth
            rgb_image = np.array(Image.open(impath))

            depthmap = Image.open(depthpath)
            depth_h, depth_w = depthmap.size[1], depthmap.size[0]
            depthmap = np.frombuffer(np.array(depthmap, dtype=np.uint16), dtype=np.float16).astype(np.float32).reshape((depth_h, depth_w))
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

            # load camera params
            camera_pose = np.eye(4)
            R = np.array(anno['R'])
            T = np.array(anno['T'])

            image_size = np.array([rgb_image.shape[0], rgb_image.shape[1]])
            focal_length = np.array(anno['focal_length'])
            principal_point = np.array(anno['principal_point'])

            R, tvec, camera_intrinsics = opencv_from_cameras_projection(R, T, focal_length, principal_point, image_size)
            camera_pose[:3, :3] = R
            camera_pose[:3, 3] = tvec
            camera_pose = np.linalg.inv(camera_pose)

            if mask_bg:
                maskpath = impath.replace('/images/', '/masks/').replace('.jpg', '.png')
                maskmap = np.array(Image.open(maskpath)).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1
                depthmap *= maskmap

            frame_trajs = trajs_2d[t_i].copy() if has_tracks else None

            rgb_image, depthmap, K, _, _, frame_trajs, _ = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics, resolution, rng=rng, info=impath,
                trajs_2d=frame_trajs)

            pts3d_i, valid_mask_i, depthmap = self._process_depth(
                depthmap, K, camera_pose,
                label=f'{self.dataset_label}/{clip_label}', frame_id=str(idx))

            images.append(self.transform(rgb_image))
            depths.append(depthmap.astype(np.float32))
            poses.append(camera_pose.astype(np.float32))
            intrinsics.append(K)
            instances.append(osp.split(impath)[1])
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

    dataset = CO3DV2Dataset(
        data_root='/data2/d4rt/datasets/Co3Dv2',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        mask_bg=False,
        seed=42,
        sampling_mode="stride",
    )

    dataset[0].save_as_rrd('Co3Dv2_clip.rrd')
    visualize_dataset(dataset, start_idx=0)
