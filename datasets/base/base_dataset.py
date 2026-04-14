from datasets.base.easy_dataset import EasyDataset
from datasets.base.types import UnifiedClip
from utils.geometry import depthmap_to_absolute_camera_coordinates
import numpy as np
import os
import PIL
import utils.cropping as cropping
import torchvision.transforms as tvf
import torchvision.transforms.v2 as tv2
import torch
from omegaconf import OmegaConf
from .transforms import *
import pandas as pd
from .utils import *
from pathlib import Path
from typing import Optional


def load_precomputed_fast(npz_path, frame_indices):
    """Load precomputed tracks/normals for specific frame indices.

    Prefers .h5 over .npz for faster random access.
    Returns dict with arrays indexed to frame_indices, or None.
    """
    npz_path = Path(npz_path)
    h5_path = npz_path.with_suffix('.h5')

    if h5_path.exists():
        import h5py
        sorted_idx = sorted(set(frame_indices))
        idx_map = {v: i for i, v in enumerate(sorted_idx)}
        reorder = [idx_map[i] for i in frame_indices]
        needs_reorder = reorder != list(range(len(frame_indices)))

        result = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                ds = f[key]
                if ds.ndim >= 1 and ds.shape[0] > 1:
                    data = ds[sorted_idx]
                    if needs_reorder:
                        data = data[reorder]
                    result[key] = data
                else:
                    result[key] = ds[()]
        return result

    elif npz_path.exists():
        raw = np.load(npz_path, allow_pickle=True)
        result = {}
        for k in raw.files:
            arr = raw[k]
            if arr.ndim >= 1 and arr.shape[0] > 1:
                result[k] = arr[np.array(frame_indices)]
            else:
                result[k] = arr[()]
        return result

    return None


def sample_frame_indices_stride(
    T_total,
    frame_num,
    rng,
    stride_candidates,
    stride_weights,
):
    """
    Helper function for stride-based frame sampling.

    This is a utility function that can be called by dataset subclasses
    in their _sample_frame_indices implementation.

    Args:
        T_total: Total number of frames
        frame_num: Number of frames to sample
        rng: numpy random generator
        stride_candidates: List of candidate strides (e.g., [1, 2, 4])
        stride_weights: Weights for each stride (e.g., [0.5, 0.3, 0.2])

    Returns:
        indices: array of frame indices, sorted
    """
    if T_total < frame_num:
        # Need replacement
        return np.sort(rng.choice(T_total, frame_num, replace=True))

    # Filter valid strides
    valid_strides = []
    valid_weights = []
    for stride, weight in zip(stride_candidates, stride_weights):
        if T_total >= 1 + (frame_num - 1) * stride:
            valid_strides.append(stride)
            valid_weights.append(weight)

    if not valid_strides:
        # Fallback to stride=1
        stride = 1
    else:
        # Weighted random choice
        total_w = sum(valid_weights)
        probs = [w / total_w for w in valid_weights]
        stride = rng.choice(valid_strides, p=probs)

    # Sample start index
    max_start = T_total - (frame_num - 1) * stride
    if max_start <= 0:
        # Fallback: take first frame_num frames
        return np.arange(min(frame_num, T_total))

    start_idx = rng.integers(0, max_start)
    return np.arange(start_idx, start_idx + frame_num * stride, stride)

class BaseDataset(EasyDataset):
    def __init__(
        self,
        seed=2024,
        resolution=None,            # (width, height) or list of (width, height) or list of int
        aug_crop=False,             # False or int, slightly scale the image a bit larger than the target resolution
        aug_focal=False,            # False or float in [0, 1]
        z_far=0,
        frame_num=2,
        transform=tvf.ToTensor(),
        cache_file=None,
        save_cache=False,
        mode='train',
        cache_name=None,
        max_refetch=3,
        random_sample_thres=0.1,
        use_sparse_depth=False,
        sampling_mode='random',     # 'random' or 'stride'
    ):
        super().__init__()
        self.frame_num = frame_num
        self.sampling_mode = sampling_mode

        self.transform = transform

        self.use_sparse_depth = use_sparse_depth

        self._rng = np.random.default_rng(seed)
        self._set_resolutions(resolution)

        self.aug_crop = aug_crop
        self.aug_focal = aug_focal

        self.z_far = z_far

        self.dataset_label = 'BaseDataset'

        self.save_cache = save_cache
        self.cache_loaded = False
        self.cache_name = cache_name
        if cache_file is not None:
            print(f'[BaseDataset] Loading cache from {cache_file}..')
            res = self.load_cache(cache_file)
            if res:
                self.cache_loaded = True
                print(f'[BaseDataset] Cache is loaded.')

        self.mode = mode
        self.max_refetch = max_refetch

        self.random_sample_thres = random_sample_thres  # default not to do that

    def _sample_frame_indices(self, T_total, rng):
        """
        Sample frame indices from a sequence.

        Subclasses should define STRIDE_CANDIDATES and STRIDE_WEIGHTS as class attributes,
        or override this method for custom sampling logic.

        Args:
            T_total: Total number of frames in sequence
            rng: numpy random generator

        Returns:
            indices: array of frame indices
        """
        if T_total < self.frame_num:
            # Need replacement
            return np.sort(rng.choice(T_total, self.frame_num, replace=True))

        if self.sampling_mode == 'random':
            # Pure random sampling without replacement
            return np.sort(rng.choice(T_total, self.frame_num, replace=False))

        elif self.sampling_mode == 'stride':
            # Subclasses must define STRIDE_CANDIDATES and STRIDE_WEIGHTS
            if not hasattr(self.__class__, 'STRIDE_CANDIDATES'):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must define STRIDE_CANDIDATES and STRIDE_WEIGHTS "
                    f"class attributes or override _sample_frame_indices method"
                )

            return sample_frame_indices_stride(
                T_total, self.frame_num, rng,
                self.__class__.STRIDE_CANDIDATES,
                self.__class__.STRIDE_WEIGHTS
            )

        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

    def convert_attributes(self):
        """
        Avoid memory leak caused by python list or python dict
        https://github.com/pytorch/pytorch/issues/13246
        """

        def _is_equivalent(original, converted):
            """
            Check if the converted data structure is equivalent to the original.
            """
            try:
                return original == converted
            except Exception:
                return False

        for attr_name in dir(self):
            if attr_name.startswith("__") or callable(getattr(self, attr_name)):
                continue
            
            attr_value = getattr(self, attr_name)
            
            if isinstance(attr_value, list):
                try:
                    converted_value = np.array(attr_value)
                    
                    if _is_equivalent(attr_value, converted_value.tolist()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)
            
            elif isinstance(attr_value, dict):
                try:
                    converted_value = pd.Series(attr_value)
                    if _is_equivalent(attr_value, converted_value.to_dict()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'
        if OmegaConf.is_config(resolutions):
            resolutions = OmegaConf.to_object(resolutions)

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            # assert width >= height
            # self._resolutions.append((width, height))
            self._resolutions.append([width, height])

        self.num_resolutions = len(self._resolutions)

    def _get_clip(self, idx, resolution, rng) -> UnifiedClip:
        """Load and transform data for a sequence. Subclasses must override.

        Should call _crop_resize_if_necessary per frame to apply crop/resize
        and return a UnifiedClip at the target resolution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_clip()"
        )

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info='', normal=None, far_mask=None, trajs_2d=None, visibility=None):
        """Crop/resize a single frame, synchronously transforming tracks and visibility.

        Steps:
            1. Principal-point centered crop (symmetric around optical center)
            2. Optional random focal augmentation (center crop)
            3. Lanczos rescale to target resolution
            4. Final center crop to exact resolution

        Track/visibility transforms are handled by cropping functions.

        Args:
            image: PIL Image or numpy array [H, W, 3]
            depthmap: numpy array [H, W]
            intrinsics: numpy array [3, 3]
            resolution: target (width, height)
            rng: numpy RNG for augmentation
            info: string for error messages
            trajs_2d: numpy array [N, 2] or None
            visibility: numpy array [N] bool or None

        Returns:
            (image, depthmap, intrinsics, trajs_2d, visibility)
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        target_resolution = np.array(resolution)

        # --- Step 1: principal-point centered crop ---
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask, trajs_2d, visibility = cropping.crop_image_depthmap(
            crop_bbox, image, depthmap, intrinsics, normal=normal, far_mask=far_mask, trajs_2d=trajs_2d, visibility=visibility)

        W, H = image.size  # size after principal-point crop

        # --- Step 2: optional focal augmentation (center crop) ---
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * rng.beta(0.5, 0.5) if rng is not None else self.aug_focal
            image, depthmap, intrinsics, normal, far_mask, trajs_2d, visibility = cropping.center_crop_image_depthmap(
                crop_scale, image, depthmap, intrinsics, normal=normal, far_mask=far_mask, trajs_2d=trajs_2d, visibility=visibility)

        # --- Step 3: Lanczos rescale ---
        if self.aug_crop > 1:
            target_resolution = target_resolution + rng.integers(0, self.aug_crop)
            image, depthmap, intrinsics, normal, far_mask, trajs_2d, visibility = cropping.rescale_image_depthmap(
            target_resolution, image, depthmap, intrinsics, normal=normal, far_mask=far_mask, trajs_2d=trajs_2d, visibility=visibility)
        elif image.size[0] < target_resolution[0] or image.size[1] < target_resolution[1]:
            # Image smaller than target after crop — must upscale to avoid negative margins in Step 4
            image, depthmap, intrinsics, normal, far_mask, trajs_2d, visibility = cropping.rescale_image_depthmap(
            target_resolution, image, depthmap, intrinsics, normal=normal, far_mask=far_mask, trajs_2d=trajs_2d, visibility=visibility)

        # --- Step 4: final center crop to exact resolution ---
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask, trajs_2d, visibility = cropping.crop_image_depthmap(
            crop_bbox, image, depthmap, intrinsics, normal=normal, far_mask=far_mask, trajs_2d=trajs_2d, visibility=visibility)

        return image, depthmap, intrinsics2, normal, far_mask, trajs_2d, visibility

    def _process_depth(self, depthmap, intrinsics, camera_pose, label='', frame_id=''):
        """Compute pts3d and valid_mask for a single frame.

        Args:
            depthmap: numpy [H, W]
            intrinsics: numpy [3, 3]
            camera_pose: numpy [4, 4]
            label: dataset/scene label for error messages
            frame_id: frame identifier for error messages

        Returns:
            (pts3d, valid_mask, depthmap) where invalid depths are zeroed.
        """
        assert np.isfinite(depthmap).all(), f'NaN in depthmap for frame {frame_id}'
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
            depthmap=depthmap,
            camera_intrinsics=intrinsics,
            camera_pose=camera_pose,
            z_far=self.z_far,
        )
        valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
        depthmap[~valid_mask] = 0.0
        if valid_mask.sum() == 0:
            raise ValueError(
                f"All pixels invalid in depthmap for frame {frame_id} of {label}"
            )
        return pts3d, valid_mask, depthmap

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            if len(idx) == 3:
                idx, ar_idx, frame_num = idx
                self.frame_num = frame_num
            else:
                idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)

        error = None
        for _ in range(10):
            try:
                clip = self._get_clip(idx, resolution, self._rng)
                T = clip.images.shape[0]
                h, w = clip.images.shape[-2:]  # [T, 3, H, W] -> H, W
                clip.true_shape = np.array([[h, w]] * T, dtype=np.int32)
                clip.z_far = self.z_far
                clip.idx = (idx, ar_idx)

                return clip

            except Exception as e:
                if hasattr(self, 'this_views_info'):
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} ({self.this_views_info}) for error {e}.", flush=True
                    )
                else:
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} for error {e}.", flush=True
                    )
                idx = np.random.randint(0, len(self))
                error = e

        raise error
    
    def load_cache(self, cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True).item()
            # Step 2: 遍历字典中的每个键值对，并将其赋值为类实例的属性
            if isinstance(data, dict):
                for key, value in data.items():
                    setattr(self, key, value)
                return True
            else:
                print("Error: The npy file does not contain a dictionary.")
                return False
        except Exception as e:
            print(f"An error occurred while loading the cache: {e}")
            return False

    def _save_cache(self, keys, desc=None):
        if desc is None:
            save_path = f'data/dataset_cache/{self.dataset_label}_cache.npy'
        else:
            save_path = f'data/dataset_cache/{self.dataset_label}_{desc}_cache.npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {}
        for key in keys:
            save_dict[key] = getattr(self, key)
        
        np.save(save_path, save_dict)

        print(f'Saved cache to {save_path}.', flush=True)

