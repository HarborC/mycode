from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class UnifiedClip:
    """Clip-level data container for all datasets.

    Contains both raw loaded data and post-processed fields
    (pts3d, valid_mask, true_shape, etc.) computed by ``BaseDataset.__getitem__``.

    Fields
    ------
    images : ndarray
        ``[T, H, W, 3]`` uint8 RGB images (from _get_clip),
        or ``[T, 3, H, W]`` float32 tensor (after transform in __getitem__).
    depths : ndarray
        ``[T, H, W]`` float32 depth maps.
    camera_poses : ndarray
        ``[T, 4, 4]`` camera-to-world (c2w) poses.
    intrinsics : ndarray
        ``[T, 3, 3]`` pinhole intrinsics.

    trajs_2d : ndarray | None
        ``[T, N, 2]`` float32 2-D track pixel coordinates.
    trajs_3d_world : ndarray | None
        ``[T, N, 3]`` float32 3-D world-space track positions.
    visibility : ndarray | None
        ``[T, N]`` bool  track visibility mask.
    valids : ndarray | None
        ``[T, N]`` bool  track validity mask (finite coords + positive depth).

    pts3d : ndarray | None
        ``[T, H, W, 3]`` float32 3-D point maps.
        Computed by ``BaseDataset.__getitem__`` from depth + intrinsics + pose.
    valid_mask : ndarray | None
        ``[T, H, W]`` bool valid-pixel masks.
    normal : ndarray | None
        ``[T, H, W, 3]`` float32 surface normals, or None.

    true_shape : ndarray
        ``[T, 2]`` int32 original (height, width) before crop/resize.
    idx : tuple
        ``(dataset_idx, ar_idx, view_idx)`` sample index.
    z_far : float
        Far clipping distance for depth filtering.

    dataset : str
    label : str
    instances : list[str]
    metadata : dict
    """

    images: np.ndarray             # [T, H, W, 3] uint8 or [T, 3, H, W] float32 tensor
    depths: np.ndarray             # [T, H, W] float32
    camera_poses: np.ndarray       # [T, 4, 4] c2w
    intrinsics: np.ndarray         # [T, 3, 3]

    trajs_2d: Optional[np.ndarray] = None
    trajs_3d_world: Optional[np.ndarray] = None
    visibility: Optional[np.ndarray] = None
    valids: Optional[np.ndarray] = None

    pts3d: Optional[np.ndarray] = None      # [T, H, W, 3] float32
    valid_mask: Optional[np.ndarray] = None  # [T, H, W] bool
    normal: Optional[np.ndarray] = None      # [T, H, W, 3] float32

    true_shape: Optional[np.ndarray] = None   # [T, 2]
    idx: Optional[tuple] = None
    z_far: float = 0.0

    dataset: str = ""
    label: str = ""
    instances: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


    def save_as_rrd(self, path: str):
        # todo
        pass