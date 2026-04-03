from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class UnifiedClip:
    dataset_name: str
    sequence_name: str
    frame_paths: Optional[list[str]]
    images: list[np.ndarray]                  # [T][H,W,3]
    depths: Optional[list[np.ndarray]]        # [T][H,W]
    normals: Optional[list[np.ndarray]]       # [T][H,W,3]
    trajs_2d: Optional[np.ndarray]            # [T,N,2]
    trajs_3d_world: Optional[np.ndarray]      # [T,N,3]
    valids: Optional[np.ndarray]              # [T,N]
    visibs: Optional[np.ndarray]              # [T,N]
    intrinsics: np.ndarray                    # [T,3,3]
    extrinsics: np.ndarray                    # [T,4,4]
    flows: Optional[list[np.ndarray]] = None  # [T][H,W,2], optical flow
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.images)

    @property
    def image_size(self) -> tuple[int, int]:
        h, w = self.images[0].shape[:2]
        return h, w
