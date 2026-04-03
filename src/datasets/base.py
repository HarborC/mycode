from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.datasets.types import UnifiedClip


def load_precomputed_fast(
    npz_path: Path,
    frame_indices: list[int],
) -> Optional[dict]:
    """
    Load precomputed tracks/normals for specific frame indices.

    Prefers .h5 (chunked HDF5, O(frames) random access) over .npz
    (requires full zlib decompression of the entire array).
    Falls back to .npz if .h5 is not found.

    Returns a dict with arrays already indexed to frame_indices order,
    or None if neither .h5 nor .npz exists.

    Run computer/convert_precomputed_to_h5.py once to generate .h5 files.
    """
    npz_path = Path(npz_path)
    h5_path = npz_path.with_suffix('.h5')

    if h5_path.exists():
        import h5py
        # h5py fancy indexing requires sorted unique indices
        sorted_idx = sorted(set(frame_indices))
        idx_map = {v: i for i, v in enumerate(sorted_idx)}
        reorder = [idx_map[i] for i in frame_indices]
        needs_reorder = reorder != list(range(len(frame_indices)))

        result: dict = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                ds = f[key]
                if ds.ndim >= 1 and ds.shape[0] > 1:
                    data = ds[sorted_idx]       # reads only the needed chunks
                    if needs_reorder:
                        data = data[reorder]
                    result[key] = data
                else:
                    result[key] = ds[()]        # scalar / metadata
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


class BaseDataset(ABC):
    dataset_name: str = "base"

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def list_sequences(self) -> list[str]:
        pass

    @abstractmethod
    def get_sequence_name(self, index: int) -> str:
        pass

    @abstractmethod
    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        pass

    def get_num_frames(self, sequence_name: str) -> int:
        """Return number of frames for sequence_name without expensive I/O.

        Subclasses should override this if they can return num_frames directly
        from their in-memory index without loading annotation files.
        Default falls back to get_sequence_info (may be slow for some datasets).
        """
        return self.get_sequence_info(sequence_name)['num_frames']

    @abstractmethod
    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        pass

    @abstractmethod
    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        pass
