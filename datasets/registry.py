"""
Dataset adapter registry for D4RT.

Maintains a mapping from dataset name to adapter class.
Allows instantiation of adapters from configuration.

Usage:
    adapter = create_adapter('pointodyssey', root='/path/to/data', split='train')

    # Or get all available datasets
    names = list_datasets()
"""

from __future__ import annotations

from typing import Type

from src.datasets.base import BaseDataset
from src.datasets.blendedmvs import BlendedMVSDataset
from src.datasets.co3dv2 import Co3Dv2Dataset
from src.datasets.dynamic_replica import DynamicReplicaDataset
from src.datasets.kubric import KubricDataset
from src.datasets.mvssynth import MVSSynthDataset
from src.datasets.pointodyssey import PointOdysseyDataset
from src.datasets.scannet import ScanNetDataset
from src.datasets.tartanair import TartanAirDataset
from src.datasets.virtual_kitti import VKITTI2Dataset


# Dataset name -> Dataset class mapping
DATASET_REGISTRY: dict[str, Type[BaseDataset]] = {
    "pointodyssey": PointOdysseyDataset,
    "scannet": ScanNetDataset,
    "co3dv2": Co3Dv2Dataset,
    "kubric": KubricDataset,
    "blendedmvs": BlendedMVSDataset,
    "mvssynth": MVSSynthDataset,
    "dynamic_replica": DynamicReplicaDataset,
    "tartanair": TartanAirDataset,
    "vkitti2": VKITTI2Dataset,
}

# Lazy-loaded datasets (requires special dependencies)
LAZY_ADAPTERS = {
    "waymo": ("src.datasets.waymo", "WaymoDataset"),  # requires tensorflow
}


def register_dataset(name: str, adapter_class: Type[BaseDataset]) -> None:
    """Register a new dataset."""
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' already registered")
    DATASET_REGISTRY[name] = adapter_class


def _load_lazy_adapter(name: str) -> Type[BaseDataset]:
    """Load a lazy dataset on demand."""
    if name not in LAZY_ADAPTERS:
        return None

    module_path, class_name = LAZY_ADAPTERS[name]
    try:
        import importlib
        module = importlib.import_module(module_path)
        adapter_class = getattr(module, class_name)
        # Cache it for future use
        DATASET_REGISTRY[name] = adapter_class
        return adapter_class
    except Exception as e:
        raise ImportError(
            f"Failed to load adapter '{name}' from {module_path}.{class_name}: {e}"
        )


def create_adapter(name: str, **kwargs) -> BaseDataset:
    """
    Create an adapter instance by name.

    Args:
        name: Dataset name (e.g., 'pointodyssey', 'scannet')
        **kwargs: Arguments passed to adapter constructor

    Returns:
        Dataset instance

    Example:
        adapter = create_adapter('pointodyssey', root='/data/pointodyssey', split='train')  # noqa: E501
    """
    # Check if already loaded
    if name in DATASET_REGISTRY:
        adapter_class = DATASET_REGISTRY[name]
        return adapter_class(**kwargs)

    # Try lazy loading
    if name in LAZY_ADAPTERS:
        adapter_class = _load_lazy_adapter(name)
        return adapter_class(**kwargs)

    # Not found
    available = ", ".join(list(DATASET_REGISTRY.keys()) + list(LAZY_ADAPTERS.keys()))
    raise ValueError(f"Unknown dataset '{name}'. Available: {available}")


def list_datasets() -> list[str]:
    """Get list of registered dataset names."""
    return list(DATASET_REGISTRY.keys()) + list(LAZY_ADAPTERS.keys())


def get_adapter_class(name: str) -> Type[BaseDataset]:
    """Get adapter class by name."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]

    if name in LAZY_ADAPTERS:
        return _load_lazy_adapter(name)

    available = ", ".join(list(DATASET_REGISTRY.keys()) + list(LAZY_ADAPTERS.keys()))
    raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
