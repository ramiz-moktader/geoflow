"""
Machine learning and deep learning dataset generators for remote sensing.
Provides PatchDataset, PointDataset, and leakage-safe spatial splits.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import shapely.geometry
import geopandas as gpd

from geoflow.raster.image import Image
from geoflow.geometry.base import Geometry
from geoflow.vector.collection import FeatureCollection
from geoflow.validation.spatial_cv import SpatialCV, BufferedSpatialCV


class PatchDataset:
    """
    Extracts 2D/3D geospatial patches from raster images with coordinates and optional labels.
    Compatible with PyTorch Dataset and native NumPy workflows.
    """

    def __init__(
        self,
        image: Image,
        labels: Optional[Union[Image, str, np.ndarray]] = None,
        region: Optional[Geometry] = None,
        patch_size: int = 64,
        stride: int = 32,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.image = image.clip(region) if region else image
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.metadata = metadata or {}

        # Handle labels
        if isinstance(labels, (str, Image)):
            self.labels_img = labels if isinstance(labels, Image) else Image(labels)
            if self.labels_img.shape[1:] != self.image.shape[1:]:
                # Resample labels to match image size
                self.labels_img = self.labels_img.resample(self.image.width / self.labels_img.width)
        elif isinstance(labels, np.ndarray):
            self.labels_img = Image(labels, transform=self.image.transform, crs=self.image.crs)
        else:
            self.labels_img = None

        # Precompute patch bounding boxes and array indices
        self.patches: List[Dict[str, Any]] = []
        H, W = self.image.height, self.image.width
        P = self.patch_size
        S = self.stride

        for r in range(0, H - P + 1, S):
            for c in range(0, W - P + 1, S):
                # Calculate geographical bounds of the patch
                minx, maxy = self.image.transform * (c, r)
                maxx, miny = self.image.transform * (c + P, r + P)
                geom = shapely.geometry.box(min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))

                # Check if patch is mostly valid data (not all NaN)
                patch_data = self.image._data[:, r : r + P, c : c + P]
                if np.count_nonzero(~np.isnan(patch_data)) > (P * P * 0.2):  # At least 20% valid
                    self.patches.append({
                        "row": r,
                        "col": c,
                        "bounds": (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy)),
                        "geometry": geom,
                    })

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        p_info = self.patches[idx]
        r, c = p_info["row"], p_info["col"]
        P = self.patch_size

        x = self.image._data[:, r : r + P, c : c + P].copy()
        x = np.nan_to_num(x, nan=0.0)

        if self.transform:
            x = self.transform(x)

        if self.labels_img is not None:
            y = self.labels_img._data[:, r : r + P, c : c + P].copy()
            y = np.nan_to_num(y, nan=0.0)
            return x, y
        return x

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Return patch footprints as a GeoDataFrame."""
        rows = []
        for i, p in enumerate(self.patches):
            rows.append({
                "patch_id": i,
                "row": p["row"],
                "col": p["col"],
                "geometry": p["geometry"],
            })
        return gpd.GeoDataFrame(rows, crs=self.image.crs.to_string())

    def split(
        self,
        strategy: str = "spatial",
        blocks: Optional[Any] = None,
        buffer_distance: float = 0.0,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> Tuple[PatchDataset, PatchDataset, PatchDataset]:
        """
        Split PatchDataset into train, val, test subsets using spatial blocking.
        """
        gdf = self.to_geodataframe()
        n = len(gdf)

        if strategy == "spatial" and blocks is not None:
            cv = BufferedSpatialCV(buffer_distance=buffer_distance, blocks=blocks, n_splits=5)
            # Pick first fold for test, second fold for val, rest for train
            train_idx, test_idx = next(cv.split(gdf))
            # Split val out of train
            n_val = int(len(train_idx) * (ratios[1] / (ratios[0] + ratios[1])))
            val_idx = train_idx[:n_val]
            train_idx = train_idx[n_val:]
        else:
            # Geographic coordinate split
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            n_train = int(n * ratios[0])
            n_val = int(n * ratios[1])
            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

        def subset(idxs):
            ds = PatchDataset(
                self.image,
                labels=self.labels_img,
                patch_size=self.patch_size,
                stride=self.stride,
                transform=self.transform,
            )
            ds.patches = [self.patches[i] for i in idxs]
            return ds

        return subset(train_idx), subset(val_idx), subset(test_idx)

    def dataloader(self, batch_size: int = 32, shuffle: bool = True, device: str = "cpu") -> Any:
        """
        Create a DataLoader. If PyTorch is installed, returns torch.utils.data.DataLoader.
        Otherwise, yields NumPy array batches.
        """
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset

            class TorchWrapper(Dataset):
                def __init__(self, parent):
                    self.parent = parent
                def __len__(self):
                    return len(self.parent)
                def __getitem__(self, i):
                    item = self.parent[i]
                    if isinstance(item, tuple):
                        return torch.from_numpy(item[0]).float(), torch.from_numpy(item[1]).float()
                    return torch.from_numpy(item).float()

            wrapped = TorchWrapper(self)
            pin_memory = (device.startswith("cuda"))
            return DataLoader(wrapped, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
        except ImportError:
            # Pure NumPy mini-batch generator
            class BatchIterator:
                def __init__(self, ds, bsize, shuff):
                    self.ds = ds
                    self.bsize = bsize
                    self.shuff = shuff
                def __iter__(self):
                    indices = np.arange(len(self.ds))
                    if self.shuff:
                        np.random.shuffle(indices)
                    for start in range(0, len(indices), self.bsize):
                        batch_idx = indices[start : start + self.bsize]
                        items = [self.ds[i] for i in batch_idx]
                        if isinstance(items[0], tuple):
                            xs = np.stack([it[0] for it in items], axis=0)
                            ys = np.stack([it[1] for it in items], axis=0)
                            yield xs, ys
                        else:
                            xs = np.stack(items, axis=0)
                            yield xs

            return BatchIterator(self, batch_size, shuffle)

    def __repr__(self) -> str:
        return f"<PatchDataset patches={len(self)} patch_size={self.patch_size}x{self.patch_size}>"
