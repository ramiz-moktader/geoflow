"""
Spatial Cross-Validation engines (SpatialCV, BufferedSpatialCV).
Prevents spatial autocorrelation data leakage between training and validation splits.
"""

from __future__ import annotations
from typing import Any, Iterator, List, Optional, Tuple, Union
import numpy as np
import geopandas as gpd
import shapely.geometry

from geoflow.vector.collection import FeatureCollection
from geoflow.geometry.base import Geometry


class SpatialCV:
    """
    Spatially blocked cross-validator.
    Ensures observations located within the same spatial block are never split across train and test.
    """

    def __init__(self, blocks: Union[FeatureCollection, gpd.GeoDataFrame, int] = 5, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.blocks = blocks
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data: Any, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices for training and validation splits."""
        rng = np.random.RandomState(self.random_state)

        # Extract coordinates and assign blocks
        if isinstance(data, (gpd.GeoDataFrame, FeatureCollection)):
            gdf = data.gdf if hasattr(data, "gdf") else data
            n_samples = len(gdf)

            if isinstance(self.blocks, (FeatureCollection, gpd.GeoDataFrame)):
                blocks_gdf = self.blocks.gdf if hasattr(self.blocks, "gdf") else self.blocks
                # Spatial join to find which block each sample falls into
                joined = gpd.sjoin(gdf, blocks_gdf, how="left", predicate="intersects")
                # Handle possible multiple intersections by picking first
                joined = joined[~joined.index.duplicated(keep="first")]
                block_ids = joined["block_id"].fillna(0).to_numpy()
            else:
                # Assign to k spatial clusters using KMeans on coordinates
                from sklearn.cluster import KMeans
                coords = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])
                k = self.blocks if isinstance(self.blocks, int) else self.n_splits
                block_ids = KMeans(n_clusters=k, random_state=self.random_state, n_init=10).fit_predict(coords)
        else:
            n_samples = len(data)
            block_ids = np.arange(n_samples) % self.n_splits

        unique_blocks = np.unique(block_ids)
        if self.shuffle:
            rng.shuffle(unique_blocks)

        folds = np.array_split(unique_blocks, self.n_splits)

        for fold_blocks in folds:
            test_mask = np.isin(block_ids, fold_blocks)
            train_mask = ~test_mask

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(test_mask)[0]
            yield train_indices, val_indices

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits


class BufferedSpatialCV:
    """
    Buffered Spatial Cross-Validator.
    Enforces a strict spatial exclusion buffer around test blocks to eradicate
    spatial autocorrelation leakage from nearby training samples.
    """

    def __init__(
        self,
        buffer_distance: float = 1000.0,
        n_splits: int = 5,
        blocks: Optional[Union[FeatureCollection, gpd.GeoDataFrame]] = None,
        random_state: int = 42,
    ):
        self.buffer_distance = buffer_distance
        self.n_splits = n_splits
        self.blocks = blocks
        self.random_state = random_state

    def split(self, data: Any, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields (train_indices, val_indices) where train_indices excludes any sample
        within buffer_distance of any sample in val_indices.
        """
        gdf = data.gdf if hasattr(data, "gdf") else data
        n_samples = len(gdf)

        # Base spatial CV
        base_cv = SpatialCV(blocks=self.blocks or self.n_splits, n_splits=self.n_splits, random_state=self.random_state)
        sample_geoms = list(gdf.geometry)

        for train_idx, val_idx in base_cv.split(gdf):
            # Create union of validation geometries and buffer it
            val_union = shapely.ops.unary_union([sample_geoms[i] for i in val_idx])
            buffer_zone = val_union.buffer(self.buffer_distance)

            # Drop training samples intersecting the buffer zone
            leak_free_train = []
            for idx in train_idx:
                if not sample_geoms[idx].intersects(buffer_zone):
                    leak_free_train.append(idx)

            yield np.array(leak_free_train, dtype=int), val_idx
