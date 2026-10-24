"""
Spatial Cross-Validation engines (SpatialCV, BufferedSpatialCV).
Prevents spatial autocorrelation data leakage between training and validation splits.
"""

from __future__ import annotations
from typing import Any, Iterator, List, Optional, Tuple, Union
import warnings
import numpy as np
import geopandas as gpd
import shapely.geometry

from geoflow.vector.collection import FeatureCollection
from geoflow.geometry.base import Geometry


def assign_blocks(samples: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> np.ndarray:
    """
    Return one block id per sample (in sample order). Samples are matched by their centroid,
    so each lands in exactly one block; samples outside every block go to the nearest block.
    Blocks are identified by their 'block_id' column, or by row position if it is absent.
    """
    if samples.crs is not None and blocks.crs is not None and samples.crs != blocks.crs:
        blocks = blocks.to_crs(samples.crs)
    ids = blocks["block_id"].to_numpy() if "block_id" in blocks.columns else np.arange(len(blocks))
    blk = gpd.GeoDataFrame({"_gf_block": ids}, geometry=blocks.geometry.to_numpy(), crs=blocks.crs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # centroid in a geographic CRS is fine for block assignment
        centroids = samples.geometry.centroid.to_numpy()
    pts = gpd.GeoDataFrame({"_gf_pos": np.arange(len(samples))}, geometry=centroids, crs=samples.crs)

    joined = gpd.sjoin(pts, blk, how="left", predicate="intersects")
    joined = joined.drop_duplicates("_gf_pos").set_index("_gf_pos").sort_index()
    block_ids = joined["_gf_block"].to_numpy(dtype=float).copy()

    missing = np.isnan(block_ids)
    if missing.any():
        nearest = gpd.sjoin_nearest(pts[missing], blk, how="left")
        nearest = nearest.drop_duplicates("_gf_pos").set_index("_gf_pos")
        block_ids[nearest.index.to_numpy()] = nearest["_gf_block"].to_numpy(dtype=float)
    return block_ids


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
                block_ids = assign_blocks(gdf, blocks_gdf)
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
        if len(unique_blocks) < self.n_splits:
            raise ValueError(
                f"Only {len(unique_blocks)} spatial blocks contain samples, fewer than n_splits={self.n_splits}. "
                "Use smaller blocks or fewer splits."
            )
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
        unit: str = "meters",
    ):
        """
        unit: 'meters' (default) buffers in metres; data in a geographic CRS is projected to its
              local UTM zone for buffering. 'crs' buffers in the data's own CRS units.
        """
        if unit.lower() not in ("meters", "metres", "m", "crs"):
            raise ValueError(f"Unsupported unit {unit!r}; use 'meters' or 'crs'.")
        self.buffer_distance = buffer_distance
        self.n_splits = n_splits
        self.blocks = blocks
        self.random_state = random_state
        self.unit = unit.lower()

    def split(self, data: Any, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields (train_indices, val_indices) where train_indices excludes any sample
        within buffer_distance of any sample in val_indices.
        """
        gdf = data.gdf if hasattr(data, "gdf") else data

        # Base spatial CV (blocks are assigned in the data's own CRS)
        base_cv = SpatialCV(blocks=self.blocks if self.blocks is not None else self.n_splits,
                            n_splits=self.n_splits, random_state=self.random_state)

        # Buffer in metres: project geographic data to its local UTM zone
        geoms = gdf.geometry
        if self.unit != "crs" and gdf.crs is not None and gdf.crs.is_geographic:
            geoms = geoms.to_crs(gdf.estimate_utm_crs())
        geoms = geoms.reset_index(drop=True)

        for train_idx, val_idx in base_cv.split(gdf):
            buffer_zone = geoms.iloc[val_idx].union_all().buffer(self.buffer_distance)
            # Drop training samples intersecting the buffer zone
            keep = ~geoms.iloc[train_idx].intersects(buffer_zone).to_numpy()
            yield np.asarray(train_idx)[keep].astype(int), val_idx
