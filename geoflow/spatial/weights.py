"""
Spatial weights matrix construction for GeoFlow.
Provides KNN, distance-band, and grid-based neighborhood weights with row standardization.
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import scipy.spatial
import geopandas as gpd
import shapely.geometry


class SpatialWeights:
    """Sparse spatial weight matrix wrapper with neighbor lookups."""

    def __init__(self, weights_dict: dict[int, dict[int, float]], n: int):
        self.weights = weights_dict
        self.n = n

    def row_standardize(self) -> SpatialWeights:
        """Standardize weights such that row sums equal 1.0."""
        new_w = {}
        for i, neighbors in self.weights.items():
            s = sum(neighbors.values())
            if s > 0:
                new_w[i] = {j: val / s for j, val in neighbors.items()}
            else:
                new_w[i] = {}
        return SpatialWeights(new_w, self.n)

    def to_dense(self) -> np.ndarray:
        w_mat = np.zeros((self.n, self.n), dtype=np.float64)
        for i, neighbors in self.weights.items():
            for j, val in neighbors.items():
                w_mat[i, j] = val
        return w_mat


def _extract_xy(data: Any) -> np.ndarray:
    if isinstance(data, gpd.GeoDataFrame):
        coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in data.geometry])
        return coords
    elif hasattr(data, "gdf"):
        return _extract_xy(data.gdf)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Cannot extract coordinates from {type(data)}")


def knn(data: Any, k: int = 8) -> SpatialWeights:
    """Construct k-nearest neighbor spatial weights."""
    coords = _extract_xy(data)
    n = len(coords)
    k = min(k, n - 1)
    tree = scipy.spatial.KDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)  # +1 because index 0 is self

    w = {}
    for i in range(n):
        neighbors = {}
        for neighbor_idx in indices[i]:
            if neighbor_idx != i:
                neighbors[int(neighbor_idx)] = 1.0
        w[i] = neighbors

    return SpatialWeights(w, n).row_standardize()


def distance(data: Any, threshold: Optional[float] = None) -> SpatialWeights:
    """Construct distance-band spatial weights."""
    coords = _extract_xy(data)
    n = len(coords)
    tree = scipy.spatial.KDTree(coords)

    if threshold is None:
        # Default to median minimum distance to nearest neighbor
        distances, _ = tree.query(coords, k=2)
        threshold = float(np.median(distances[:, 1]) * 1.5)

    pairs = tree.query_pairs(r=threshold)
    w = {i: {} for i in range(n)}
    for i, j in pairs:
        w[i][j] = 1.0
        w[j][i] = 1.0

    return SpatialWeights(w, n).row_standardize()
