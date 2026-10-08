"""
Semivariogram and spatial dependence modeling for GeoFlow.
"""

from __future__ import annotations
from typing import Any, Optional, Union
import numpy as np
import scipy.spatial
import geopandas as gpd
import matplotlib.pyplot as plt


class VariogramResult:
    """Empirical semivariogram container and plotter."""

    def __init__(self, lag_distances: np.ndarray, semivariances: np.ndarray, npairs: np.ndarray):
        self.lag_distances = lag_distances
        self.semivariances = semivariances
        self.npairs = npairs

    @property
    def nugget(self) -> float:
        """Estimated nugget variance."""
        return float(self.semivariances[0]) if len(self.semivariances) > 0 else 0.0

    @property
    def sill(self) -> float:
        """Estimated sill (plateau)."""
        return float(np.max(self.semivariances)) if len(self.semivariances) > 0 else 0.0

    def plot(self, ax: Any = None, title: str = "Experimental Semivariogram"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(self.lag_distances, self.semivariances, color="#2563eb", s=40, zorder=3)
        ax.plot(self.lag_distances, self.semivariances, color="#60a5fa", linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Lag Distance (h)")
        ax.set_ylabel("Semivariance γ(h)")
        ax.grid(True, linestyle=":", alpha=0.6)
        return ax

    def __repr__(self) -> str:
        return f"<VariogramResult lags={len(self.lag_distances)} sill={self.sill:.3f}>"


def variogram(
    data: Any,
    value: Union[str, np.ndarray],
    n_lags: int = 15,
    max_dist: Optional[float] = None,
) -> VariogramResult:
    """
    Compute experimental semivariogram for spatial point observations.
    """
    if isinstance(data, gpd.GeoDataFrame):
        coords = np.array([[g.centroid.x, g.centroid.y] for g in data.geometry])
        y = data[value].to_numpy().astype(np.float64) if isinstance(value, str) else np.asarray(value)
    elif hasattr(data, "gdf"):
        coords = np.array([[g.centroid.x, g.centroid.y] for g in data.gdf.geometry])
        y = data.gdf[value].to_numpy().astype(np.float64) if isinstance(value, str) else np.asarray(value)
    else:
        raise TypeError("variogram expects GeoDataFrame or FeatureCollection")

    # Pairwise distances and squared differences
    pdist = scipy.spatial.distance.pdist(coords)
    n = len(y)
    diff_sq = []
    for i in range(n):
        for j in range(i + 1, n):
            diff_sq.append(0.5 * ((y[i] - y[j])**2))
    diff_sq = np.array(diff_sq)

    max_d = max_dist or (np.max(pdist) / 2.0)
    valid_mask = pdist <= max_d
    pdist = pdist[valid_mask]
    diff_sq = diff_sq[valid_mask]

    bins = np.linspace(0, max_d, n_lags + 1)
    lag_dist = []
    semivars = []
    npairs = []

    for k in range(n_lags):
        in_bin = (pdist >= bins[k]) & (pdist < bins[k + 1])
        if np.any(in_bin):
            lag_dist.append(float(np.mean(pdist[in_bin])))
            semivars.append(float(np.mean(diff_sq[in_bin])))
            npairs.append(int(np.sum(in_bin)))

    return VariogramResult(np.array(lag_dist), np.array(semivars), np.array(npairs))
