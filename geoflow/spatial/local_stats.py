"""
Local spatial autocorrelation statistics (LISA: Local Moran's I, Getis-Ord Local Gi*).
"""

from __future__ import annotations
from typing import Any, Optional, Union
import numpy as np
import scipy.stats
import geopandas as gpd

from geoflow.spatial.weights import SpatialWeights, knn


def local_morans_i(
    data: Any,
    value: Union[str, np.ndarray],
    weights: Optional[SpatialWeights] = None,
    k: int = 8,
    p_threshold: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Compute Local Indicators of Spatial Association (LISA / Local Moran's I).
    Returns GeoDataFrame with columns ['value', 'I', 'z_score', 'p_value', 'cluster'] (HH, HL, LH, LL, not significant).
    """
    if isinstance(data, gpd.GeoDataFrame):
        base_gdf = data.copy()
        y = data[value].to_numpy().astype(np.float64) if isinstance(value, str) else np.asarray(value)
        val_name = value if isinstance(value, str) else "value"
    elif hasattr(data, "gdf"):
        base_gdf = data.gdf.copy()
        y = base_gdf[value].to_numpy().astype(np.float64) if isinstance(value, str) else np.asarray(value)
        val_name = value if isinstance(value, str) else "value"
    else:
        raise TypeError("local_morans_i expects GeoDataFrame or FeatureCollection")

    n = len(y)
    if weights is None:
        weights = knn(base_gdf, k=k)

    w_mat = weights.to_dense()

    # Standardized deviations
    y_mean = np.mean(y)
    z = y - y_mean
    s2 = np.sum(z**2) / n
    if s2 == 0:
        base_gdf["I"] = 0.0
        base_gdf["z_score"] = 0.0
        base_gdf["p_value"] = 1.0
        base_gdf["cluster"] = "not significant"
        return base_gdf

    # Spatial lag: W * z
    spatial_lag = np.dot(w_mat, z)

    # Local Moran's I_i: (z_i / s^2) * sum_j w_ij * z_j
    I_local = (z / s2) * spatial_lag

    # Analytical expectation and variance for local Moran
    expected_local = -w_mat.sum(axis=1) / (n - 1)

    # Permutation-based or analytical z-score
    # Analytical approximation of variance
    w_sq = np.sum(w_mat**2, axis=1)
    b2 = (np.sum(z**4) / n) / (s2**2)
    var_local = ((n - b2) / (n - 1)) * w_sq - (expected_local**2)
    var_local = np.maximum(var_local, 1e-10)

    z_scores = (I_local - expected_local) / np.sqrt(var_local)
    p_values = 2.0 * (1.0 - scipy.stats.norm.cdf(np.abs(z_scores)))

    # Cluster classification: HH, LL, HL, LH, not significant
    clusters = []
    for i in range(n):
        if p_values[i] > p_threshold:
            clusters.append("not significant")
        else:
            if z[i] > 0 and spatial_lag[i] > 0:
                clusters.append("HH")
            elif z[i] < 0 and spatial_lag[i] < 0:
                clusters.append("LL")
            elif z[i] > 0 and spatial_lag[i] < 0:
                clusters.append("HL")
            else:
                clusters.append("LH")

    base_gdf[f"{val_name}_std"] = z
    base_gdf["I"] = I_local
    base_gdf["z_score"] = z_scores
    base_gdf["p_value"] = p_values
    base_gdf["cluster"] = clusters

    return base_gdf
