"""
Global spatial autocorrelation statistics: Moran's I, Geary's C, Getis-Ord General G.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Union
import numpy as np
import scipy.stats
import geopandas as gpd

from geoflow.spatial.weights import SpatialWeights, knn


class MoranResult:
    """Encapsulates Global Moran's I statistic and hypothesis testing."""

    def __init__(self, I: float, expected: float, variance: float, z_score: float, p_value: float, n: int):
        self.I = float(I)
        self.expected = float(expected)
        self.variance = float(variance)
        self.z_score = float(z_score)
        self.p_value = float(p_value)
        self.n = n

    def summary(self) -> str:
        sig = "Significant" if self.p_value < 0.05 else "Not Significant"
        interp = "Positive autocorrelation (Clustering)" if self.I > self.expected else "Negative autocorrelation (Dispersion)"
        return (
            f"--- Global Moran's I Statistics ---\n"
            f"Observed I  : {self.I:.4f}\n"
            f"Expected I  : {self.expected:.4f}\n"
            f"Variance    : {self.variance:.6f}\n"
            f"z-score     : {self.z_score:.4f}\n"
            f"p-value     : {self.p_value:.6e} ({sig} at alpha=0.05)\n"
            f"Conclusion  : {interp}\n"
            f"Sample Size : {self.n}"
        )

    def __repr__(self) -> str:
        return f"<MoranResult I={self.I:.4f} z={self.z_score:.2f} p={self.p_value:.4e}>"


def _values(data: Any, value: Union[str, np.ndarray]) -> np.ndarray:
    """Extract the observation vector, rejecting NaN (weights are built over all rows)."""
    if isinstance(value, str):
        gdf = data if isinstance(data, gpd.GeoDataFrame) else data.gdf
        y = gdf[value].to_numpy(dtype=np.float64)
    else:
        y = np.asarray(value, dtype=np.float64)
    if np.isnan(y).any():
        raise ValueError(
            f"{int(np.isnan(y).sum())} missing values in '{value if isinstance(value, str) else 'value'}'. "
            "Drop those rows (e.g. gdf.dropna(subset=[...])) before computing spatial statistics."
        )
    return y


def morans_i(
    data: Any,
    value: Union[str, np.ndarray],
    weights: Optional[SpatialWeights] = None,
    k: int = 8,
) -> MoranResult:
    """
    Compute Global Moran's I for spatial autocorrelation (analytical inference under randomization).
    """
    y = _values(data, value)

    n = len(y)
    if n < 4:
        raise ValueError("Moran's I inference requires at least 4 spatial observations")

    if weights is None:
        weights = knn(data, k=k)

    # Standardize values: z = y - mean(y)
    y_mean = np.mean(y)
    z = y - y_mean
    ss = np.sum(z**2)
    if ss == 0:
        return MoranResult(0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0, n)

    w_mat = weights.to_sparse()
    s0 = w_mat.sum()
    if s0 == 0:
        return MoranResult(0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0, n)

    # Compute numerator: sum_i sum_j w_ij * z_i * z_j
    numerator = float(z @ (w_mat @ z))
    I = (n / s0) * (numerator / ss)

    # Analytical Expectation and Variance under randomization
    expected = -1.0 / (n - 1)

    s1 = 0.5 * (w_mat + w_mat.T).power(2).sum()
    s2 = np.sum((np.asarray(w_mat.sum(axis=1)).ravel() + np.asarray(w_mat.sum(axis=0)).ravel())**2)
    m4 = np.sum(z**4) / n
    m2 = ss / n
    b2 = m4 / (m2**2)  # Kurtosis

    var_top = n * ((n**2 - 3*n + 3)*s1 - n*s2 + 3*(s0**2)) - b2 * ((n**2 - n)*s1 - 2*n*s2 + 6*(s0**2))
    var_bottom = (n - 1) * (n - 2) * (n - 3) * (s0**2)
    variance = max(1e-10, (var_top / var_bottom) - (expected**2))

    z_score = (I - expected) / np.sqrt(variance)
    p_value = 2.0 * (1.0 - scipy.stats.norm.cdf(abs(z_score)))

    return MoranResult(I, expected, variance, z_score, p_value, n)


def gearys_c(data: Any, value: Union[str, np.ndarray], weights: Optional[SpatialWeights] = None, k: int = 8) -> float:
    """Compute Geary's C statistic."""
    y = _values(data, value)

    n = len(y)
    if weights is None:
        weights = knn(data, k=k)
    w_coo = weights.to_sparse().tocoo()
    s0 = w_coo.sum()

    z = y - np.mean(y)
    ss = np.sum(z**2)
    if s0 == 0 or ss == 0:
        return 1.0

    weighted_diff = np.sum(w_coo.data * (y[w_coo.row] - y[w_coo.col])**2)
    C = ((n - 1) / (2 * s0)) * (weighted_diff / ss)
    return float(C)
