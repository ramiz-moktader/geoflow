"""
Spatial model evaluation and prediction diagnostics.
Computes RMSE, MAE, R², and residual Moran's I to test for spatial bias.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Union
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from geoflow.spatial.autocorrelation import morans_i, MoranResult


class SpatialEvalReport:
    """Contains regression metrics and spatial residual autocorrelation diagnostics."""

    def __init__(
        self,
        rmse: float,
        mae: float,
        r2: float,
        residuals: np.ndarray,
        moran: Optional[MoranResult] = None,
        geometry: Any = None,
    ):
        self.rmse = float(rmse)
        self.mae = float(mae)
        self.r2 = float(r2)
        self.residuals = residuals
        self.moran = moran
        self.geometry = geometry

    def summary(self) -> str:
        rep = (
            f"=== SPATIAL PREDICTION EVALUATION REPORT ===\n"
            f"RMSE                  : {self.rmse:.4f}\n"
            f"MAE                   : {self.mae:.4f}\n"
            f"R² Score              : {self.r2:.4f}\n"
        )
        if self.moran:
            sig = "SIGNIFICANT (Spatial Bias Alert)" if self.moran.p_value < 0.05 else "Not Significant (White Noise)"
            rep += (
                f"Residual Moran's I    : {self.moran.I:.4f}\n"
                f"Residuals z-score     : {self.moran.z_score:.2f}\n"
                f"Residuals p-value     : {self.moran.p_value:.4e} ({sig})\n"
                f"Interpretation        : "
            )
            if self.moran.p_value < 0.05 and self.moran.I > 0:
                rep += "Residuals exhibit spatial clustering. The model suffers from omitted spatial covariates.\n"
            else:
                rep += "Residuals are spatially random. Model errors do not show spatial dependence.\n"
        rep += "============================================"
        return rep

    def residual_map(self, ax: Any = None):
        """Plot residuals spatially."""
        if self.geometry is None:
            raise ValueError("No geometry provided for residual map")

        gdf = gpd.GeoDataFrame({"residual": self.residuals, "geometry": self.geometry})
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        gdf.plot(
            column="residual",
            cmap="coolwarm",
            legend=True,
            ax=ax,
            markersize=25,
            edgecolor="none",
        )
        ax.set_title("Spatial Prediction Residuals (y_true - y_pred)")
        return ax

    error_map = residual_map

    def __repr__(self) -> str:
        return f"<SpatialEvalReport RMSE={self.rmse:.3f} R2={self.r2:.3f}>"


def evaluate_spatial(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    geometry: Optional[Any] = None,
    k: int = 8,
) -> SpatialEvalReport:
    """
    Evaluate spatial prediction with standard metrics and residual Moran's I.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    residuals = yt - yp
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yt - np.mean(yt))**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    moran = None
    geoms = None
    if geometry is not None:
        if isinstance(geometry, gpd.GeoDataFrame):
            geoms = geometry.geometry
        elif hasattr(geometry, "gdf"):
            geoms = geometry.gdf.geometry
        else:
            geoms = geometry

        gdf = gpd.GeoDataFrame({"residual": residuals, "geometry": list(geoms)})
        moran = morans_i(gdf, value="residual", k=k)

    return SpatialEvalReport(rmse, mae, r2, residuals, moran=moran, geometry=geoms)
