"""
Leakage diagnostic suite for spatial and remote-sensing machine learning.
Detects overlapping spatial extents, duplicate coordinates, and buffer proximity violations.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import numpy as np
import scipy.spatial
import geopandas as gpd
import shapely.geometry


class LeakageReport:
    """Encapsulates spatial data leakage diagnostics between train and test partitions."""

    def __init__(
        self,
        has_overlap: bool,
        duplicate_count: int,
        min_distance: float,
        buffer_violation: bool,
        buffer_threshold: float,
        train_count: int,
        test_count: int,
    ):
        self.has_overlap = has_overlap
        self.duplicate_count = duplicate_count
        self.min_distance = min_distance
        self.buffer_violation = buffer_violation
        self.buffer_threshold = buffer_threshold
        self.train_count = train_count
        self.test_count = test_count

    @property
    def has_leakage(self) -> bool:
        # Hull overlap alone is not leakage: interleaved spatial blocks always overlap at hull level
        return (self.duplicate_count > 0) or self.buffer_violation

    @property
    def proximity_checked(self) -> bool:
        return self.buffer_threshold > 0

    def summary(self) -> str:
        status = "CRITICAL LEAKAGE DETECTED" if self.has_leakage else "CLEAN: NO LEAKAGE DETECTED"
        report_str = (
            f"==================================================\n"
            f"          SPATIAL ML LEAKAGE DIAGNOSTIC REPORT     \n"
            f"==================================================\n"
            f"Status                 : {status}\n"
            f"Train Sample Count     : {self.train_count}\n"
            f"Test Sample Count      : {self.test_count}\n"
            f"Convex Hull Overlap    : {'YES (informational)' if self.has_overlap else 'NO'}\n"
            f"Duplicate Coordinates  : {self.duplicate_count}\n"
            f"Minimum Separation     : {self.min_distance:.2f}\n"
            f"Buffer Threshold       : {self.buffer_threshold:.2f}\n"
            f"Buffer Proximity Alert : {('VIOLATION' if self.buffer_violation else 'PASS') if self.proximity_checked else 'NOT CHECKED'}\n"
            f"--------------------------------------------------\n"
        )
        if self.has_leakage:
            report_str += (
                "RECOMMENDATION:\n"
                "Use `geoflow.validation.BufferedSpatialCV` with an exclusion buffer\n"
                "exceeding the range of spatial autocorrelation to eliminate inflated validation metrics.\n"
            )
        elif not self.proximity_checked:
            report_str += (
                "RECOMMENDATION:\nNo duplicates found, but proximity was not assessed. Set buffer_threshold to\n"
                "the range of spatial autocorrelation (e.g. from geoflow.spatial.variogram) to check separation.\n"
            )
        else:
            report_str += "RECOMMENDATION:\nNo train/test pair is closer than the buffer threshold.\n"
        report_str += "=================================================="
        return report_str

    def __repr__(self) -> str:
        return f"<LeakageReport leakage={self.has_leakage} min_dist={self.min_distance:.2f}>"


def leakage_report(
    train: Any,
    test: Any,
    buffer_threshold: float = 0.0,
    unit: str = "meters",
) -> LeakageReport:
    """
    Audit train and test partitions for spatial leakage.

    Leakage is flagged for duplicate locations, or for train/test pairs closer than
    buffer_threshold. Convex-hull overlap is reported for information only: interleaved
    spatial blocks always overlap at the hull level without leaking.

    unit: 'meters' (default) measures distances in metres, projecting geographic data to its
          local UTM zone; 'crs' uses the data's own CRS units.
    """
    def to_gdf(obj):
        if isinstance(obj, gpd.GeoDataFrame):
            return obj
        if hasattr(obj, "gdf"):
            return obj.gdf
        if hasattr(obj, "geometry"):
            geom = obj.geometry
            crs = getattr(geom, "crs", None)
            return gpd.GeoDataFrame(
                geometry=[geom.shapely if hasattr(geom, "shapely") else geom],
                crs=crs.to_string() if hasattr(crs, "to_string") else crs,
            )
        raise TypeError(f"Cannot extract spatial geometry from {type(obj)}")

    train_gdf, test_gdf = to_gdf(train), to_gdf(test)
    if train_gdf.crs is not None and test_gdf.crs is not None and train_gdf.crs != test_gdf.crs:
        test_gdf = test_gdf.to_crs(train_gdf.crs)
    if unit.lower() != "crs" and train_gdf.crs is not None and train_gdf.crs.is_geographic:
        utm = train_gdf.estimate_utm_crs()
        train_gdf, test_gdf = train_gdf.to_crs(utm), test_gdf.to_crs(utm)

    def coords_and_hull(gdf):
        coords = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])
        if len(coords) >= 3:
            hull = shapely.geometry.MultiPoint(coords).convex_hull
        else:
            hull = gdf.union_all()
        return coords, hull

    train_coords, train_geom = coords_and_hull(train_gdf)
    test_coords, test_geom = coords_and_hull(test_gdf)

    has_overlap = train_geom.intersects(test_geom)

    # Minimum distance between train and test using KDTree
    tree = scipy.spatial.KDTree(train_coords)
    distances, _ = tree.query(test_coords)
    min_dist = float(np.min(distances)) if len(distances) > 0 else float("inf")

    duplicate_count = int(np.sum(distances < 1e-7))
    buffer_violation = bool(buffer_threshold > 0 and min_dist < buffer_threshold)

    return LeakageReport(
        has_overlap=has_overlap,
        duplicate_count=duplicate_count,
        min_distance=min_dist,
        buffer_violation=buffer_violation,
        buffer_threshold=buffer_threshold,
        train_count=len(train_coords),
        test_count=len(test_coords),
    )
