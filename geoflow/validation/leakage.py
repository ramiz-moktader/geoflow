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
        return self.has_overlap or (self.duplicate_count > 0) or self.buffer_violation

    def summary(self) -> str:
        status = "CRITICAL LEAKAGE DETECTED" if self.has_leakage else "CLEAN: NO LEAKAGE DETECTED"
        report_str = (
            f"==================================================\n"
            f"          SPATIAL ML LEAKAGE DIAGNOSTIC REPORT     \n"
            f"==================================================\n"
            f"Status                 : {status}\n"
            f"Train Sample Count     : {self.train_count}\n"
            f"Test Sample Count      : {self.test_count}\n"
            f"Bounding Box Overlap   : {'YES (Risk of leakage)' if self.has_overlap else 'NO'}\n"
            f"Duplicate Coordinates  : {self.duplicate_count}\n"
            f"Minimum Separation     : {self.min_distance:.2f} units\n"
            f"Buffer Threshold       : {self.buffer_threshold:.2f} units\n"
            f"Buffer Proximity Alert : {'VIOLATION' if self.buffer_violation else 'PASS'}\n"
            f"--------------------------------------------------\n"
        )
        if self.has_leakage:
            report_str += (
                "RECOMMENDATION:\n"
                "Use `geoflow.validation.BufferedSpatialCV` with an exclusion buffer\n"
                "exceeding the range of spatial autocorrelation to eliminate inflated validation metrics.\n"
            )
        else:
            report_str += "RECOMMENDATION:\nValidation partition is isolated. Safe for spatial generalization claims.\n"
        report_str += "=================================================="
        return report_str

    def __repr__(self) -> str:
        return f"<LeakageReport leakage={self.has_leakage} min_dist={self.min_distance:.2f}>"


def leakage_report(
    train: Any,
    test: Any,
    buffer_threshold: float = 0.0,
) -> LeakageReport:
    """
    Audit train and test partitions for spatial leakage.
    """
    def extract_coords_and_bounds(obj):
        if isinstance(obj, gpd.GeoDataFrame):
            coords = np.array([[g.centroid.x, g.centroid.y] for g in obj.geometry])
            bounds = shapely.geometry.box(*obj.total_bounds)
        elif hasattr(obj, "gdf"):
            coords = np.array([[g.centroid.x, g.centroid.y] for g in obj.gdf.geometry])
            bounds = shapely.geometry.box(*obj.gdf.total_bounds)
        elif hasattr(obj, "geometry"):
            coords = np.array([[obj.geometry.centroid.x, obj.geometry.centroid.y]])
            bounds = obj.geometry.shapely
        else:
            raise TypeError(f"Cannot extract spatial geometry from {type(obj)}")
        return coords, bounds

    train_coords, train_box = extract_coords_and_bounds(train)
    test_coords, test_box = extract_coords_and_bounds(test)

    has_overlap = train_box.intersects(test_box)

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
