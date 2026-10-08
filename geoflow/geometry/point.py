"""
Point and MultiPoint geometry implementations for GeoFlow.
"""

from __future__ import annotations
from typing import List, Tuple, Union
import shapely.geometry
from geoflow.geometry.base import Geometry
from geoflow.core.crs import CRS


class Point(Geometry):
    """GEE-like Point geometry."""

    def __init__(self, coords: Union[List[float], Tuple[float, float]], crs: Union[str, int, CRS] = "EPSG:4326"):
        if isinstance(coords, shapely.geometry.Point):
            geom = coords
        else:
            if len(coords) < 2:
                raise ValueError("Point coordinates must have at least 2 values: [x, y]")
            geom = shapely.geometry.Point(coords[0], coords[1])
        super().__init__(geom, crs=crs)

    @property
    def x(self) -> float:
        return self._geom.x

    @property
    def y(self) -> float:
        return self._geom.y


class MultiPoint(Geometry):
    """GEE-like MultiPoint geometry."""

    def __init__(self, coords: Union[List[List[float]], shapely.geometry.MultiPoint], crs: Union[str, int, CRS] = "EPSG:4326"):
        if isinstance(coords, shapely.geometry.MultiPoint):
            geom = coords
        else:
            geom = shapely.geometry.MultiPoint(coords)
        super().__init__(geom, crs=crs)
