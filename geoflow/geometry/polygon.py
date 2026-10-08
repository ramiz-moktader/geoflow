"""
Polygon, MultiPolygon, and Rectangle geometry implementations for GeoFlow.
"""

from __future__ import annotations
from typing import List, Tuple, Union
import shapely.geometry
from geoflow.geometry.base import Geometry
from geoflow.core.crs import CRS


class Polygon(Geometry):
    """GEE-like Polygon geometry."""

    def __init__(self, coords: Any, crs: Union[str, int, CRS] = "EPSG:4326"):
        if isinstance(coords, shapely.geometry.Polygon):
            geom = coords
        elif isinstance(coords, (list, tuple)):
            # If coordinates are nested like [[[x,y], [x,y], ...]]
            if len(coords) > 0 and isinstance(coords[0], (list, tuple)) and isinstance(coords[0][0], (list, tuple)):
                shell = coords[0]
                holes = coords[1:] if len(coords) > 1 else None
                geom = shapely.geometry.Polygon(shell, holes)
            else:
                geom = shapely.geometry.Polygon(coords)
        else:
            raise TypeError(f"Cannot construct Polygon from {type(coords)}")
        super().__init__(geom, crs=crs)


class MultiPolygon(Geometry):
    """GEE-like MultiPolygon geometry."""

    def __init__(self, coords: Any, crs: Union[str, int, CRS] = "EPSG:4326"):
        if isinstance(coords, shapely.geometry.MultiPolygon):
            geom = coords
        elif isinstance(coords, (list, tuple)):
            polys = []
            for p in coords:
                if isinstance(p, Polygon):
                    polys.append(p.shapely)
                elif isinstance(p, shapely.geometry.Polygon):
                    polys.append(p)
                else:
                    polys.append(shapely.geometry.Polygon(p))
            geom = shapely.geometry.MultiPolygon(polys)
        else:
            raise TypeError(f"Cannot construct MultiPolygon from {type(coords)}")
        super().__init__(geom, crs=crs)


class Rectangle(Polygon):
    """GEE-like Rectangle geometry: [minx, miny, maxx, maxy]."""

    def __init__(self, coords: Union[List[float], Tuple[float, float, float, float]], crs: Union[str, int, CRS] = "EPSG:4326"):
        if len(coords) != 4:
            raise ValueError("Rectangle coords must be [minx, miny, maxx, maxy]")
        minx, miny, maxx, maxy = coords
        geom = shapely.geometry.box(minx, miny, maxx, maxy)
        super().__init__(geom, crs=crs)
