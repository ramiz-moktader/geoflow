"""
Spatial blocking engine for leakage-safe cross-validation and sampling.
Generates square, grid, and hexagonal blocks across areas of interest.
"""

from __future__ import annotations
from typing import Any, List, Union
import math
import shapely.geometry
import geopandas as gpd

from geoflow.geometry.base import Geometry
from geoflow.vector.collection import FeatureCollection
from geoflow.core.crs import CRS


def _utm_epsg_for_lon_lat(lon: float, lat: float) -> int:
    """Determine best UTM EPSG code for a longitude and latitude."""
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def block(
    aoi: Union[Geometry, Any],
    size: float = 5000.0,
    unit: str = "meters",
    shape: str = "square",
) -> FeatureCollection:
    """
    Generate spatial blocking polygons across an AOI.
    shape: 'square', 'grid', or 'hexagonal'
    size: block dimension (e.g. 5000 for 5km)
    """
    geom_obj = aoi if isinstance(aoi, Geometry) else Geometry(aoi)
    orig_crs = geom_obj.crs

    # If unit is meters and CRS is geographic, project to local UTM
    if unit.lower() in ("meters", "m") and orig_crs.is_geographic:
        minx, miny, maxx, maxy = geom_obj.bounds
        center_lon, center_lat = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        utm_epsg = _utm_epsg_for_lon_lat(center_lon, center_lat)
        proj_geom = geom_obj.reproject(f"EPSG:{utm_epsg}")
        working_geom = proj_geom
        working_crs = proj_geom.crs
    else:
        working_geom = geom_obj
        working_crs = orig_crs

    minx, miny, maxx, maxy = working_geom.bounds

    blocks = []
    block_id = 0

    if shape.lower() in ("square", "grid"):
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                poly = shapely.geometry.box(x, y, x + size, y + size)
                if poly.intersects(working_geom.shapely):
                    blocks.append({
                        "block_id": block_id,
                        "geometry": poly,
                    })
                    block_id += 1
                y += size
            x += size

    elif shape.lower() == "hexagonal":
        # Pointy-top hexagons of width `size` (flat-to-flat): circumradius R = size / sqrt(3),
        # centres sqrt(3)*R apart within a row, rows 1.5*R apart, odd rows offset by sqrt(3)/2*R.
        radius = size / math.sqrt(3)
        col_step = math.sqrt(3) * radius
        row_step = 1.5 * radius
        angles = [math.radians(a) for a in range(30, 390, 60)]

        row = 0
        y = miny - row_step
        while y < maxy + row_step:
            x_offset = (col_step / 2.0) if (row % 2 == 1) else 0.0
            x = minx - col_step + x_offset
            while x < maxx + col_step:
                hex_pts = [(x + radius * math.cos(a), y + radius * math.sin(a)) for a in angles]
                poly = shapely.geometry.Polygon(hex_pts)
                if poly.intersects(working_geom.shapely):
                    blocks.append({
                        "block_id": block_id,
                        "geometry": poly,
                    })
                    block_id += 1
                x += col_step
            y += row_step
            row += 1
    else:
        raise ValueError(f"Unsupported block shape: {shape}. Choose 'square', 'grid', or 'hexagonal'.")

    gdf = gpd.GeoDataFrame(blocks, crs=working_crs.to_string())
    # Reproject back to original CRS if projected
    if gdf.crs != orig_crs.pyproj_crs:
        gdf = gdf.to_crs(orig_crs.to_string())

    return FeatureCollection(gdf, crs=orig_crs)
