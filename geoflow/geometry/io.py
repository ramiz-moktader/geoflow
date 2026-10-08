"""
Geometry I/O utilities: reading from GeoJSON, Shapefiles, GeoPackages, and bounding boxes.
"""

from __future__ import annotations
from typing import Any, Union
import json
import shapely.geometry
import geopandas as gpd

from geoflow.geometry.base import Geometry
from geoflow.core.crs import CRS


def from_geojson(data: Union[str, dict], crs: Union[str, int, CRS] = "EPSG:4326") -> Geometry:
    """Load Geometry from GeoJSON string or dictionary."""
    if isinstance(data, str):
        data = json.loads(data)

    if data.get("type") == "FeatureCollection":
        geoms = [shapely.geometry.shape(f["geometry"]) for f in data["features"]]
        geom = shapely.ops.unary_union(geoms)
    elif data.get("type") == "Feature":
        geom = shapely.geometry.shape(data["geometry"])
    else:
        geom = shapely.geometry.shape(data)

    return Geometry(geom, crs=crs)


def from_file(filepath: str, crs: Union[str, int, CRS, None] = None) -> Geometry:
    """Load Geometry from vector file (GeoJSON, Shapefile, GeoPackage, etc.)."""
    gdf = gpd.read_file(filepath)
    file_crs = gdf.crs if gdf.crs else "EPSG:4326"
    target_crs = crs if crs is not None else file_crs
    union_geom = gdf.unary_union
    geom_obj = Geometry(union_geom, crs=file_crs)
    if crs is not None and geom_obj.crs != CRS(target_crs):
        geom_obj = geom_obj.reproject(target_crs)
    return geom_obj
