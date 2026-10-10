"""
Base Geometry class for GeoFlow, wrapping Shapely geometries with CRS management.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
import json
import shapely
from shapely.geometry.base import BaseGeometry
import geopandas as gpd

from geoflow.core.base import EarthObject
from geoflow.core.crs import CRS, check_crs_compatibility, reproject_bounds


class Geometry(EarthObject):
    """GEE-like Geometry object wrapping Shapely geometries with strict CRS support."""

    def __init__(self, shapely_geom: BaseGeometry, crs: Union[str, int, CRS] = "EPSG:4326", metadata: dict | None = None):
        super().__init__(metadata=metadata)
        if not isinstance(shapely_geom, BaseGeometry):
            raise TypeError(f"Expected Shapely geometry, got {type(shapely_geom)}")
        self._geom = shapely_geom
        self._crs = crs if isinstance(crs, CRS) else CRS(crs)

    @property
    def crs(self) -> CRS:
        return self._crs

    def __call__(self) -> Geometry:
        """Allow calling geometry as a method (e.g. img.geometry() or ee.Image.geometry())."""
        return self

    @property
    def shapely(self) -> BaseGeometry:
        return self._geom

    def to_shapely(self) -> BaseGeometry:
        return self._geom

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns (minx, miny, maxx, maxy)."""
        return self._geom.bounds

    @property
    def area(self) -> float:
        """Area in CRS units (e.g., m^2 for projected or deg^2 for geographic)."""
        return self._geom.area

    @property
    def length(self) -> float:
        """Length / perimeter in CRS units."""
        return self._geom.length

    def centroid(self) -> Geometry:
        return Geometry(self._geom.centroid, crs=self._crs)

    def buffer(self, distance: float, quad_segs: int = 16) -> Geometry:
        """Buffer geometry by distance."""
        res = self._geom.buffer(distance, quad_segs=quad_segs)
        return Geometry(res, crs=self._crs)

    def simplify(self, tolerance: float, preserve_topology: bool = True) -> Geometry:
        res = self._geom.simplify(tolerance, preserve_topology=preserve_topology)
        return Geometry(res, crs=self._crs)

    def convex_hull(self) -> Geometry:
        return Geometry(self._geom.convex_hull, crs=self._crs)

    convexHull = convex_hull

    def intersection(self, other: Union[Geometry, BaseGeometry]) -> Geometry:
        other_geom, other_crs = self._parse_other(other)
        check_crs_compatibility(self._crs, other_crs, "in intersection")
        res = self._geom.intersection(other_geom)
        return Geometry(res, crs=self._crs)

    def union(self, other: Union[Geometry, BaseGeometry]) -> Geometry:
        other_geom, other_crs = self._parse_other(other)
        check_crs_compatibility(self._crs, other_crs, "in union")
        res = self._geom.union(other_geom)
        return Geometry(res, crs=self._crs)

    def difference(self, other: Union[Geometry, BaseGeometry]) -> Geometry:
        other_geom, other_crs = self._parse_other(other)
        check_crs_compatibility(self._crs, other_crs, "in difference")
        res = self._geom.difference(other_geom)
        return Geometry(res, crs=self._crs)

    def symmetric_difference(self, other: Union[Geometry, BaseGeometry]) -> Geometry:
        other_geom, other_crs = self._parse_other(other)
        check_crs_compatibility(self._crs, other_crs, "in symmetricDifference")
        res = self._geom.symmetric_difference(other_geom)
        return Geometry(res, crs=self._crs)

    symmetricDifference = symmetric_difference

    def intersects(self, other: Union[Geometry, BaseGeometry]) -> bool:
        other_geom, _ = self._parse_other(other)
        return self._geom.intersects(other_geom)

    def contains(self, other: Union[Geometry, BaseGeometry]) -> bool:
        other_geom, _ = self._parse_other(other)
        return self._geom.contains(other_geom)

    def reproject(self, dst_crs: Union[str, int, CRS]) -> Geometry:
        """Reproject geometry to destination CRS."""
        dst = dst_crs if isinstance(dst_crs, CRS) else CRS(dst_crs)
        if self._crs == dst:
            return self
        from pyproj import Transformer
        from shapely.ops import transform
        transformer = Transformer.from_crs(self._crs.pyproj_crs, dst.pyproj_crs, always_xy=True)
        reprojected_geom = transform(transformer.transform, self._geom)
        return Geometry(reprojected_geom, crs=dst)

    def to_geojson(self) -> Dict[str, Any]:
        return shapely.geometry.mapping(self._geom)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame([{"geometry": self._geom}], crs=self._crs.to_string())

    def _parse_other(self, other: Any) -> Tuple[BaseGeometry, CRS]:
        if isinstance(other, Geometry):
            return other.shapely, other.crs
        if isinstance(other, BaseGeometry):
            return other, self._crs
        raise TypeError(f"Unsupported geometry type: {type(other)}")

    def _repr_svg_(self) -> str:
        return self._geom._repr_svg_()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Geometry):
            return False
        return self._crs == other._crs and bool(self._geom.equals(other._geom))

    def __repr__(self) -> str:
        return f"<Geometry {self._geom.geom_type} (CRS: {self._crs})>"

    # -------------------------------------------------------------
    # GEE Factory Methods
    # -------------------------------------------------------------
    @classmethod
    def Point(cls, coords: Union[List[float], Tuple[float, float]], crs: Union[str, int, CRS] = "EPSG:4326") -> Geometry:
        from geoflow.geometry.point import Point
        return Point(coords, crs=crs)

    @classmethod
    def Polygon(cls, coords: Any, crs: Union[str, int, CRS] = "EPSG:4326") -> Geometry:
        from geoflow.geometry.polygon import Polygon
        return Polygon(coords, crs=crs)

    @classmethod
    def MultiPolygon(cls, coords: Any, crs: Union[str, int, CRS] = "EPSG:4326") -> Geometry:
        from geoflow.geometry.polygon import MultiPolygon
        return MultiPolygon(coords, crs=crs)

    @classmethod
    def Rectangle(cls, coords: Union[List[float], Tuple[float, float, float, float]], crs: Union[str, int, CRS] = "EPSG:4326") -> Geometry:
        from geoflow.geometry.polygon import Rectangle
        return Rectangle(coords, crs=crs)

    @classmethod
    def BBox(cls, minx: float, miny: float, maxx: float, maxy: float, crs: Union[str, int, CRS] = "EPSG:4326") -> Geometry:
        from geoflow.geometry.polygon import Rectangle
        return Rectangle([minx, miny, maxx, maxy], crs=crs)

    @classmethod
    def from_geojson(cls, data: Union[str, dict], crs: Union[str, int, CRS] = "EPSG:4326") -> Geometry:
        from geoflow.geometry.io import from_geojson
        return from_geojson(data, crs=crs)

    @classmethod
    def from_file(cls, filepath: str, crs: Union[str, int, CRS, None] = None) -> Geometry:
        from geoflow.geometry.io import from_file
        return from_file(filepath, crs=crs)

    @classmethod
    def from_shapely(cls, geom: BaseGeometry, crs: Union[str, int, CRS] = "EPSG:4326") -> Geometry:
        return cls(geom, crs=crs)
