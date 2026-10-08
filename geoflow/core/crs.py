"""
CRS Management and Coordinates Transformation for GeoFlow.
Ensures coordinate reference systems are never silently mixed or corrupted.
"""

from __future__ import annotations
import warnings
from typing import Tuple, Union
import pyproj
from pyproj import Transformer


class CRS:
    """Wrapper for Coordinate Reference Systems with validation."""

    def __init__(self, crs_input: Union[str, int, pyproj.CRS, None] = "EPSG:4326"):
        if crs_input is None:
            self._crs = pyproj.CRS.from_user_input("EPSG:4326")
        elif isinstance(crs_input, pyproj.CRS):
            self._crs = crs_input
        else:
            self._crs = pyproj.CRS.from_user_input(crs_input)

    @property
    def pyproj_crs(self) -> pyproj.CRS:
        return self._crs

    @property
    def epsg(self) -> int | None:
        return self._crs.to_epsg()

    @property
    def is_geographic(self) -> bool:
        return self._crs.is_geographic

    @property
    def is_projected(self) -> bool:
        return self._crs.is_projected

    def to_string(self) -> str:
        epsg = self.epsg
        if epsg:
            return f"EPSG:{epsg}"
        return self._crs.to_wkt()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (CRS, str, int, pyproj.CRS)):
            return False
        if not isinstance(other, CRS):
            other = CRS(other)
        return self._crs == other._crs

    def __repr__(self) -> str:
        epsg = self.epsg
        if epsg:
            return f"CRS('EPSG:{epsg}')"
        return f"CRS('{self._crs.name}')"


def reproject_bounds(
    bounds: Tuple[float, float, float, float],
    src_crs: Union[str, int, pyproj.CRS, CRS],
    dst_crs: Union[str, int, pyproj.CRS, CRS],
) -> Tuple[float, float, float, float]:
    """
    Reproject a (minx, miny, maxx, maxy) bounding box from src_crs to dst_crs.
    """
    src = src_crs.pyproj_crs if isinstance(src_crs, CRS) else pyproj.CRS.from_user_input(src_crs)
    dst = dst_crs.pyproj_crs if isinstance(dst_crs, CRS) else pyproj.CRS.from_user_input(dst_crs)

    if src == dst:
        return bounds

    transformer = Transformer.from_crs(src, dst, always_xy=True)
    minx, miny, maxx, maxy = bounds
    xs = [minx, maxx, minx, maxx]
    ys = [miny, miny, maxy, maxy]
    tx, ty = transformer.transform(xs, ys)
    return (min(tx), min(ty), max(tx), max(ty))


def check_crs_compatibility(crs1: Union[str, CRS], crs2: Union[str, CRS], context: str = ""):
    """Issue warning if two spatial operations use conflicting CRS."""
    c1 = crs1 if isinstance(crs1, CRS) else CRS(crs1)
    c2 = crs2 if isinstance(crs2, CRS) else CRS(crs2)
    if c1 != c2:
        warnings.warn(
            f"CRS mismatch detected {context}: {c1} vs {c2}. "
            "Data should be reprojected before spatial operations to prevent distortion.",
            UserWarning,
            stacklevel=2,
        )
