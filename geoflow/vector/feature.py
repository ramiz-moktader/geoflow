"""
GEE-compatible Feature implementation for GeoFlow.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Union
import shapely.geometry
from geoflow.core.base import EarthObject
from geoflow.geometry.base import Geometry
from geoflow.core.crs import CRS


class Feature(EarthObject):
    """GEE-like Feature combining a Geometry and property dictionary."""

    def __init__(self, geometry: Union[Geometry, shapely.geometry.base.BaseGeometry, Any], properties: Optional[Dict[str, Any]] = None):
        super().__init__(metadata=properties)
        if isinstance(geometry, Geometry):
            self._geom = geometry
        else:
            self._geom = Geometry(geometry)
        self.properties = dict(properties or {})

    @property
    def geometry(self) -> Geometry:
        return self._geom

    @property
    def bounds(self):
        return self._geom.bounds

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def set(self, key: str, value: Any) -> Feature:
        self.properties[key] = value
        return self

    def buffer(self, distance: float) -> Feature:
        return Feature(self._geom.buffer(distance), properties=dict(self.properties))

    def centroid(self) -> Feature:
        return Feature(self._geom.centroid(), properties=dict(self.properties))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": self._geom.to_geojson(),
            "properties": dict(self.properties),
        }

    def __repr__(self) -> str:
        return f"<geoflow.Feature geom={self._geom.shapely.geom_type} props={list(self.properties.keys())}>"
