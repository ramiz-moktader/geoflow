"""
Base class for all GeoFlow geospatial and Earth observation objects.
Tracks provenance, CRS, operations, and metadata.
"""

from __future__ import annotations
from typing import Any, Dict, List
import datetime


class EarthObject:
    """Base class for GeoFlow objects (Image, ImageCollection, Geometry, FeatureCollection)."""

    def __init__(self, metadata: Dict[str, Any] | None = None):
        self._metadata = metadata or {}
        self._provenance: List[Dict[str, Any]] = [
            {
                "action": "created",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ]

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def get(self, property_name: str, default: Any = None) -> Any:
        return self._metadata.get(property_name, default)

    def set(self, property_name: str, value: Any) -> EarthObject:
        self._metadata[property_name] = value
        return self

    def track_action(self, action_name: str, params: Dict[str, Any] | None = None):
        self._provenance.append({
            "action": action_name,
            "params": params or {},
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._provenance)

    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Apply a function to self in a fluent pipeline (matching Pandas / Xarray idioms).
        Example:
            result = img.pipe(custom_filter, threshold=0.5).pipe(calc_metrics)
        """
        return func(self, *args, **kwargs)
