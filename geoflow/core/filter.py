"""
GEE-compatible Filter engine for GeoFlow.
Provides rich declarative filtering across metadata, geometries, dates, and properties.
"""

from __future__ import annotations
from typing import Any, Callable, List, Union
import datetime
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry


def _parse_date(value: Any) -> datetime.date | None:
    """
    Parse a date-like value to a calendar date (UTC for timezone-aware inputs).
    Accepts date/datetime/pandas/numpy datetimes and strings in ISO 8601 ('2023-01-31',
    '2023-01-31T10:00:00Z'), compact 'YYYYMMDD', or ordinal 'YYYYDDD' (as in HLS granule
    names, e.g. '2023001T042149'). Returns None if the value cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value).to_pydatetime()
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc)
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if not isinstance(value, str):
        return None

    s = value.strip()
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return _parse_date(dt)
    except ValueError:
        pass
    day = s.split("T")[0].split(" ")[0]
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y%j"):
        if fmt == "%Y%m%d" and len(day) != 8 or fmt == "%Y%j" and len(day) != 7:
            continue
        try:
            return datetime.datetime.strptime(day, fmt).date()
        except ValueError:
            continue
    return None


class Filter:
    """GEE-like Filter class supporting comparison, spatial, temporal, and boolean combinations."""

    def __init__(self, predicate_fn: Callable[[Any], bool], description: str = "Filter"):
        self._predicate = predicate_fn
        self.description = description

    def __call__(self, item: Any) -> bool:
        """Evaluate the filter against an item (dict, Feature, Image, or row)."""
        try:
            return bool(self._predicate(item))
        except Exception:
            return False

    def __and__(self, other: Filter) -> Filter:
        return Filter.and_(self, other)

    def __or__(self, other: Filter) -> Filter:
        return Filter.or_(self, other)

    def __invert__(self) -> Filter:
        return Filter.not_(self)

    def __repr__(self) -> str:
        return f"<Filter: {self.description}>"

    # -------------------------------------------------------------
    # Helper to extract value from item
    # -------------------------------------------------------------
    @staticmethod
    def _extract_val(item: Any, name: str) -> Any:
        if isinstance(item, (dict, pd.Series)):
            # Series rows: index lookup, never attribute access (row.name / row.count are pandas members)
            return item.get(name)
        if hasattr(item, "properties") and isinstance(item.properties, dict):
            return item.properties.get(name)
        if hasattr(item, "metadata") and isinstance(item.metadata, dict):
            return item.metadata.get(name)
        if hasattr(item, name):
            return getattr(item, name)
        return None

    # -------------------------------------------------------------
    # Comparison Filters
    # -------------------------------------------------------------
    @classmethod
    def eq(cls, name: str, value: Any) -> Filter:
        """Filter where property equals value."""
        return cls(
            lambda item: cls._extract_val(item, name) == value,
            description=f"{name} == {value}",
        )

    equals = eq

    @classmethod
    def neq(cls, name: str, value: Any) -> Filter:
        """Filter where property does not equal value."""
        return cls(
            lambda item: cls._extract_val(item, name) != value,
            description=f"{name} != {value}",
        )

    notEquals = neq

    @classmethod
    def gt(cls, name: str, value: Any) -> Filter:
        """Filter where property is strictly greater than value."""
        return cls(
            lambda item: (v := cls._extract_val(item, name)) is not None and v > value,
            description=f"{name} > {value}",
        )

    greaterThan = gt

    @classmethod
    def gte(cls, name: str, value: Any) -> Filter:
        """Filter where property is greater than or equal to value."""
        return cls(
            lambda item: (v := cls._extract_val(item, name)) is not None and v >= value,
            description=f"{name} >= {value}",
        )

    greaterThanOrEquals = gte

    @classmethod
    def lt(cls, name: str, value: Any) -> Filter:
        """Filter where property is strictly less than value."""
        return cls(
            lambda item: (v := cls._extract_val(item, name)) is not None and v < value,
            description=f"{name} < {value}",
        )

    lessThan = lt

    @classmethod
    def lte(cls, name: str, value: Any) -> Filter:
        """Filter where property is less than or equal to value."""
        return cls(
            lambda item: (v := cls._extract_val(item, name)) is not None and v <= value,
            description=f"{name} <= {value}",
        )

    lessThanOrEquals = lte

    # -------------------------------------------------------------
    # Range & Date Filters
    # -------------------------------------------------------------
    @classmethod
    def rangeContains(cls, name: str, min_val: Any, max_val: Any) -> Filter:
        """Filter where property is within [min_val, max_val]."""
        return cls(
            lambda item: (v := cls._extract_val(item, name)) is not None and min_val <= v <= max_val,
            description=f"{min_val} <= {name} <= {max_val}",
        )

    @classmethod
    def date(cls, start: Union[str, datetime.date, datetime.datetime], end: Union[str, datetime.date, datetime.datetime]) -> Filter:
        """Filter items by temporal window [start, end], inclusive, compared at day granularity."""
        dt_start = _parse_date(start)
        dt_end = _parse_date(end)
        if start is not None and dt_start is None or end is not None and dt_end is None:
            raise ValueError(f"Could not parse date window [{start!r}, {end!r}]")

        def _pred(item):
            # check date or timestamp or datetime in item
            raw = (
                cls._extract_val(item, "date")
                or cls._extract_val(item, "datetime")
                or cls._extract_val(item, "timestamp")
                or cls._extract_val(item, "time_start")
            )
            if raw is None:
                return True
            dt_item = _parse_date(raw)
            if dt_item is None:
                return False
            if dt_start and dt_item < dt_start:
                return False
            if dt_end and dt_item > dt_end:
                return False
            return True

        return cls(_pred, description=f"date in [{start}, {end}]")

    @classmethod
    def calendarRange(cls, start: int, end: int, field: str = "month") -> Filter:
        """Filter by calendar unit: 'month' (1-12) or 'day_of_year' (1-366)."""
        def _pred(item):
            raw = (
                cls._extract_val(item, "date")
                or cls._extract_val(item, "datetime")
                or cls._extract_val(item, "timestamp")
            )
            if raw is None:
                return True
            dt = _parse_date(raw)
            if dt is None:
                return False

            val = dt.month if field.lower() == "month" else dt.timetuple().tm_yday
            if start <= end:
                return start <= val <= end
            # wraps around year (e.g. Nov to Feb: 11 to 2)
            return val >= start or val <= end

        return cls(_pred, description=f"{field} in [{start}, {end}]")

    @classmethod
    def dayOfYear(cls, start: int, end: int) -> Filter:
        return cls.calendarRange(start, end, field="day_of_year")

    # -------------------------------------------------------------
    # Spatial Bounds Filters
    # -------------------------------------------------------------
    @classmethod
    def bounds(cls, geometry: Any) -> Filter:
        """Filter items whose geometry intersects the given geometry or bbox."""
        from shapely.geometry import box

        target_geom = None
        if hasattr(geometry, "to_shapely"):
            target_geom = geometry.to_shapely()
        elif isinstance(geometry, BaseGeometry):
            target_geom = geometry
        elif isinstance(geometry, (list, tuple)) and len(geometry) == 4:
            target_geom = box(*geometry)

        def _pred(item):
            item_geom = None
            if hasattr(item, "geometry"):
                g = item.geometry
                item_geom = g.to_shapely() if hasattr(g, "to_shapely") else g
            elif hasattr(item, "bounds"):
                b = item.bounds
                item_geom = box(*b)
            if item_geom is not None and target_geom is not None:
                return item_geom.intersects(target_geom)
            return True

        return cls(_pred, description=f"bounds intersects {geometry}")

    intersects = bounds

    # -------------------------------------------------------------
    # List & String Filters
    # -------------------------------------------------------------
    @classmethod
    def inList(cls, name: str, values: list) -> Filter:
        """Filter where property is contained in list/set."""
        vset = set(values)
        return cls(
            lambda item: cls._extract_val(item, name) in vset,
            description=f"{name} in {values}",
        )

    @classmethod
    def notInList(cls, name: str, values: list) -> Filter:
        """Filter where property is not contained in list/set."""
        vset = set(values)
        return cls(
            lambda item: cls._extract_val(item, name) not in vset,
            description=f"{name} not in {values}",
        )

    @classmethod
    def stringContains(cls, name: str, substring: str, case_sensitive: bool = True) -> Filter:
        """Filter where string property contains substring."""
        def _pred(item):
            val = cls._extract_val(item, name)
            if not isinstance(val, str):
                return False
            return substring in val if case_sensitive else substring.lower() in val.lower()

        return cls(_pred, description=f"{name} contains '{substring}'")

    @classmethod
    def stringStartsWith(cls, name: str, prefix: str) -> Filter:
        return cls(
            lambda item: isinstance(v := cls._extract_val(item, name), str) and v.startswith(prefix),
            description=f"{name} startsWith '{prefix}'",
        )

    @classmethod
    def stringEndsWith(cls, name: str, suffix: str) -> Filter:
        return cls(
            lambda item: isinstance(v := cls._extract_val(item, name), str) and v.endswith(suffix),
            description=f"{name} endsWith '{suffix}'",
        )

    # -------------------------------------------------------------
    # Logical Combinators
    # -------------------------------------------------------------
    @classmethod
    def and_(cls, *filters: Filter) -> Filter:
        """Logical AND across multiple filters."""
        return cls(
            lambda item: all(f(item) for f in filters),
            description=" AND ".join(f.description for f in filters),
        )

    @classmethod
    def or_(cls, *filters: Filter) -> Filter:
        """Logical OR across multiple filters."""
        return cls(
            lambda item: any(f(item) for f in filters),
            description=" OR ".join(f.description for f in filters),
        )

    @classmethod
    def not_(cls, filter_obj: Filter) -> Filter:
        """Logical NOT of a filter."""
        return cls(
            lambda item: not filter_obj(item),
            description=f"NOT ({filter_obj.description})",
        )
