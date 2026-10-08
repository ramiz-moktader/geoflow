"""
GEE-compatible Reducer engine for GeoFlow.
Provides statistical, temporal, and spatial reduction operations over arrays and collections.
"""

from __future__ import annotations
from typing import Any, Callable, List, Optional
import numpy as np


class Reducer:
    """GEE-like Reducer for aggregating arrays, image collections, and spatial regions."""

    def __init__(self, name: str, fn: Callable[[np.ndarray, Optional[int]], Any], params: dict | None = None):
        self.name = name
        self._fn = fn
        self.params = params or {}

    def reduce(self, data: np.ndarray, axis: int | None = None) -> Any:
        """Apply reduction to a numpy array along specified axis."""
        # Handle nan-safe reduction
        return self._fn(data, axis)

    def __repr__(self) -> str:
        return f"<Reducer: {self.name}>"

    # -------------------------------------------------------------
    # Standard Reducers
    # -------------------------------------------------------------
    @classmethod
    def mean(cls) -> Reducer:
        return cls("mean", lambda a, ax: np.nanmean(a, axis=ax))

    @classmethod
    def median(cls) -> Reducer:
        return cls("median", lambda a, ax: np.nanmedian(a, axis=ax))

    @classmethod
    def min(cls) -> Reducer:
        return cls("min", lambda a, ax: np.nanmin(a, axis=ax))

    @classmethod
    def max(cls) -> Reducer:
        return cls("max", lambda a, ax: np.nanmax(a, axis=ax))

    @classmethod
    def sum(cls) -> Reducer:
        return cls("sum", lambda a, ax: np.nansum(a, axis=ax))

    @classmethod
    def stdDev(cls) -> Reducer:
        return cls("stdDev", lambda a, ax: np.nanstd(a, axis=ax))

    @classmethod
    def variance(cls) -> Reducer:
        return cls("variance", lambda a, ax: np.nanvar(a, axis=ax))

    @classmethod
    def count(cls) -> Reducer:
        return cls("count", lambda a, ax: np.count_nonzero(~np.isnan(a), axis=ax))

    @classmethod
    def mode(cls) -> Reducer:
        from scipy import stats
        def _mode_fn(a, ax):
            m = stats.mode(a, axis=ax, nan_policy="omit", keepdims=False)
            return m.mode
        return cls("mode", _mode_fn)

    @classmethod
    def percentile(cls, percentiles: Union[int, float, List[float]]) -> Reducer:
        p = [percentiles] if isinstance(percentiles, (int, float)) else percentiles
        return cls(
            f"percentile_{p}",
            lambda a, ax: np.nanpercentile(a, p, axis=ax),
            params={"percentiles": p},
        )

    @classmethod
    def minMax(cls) -> Reducer:
        return cls(
            "minMax",
            lambda a, ax: (np.nanmin(a, axis=ax), np.nanmax(a, axis=ax)),
        )

    @classmethod
    def histogram(cls, bins: int = 50) -> Reducer:
        def _hist(a, ax):
            valid = a[~np.isnan(a)]
            counts, edges = np.histogram(valid, bins=bins)
            return {"counts": counts.tolist(), "edges": edges.tolist()}
        return cls("histogram", _hist, params={"bins": bins})

    @classmethod
    def first(cls) -> Reducer:
        return cls("first", lambda a, ax: np.take(a, 0, axis=ax) if ax is not None else a.flat[0])

    @classmethod
    def last(cls) -> Reducer:
        return cls("last", lambda a, ax: np.take(a, -1, axis=ax) if ax is not None else a.flat[-1])
