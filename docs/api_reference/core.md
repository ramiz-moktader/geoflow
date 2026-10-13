# Core API Reference (`geoflow.core`)

The core package holds the building blocks shared by every other subsystem.
Everything here is importable from `geoflow.core`; `Filter`, `Reducer`, `CRS`
and `reproject_bounds` are also re-exported at the top level (`gf.Filter`, ...).

| Symbol | Kind | Source |
|---|---|---|
| [`EarthObject`](#earthobject) | base class | `geoflow/core/base.py` |
| [`config`](#config) | singleton | `geoflow/core/config.py` |
| [`CRS`](#crs) | class | `geoflow/core/crs.py` |
| [`reproject_bounds`](#reproject_bounds) | function | `geoflow/core/crs.py` |
| [`check_crs_compatibility`](#check_crs_compatibility) | function | `geoflow/core/crs.py` |
| [`Filter`](#filter) | class | `geoflow/core/filter.py` |
| [`Reducer`](#reducer) | class | `geoflow/core/reducer.py` |

---

## `EarthObject`

```python
EarthObject(metadata: dict | None = None)
```

Base class of `Image`, `ImageCollection`, `Geometry`, `Feature`,
`FeatureCollection` and `GEDI`. It provides a metadata dictionary and a
provenance log.

| Member | Signature | Description |
|---|---|---|
| `metadata` | property → `dict` | The live metadata dict (mutations are visible). |
| `get` | `get(property_name, default=None)` | Read a metadata key. |
| `set` | `set(property_name, value) -> self` | Write a metadata key **in place** and return `self` (unlike GEE, this mutates). |
| `track_action` | `track_action(action_name, params=None)` | Append an entry to the provenance log. |
| `history` | property → `list[dict]` | Copy of the provenance log. Starts with a `"created"` entry (UTC ISO timestamp). |
| `pipe` | `pipe(func, *args, **kwargs)` | Returns `func(self, *args, **kwargs)`, so custom steps can be chained fluently. |

!!! note
    Built-in operations do **not** call `track_action` yet, so `history`
    only contains the creation entry plus anything you add yourself.

```python
img = gf.Image(arr).set("sensor", "S2A")
img.get("sensor")                     # "S2A"
img.pipe(lambda im, k: im * k, 2.0)   # same as img * 2.0
```

---

## `config`

Module-level singleton instance of `Config`. It is created at import time and
**creates `~/.geoflow/cache` as a side effect**.

| Attribute | Default | Used by |
|---|---|---|
| `default_crs` | `"EPSG:4326"` | Informational (CLI `info`). |
| `cache_dir` | `~/.geoflow/cache` | Informational. |
| `download_dir` | `<cwd>/data` | Informational (downloaders take explicit `output_dir`). |
| `backend` | `"auto"` | Informational (`"auto"`, `"cpu"`, `"cuda"`, `"mps"`). |
| `workers` | `4` | Informational. |
| `strict_crs` | `False` | Not enforced yet. |
| `verbose` | `True` | Not used yet. |

| Method | Description |
|---|---|
| `set_cache_dir(path)` | Set and create the cache directory. |
| `set_backend(name)` | Validate and set the backend; raises `ValueError` for unknown names. |

```python
from geoflow.core import config
config.set_backend("cuda")
```

---

## `CRS`

```python
CRS(crs_input: str | int | pyproj.CRS | None = "EPSG:4326")
```

Thin wrapper around `pyproj.CRS`. Accepts anything
`pyproj.CRS.from_user_input` accepts (`"EPSG:32646"`, `32646`, WKT, PROJ strings).
`None` resolves to EPSG:4326.

| Member | Returns | Description |
|---|---|---|
| `pyproj_crs` | `pyproj.CRS` | Underlying object. |
| `epsg` | `int \| None` | EPSG code if one can be determined. |
| `is_geographic` | `bool` | Degrees-based CRS. |
| `is_projected` | `bool` | Projected CRS (usually metres). |
| `to_string()` | `str` | `"EPSG:xxxx"` when possible, otherwise WKT. |
| `==` | `bool` | Compares against `CRS`, `str`, `int` or `pyproj.CRS`. |

!!! warning
    `CRS` defines `__eq__` without `__hash__`, so instances **cannot be used as
    dict keys or in sets**. Use `crs.to_string()` as the key instead.
    `str(crs)` returns the repr (`"CRS('EPSG:4326')"`), not `"EPSG:4326"`.

---

## `reproject_bounds`

```python
reproject_bounds(bounds, src_crs, dst_crs) -> (minx, miny, maxx, maxy)
```

Transforms the four corners of a bounding box and returns their envelope.
It returns the input unchanged when both CRSs are equal. Only corners are
sampled, so very large boxes in curved projections can be slightly
under-estimated.

## `check_crs_compatibility`

```python
check_crs_compatibility(crs1, crs2, context: str = "")
```

Emits a `UserWarning` (it doesn't raise) when two CRSs differ. Used by
`Image` band math and `Geometry` set operations.

---

## `Filter`

```python
Filter(predicate_fn: Callable[[Any], bool], description: str = "Filter")
```

A declarative predicate, modelled on `ee.Filter`. Filters are callables:
`f(item) -> bool`. They are consumed by `ImageCollection.filter`,
`FeatureCollection.filter`, `filterBounds`, `filterDate`, and so on.

### How property values are looked up

For a property `name`, `Filter` looks in this order:

1. `item[name]` if `item` is a `dict`
2. `item.properties[name]` (e.g. `Feature`)
3. `item.metadata[name]` (e.g. `Image`)
4. `getattr(item, name)` (e.g. a pandas row)
5. otherwise `None`

!!! warning "Silent failures"
    `Filter.__call__` catches **every exception** and returns `False`. A typo
    in a property name, or comparing a string to a number, quietly removes all
    items instead of raising. If a filtered collection is unexpectedly empty,
    test the filter against one item by hand.

### Comparison filters

| Classmethod | GEE alias | Matches when |
|---|---|---|
| `eq(name, value)` | `equals` | `value == v` |
| `neq(name, value)` | `notEquals` | `value != v` (also matches when the property is missing) |
| `gt(name, value)` | `greaterThan` | `v > value` (missing → `False`) |
| `gte(name, value)` | `greaterThanOrEquals` | `v >= value` |
| `lt(name, value)` | `lessThan` | `v < value` |
| `lte(name, value)` | `lessThanOrEquals` | `v <= value` |
| `rangeContains(name, lo, hi)` | – | `lo <= v <= hi` |
| `inList(name, values)` | – | `v in values` |
| `notInList(name, values)` | – | `v not in values` |
| `stringContains(name, sub, case_sensitive=True)` | – | substring match (non-strings → `False`) |
| `stringStartsWith(name, prefix)` | – | prefix match |
| `stringEndsWith(name, suffix)` | – | suffix match |

### Temporal filters

| Classmethod | Description |
|---|---|
| `date(start, end)` | Inclusive window. `start`/`end` can be `str` (ISO), `date` or `datetime`. Strings are truncated to the **date part**. |
| `calendarRange(start, end, field="month")` | `field` is `"month"` (1–12) or `"day_of_year"` (1–366). Wraps around the year when `start > end` (e.g. `11, 2` = Nov–Feb). |
| `dayOfYear(start, end)` | Shortcut for `calendarRange(..., field="day_of_year")`. |

The item date is read from the first non-empty of `date`, `datetime`,
`timestamp` or `time_start` (`calendarRange` skips `time_start`).

!!! note
    Items **without** a parseable date **pass** temporal filters (lenient by
    design). Compact strings such as `"2023001T042149"` (HLS) cannot be parsed
    and therefore always pass.

### Spatial filters

| Classmethod | Description |
|---|---|
| `bounds(geometry)` / `intersects(geometry)` | `geometry` can be a `Geometry`, a Shapely geometry or a `[minx, miny, maxx, maxy]` list. Uses `item.geometry` if present, otherwise `box(*item.bounds)`. **No CRS reprojection is done.** Items with no geometry pass. |

### Boolean combinators

| Form | Equivalent |
|---|---|
| `Filter.and_(f1, f2, ...)` | `f1 & f2` |
| `Filter.or_(f1, f2, ...)` | `f1 \| f2` |
| `Filter.not_(f)` | `~f` |

```python
f = (gf.Filter.eq("platform", "Sentinel-2A")
     & gf.Filter.date("2023-01-01", "2023-06-30")
     & ~gf.Filter.gt("cloud_cover", 20))
f({"platform": "Sentinel-2A", "date": "2023-03-15", "cloud_cover": 5})  # True
```

---

## `Reducer`

```python
Reducer(name: str, fn: Callable[[np.ndarray, int | None], Any], params: dict | None = None)
```

Wraps an aggregation function. `reducer.reduce(data, axis=None)` calls
`fn(data, axis)`. Built-in reducers ignore NaN.

| Factory | Output for `axis=None` | Notes |
|---|---|---|
| `Reducer.mean()` | scalar | `np.nanmean` |
| `Reducer.median()` | scalar | `np.nanmedian` |
| `Reducer.min()` / `max()` | scalar | `np.nanmin` / `np.nanmax` |
| `Reducer.sum()` | scalar | `np.nansum` |
| `Reducer.stdDev()` | scalar | population std (`ddof=0`) |
| `Reducer.variance()` | scalar | population variance |
| `Reducer.count()` | int | number of non-NaN values |
| `Reducer.mode()` | scalar | `scipy.stats.mode(nan_policy="omit")` |
| `Reducer.percentile(p)` | array of `len(p)` | `p` is a number or list |
| `Reducer.minMax()` | `(min, max)` tuple | |
| `Reducer.histogram(bins=50)` | `{"counts": [...], "edges": [...]}` | always flattens and ignores `axis` |
| `Reducer.first()` / `last()` | element | no NaN skipping |

!!! warning "Collection compositing"
    `ImageCollection` composites only work with reducers that return **one
    value per pixel** (`mean`, `median`, `min`, `max`, `sum`, `stdDev`,
    `variance`, `count`, `mode`, `first`, `last`). `percentile`, `minMax` and
    `histogram` change the output shape and fail.

### Custom reducers

```python
iqr = gf.Reducer("iqr", lambda a, ax: np.nanpercentile(a, 75, axis=ax) - np.nanpercentile(a, 25, axis=ax))
img.reduceRegion(iqr, geometry=aoi)
```
