# Core API Reference

The `geoflow.core` module establishes the fundamental abstractions mapping to GEE paradigms.

## `gf.Filter`

Declarative filters for data querying.

- `gf.Filter.eq(key, value)`
- `gf.Filter.neq(key, value)`
- `gf.Filter.gt(key, value)`, `gf.Filter.gte(key, value)`
- `gf.Filter.lt(key, value)`, `gf.Filter.lte(key, value)`
- `gf.Filter.date(start, end)`: Filters by temporal range.
- `gf.Filter.bounds(geometry)`: Filters by spatial intersection.

## `gf.Reducer`

Declarative aggregations.

- `gf.Reducer.mean()`
- `gf.Reducer.median()`
- `gf.Reducer.mode()`
- `gf.Reducer.min()`
- `gf.Reducer.max()`
- `gf.Reducer.sum()`
- `gf.Reducer.stdDev()`
- `gf.Reducer.percentile(percentiles)`
- `gf.Reducer.histogram()`

## `gf.CRS`

Coordinate Reference System abstraction.

- `gf.CRS(crs_string)`: Initialize from EPSG code or PROJ string.
- `.is_geographic`: Boolean indicating if the CRS uses lat/long degrees.
- `.is_projected`: Boolean indicating if the CRS uses projected meters.
