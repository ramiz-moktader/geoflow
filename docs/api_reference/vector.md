# Vector & Geometry API Reference (`geoflow.vector` & `geoflow.geometry`)

The vector subsystem wraps Shapely and GeoPandas into Google Earth Engine-styled declarative interfaces with coordinate reference system validation and local file/database I/O.

| Class / Function | Source File | Description |
|---|---|---|
| [`Geometry`](#geometry) | `geoflow/geometry/base.py` | Base geometry with CRS awareness and spatial operations |
| [`Point`, `MultiPoint`](#point-multipoint) | `geoflow/geometry/point.py` | Point geometry constructs |
| [`Polygon`, `MultiPolygon`, `Rectangle`](#polygon-multipolygon-rectangle) | `geoflow/geometry/polygon.py` | Polygonal geometry constructs |
| [`from_geojson`, `from_file`](#geometry-io) | `geoflow/geometry/io.py` | Vector geometry deserializers |
| [`Feature`](#feature) | `geoflow/vector/feature.py` | Pair of `Geometry` and metadata properties dict |
| [`FeatureCollection`](#featurecollection) (alias: `Vector`) | `geoflow/vector/collection.py` | Collection of features backed by a `GeoDataFrame` |

---

## `Geometry`

```python
gf.Geometry(
    shapely_geom: shapely.geometry.base.BaseGeometry,
    crs: str | int | CRS = "EPSG:4326",
    metadata: dict | None = None
)
```

Base class for all vector geometries. Implements Jupyter SVG display protocol (`_repr_svg_`).

### Properties

| Property | Type | Description |
|---|---|---|
| `.crs` | `CRS` | Coordinate Reference System of the geometry. |
| `.shapely` / `.to_shapely()` | `BaseGeometry` | Underlying Shapely geometry instance. |
| `.bounds` | `tuple[float, float, float, float]` | Bounding box `(minx, miny, maxx, maxy)`. |
| `.area` | `float` | Area in CRS units (e.g., $m^2$ for projected, $\text{deg}^2$ for geographic). |
| `.length` | `float` | Perimeter or line length in CRS units. |

### Spatial Methods

| Method | Return Type | Description |
|---|---|---|
| `.centroid()` | `Geometry` | Geometric center point. |
| `.buffer(distance, quad_segs=16)` | `Geometry` | Dilates or erodes geometry by `distance` in CRS units. |
| `.simplify(tolerance, preserve_topology=True)` | `Geometry` | Douglas-Peucker line/polygon simplification. |
| `.convex_hull()` (alias: `convexHull`) | `Geometry` | Smallest convex polygon containing geometry. |
| `.intersection(other)` | `Geometry` | Set intersection with another geometry. |
| `.union(other)` | `Geometry` | Set union with another geometry. |
| `.difference(other)` | `Geometry` | Spatial difference (`self - other`). |
| `.symmetric_difference(other)` | `Geometry` | Spatial symmetric difference (XOR). |
| `.intersects(other)` | `bool` | Tests topological intersection. |
| `.contains(other)` | `bool` | Tests topological containment. |
| `.reproject(dst_crs)` | `Geometry` | Reprojects coordinates to `dst_crs` via pyproj transformer. |
| `.to_geojson()` | `dict` | GeoJSON mapping dictionary. |
| `.to_geodataframe()` | `geopandas.GeoDataFrame` | Wraps geometry into a 1-row GeoDataFrame. |

### Geometry Factory Classmethods

| Factory Method | Signature | Description |
|---|---|---|
| `gf.Geometry.Point(coords, crs="EPSG:4326")` | `Point([lon, lat])` | Constructs a `Point`. |
| `gf.Geometry.Polygon(coords, crs="EPSG:4326")` | `Polygon([[[x, y], ...]])` | Constructs a `Polygon`. |
| `gf.Geometry.MultiPolygon(coords, crs="EPSG:4326")` | `MultiPolygon(...)` | Constructs a `MultiPolygon`. |
| `gf.Geometry.Rectangle(coords, crs="EPSG:4326")` | `Rectangle([minx, miny, maxx, maxy])` | Constructs a bounding rectangle. |
| `gf.Geometry.BBox(minx, miny, maxx, maxy, crs="EPSG:4326")` | `BBox(...)` | Alias for `Rectangle`. |
| `gf.Geometry.from_geojson(data, crs="EPSG:4326")` | `from_geojson(dict | str)` | Parses GeoJSON string or dict. |
| `gf.Geometry.from_file(filepath, crs=None)` | `from_file(path)` | Reads Shapefile, GeoPackage, or GeoJSON file and dissolves to a single union geometry. |
| `gf.Geometry.from_shapely(geom, crs="EPSG:4326")` | `from_shapely(geom)` | Wraps pre-existing Shapely geometry. |

---

## `Point`, `MultiPoint`

```python
gf.Point(coords: list[float] | tuple[float, float], crs="EPSG:4326")
gf.MultiPoint(coords: list[list[float]] | shapely.geometry.MultiPoint, crs="EPSG:4326")
```

- For `Point`, `.x` and `.y` properties expose longitude and latitude directly.

---

## `Polygon`, `MultiPolygon`, `Rectangle`

```python
gf.Polygon(coords, crs="EPSG:4326")
gf.MultiPolygon(coords, crs="EPSG:4326")
gf.Rectangle(coords: list[float] | tuple[float, float, float, float], crs="EPSG:4326")
```

- `Rectangle` accepts `[minx, miny, maxx, maxy]` (Western longitude, Southern latitude, Eastern longitude, Northern latitude).

---

## Geometry I/O

```python
from geoflow.geometry.io import from_geojson, from_file

# Load from file (automatically handles Shapefiles, GeoJSON, FlatGeoBuf, GeoPackage)
aoi = from_file("study_boundary.geojson")

# Load from GeoJSON string or dict
aoi = from_geojson('{"type": "Polygon", "coordinates": [[[91.7, 22.2], [92.0, 22.2], [92.0, 22.5], [91.7, 22.5], [91.7, 22.2]]]}')
```

---

## `Feature`

```python
gf.Feature(
    geometry: Geometry | shapely.geometry.base.BaseGeometry | Any,
    properties: dict[str, Any] | None = None
)
```

Represents a single GIS feature combining a geometry with attribute properties.

| Member | Signature / Type | Description |
|---|---|---|
| `.geometry` | `Geometry` | Geometry of the feature. |
| `.properties` | `dict[str, Any]` | Dictionary of attribute fields. |
| `.bounds` | `tuple` | Extent of feature geometry. |
| `.get(key, default=None)` | `Any` | Read attribute value. |
| `.set(key, value)` | `Feature` | Write attribute value in-place. |
| `.buffer(distance)` | `Feature` | Buffers feature geometry while preserving properties. |
| `.centroid()` | `Feature` | Returns centroid point while preserving properties. |
| `.to_dict()` | `dict` | Serializes to standard GeoJSON Feature mapping dict. |

---

## `FeatureCollection`

```python
gf.FeatureCollection(
    data: geopandas.GeoDataFrame | list[Feature] | str | Path,
    crs: str | int | CRS = "EPSG:4326",
    metadata: dict[str, Any] | None = None
)
```

GEE-compatible container for spatial feature sets. Backed internally by a `geopandas.GeoDataFrame`.

### Properties

| Property | Type | Description |
|---|---|---|
| `.gdf` | `geopandas.GeoDataFrame` | Live underlying GeoDataFrame. |
| `.crs` | `CRS` | Coordinate Reference System. |
| `.columns` | `list[str]` | List of attribute columns (including `'geometry'`). |
| `.bounds` | `tuple[float, float, float, float]` | Total bounding box across all features. |
| `.geometry` | `Geometry` | Unary union of all feature geometries. |
| `.__geo_interface__` | `dict` | Python geo-interface protocol dict. |

### Collection Operations

| Method | Signature | Description |
|---|---|---|
| `.size()` / `len(fc)` | `int` | Number of features in collection. |
| `.first()` | `Feature | None` | Returns first feature or None if empty. |
| `.filter(filter_obj)` | `filter(filter: Filter) -> FeatureCollection` | Evaluates declarative [`Filter`](core.md#filter) against each feature row. |
| `.filterBounds(geometry)` | `filterBounds(geom) -> FeatureCollection` | Spatially filters features that intersect `geometry`. |
| `.select(columns)` | `select(cols: list[str]) -> FeatureCollection` | Subsets attribute columns (preserves `'geometry'`). |
| `.sort(column, ascending=True)` | `sort(...) -> FeatureCollection` | Sorts features by attribute column. |
| `.limit(n)` | `limit(n: int) -> FeatureCollection` | Truncates to first $n$ features. |
| `.buffer(distance)` | `buffer(dist: float) -> FeatureCollection` | Buffers all feature geometries. |
| `.centroid()` | `centroid() -> FeatureCollection` | Replaces geometries with their centroids. |
| `.simplify(tolerance)` | `simplify(tol: float) -> FeatureCollection` | Simplifies geometries. |
| `.dissolve(by=None, aggfunc="first")` | `dissolve(...) -> FeatureCollection` | Dissolves geometries grouped by column. |
| `.spatial_join(other, how="inner", predicate="intersects")` | `spatial_join(...) -> FeatureCollection` | Spatial join with another `FeatureCollection`. |

### Point & Polygon Raster Sampling

```python
fc.sample_raster(image: Image, bands: list[str] | None = None, scale: float | None = None) -> FeatureCollection
```

Samples pixel values from a multi-band `Image` at feature geometry coordinates and appends sampled values as new attribute columns.
- For `Point` features: Samples exact pixel values under coordinates.
- For `Polygon` features: Uses the polygon centroid.
- **CRS Handling**: If `fc.crs` differs from `image.crs`, sample coordinates are dynamically reprojected to the image's coordinate space without reprojecting the entire raster grid.

```python
# Sample Sentinel-2 NDVI and Blue bands at GEDI field footprint points
sampled_fc = gedi_fc.sample_raster(sentinel_img, bands=["B2", "NDVI"])
print(sampled_fc.gdf[["rh98", "NDVI", "B2"]].head())
```

### Attribute Reduction

```python
fc.reduceColumns(reducer: Reducer, selectors: list[str]) -> dict[str, Any]
```

Aggregates specified columns using a [`Reducer`](core.md#reducer).

```python
stats = fc.reduceColumns(gf.Reducer.mean(), selectors=["rh98", "sensitivity"])
```

### Cartography & Data Export

| Method | Description |
|---|---|
| `.plot(column=None, cmap="viridis", ...)` | Visualizes features using GeoPandas / Matplotlib. |
| `.plot_map(**kwargs)` | Publication-grade cartographic map rendering with north arrow, scale bar, and coordinate graticules. |
| `.edit_layout(**kwargs)` | Launches in-notebook interactive ArcGIS-style Layout Editor. |
| `.to_geodataframe()` | Returns a copy of the underlying GeoDataFrame. |
| `.to_dataframe()` / `.to_pandas()` | Returns standard pandas DataFrame. |
| `.to_csv(filepath, index=False)` | Exports attributes to CSV. |
| `.to_parquet(filepath)` | Exports to GeoParquet / Apache Parquet. |
| `.to_file(filepath, driver=None)` | Writes vector dataset to Shapefile, GeoPackage, or GeoJSON. |
| `.to_geojson()` | Returns GeoJSON FeatureCollection dictionary. |
