# Raster API Reference (`geoflow.raster`)

The raster subsystem provides Google Earth Engine-compatible `Image` (or `Raster`) and `ImageCollection` (or `RasterCollection`) objects that execute natively in Python on NumPy arrays and rasterio transforms, with first-class `xarray` and `rioxarray` integration.

| Class / Alias | Source File | Description |
|---|---|---|
| [`Image`](#image) (alias: `Raster`) | `geoflow/raster/image.py` | Multi-band geospatial raster representation |
| [`ImageCollection`](#imagecollection) (alias: `RasterCollection`) | `geoflow/raster/collection.py` | Spatiotemporal collection of `Image` granules |

---

## `Image`

```python
gf.Image(
    source: str | Path | np.ndarray | Image,
    transform: Affine | None = None,
    crs: str | int | CRS = "EPSG:4326",
    band_names: list[str] | None = None,
    nodata: float | None = None,
    metadata: dict[str, Any] | None = None,
)
```

Wraps multi-band raster data. Inherits from [`EarthObject`](core.md#earthobject).

### Constructors & Factories

| Method | Signature | Description |
|---|---|---|
| `open` / `from_file` | `open(filepath, band_names=None, **kwargs) -> Image` | Open a single raster file (GeoTIFF, etc.) via rasterio. |
| `from_files` | `from_files(filepaths: list[str | Path], band_names=None) -> Image` | Stack multiple single-band rasters into a single multi-band `Image`. Auto-detects band names from file suffixes (e.g., `...B04.tif` -> `"B04"`). |
| `from_xarray` | `from_xarray(da: xarray.DataArray) -> Image` | Reconstructs a GeoFlow `Image` from an xarray DataArray, extracting spatial transform and CRS from `.rio` or coordinates. |

### Properties

| Property | Type | Description |
|---|---|---|
| `.data` | `np.ndarray` | Raw 3D NumPy array of shape `(bands, height, width)` with `float32` dtype. |
| `.shape` | `tuple[int, int, int]` | Dimensions `(bands, height, width)`. |
| `.bands` | `list[str]` | List of band names (e.g. `["B2", "B3", "B4", "B8"]`). |
| `.count` | `int` | Number of spectral bands. |
| `.height` | `int` | Height in pixels (rows). |
| `.width` | `int` | Width in pixels (columns). |
| `.transform` | `rasterio.Affine` | Affine geotransform matrix mapping pixel coordinates to CRS space. |
| `.crs` | `CRS` | Coordinate Reference System wrapper. |
| `.nodata` | `float` | Nodata value (defaults to `np.nan` if unset). |
| `.bounds` | `tuple[float, float, float, float]` | Spatial extent as `(minx, miny, maxx, maxy)`. |
| `.geometry` | `Geometry` | Bounding rectangle geometry in the image's CRS. |

### Band Selection & Renaming

| Method | Description |
|---|---|
| `.select(bands, new_names=None)` | Select one or more bands by string name or integer 0-index. Returns a new `Image`. Optional `new_names` renames the selected bands. |
| `.rename(new_names)` | Renames all bands in the current image. Length of `new_names` must match `.count`. |

```python
b4_b8 = img.select(["B4", "B8"], new_names=["Red", "NIR"])
```

### Band Math & Arithmetic

Arithmetic operations can be performed using native Python operators or method equivalents:

| Method | Operator | Description |
|---|---|---|
| `.add(other)` | `+` | Pixel-wise addition with a scalar or another `Image`. |
| `.subtract(other)` | `-` | Pixel-wise subtraction. |
| `.multiply(other)` | `*` | Pixel-wise multiplication. |
| `.divide(other)` | `/` | Safe pixel-wise division (divide-by-zero produces `np.nan` without crashing). |
| `.pow(power)` | `**` | Exponentiation. |
| `.abs()` | – | Absolute value. |
| `.sqrt()` | – | Square root (negative values produce `np.nan`). |
| `.log()`, `.log10()` | – | Natural logarithm and Base-10 logarithm. |
| `.exp()` | – | Exponential function ($e^x$). |
| `.sin()`, `.cos()`, `.tan()` | – | Trigonometric functions. |
| `.floor()`, `.ceil()`, `.round()` | – | Rounding functions. |
| `__neg__()` | `-img` | Negates array values. |

#### Comparison Operators (Thresholding Masks)

| Expression | Operator | Output |
|---|---|---|
| `img > threshold` | `__gt__` | 1.0 where true, 0.0 where false |
| `img >= threshold` | `__ge__` (alias: `__gte__`) | 1.0 where true, 0.0 where false |
| `img < threshold` | `__lt__` | 1.0 where true, 0.0 where false |
| `img <= threshold` | `__le__` (alias: `__lte__`) | 1.0 where true, 0.0 where false |
| `img == other` | `__eq__` | 1.0 where true, 0.0 where false |
| `img != other` | `__ne__` | 1.0 where true, 0.0 where false |

### Expressions

```python
img.expression(expr_str: str, band_dict: dict[str, Image | float] | None = None) -> Image
```

Evaluates a mathematical string expression with band variables and mathematical functions (`sin`, `cos`, `tan`, `log`, `exp`, `sqrt`, `abs`).

```python
# Variables from dictionary or automatically inferred from image band names
custom_index = img.expression(
    "(2.5 * (NIR - RED)) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)",
    {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")}
)
```

### Spectral Indices

Built-in indices include intelligent auto-detection of common band aliases (e.g., `"B8"`, `"B8A"`, `"B08"`, `"NIR"`, `"B4"`, `"B04"`, `"RED"`, `"B3"`, `"B03"`, `"GREEN"`):

| Method | Formula | Band Alias Resolution |
|---|---|---|
| `.normalizedDifference([b1, b2])` | $(b_1 - b_2) / (b_1 + b_2)$ | Required list of 2 band names |
| `.ndvi(nir=None, red=None)` | Normalized Difference Vegetation Index | Auto-detects NIR (`B8`, `B8A`, `B08`, `NIR`) / Red (`B4`, `B04`, `RED`) |
| `.ndwi(green=None, nir=None)` | Normalized Difference Water Index | Auto-detects Green (`B3`, `B03`, `GREEN`) / NIR (`B8`, `B8A`, `B08`, `NIR`) |
| `.ndbi(swir=None, nir=None)` | Normalized Difference Built-Up Index | Auto-detects SWIR (`B11`, `SWIR`) / NIR (`B8`, `B8A`, `B08`, `NIR`) |
| `.nbr(nir=None, swir2=None)` | Normalized Burn Ratio | Auto-detects NIR (`B8`, `B8A`, `B08`) / SWIR-2 (`B12`, `SWIR2`) |
| `.evi(nir=None, red=None, blue=None)` | Enhanced Vegetation Index | Auto-detects NIR, Red, Blue |
| `.savi(nir=None, red=None, L=0.5)` | Soil Adjusted Vegetation Index | Auto-detects NIR, Red, Canopy factor L |

### Masking, QA Cloud Filtering & Radiometric Scaling

| Method | Signature | Description |
|---|---|---|
| `.mask_clouds(...)` | `mask_clouds(qa_band=None, method="auto", keep_clear=True) -> Image` | Masks clouds, cloud shadows, and cirrus using QA bands (supports HLS Fmask, Sentinel-2 SCL, and Landsat Collection 2 QA_PIXEL). |
| `.auto_scale(...)` | `auto_scale(scale=None, offset=None, dataset=None, exclude_bands=None) -> Image` | Applies radiometric scaling factors and offsets (e.g. 0.0001 for HLS/Sentinel-2; 0.0000275 and -0.2 for Landsat C2). QA bands are automatically protected from scaling. |
| `.clip(geometry)` | `clip(geometry: Geometry | Any) -> Image` | Clips image using polygon boundary; pixels outside geometry are set to `np.nan`. |
| `.updateMask(mask)` | `updateMask(mask: Image | np.ndarray) -> Image` | Retains pixels where mask is non-zero / True; others set to `np.nan`. |
| `.unmask(value=0.0)` | `unmask(value: float = 0.0) -> Image` | Replaces `nan` values with a constant number. |
| `.where(condition, value)` | `where(condition, value) -> Image` | Replaces pixels with `value` where `condition` is True. |
| `.mask()` | `mask() -> Image` | Returns single-band binary mask (1.0 for valid pixels, 0.0 for `nan`). |
| `.normalize(min_val=None, max_val=None)` | `normalize(...) -> Image` | Scales values to $[0.0, 1.0]$ using empirical min/max or explicit values. |
| `.clamp(low, high)` | `clamp(low: float, high: float) -> Image` | Restricts pixel values to `[low, high]`. |
| `.reproject(dst_crs, scale=None, resampling="bilinear")` | `reproject(...) -> Image` | Reprojects raster to target CRS with optional target pixel resolution scale. Preserves custom subclasses. |
| `.resample(scale_factor=1.0, method="bilinear")` | `resample(...) -> Image` | Resizes grid by `scale_factor` (>1 upsamples, <1 downsamples). Preserves custom subclasses. |

### Focal & Neighborhood Filters

Calculates spatial convolution filters across circular, gaussian, or square kernels:

| Method | Signature | Description |
|---|---|---|
| `.focal_mean(radius=3, kernel="circle")` | `focal_mean(...) -> Image` | Neighborhood mean filter (`circle`, `gaussian`, `square`). |
| `.focal_median(radius=3)` | `focal_median(...) -> Image` | Neighborhood median filter (removes salt-and-pepper noise). |
| `.focal_min(radius=3)` | `focal_min(...) -> Image` | Neighborhood minimum filter (erosion). |
| `.focal_max(radius=3)` | `focal_max(...) -> Image` | Neighborhood maximum filter (dilation). |
| `.focal_std(radius=3)` | `focal_std(...) -> Image` | Local standard deviation (texture analysis). |

### Terrain Analysis (DEM Operations)

Applicable to single-band Digital Elevation Models:

| Method | Returns | Description |
|---|---|---|
| `.slope()` | `Image` | Terrain slope in degrees ($0^\circ - 90^\circ$). |
| `.aspect()` | `Image` | Terrain aspect in degrees ($0^\circ - 360^\circ$ clockwise from North). |
| `.hillshade(azimuth=315.0, elevation=45.0)` | `Image` | 8-bit shaded relief illumination array ($0 - 255$). |

### Zonal Reductions & Regional Aggregations

```python
img.reduceRegion(reducer: Reducer, geometry: Geometry | None = None, scale: float | None = None) -> dict[str, float]
```

Aggregates all pixel values inside `geometry` (or the full image extent if `None`) using a [`Reducer`](core.md#reducer).

```python
stats = img.reduceRegion(gf.Reducer.mean(), geometry=study_aoi)
# Returns: {"B2": 450.2, "B3": 512.4, "B4": 490.1, "B8": 2810.5}
```

### Visualization & Cartographic Export

| Method | Description |
|---|---|
| `.visualize(vis_params=None)` | Renders an 8-bit RGB array `(H, W, 3)` suitable for PIL/Leaflet. Supports multi-band `{"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}` or single-band colormaps `{"bands": ["NDVI"], "palette": "RdYlGn"}`. |
| `.plot(bands=None, cmap="viridis", ...)` | Quick Matplotlib plot. |
| `.plot_map(**kwargs)` | Generates a publication-grade map complete with dynamic scale bar, 3D North arrow, and degree-formatted coordinates. |
| `.edit_layout(**kwargs)` | Launches the in-notebook interactive ArcGIS-style Layout Editor. |
| `.save(filepath, driver="GTiff", compress="lzw")` | Writes raster to disk as GeoTIFF. |
| `.to_numpy()` | Returns copy of internal 3D NumPy array. |
| `.to_xarray(name=None)` | Converts to `xarray.DataArray` with `.rio` geospatial accessor. |

---

## `ImageCollection`

```python
gf.ImageCollection(
    images: list[Image] | None = None,
    dataset_name: str | None = "CustomCollection",
    metadata: dict[str, Any] | None = None,
)
```

Manages sequences of `Image` granules across space and time.

!!! warning "String constructor gotcha"
    Calling `gf.ImageCollection("SENTINEL-2")` will treat the string as an iterable of 10 characters rather than loading a collection. To load collections by name, use `gf.ImageCollection.from_earthdata(...)` or `gf.ImageCollection.from_stac(...)`.

### Factory Methods (Cloud-Native & NASA Ingestion)

| Factory | Signature | Description |
|---|---|---|
| `from_stac` | `from_stac(collection, catalog="planetary-computer", geometry=None, start=None, end=None, limit=100, bands=None, resolution=None)` | Streams analysis-ready data lazily using STAC + `odc-stac` + Dask without downloading raw granules. |
| `from_earthdata` | `from_earthdata(dataset, region=None, start=None, end=None, limit=10, bands=None, cloud_cover=None, **kwargs)` | Searches NASA CMR / earthaccess, downloads granules to local disk, groups single-band files by granule, and returns an initialized collection. |

### Collection Filtering & Transformation

| Method | Signature | Description |
|---|---|---|
| `.filterBounds(geometry)` | `filterBounds(geom) -> ImageCollection` | Retains images whose spatial extent intersects `geometry`. |
| `.filterDate(start, end)` | `filterDate(start, end) -> ImageCollection` | Retains images whose acquisition date falls within `[start, end]`. |
| `.filter(filter_obj)` | `filter(filter_obj: Filter) -> ImageCollection` | Applies arbitrary declarative [`Filter`](core.md#filter). |
| `.filterMetadata(name, op, val)` | `filterMetadata(...) -> ImageCollection` | Filters on metadata attribute (e.g. `op="<="`, `val=20`). |
| `.filterQuality()` | `filterQuality() -> ImageCollection` | Retains images with cloud cover $\le 30\%$ and good quality flags. |
| `.select(bands)` | `select(bands: str | list[str]) -> ImageCollection` | Subsets bands across all images in the collection. |
| `.sort(prop, ascending=True)` | `sort(...) -> ImageCollection` | Sorts images by metadata property. |
| `.limit(count)` | `limit(count: int) -> ImageCollection` | Truncates collection to first $N$ images. |
| `.map(func)` | `map(func: Callable[[Image], Image]) -> ImageCollection` | Applies custom Python function to each image. |

### Compositing & Reductions

| Method | Description |
|---|---|
| `.mean()` | Pixel-wise temporal mean composite. |
| `.median()` | Pixel-wise temporal median composite. |
| `.min()`, `.max()` | Pixel-wise temporal minimum or maximum composite. |
| `.sum()` | Pixel-wise temporal sum composite. |
| `.mosaic()` | Stitches images across space, with later images layered on top. |
| `.qualityMosaic(quality_band)` | For each pixel, picks values from the image granule that has the maximum value in `quality_band` (e.g., greenest-pixel composite using `"NDVI"`). |

### 4D Datacubes & Xarray Export

| Method | Return Type | Description |
|---|---|---|
| `.to_xarray()` | `xarray.DataArray` | Stacks all images into a 4D datacube with dimensions `('time', 'band', 'y', 'x')`. |
| `.to_dataset()` | `xarray.Dataset` | Converts collection into an xarray Dataset where each band is an individual variable with dimensions `('time', 'y', 'x')`. |
