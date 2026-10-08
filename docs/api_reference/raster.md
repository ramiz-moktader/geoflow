# Raster API Reference

The `geoflow.raster` module provides `Image` and `ImageCollection` classes, mirroring the behavior of Google Earth Engine while executing locally via `xarray` and `rasterio`.

## `gf.Image`

Represents a single multi-band raster image.

### Initialization
```python
gf.Image(data, transform=None, crs=None, band_names=None)
```
- `data`: Can be a file path (`str`), a numpy array, or an `xarray.DataArray`.

### Core Methods
- `.select(bands)`: Selects one or more bands by name or index.
- `.clip(geometry)`: Clips the image to a `Geometry` or `Feature`.
- `.rename(names)`: Renames the image bands.
- `.mask(mask_image)`: Applies a boolean mask to the image.

### Band Math & Expressions
- `.add(other)`, `.subtract(other)`, `.multiply(other)`, `.divide(other)`
- `.normalizedDifference(bands)`: Computes `(band1 - band2) / (band1 + band2)`.
- `.expression(expr, context)`: Evaluates a string expression (e.g., `"(B5 - B4) / (B5 + B4)"`).

### Spectral Indices
- `.ndvi()`, `.ndwi()`, `.ndbi()`, `.evi()`, `.savi()`, `.nbr()`

### Ecosystem Integration
- `.to_xarray()`: Converts the image to an `xarray.DataArray` with spatial coordinates.
- `.from_xarray(da)`: Constructs an `Image` from an `xarray.DataArray`.
- `.plot_map(**kwargs)`: Renders a publication-ready cartographic map.

---

## `gf.ImageCollection`

Represents a stack or timeseries of `Image` objects.

### Core Methods
- `.filterBounds(geometry)`: Filters the collection to images intersecting the geometry.
- `.filterDate(start, end)`: Filters the collection by time range.
- `.select(bands)`: Selects bands across all images in the collection.

### Compositing and Reduction
- `.mosaic()`: Mosaics the collection (last on top).
- `.mean()`, `.median()`, `.min()`, `.max()`: Temporal reduction.
- `.qualityMosaic(quality_band)`: Composites pixels based on the maximum value of a specific quality band.

### Ecosystem Integration
- `.to_xarray()`: Converts the collection into a 4D `xarray.DataArray` (`time`, `band`, `y`, `x`).
- `.to_dataset()`: Converts the collection into an `xarray.Dataset` (where each band is a variable).
