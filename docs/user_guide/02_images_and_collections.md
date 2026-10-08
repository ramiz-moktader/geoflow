# Images and Collections User Guide

GeoFlow brings the ease of Google Earth Engine's declarative image processing to the Python data science ecosystem.

## `gf.Image`

The `Image` object is the core raster data structure.

### Instantiation
You can load an image from a local file, a NumPy array, or an xarray DataArray.

```python
import geoflow as gf

img = gf.Image("sentinel2.tif")
```

### Band Math & Indices
Calculations are lazy and chainable.

```python
# Select specific bands
b4 = img.select("B4")
b8 = img.select("B8")

# Basic math
ndvi_manual = (b8 - b4) / (b8 + b4)

# Or use built-in indices
ndvi = img.ndvi(nir="B8", red="B4")
```

### Xarray Integration
Unlike GEE, you can effortlessly drop into `xarray` for advanced multi-dimensional processing.

```python
da = img.to_xarray()
# da is an xarray.DataArray with dims ('band', 'y', 'x')
# ... do custom xarray processing ...

# Convert back to GeoFlow Image
processed_img = gf.Image.from_xarray(da)
```

## `gf.ImageCollection`

An `ImageCollection` handles stacks of images (spatiotemporal datacubes).

### Filtering
You can filter large collections dynamically before processing.

```python
collection = (
    gf.ImageCollection("LANDSAT-8")
    .filterBounds(aoi)
    .filterDate("2020-01-01", "2020-12-31")
)
```

### Compositing and Temporal Reduction
Combine timeseries of images into single composite images.

```python
# Compute the median pixel value over time
median_composite = collection.median()

# Create a cloud-free mosaic using a quality band
quality_mosaic = collection.qualityMosaic("NDVI")
```

### 4D Datacubes
You can convert an entire collection into a 4-dimensional `xarray.DataArray` (time, band, y, x).

```python
cube = collection.to_xarray()
```
