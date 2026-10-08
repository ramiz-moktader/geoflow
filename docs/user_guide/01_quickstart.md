# Quickstart

Welcome to GeoFlow! This guide will get you up and running with the core concepts.

## Installation

```bash
pip install git+https://github.com/ramiz-moktader/geoflow.git
```

## First Steps

GeoFlow's syntax is inspired by Google Earth Engine (GEE). You can chain operations seamlessly.

### Working with Collections

```python
import geoflow as gf

# Define an Area of Interest
aoi = gf.Rectangle([91.7, 22.2, 92.0, 22.5], crs="EPSG:4326")

# Filter an ImageCollection
collection = (
    gf.ImageCollection("SENTINEL-2")
    .filterBounds(aoi)
    .filterDate("2023-01-01", "2023-12-31")
)

# Create a composite image
median_img = collection.median().clip(aoi)

# Calculate a Vegetation Index
ndvi = median_img.ndvi()
```
### Authenticating and Downloading NASA Earthdata

GeoFlow seamlessly integrates with NASA's `earthaccess` to let you securely search and download data right to your environment (like Google Colab):

```python
import geoflow as gf
import earthaccess
import os

# 1. Login to NASA Earthdata (prompts for credentials)
gf.auth.login(strategy="interactive")

# 2. Search for Sentinel-2 HLS data
aoi = gf.Rectangle([91.7, 22.2, 92.0, 22.5])
results = gf.earthdata.search(
    dataset="SENTINEL-2",
    region=aoi,
    start="2023-01-01",
    end="2023-01-15",
    limit=2
)

# 3. Flawless Direct Download
if len(results) > 0:
    os.makedirs("./downloads", exist_ok=True)
    earthaccess.download(results, local_path="./downloads")
```


### Extensibility

Unlike GEE, GeoFlow operates locally or on your own clusters, meaning you have full access to standard Python tools like `xarray` and `geopandas`.

```python
# Convert to xarray DataArray
da = median_img.to_xarray()
```

GeoFlow also preserves your custom Python subclasses through method chains:

```python
class MyCustomImage(gf.Image):
    pass

img = MyCustomImage("data.tif")
result = img.ndvi()
assert isinstance(result, MyCustomImage)  # Type is preserved!
```
