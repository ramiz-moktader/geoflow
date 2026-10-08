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

GeoFlow seamlessly integrates with NASA's `earthaccess` to let you securely search and download data right to your environment (like Google Colab), completely abstracted behind a GEE-like API:

```python
import geoflow as gf

# 1. Login to NASA Earthdata (prompts for credentials in Colab)
gf.auth.login(strategy="interactive")
aoi = gf.Rectangle([91.7, 22.2, 92.0, 22.5])

# 2. Seamlessly search, download, and load into a GEE-like ImageCollection!
collection = gf.ImageCollection.from_earthdata(
    dataset="SENTINEL-2", 
    region=aoi, 
    start="2023-01-01", 
    end="2023-01-15",
    bands=["B04", "B8A"]
)

# 3. Calculate median composite & NDVI
ndvi = collection.median().ndvi(nir="B8A", red="B04")

# 4. Generate a publication-ready map (saves directly to disk)
ndvi.plot_carto_map(
    cmap="RdYlGn", vmin=-0.2, vmax=0.8,
    title="Vegetation Index (NDVI) of Chattogram",
    colorbar_label="NDVI",
    save_path="./downloads/chattogram_ndvi_map.png"
)
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
