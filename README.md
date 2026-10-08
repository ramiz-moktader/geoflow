# 🌍 GeoFlow (`gf`)

> **A Google Earth Engine-like Earth Observation, GIS, Spatial Statistics, and Geospatial ML Framework in Python.**

```python
import geoflow as gf

# GEE-like declarative data discovery & acquisition
aoi = gf.Geometry.from_file("study_area.geojson")

gedi = (
    gf.GEDI("L2A")
      .filterBounds(aoi)
      .filterDate("2021-01-01", "2023-12-31")
      .filterQuality()
      .select(["rh98", "sensitivity"])
)

# GEE-like raster compositing & index calculation
image = (
    gf.ImageCollection("SENTINEL-2")
      .filterBounds(aoi)
      .filterDate("2022-01-01", "2022-12-31")
      .median()
      .clip(aoi)
)

ndvi = image.ndvi()
```

---

## 🌟 Why GeoFlow?

Google Earth Engine revolutionized cloud-based remote sensing with its declarative syntax (`Image`, `ImageCollection`, `FeatureCollection`, `Filter`, `Reducer`, `Map`). However, serious geospatial ML and statistical research require the flexibility of the Python scientific ecosystem (**PyTorch, Scikit-Learn, SciPy, PySAL, GeoPandas, Rasterio**).

**GeoFlow** unites the two worlds:
1. **GEE-like Developer Experience**: Write code with familiar `.filterBounds()`, `.filterDate()`, `.select()`, `.clip()`, `.reduceRegion()`, and `Map.addLayer()`.
2. **First-Class Spatial Statistics**: Native Global Moran's I, Local Moran's I (LISA: HH, HL, LH, LL), Geary's C, semivariograms, and Spatial Effective Sample Size (ESS) to prevent pseudoreplication.
3. **Leakage-Safe Spatial ML**: Built-in spatial blocking (square, grid, hexagonal), `BufferedSpatialCV` with exclusionary dead-zones, and automated leakage diagnostics (`leakage_report`).
4. **Deep Learning Ready**: `PatchDataset` with georeferenced bounding boxes and PyTorch `DataLoader` batch generators.
5. **Interactive & Publication Visualizations**: Fast interactive Leaflet maps (`gf.Map()`) with zero mandatory external web dependencies, plus Nature/Science publication themes and LaTeX tables.

---

## 🚀 Quickstart

### Installation

```bash
pip install -e .
```

### Complete End-to-End Workflow

```python
import geoflow as gf

# 1. Define Area of Interest (AOI)
aoi = gf.Rectangle([91.7, 22.2, 92.0, 22.5], crs="EPSG:4326")

# 2. Acquire & Process GEDI LiDAR
gedi = (
    gf.GEDI("L2A")
      .filterBounds(aoi)
      .filterQuality()
      .select(["rh98", "sensitivity"])
)
gdf = gedi.to_geodataframe()

# 3. Interactive Visualization
m = gf.Map()
m.centerObject(aoi)
m.addLayer(aoi, name="Study Boundary")
m.addLayer(gedi, name="GEDI Canopy Height")
m.save("interactive_map.html")

# 4. Spatial Autocorrelation & Clustering
moran = gf.spatial.morans_i(gdf, value="rh98", k=8)
print(moran.summary())

local_lisa = gf.spatial.local_morans_i(gdf, value="rh98", k=8)
# local_lisa['cluster'] contains: 'HH', 'LL', 'HL', 'LH', or 'not significant'

# 5. Spatial Blocking (Square or Hexagonal)
blocks = gf.spatial.block(aoi, size=5000, unit="meters", shape="hexagonal")

# 6. Leakage-Safe Cross-Validation
cv = gf.validation.BufferedSpatialCV(buffer_distance=1000, blocks=blocks, n_splits=5)
train_idx, val_idx = next(cv.split(gdf))

# 7. Audit for Data Leakage
report = gf.validation.leakage_report(gdf.iloc[train_idx], gdf.iloc[val_idx], buffer_threshold=1000)
print(report.summary())

# 8. ML Patch Extraction & PyTorch DataLoader
image = gf.Image("sentinel2_chattogram.tif")
dataset = gf.ml.PatchDataset(image=image, patch_size=64, stride=32)
train_loader = dataset.dataloader(batch_size=32, shuffle=True)
```

---

## 🧩 GEE Parity Reference

| Google Earth Engine (GEE) | GeoFlow (`gf`) |
|---|---|
| `ee.Image` | `gf.Image` (or `gf.Raster`) |
| `ee.ImageCollection` | `gf.ImageCollection` |
| `ee.Geometry.Point([x, y])` | `gf.Point([x, y])` |
| `ee.Geometry.Polygon(coords)` | `gf.Polygon(coords)` |
| `ee.Geometry.Rectangle([w, s, e, n])` | `gf.Rectangle([minx, miny, maxx, maxy])` |
| `ee.Feature` | `gf.Feature` |
| `ee.FeatureCollection` | `gf.FeatureCollection` (or `gf.Vector`) |
| `ee.Filter.eq(k, v)` | `gf.Filter.eq(k, v)` |
| `ee.Filter.date(start, end)` | `gf.Filter.date(start, end)` |
| `ee.Filter.bounds(geometry)` | `gf.Filter.bounds(geometry)` |
| `ee.Reducer.mean()` | `gf.Reducer.mean()` |
| `ee.Reducer.median()` | `gf.Reducer.median()` |
| `image.normalizedDifference(['B8', 'B4'])` | `image.normalizedDifference(['B8', 'B4'])` |
| `image.expression(expr, vars)` | `image.expression(expr, vars)` |
| `image.clip(geometry)` | `image.clip(geometry)` |
| `image.reduceRegion(reducer, aoi)` | `image.reduceRegion(reducer, aoi)` |
| `Map.centerObject(aoi)` | `m.centerObject(aoi)` |
| `Map.addLayer(image, vis_params)` | `m.addLayer(image, vis_params)` |

---

---

## 🐍 Native Python Subclassing & Class Propagation

GeoFlow is designed to be deeply extensible. You can subclass `gf.Image`, `gf.ImageCollection`, or `gf.FeatureCollection` with your own domain methods and attributes—and GeoFlow's fluent pipelines (`.select()`, `.add()`, `.ndvi()`, `.clip()`, etc.) will **automatically propagate your custom subclass** and preserve custom instance attributes!

```python
class ForestryImage(gf.Image):
    def __init__(self, *args, site_id="SITE-01", **kwargs):
        super().__init__(*args, **kwargs)
        self.site_id = site_id

    def canopy_mask(self, threshold=0.4):
        return (self.ndvi() > threshold).rename(["canopy_mask"])

# Subclass and custom attributes persist through operations
img = ForestryImage("sentinel2.tif", site_id="CHATTOGRAM-HILL-TRACTS")
processed = img.select(["B4", "B8"]).canopy_mask()
assert isinstance(processed, ForestryImage)
assert processed.site_id == "CHATTOGRAM-HILL-TRACTS"

# Fluent functional composition via .pipe()
result = img.pipe(lambda x: x * 1.5).canopy_mask()
```

---

## 🧊 Modern `xarray` & `rioxarray` Integration

GeoFlow provides first-class bidirectional conversion with `xarray` and `rioxarray`:

```python
# 1. Convert Image to xarray.DataArray (with coordinates, CRS, and affine transform)
da = img.to_xarray()
# da has dims ('band', 'y', 'x') with coordinates: x (lon), y (lat), band

# 2. Convert xarray.DataArray back into geoflow.Image
reconstructed_img = gf.Image.from_xarray(da)

# 3. Create a 4D spatiotemporal datacube from an ImageCollection
datacube_4d = collection.to_xarray()
# dims: ('time', 'band', 'y', 'x')

# 4. Convert ImageCollection to xarray.Dataset (each band as a variable)
dataset = collection.to_dataset()
```

---

## 🗺️ Publication Cartographic Map Maker

Generate publication-ready maps with an authentic **North Arrow**, an accurate **dynamic Scale Bar in km**, **Lat/Long Graticule in Degrees** (e.g. `22.5° N`, `91.8° E`), and **Legends/Colorbars** in a single line of code.

### Flexible Graticule Orientation, 4-Sided Coordinates & Typography
Customize graticule coordinates on all four sides with vertical or horizontal orientations and custom typography:

```python
# 1. True Color RGB map with 4-sided coordinates, vertical latitudes & custom typography
img.plot_map(
    bands=["B4", "B3", "B2"],
    title="Chattogram Region — Sentinel-2 L2A",
    subtitle="Datum: WGS84 | Resolution: 10m",
    
    # 4-Sided Display: Show coordinates on top and right as well
    show_top=True,
    show_right=True,
    show_bottom=True,
    show_left=True,
    
    # Orientation: Vertical latitude labels (rotated 90°), Horizontal longitudes
    lat_orientation="vertical",
    lon_orientation="horizontal",
    
    # Typography & Styling
    tick_fontsize=9,
    tick_fontfamily="serif",
    tick_fontweight="bold",
    tick_color="#1e3a8a",
    degree_precision=3,
    
    # Clean corners: prevents first/last lat/long coordinate text from colliding
    prune_corners=True,
    
    # Cartographic elements
    north_arrow=True,
    scale_bar=True,
    grid=True,
    save_path="sentinel_map_4sided.png",
    dpi=300
)

# 2. Vegetation Index (NDVI) map with continuous colorbar
img.ndvi().plot_map(
    cmap="RdYlGn",
    vmin=-0.1,
    vmax=0.8,
    title="Vegetation Index (NDVI)",
    colorbar_label="NDVI",
    lat_orientation="vertical",
    show_top=True,
    show_right=True,
    north_arrow=True,
    scale_bar=True,
    save_path="ndvi_map.png"
)
```

---

## 🔬 Spatial Statistics & ML Extensions

GeoFlow goes beyond GEE by adding native spatial statistics, inferential testing, and spatial ML diagnostics:

- **Global Spatial Autocorrelation**: `gf.spatial.morans_i()`, `gf.spatial.gearys_c()`
- **Local Spatial Autocorrelation (LISA)**: `gf.spatial.local_morans_i()`
- **Spatial Weights**: `gf.spatial.knn()`, `gf.spatial.distance()`
- **Semivariograms**: `gf.spatial.variogram()`
- **Spatial Effective Sample Size (ESS)**: `gf.stats.effective_sample_size()` (combats pseudoreplication)
- **Spatial Hypothesis Testing**: `gf.stats.test()` (Welch t-test, Mann-Whitney U, ANOVA with Cohen's d / $\eta^2$)
- **Spatial Cross-Validation**: `gf.validation.SpatialCV`, `gf.validation.BufferedSpatialCV`
- **Leakage Auditing**: `gf.validation.leakage_report()`
- **Spatial Error Diagnostics**: `gf.validation.evaluate_spatial()` (RMSE, R², and residual Moran's I)
- **Publication Figures & Tables**: `gf.viz.theme("nature")`, `gf.viz.table().to_latex()`

---

## 📜 License

Apache License 2.0. Built for scientific reproducibility and open geospatial research.
