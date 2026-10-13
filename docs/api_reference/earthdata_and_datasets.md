# Earthdata & Datasets API Reference (`geoflow.earthdata`, `geoflow.auth`, `geoflow.datasets`)

GeoFlow provides integrated pipelines for NASA Earthdata authentication, CMR discovery, parallel downloading, cloud-native STAC streaming, and mission-specific adapters such as NASA GEDI LiDAR.

| Module | Purpose |
|---|---|
| [`geoflow.auth`](#authentication-geoflowauth) | NASA Earthdata Login session management |
| [`geoflow.earthdata.search`](#data-discovery-geoflowearthdatasearch) | Common Metadata Repository (CMR) granule queries |
| [`geoflow.earthdata.downloader`](#resilient-downloader-geoflowearthdatadownloader) | Multi-threaded download engine with resume & caching |
| [`geoflow.earthdata.stac`](#cloud-native-stac-streaming-geoflowearthdatastac) | Direct raster streaming from AWS & Planetary Computer |
| [`geoflow.datasets.gedi`](#gedi-lidar-adapter-geoflowdatasetsgedi) | NASA GEDI Level-2A, 2B, and 4A footprint parser |

---

## Authentication (`geoflow.auth`)

```python
from geoflow.auth import login, logout, get_session, is_authenticated, get_s3_credentials
```

Handles NASA Earthdata Login authentication required to download data from DAACs (LP DAAC, ORNL DAAC, NSIDC, etc.) and in-region AWS S3 cloud direct access.

| Function | Signature | Description |
|---|---|---|
| `login` | `login(strategy: str = "interactive", persist: bool = True) -> bool` | Authenticates with Earthdata. Priority: (1) `earthaccess` backend, (2) `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` env vars, (3) `~/.netrc` (or `~/_netrc` on Windows), (4) Interactive terminal / Colab prompt. |
| `logout` | `logout() -> None` | Clears active in-memory session. |
| `get_session` | `get_session() -> Any` | Returns active session object or credential dict. |
| `is_authenticated` | `is_authenticated() -> bool` | Returns `True` if active session is authenticated. |
| `get_s3_credentials` | `get_s3_credentials(daac: str = "LPDAAC") -> dict` | Obtains temporary direct AWS S3 credentials for in-region NASA DAAC buckets (us-west-2) for zero-egress cloud streaming. |

```python
import geoflow as gf

# Login interactively or via .netrc
gf.auth.login(strategy="interactive")
print(f"Authenticated: {gf.auth.is_authenticated()}")

# In AWS us-west-2 cloud:
s3_creds = gf.auth.get_s3_credentials("LPDAAC")
```

---

## Data Discovery (`geoflow.earthdata.search`)

```python
from geoflow.earthdata import search, search_data, search_collections, open_granule
```

Queries NASA's Common Metadata Repository (CMR) API or delegates to `earthaccess` backend if installed.

| Function | Signature | Description |
|---|---|---|
| `search` / `search_data` | `search(dataset, region=None, start=None, end=None, limit=50, cloud_cover=None, **kwargs) -> list[dict]` | Searches NASA Earthdata granules with spatial bounds and temporal filters. |
| `search_collections` | `search_collections(query=None, keyword=None, limit=20, **kwargs) -> list[dict]` | Discovers NASA CMR collections / concept IDs by title or science keyword. |
| `open_granule` | `open_granule(granule, provider=None) -> Any` | Opens and streams a NASA Earthdata granule file directly into memory without full local disk download. |

```python
search(
    dataset: str,
    region: Geometry | list[float] | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 50,
    cloud_cover: float | tuple[float, float] | None = None,
    **kwargs: Any
) -> list[dict[str, Any]]
```

### Pre-configured Dataset Short Names

GeoFlow maps user-friendly names to official NASA DAAC short names via `DATASET_SHORTNAMES`:

| Alias | NASA CMR Short Name | Sensor / Description |
|---|---|---|
| `"SENTINEL-2"` | `HLSS30` | Harmonized Landsat-Sentinel (Sentinel-2 MSI, 30m) |
| `"LANDSAT-8"` | `HLSL30` | Harmonized Landsat-Sentinel (Landsat-8/9 OLI, 30m) |
| `"GEDI/L2A"` | `GEDI02_A` | GEDI Elevation & Height Metrics (25m footprint) |
| `"GEDI/L2B"` | `GEDI02_B` | GEDI Canopy Cover & Vertical Profile Metrics |
| `"GEDI/L4A"` | `GEDI04_A` | GEDI Aboveground Biomass Density (AGBD) |
| `"MODIS"` | `MOD09GA` | MODIS Surface Reflectance Daily L2G |
| `"NASADEM"` | `NASADEM_HGT` | NASA DEM Global 1 arc second (~30m) |

```python
aoi = gf.Rectangle([91.7, 22.2, 92.0, 22.5])
granules = gf.earthdata.search(
    dataset="SENTINEL-2",
    region=aoi,
    start="2023-01-01",
    end="2023-01-31",
    cloud_cover=20.0,
    limit=5
)
for g in granules:
    print(g["title"], g["time_start"])
```

---

## Resilient Downloader (`geoflow.earthdata.downloader`)

```python
from geoflow.earthdata import download
```

Downloads granules in parallel with disk checks and resume support.

```python
download(
    granules: list[dict[str, Any]] | list[str] | Any,
    output_dir: str | Path = "./data",
    workers: int = 4,
    resume: bool = True,
    cache: bool = True
) -> list[Path]
```

### Key Features
- **Disk Space Pre-check**: Verifies target volume has at least 500 MB free via `check_disk_space()`.
- **Parallel Threads**: Downloads files concurrently via Python's `ThreadPoolExecutor`.
- **Chunked Resumption**: Writes `.part` temporary files and uses HTTP `Range` headers to resume interrupted downloads.
- **Caching**: If a destination file already exists and has non-zero size, it is retained without re-downloading.

---

## Cloud-Native STAC Streaming (`geoflow.earthdata.stac`)

```python
from geoflow.earthdata.stac import search_stac, load_stac_collection
```

Allows analysis of cloud-optimized rasters (COG / Zarr) stored in AWS Open Data and Microsoft Planetary Computer without downloading gigabytes of raw data.

### `search_stac`

```python
search_stac(
    collection: str,
    catalog: str = "planetary-computer",
    geometry: Geometry | Any | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 100,
    query: dict[str, Any] | None = None
) -> pystac.ItemCollection
```

Supported catalog endpoints:
- `"planetary-computer"`: `https://planetarycomputer.microsoft.com/api/stac/v1`
- `"earth-search"`: `https://earth-search.aws.element84.com/v1`

### `load_stac_collection`

```python
load_stac_collection(
    items: pystac.ItemCollection,
    bands: list[str] | None = None,
    resolution: int | float | None = None,
    bbox: list[float] | None = None,
    crs: str = "EPSG:4326",
    chunks: dict[str, int] = {"x": 1024, "y": 1024, "time": 1},
    patch_url: Any | None = None
) -> ImageCollection
```

Uses `odc.stac` to load cloud assets into lazy, Dask-backed xarray slices and wraps them into a GeoFlow `ImageCollection`.

---

## GEDI LiDAR Adapter (`geoflow.datasets.gedi`)

```python
gf.GEDI(
    level: str = "L2A",
    files: list[str] | None = None,
    metadata: dict[str, Any] | None = None
)
```

Specialized adapter for NASA Global Ecosystem Dynamics Investigation (GEDI) spaceborne waveform LiDAR data.

### Classmethods & Factory

```python
GEDI.load(
    level: str = "L2A",
    region: Geometry | Any | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 10,
    output_dir: str = "./data",
    **kwargs: Any
) -> GEDI
```

Executes `search_data`, downloads the resulting HDF5 (`.h5`) files, and returns an initialized `GEDI` instance with active bounding box and date filters.

### Pipeline Methods

| Method | Description |
|---|---|
| `.filterBounds(aoi)` | Filters shots to bounding box of `aoi`. |
| `.filterDate(start, end)` | Sets temporal filter bounds. |
| `.filterQuality()` | Activates quality mask: retains `quality_flag == 1`, `degrade_flag == 0`, and `sensitivity >= 0.90`. |
| `.select(variables)` | Selects metrics to extract (e.g. `["rh98", "rh50", "sensitivity", "agbd"]`). |
| `.to_geodataframe()` | Parses live HDF5 files using `h5py` across all laser beams (`BEAM0000`, `BEAM0001`, etc.) and extracts georeferenced point footprints into a GeoDataFrame. |
| `.to_feature_collection()` (alias: `.to_fc()`) | Wraps the extracted footprint points into a [`FeatureCollection`](vector.md#featurecollection). |

### Extracted Footprint Columns

When parsing HDF5 files or working with GEDI point collections:
- `shot_number`: Unique 64-bit laser shot ID.
- `beam`: Laser beam identifier (e.g., `"BEAM0101"`).
- `latitude`, `longitude`, `elevation`: Ground return position.
- `quality_flag`, `sensitivity`: Signal quality indicators.
- `rh50`, `rh98`, `rh100`: Relative height percentiles (canopy top height).
- `agbd`: Aboveground Biomass Density in $Mg / ha$ (for Level-4A).
- `geometry`: `shapely.geometry.Point(longitude, latitude)`.

!!! note "Offline & Synthetic Fallback Behavior"
    If `.to_geodataframe()` is called on a `GEDI` instance that does **not** contain downloaded `.h5` files, GeoFlow generates a synthetic distribution of realistic GEDI footprints within the AOI bounds. This ensures pipeline prototyping, tests, and student labs can run offline without requiring active NASA Earthdata credentials.
